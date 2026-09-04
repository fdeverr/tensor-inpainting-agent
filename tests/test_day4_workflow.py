import json
from pathlib import Path

import numpy as np
from PIL import Image

from inpainting_research_agent.workflow_day4 import (
    Day4WorkflowConfig,
    run_day4_workflow,
)
from inpainting_research_agent.workflow_day5 import (
    Day5WorkflowConfig,
    run_day5_workflow,
)


def _write_small_image(path: Path) -> None:
    height, width = 12, 18
    y, x = np.indices((height, width), dtype=np.float32)
    base = (x + 0.7 * y) / ((width - 1) + 0.7 * (height - 1))
    image = np.stack((base, 0.85 * base + 0.05, 0.7 * base + 0.1), axis=-1)
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def test_day4_fallback_selects_and_trains_a_valid_method(tmp_path):
    image_path = tmp_path / "image.png"
    _write_small_image(image_path)
    state = run_day4_workflow(
        Day4WorkflowConfig(
            image_path=str(image_path),
            output_dir=str(tmp_path / "outputs"),
            image_size=None,
            missing_rate=0.3,
            seed=7,
            max_steps=12,
            validation_interval=3,
            patience=10,
            device="cpu",
            llm_mode="off",
            retrieval_top_k=5,
        )
    )

    assert state["stage"] == "COMPLETED"
    plan = state["results"]["method_plan"]
    assert plan["method"] in {"matrix", "cp", "tucker"}
    assert plan["selection_mode"] == "deterministic_fallback"
    assert state["selected_model"] == plan["method"]
    assert state["results"]["training"]["model_name"] == plan["method"]
    assert state["results"]["selector_diagnostics"]["llm_used"] is False

    method_plan = json.loads(Path(state["artifacts"]["method_plan"]).read_text())
    assert method_plan["ground_truth_provided_to_selector"] is False
    assert method_plan["final_metrics_provided_to_selector"] is False
    assert Path(state["artifacts"]["retrieval_result"]).is_file()

    trace_text = Path(state["artifacts"]["trace_jsonl"]).read_text()
    assert '"event": "method_selection"' in trace_text
    assert "evaluation_ground_truth" not in trace_text

    day5_state = run_day5_workflow(
        Day5WorkflowConfig(
            base_run_dir=state["artifacts"]["run_dir"],
            candidate_root=str(tmp_path / "algorithms" / "candidates"),
            output_dir=str(tmp_path / "outputs"),
            llm_mode="off",
            smoke_timeout_seconds=10,
        )
    )
    assert day5_state["stage"] == "VALIDATED"
    assert day5_state["validation"]["eligible_for_training"] is True
    manifest = json.loads(Path(day5_state["artifacts"]["manifest"]).read_text())
    assert manifest["validation_status"] == "validated"
    assert manifest["eligible_for_training"] is True
