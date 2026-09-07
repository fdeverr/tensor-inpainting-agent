from pathlib import Path

import numpy as np
from PIL import Image

from inpainting_research_agent.benchmark import aggregate_cases
from inpainting_research_agent.workflow_full import (
    FullWorkflowConfig,
    run_full_workflow,
)


def _write_image(path: Path) -> None:
    height, width = 12, 16
    y, x = np.indices((height, width), dtype=np.float32)
    base = (x + y) / ((width - 1) + (height - 1))
    image = np.stack((base, 0.8 * base + 0.1, 0.6 * base + 0.2), axis=-1)
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)


def test_one_command_workflow_writes_report_and_best_image(tmp_path):
    image_path = tmp_path / "input.png"
    _write_image(image_path)
    state = run_full_workflow(
        FullWorkflowConfig(
            image_path=str(image_path),
            output_dir=str(tmp_path / "outputs"),
            candidate_root=str(tmp_path / "candidates"),
            approved_root=str(tmp_path / "approved"),
            mask_type="block",
            missing_rate=0.3,
            seed=11,
            image_size=None,
            method_max_steps=5,
            fair_max_steps=5,
            tuning_trials=1,
            max_improvement_rounds=1,
            validation_interval=1,
            patience=5,
            device="cpu",
            llm_mode="off",
        )
    )

    assert state["stage"] == "COMPLETED"
    assert set(state["child_runs"]) == {"day4", "day5", "day6"}
    assert Path(state["artifacts"]["best_completion"]).is_file()
    report_path = Path(state["artifacts"]["report"])
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "最终指标" in report
    assert "证据边界" in report
    assert len(state["method_results"]) == 3


def test_benchmark_aggregation_reports_mean_std_and_failures():
    method = {
        "role": "tensor_baseline",
        "metrics": {"missing_psnr": 10.0, "composite_ssim": 0.5},
        "runtime_seconds": 2.0,
        "parameter_count": 100,
    }
    cases = [
        {
            "status": "completed",
            "candidate_accepted": True,
            "method_results": [method],
        },
        {
            "status": "completed",
            "candidate_accepted": False,
            "method_results": [
                {
                    **method,
                    "metrics": {"missing_psnr": 12.0, "composite_ssim": 0.7},
                }
            ],
        },
        {"status": "failed"},
    ]
    result = aggregate_cases(cases)
    tensor = result["methods_by_role"]["tensor_baseline"]
    assert result["failed_cases"] == 1
    assert result["candidate_acceptance_count"] == 1
    assert tensor["mean_missing_psnr"] == 11.0
    assert tensor["std_missing_psnr"] == 1.0
