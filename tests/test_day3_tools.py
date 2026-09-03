import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inpainting_research_agent.agent_tools import build_research_tool_registry
from inpainting_research_agent.agent_tools.framework import ToolStatus
from inpainting_research_agent.workflow import (
    Day3Workflow,
    Day3WorkflowConfig,
    WorkflowExecutionError,
    run_day3_workflow,
)


def _write_small_image(path: Path) -> None:
    height, width = 12, 16
    y, x = np.indices((height, width), dtype=np.float32)
    image = np.stack(
        (
            x / (width - 1),
            y / (height - 1),
            0.2 + 0.6 * (x + y) / (width + height - 2),
        ),
        axis=-1,
    )
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)


def _candidate():
    return [
        {
            "hyperparameters": {
                "rank_h": 3,
                "rank_w": 3,
                "rank_c": 2,
                "init_scale": 0.15,
            },
            "learning_rate": 0.04,
        }
    ]


def test_registry_exposes_six_structured_tools():
    registry = build_research_tool_registry()
    assert registry.list_tools() == [
        "analyze_image",
        "run_interpolation",
        "tune_tensor_model",
        "train_tensor_model",
        "evaluate_reconstruction",
        "compare_experiments",
    ]
    for tool in registry.get_all_tools():
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == tool.name


def test_tool_registry_returns_structured_error_with_timing(tmp_path):
    registry = build_research_tool_registry()
    response = registry.execute_tool(
        "analyze_image",
        {
            "run_id": "bad-run",
            "image_path": str(tmp_path / "missing.png"),
            "run_dir": str(tmp_path / "run"),
            "mask_type": "block",
            "missing_rate": 0.4,
            "seed": 1,
        },
    )
    assert response.status == ToolStatus.ERROR
    assert response.error_info["code"] == "INVALID_PARAM"
    assert response.stats["time_ms"] >= 0
    assert response.context["tool_name"] == "analyze_image"


def test_day3_workflow_runs_all_tools_and_preserves_gt_boundary(tmp_path):
    image_path = tmp_path / "image.png"
    _write_small_image(image_path)
    state = run_day3_workflow(
        Day3WorkflowConfig(
            image_path=str(image_path),
            output_dir=str(tmp_path / "outputs"),
            image_size=None,
            missing_rate=0.3,
            seed=19,
            max_steps=20,
            validation_interval=5,
            patience=10,
            device="cpu",
            candidates=_candidate(),
        )
    )

    assert state["stage"] == "COMPLETED"
    assert state["last_successful_stage"] == "COMPLETED"
    assert state["results"]["selected_trial"]["best_step"] >= 1
    profile = state["results"]["image_profile"]
    assert profile["missing_component_count"] >= 1
    assert 0.0 < profile["largest_missing_component_image_ratio"] < 1.0
    assert 0.0 <= profile["visible_mean_absolute_channel_correlation"] <= 1.0
    assert 0.0 <= profile["visible_local_smoothness_score"] <= 1.0
    assert profile["visible_high_frequency_energy_ratio"] >= 0.0
    assert state["results"]["training"]["observed_pixels_used"] == round(
        12 * 16 * 0.7
    )
    assert state["results"]["comparison"]["winner"] in {
        "baseline",
        "candidate",
        "tie",
    }
    for artifact_path in state["artifacts"].values():
        assert Path(artifact_path).exists()

    tuning = json.loads(Path(state["artifacts"]["tuning_result"]).read_text())
    assert tuning["ground_truth_used"] is False
    training = json.loads(
        Path(state["artifacts"]["tensor_training_result"]).read_text()
    )
    assert training["ground_truth_used"] is False

    trace_path = Path(state["artifacts"]["trace_jsonl"])
    trace_text = trace_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in trace_text.splitlines()]
    tool_order = [
        event["payload"]["tool_name"]
        for event in events
        if event["event"] == "tool_call"
    ]
    assert tool_order == [
        "analyze_image",
        "run_interpolation",
        "tune_tensor_model",
        "train_tensor_model",
        "evaluate_reconstruction",
        "evaluate_reconstruction",
        "compare_experiments",
    ]
    assert "evaluation_ground_truth" not in trace_text


def test_failure_keeps_last_successful_stage(tmp_path):
    image_path = tmp_path / "image.png"
    _write_small_image(image_path)
    workflow = Day3Workflow(
        Day3WorkflowConfig(
            image_path=str(image_path),
            output_dir=str(tmp_path / "outputs"),
            image_size=None,
            max_steps=5,
            validation_interval=1,
            device="cpu",
            candidates=[
                {
                    "hyperparameters": {"unknown_argument": 1},
                    "learning_rate": 0.03,
                }
            ],
        )
    )

    with pytest.raises(WorkflowExecutionError):
        workflow.run()

    saved_state = json.loads(workflow.state_path.read_text(encoding="utf-8"))
    assert saved_state["stage"] == "INTERPOLATED"
    assert saved_state["last_successful_stage"] == "INTERPOLATED"
    assert saved_state["last_error"]["tool_name"] == "tune_tensor_model"
