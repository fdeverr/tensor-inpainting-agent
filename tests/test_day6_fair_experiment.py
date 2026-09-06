import inspect

from inpainting_research_agent.core.experiment_judge import judge_candidate
from inpainting_research_agent.core.fair_experiment import (
    paired_trial_configurations,
    tune_model_on_observed_pixels,
)


def test_paired_trials_share_base_parameters_and_budget():
    paired = paired_trial_configurations(
        base_search_space={"rank": [4, 8, 12], "init_scale": [0.1]},
        candidate_search_space={
            "rank": [4, 8, 12],
            "init_scale": [0.1],
            "tv_weight": [0.0, 0.0001, 0.001],
        },
        trial_count=3,
        seed=42,
        anchor_base_config={"rank": 8, "init_scale": 0.1},
    )
    assert len(paired["baseline"]) == len(paired["candidate"]) == 3
    for baseline, candidate in zip(paired["baseline"], paired["candidate"]):
        assert candidate["rank"] == baseline["rank"]
        assert candidate["init_scale"] == baseline["init_scale"]
    assert paired["candidate_only_parameter_names"] == ["tv_weight"]


def test_tuner_api_cannot_receive_hidden_ground_truth():
    assert "ground_truth" not in inspect.signature(tune_model_on_observed_pixels).parameters


def _tuning(validation_mse, trial_count=2):
    return {
        "trial_count": trial_count,
        "trials": [{"runtime_seconds": 1.0}] * trial_count,
        "best": {
            "best_validation_mse": validation_mse,
            "best_step": 10,
            "hyperparameters": {"tv_weight": 0.0001},
        },
    }


def _final(psnr, ssim):
    return {
        "metrics": {"missing_psnr": psnr, "composite_ssim": ssim},
        "runtime_seconds": 1.0,
    }


def test_judge_uses_fixed_psnr_and_ssim_gate():
    accepted = judge_candidate(
        _tuning(0.03),
        _final(20.0, 0.80),
        _tuning(0.02),
        _final(20.25, 0.799),
    )
    assert accepted["decision"] == "accept"

    rejected = judge_candidate(
        _tuning(0.03),
        _final(20.0, 0.80),
        _tuning(0.02),
        _final(20.25, 0.797),
    )
    assert rejected["decision"] == "reject"
    assert rejected["next_round_constraints"]
