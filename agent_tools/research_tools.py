"""Deterministic Hello Agents tools around the Day 1/2 experiment core.

The tools exchange paths and small JSON payloads.  They never place image
arrays or model tensors in an Agent context.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ..core.data import (
    apply_observation_mask,
    load_observation_mask,
    load_rgb_image,
    save_image,
    save_mask,
)
from ..core.interpolation import nearest_neighbor_fill
from ..core.masks import generate_observation_mask
from ..core.metrics import composite_ssim, missing_region_mse, missing_region_psnr
from ..core.models import get_default_hyperparameters
from ..core.trainer import fit_tensor_model_on_all_observations, train_tensor_model
from ..schemas import SUPPORTED_MASK_TYPES, SUPPORTED_MODEL_NAMES, TrainingConfig
from .framework import Tool, ToolErrorCode, ToolParameter, ToolResponse


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")


def _required_string(parameters: Dict[str, Any], name: str) -> str:
    value = parameters.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value


def _load_observed_pair(corrupted_path: str, mask_path: str) -> tuple:
    corrupted = load_rgb_image(corrupted_path, max_size=None)
    observed_mask = load_observation_mask(mask_path)
    if corrupted.shape[:2] != observed_mask.shape:
        raise ValueError("corrupted image and mask shapes do not match")
    return corrupted, observed_mask


def _training_config(parameters: Dict[str, Any], learning_rate: float) -> TrainingConfig:
    return TrainingConfig(
        learning_rate=float(learning_rate),
        max_steps=int(parameters.get("max_steps", 200)),
        validation_observed_ratio=float(parameters.get("validation_ratio", 0.1)),
        validation_interval=int(parameters.get("validation_interval", 10)),
        early_stopping_patience=int(parameters.get("patience", 20)),
        device=str(parameters.get("device", "auto")),
    )


def _metric_payload(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
) -> Dict[str, Any]:
    psnr = missing_region_psnr(reconstruction, ground_truth, observed_mask)
    return {
        "missing_mse": missing_region_mse(
            reconstruction,
            ground_truth,
            observed_mask,
        ),
        "missing_psnr": psnr if math.isfinite(psnr) else None,
        "perfect_reconstruction": not math.isfinite(psnr),
        "composite_ssim": composite_ssim(
            reconstruction,
            ground_truth,
            observed_mask,
        ),
    }


def _default_candidates(model_name: str) -> List[Dict[str, Any]]:
    if model_name == "matrix":
        return [
            {"hyperparameters": {"rank": 8, "init_scale": 0.1}, "learning_rate": 0.03},
            {"hyperparameters": {"rank": 16, "init_scale": 0.1}, "learning_rate": 0.03},
        ]
    if model_name == "cp":
        return [
            {"hyperparameters": {"rank": 8, "init_scale": 0.2}, "learning_rate": 0.03},
            {"hyperparameters": {"rank": 12, "init_scale": 0.2}, "learning_rate": 0.03},
        ]
    return [
        {
            "hyperparameters": {
                "rank_h": 8,
                "rank_w": 8,
                "rank_c": 3,
                "init_scale": 0.15,
            },
            "learning_rate": 0.03,
        },
        {
            "hyperparameters": {
                "rank_h": 16,
                "rank_w": 16,
                "rank_c": 3,
                "init_scale": 0.15,
            },
            "learning_rate": 0.03,
        },
    ]


class ResearchTool(Tool):
    """Shared conversion of common input errors to structured responses."""

    @staticmethod
    def invalid(error: Exception) -> ToolResponse:
        return ToolResponse.error(
            code=ToolErrorCode.INVALID_PARAM,
            message=str(error),
        )

    @staticmethod
    def failed(error: Exception) -> ToolResponse:
        return ToolResponse.error(
            code=ToolErrorCode.EXECUTION_ERROR,
            message=str(error),
        )


class AnalyzeImageTool(ResearchTool):
    """Create a masked experiment case and summarize only visible pixels."""

    def __init__(self) -> None:
        super().__init__(
            name="analyze_image",
            description=(
                "创建可复现实验 mask，并根据可见像素返回图像尺寸、缺失率、"
                "通道统计和局部平滑度。"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="image_path", type="string", description="完整评测图片路径"),
            ToolParameter(name="run_dir", type="string", description="本次运行产物目录"),
            ToolParameter(name="mask_type", type="string", description="random 或 block"),
            ToolParameter(name="missing_rate", type="number", description="目标缺失率"),
            ToolParameter(name="seed", type="integer", description="随机种子"),
            ToolParameter(
                name="image_size",
                type="integer",
                description="最长边缩放尺寸",
                required=False,
                default=128,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            image_path = _required_string(parameters, "image_path")
            run_dir = Path(_required_string(parameters, "run_dir"))
            mask_type = _required_string(parameters, "mask_type")
            if mask_type not in SUPPORTED_MASK_TYPES:
                raise ValueError("mask_type must be random or block")
            missing_rate = float(parameters["missing_rate"])
            if not 0.0 < missing_rate < 1.0:
                raise ValueError("missing_rate must be strictly between 0 and 1")
            seed = int(parameters["seed"])
            raw_size = parameters.get("image_size", 128)
            image_size = None if raw_size in (None, 0) else int(raw_size)
            if image_size is not None and image_size < 8:
                raise ValueError("image_size must be at least 8")

            ground_truth = load_rgb_image(image_path, max_size=image_size)
            height, width, _ = ground_truth.shape
            observed_mask = generate_observation_mask(
                height=height,
                width=width,
                missing_rate=missing_rate,
                mask_type=mask_type,
                seed=seed,
            )
            corrupted = apply_observation_mask(ground_truth, observed_mask)
            run_dir.mkdir(parents=True, exist_ok=True)
            ground_truth_path = run_dir / "evaluation_ground_truth.png"
            corrupted_path = run_dir / "corrupted.png"
            mask_path = run_dir / "mask.png"
            profile_path = run_dir / "image_profile.json"
            save_image(str(ground_truth_path), ground_truth)
            save_image(str(corrupted_path), corrupted)
            save_mask(str(mask_path), observed_mask)

            visible_pixels = ground_truth[observed_mask]
            horizontal_pairs = observed_mask[:, 1:] & observed_mask[:, :-1]
            vertical_pairs = observed_mask[1:, :] & observed_mask[:-1, :]
            local_differences = []
            if horizontal_pairs.any():
                local_differences.append(
                    np.abs(ground_truth[:, 1:] - ground_truth[:, :-1])[horizontal_pairs]
                )
            if vertical_pairs.any():
                local_differences.append(
                    np.abs(ground_truth[1:, :] - ground_truth[:-1, :])[vertical_pairs]
                )
            mean_local_difference = (
                float(np.concatenate(local_differences).mean())
                if local_differences
                else 0.0
            )
            profile = {
                "run_id": run_id,
                "image_shape": [height, width, 3],
                "mask_type": mask_type,
                "requested_missing_rate": missing_rate,
                "actual_missing_rate": float((~observed_mask).mean()),
                "observed_pixels": int(observed_mask.sum()),
                "missing_pixels": int((~observed_mask).sum()),
                "visible_channel_mean": visible_pixels.mean(axis=0).tolist(),
                "visible_channel_std": visible_pixels.std(axis=0).tolist(),
                "visible_mean_local_absolute_difference": mean_local_difference,
                "analysis_scope": "visible_pixels_only",
            }
            _write_json(profile_path, profile)
            return ToolResponse.success(
                text=(
                    "图像分析完成：shape=%s，实际缺失率=%.4f。"
                    % (profile["image_shape"], profile["actual_missing_rate"])
                ),
                data={
                    "run_id": run_id,
                    "profile": profile,
                    "artifacts": {
                        "corrupted": str(corrupted_path),
                        "mask": str(mask_path),
                        "profile": str(profile_path),
                    },
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)


class RunInterpolationTool(ResearchTool):
    """Run nearest-neighbor filling without accessing ground truth."""

    def __init__(self) -> None:
        super().__init__(
            name="run_interpolation",
            description="仅根据 corrupted image 和 mask 运行最近邻插值并保存图片。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="corrupted_path", type="string", description="缺损图片路径"),
            ToolParameter(name="mask_path", type="string", description="观测 mask 路径"),
            ToolParameter(name="output_path", type="string", description="插值结果路径"),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            corrupted, observed_mask = _load_observed_pair(
                _required_string(parameters, "corrupted_path"),
                _required_string(parameters, "mask_path"),
            )
            output_path = Path(_required_string(parameters, "output_path"))
            reconstruction = nearest_neighbor_fill(corrupted, observed_mask)
            save_image(str(output_path), reconstruction)
            return ToolResponse.success(
                text="最近邻插值完成。",
                data={
                    "run_id": run_id,
                    "algorithm": "nearest_neighbor_manhattan",
                    "artifacts": {"reconstruction": str(output_path)},
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)


class TuneTensorModelTool(ResearchTool):
    """Select hyperparameters and step count using held-out observed pixels."""

    def __init__(self) -> None:
        super().__init__(
            name="tune_tensor_model",
            description=(
                "在观测像素内部划分 train/validation，对候选张量分解配置调参；"
                "不读取人工缺失区域 Ground Truth。"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="corrupted_path", type="string", description="缺损图片路径"),
            ToolParameter(name="mask_path", type="string", description="观测 mask 路径"),
            ToolParameter(name="output_path", type="string", description="调参结果 JSON 路径"),
            ToolParameter(name="model_name", type="string", description="matrix、cp 或 tucker"),
            ToolParameter(name="seed", type="integer", description="随机种子"),
            ToolParameter(name="candidates", type="array", description="候选配置列表", required=False),
            ToolParameter(name="max_steps", type="integer", description="每个 trial 最大步数", required=False, default=200),
            ToolParameter(name="device", type="string", description="auto、cpu 或 cuda", required=False, default="auto"),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            model_name = _required_string(parameters, "model_name")
            if model_name not in SUPPORTED_MODEL_NAMES:
                raise ValueError("model_name must be matrix, cp, or tucker")
            corrupted, observed_mask = _load_observed_pair(
                _required_string(parameters, "corrupted_path"),
                _required_string(parameters, "mask_path"),
            )
            seed = int(parameters["seed"])
            candidates = parameters.get("candidates") or _default_candidates(model_name)
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("candidates must be a non-empty list")

            trials = []
            for trial_index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    raise ValueError("each candidate must be an object")
                hyperparameters = candidate.get("hyperparameters")
                if hyperparameters is None:
                    hyperparameters = get_default_hyperparameters(model_name)
                if not isinstance(hyperparameters, dict):
                    raise ValueError("candidate hyperparameters must be an object")
                learning_rate = float(candidate.get("learning_rate", 0.03))
                config = _training_config(parameters, learning_rate)
                output = train_tensor_model(
                    model_name=model_name,
                    model_hyperparameters=dict(hyperparameters),
                    observed_image=corrupted,
                    observed_mask=observed_mask,
                    config=config,
                    seed=seed,
                )
                trials.append(
                    {
                        "trial_index": trial_index,
                        "model_name": model_name,
                        "hyperparameters": dict(hyperparameters),
                        "learning_rate": learning_rate,
                        "best_step": output.best_step,
                        "best_validation_mse": output.best_validation_mse,
                        "runtime_seconds": output.runtime_seconds,
                        "parameter_count": output.parameter_count,
                        "device": output.device,
                    }
                )

            best_trial = min(trials, key=lambda trial: trial["best_validation_mse"])
            result = {
                "run_id": run_id,
                "selection_metric": "held_out_observed_mse",
                "ground_truth_used": False,
                "trial_count": len(trials),
                "trials": trials,
                "best": best_trial,
            }
            output_path = Path(_required_string(parameters, "output_path"))
            _write_json(output_path, result)
            return ToolResponse.success(
                text=(
                    "调参完成：选择 trial %d，validation MSE=%.8f。"
                    % (best_trial["trial_index"], best_trial["best_validation_mse"])
                ),
                data={
                    "run_id": run_id,
                    "best": best_trial,
                    "trial_count": len(trials),
                    "artifacts": {"tuning_result": str(output_path)},
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)


class TrainTensorModelTool(ResearchTool):
    """Final-refit a selected model on every observed pixel."""

    def __init__(self) -> None:
        super().__init__(
            name="train_tensor_model",
            description=(
                "使用已选配置和步数，在全部观测像素上重新训练张量分解模型；"
                "不读取 Ground Truth。"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="corrupted_path", type="string", description="缺损图片路径"),
            ToolParameter(name="mask_path", type="string", description="观测 mask 路径"),
            ToolParameter(name="output_dir", type="string", description="模型产物目录"),
            ToolParameter(name="model_name", type="string", description="matrix、cp 或 tucker"),
            ToolParameter(name="hyperparameters", type="object", description="模型超参数"),
            ToolParameter(name="learning_rate", type="number", description="学习率"),
            ToolParameter(name="selected_steps", type="integer", description="调参阶段选出的步数"),
            ToolParameter(name="seed", type="integer", description="随机种子"),
            ToolParameter(name="device", type="string", description="auto、cpu 或 cuda", required=False, default="auto"),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            model_name = _required_string(parameters, "model_name")
            if model_name not in SUPPORTED_MODEL_NAMES:
                raise ValueError("model_name must be matrix, cp, or tucker")
            hyperparameters = parameters.get("hyperparameters")
            if not isinstance(hyperparameters, dict):
                raise ValueError("hyperparameters must be an object")
            selected_steps = int(parameters["selected_steps"])
            learning_rate = float(parameters["learning_rate"])
            seed = int(parameters["seed"])
            corrupted, observed_mask = _load_observed_pair(
                _required_string(parameters, "corrupted_path"),
                _required_string(parameters, "mask_path"),
            )
            config_parameters = dict(parameters)
            config_parameters["max_steps"] = max(1, selected_steps)
            config = _training_config(config_parameters, learning_rate)
            output = fit_tensor_model_on_all_observations(
                model_name=model_name,
                model_hyperparameters=dict(hyperparameters),
                observed_image=corrupted,
                observed_mask=observed_mask,
                config=config,
                selected_steps=selected_steps,
                seed=seed,
            )

            output_dir = Path(_required_string(parameters, "output_dir"))
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_path = output_dir / "model_raw.png"
            completed_path = output_dir / "model_completed.png"
            history_path = output_dir / "final_fit_history.json"
            checkpoint_path = output_dir / "model.pt"
            result_path = output_dir / "training_result.json"
            completed = output.reconstruction.copy()
            completed[observed_mask] = corrupted[observed_mask]
            save_image(str(raw_path), output.reconstruction)
            save_image(str(completed_path), completed)
            _write_json(history_path, {"history": output.history})
            torch.save(
                {
                    "model_name": model_name,
                    "hyperparameters": dict(hyperparameters),
                    "selected_steps": selected_steps,
                    "learning_rate": learning_rate,
                    "state_dict": output.state_dict,
                    "training_phase": "refit_on_all_observations",
                },
                checkpoint_path,
            )
            result = {
                "run_id": run_id,
                "model_name": model_name,
                "hyperparameters": dict(hyperparameters),
                "learning_rate": learning_rate,
                "fitted_steps": output.fitted_steps,
                "final_train_mse": output.final_train_mse,
                "observed_pixels_used": int(output.fit_mask.sum()),
                "parameter_count": output.parameter_count,
                "runtime_seconds": output.runtime_seconds,
                "device": output.device,
                "ground_truth_used": False,
            }
            _write_json(result_path, result)
            return ToolResponse.success(
                text=(
                    "最终拟合完成：使用 %d 个观测像素训练 %d 步。"
                    % (result["observed_pixels_used"], output.fitted_steps)
                ),
                data={
                    **result,
                    "artifacts": {
                        "raw_reconstruction": str(raw_path),
                        "reconstruction": str(completed_path),
                        "history": str(history_path),
                        "checkpoint": str(checkpoint_path),
                        "training_result": str(result_path),
                    },
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)


class EvaluateReconstructionTool(ResearchTool):
    """Evaluate one finished reconstruction; this is the only GT-consuming tool."""

    def __init__(self) -> None:
        super().__init__(
            name="evaluate_reconstruction",
            description="训练和选择结束后，使用 Ground Truth 计算缺失区域 PSNR 与 SSIM。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="algorithm_name", type="string", description="算法名称"),
            ToolParameter(name="reconstruction_path", type="string", description="重建图片路径"),
            ToolParameter(name="ground_truth_path", type="string", description="仅用于最终评估的完整图片"),
            ToolParameter(name="mask_path", type="string", description="观测 mask 路径"),
            ToolParameter(name="output_path", type="string", description="指标 JSON 路径"),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            algorithm_name = _required_string(parameters, "algorithm_name")
            reconstruction = load_rgb_image(
                _required_string(parameters, "reconstruction_path"),
                max_size=None,
            )
            ground_truth = load_rgb_image(
                _required_string(parameters, "ground_truth_path"),
                max_size=None,
            )
            observed_mask = load_observation_mask(
                _required_string(parameters, "mask_path")
            )
            if reconstruction.shape != ground_truth.shape:
                raise ValueError("reconstruction and ground truth shapes do not match")
            if observed_mask.shape != ground_truth.shape[:2]:
                raise ValueError("mask and image shapes do not match")
            metrics = _metric_payload(reconstruction, ground_truth, observed_mask)
            result = {
                "run_id": run_id,
                "algorithm_name": algorithm_name,
                **metrics,
                "ground_truth_usage": "final_evaluation_only",
            }
            output_path = Path(_required_string(parameters, "output_path"))
            _write_json(output_path, result)
            psnr_text = (
                "infinite" if metrics["missing_psnr"] is None else "%.4f" % metrics["missing_psnr"]
            )
            return ToolResponse.success(
                text="%s 评估完成：PSNR=%s dB。" % (algorithm_name, psnr_text),
                data={
                    **result,
                    "artifacts": {"metrics": str(output_path)},
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)


class CompareExperimentsTool(ResearchTool):
    """Compare saved metrics without rerunning either algorithm."""

    def __init__(self) -> None:
        super().__init__(
            name="compare_experiments",
            description="读取两个最终指标 JSON，以 missing PSNR 为主、SSIM 为辅选择胜者。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="run_id", type="string", description="工作流运行 ID"),
            ToolParameter(name="baseline_metrics_path", type="string", description="插值指标 JSON"),
            ToolParameter(name="candidate_metrics_path", type="string", description="张量模型指标 JSON"),
            ToolParameter(name="output_path", type="string", description="比较结果 JSON"),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            run_id = _required_string(parameters, "run_id")
            baseline_path = Path(_required_string(parameters, "baseline_metrics_path"))
            candidate_path = Path(_required_string(parameters, "candidate_metrics_path"))
            if not baseline_path.is_file() or not candidate_path.is_file():
                raise ValueError("both metric files must exist")
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

            baseline_psnr = baseline.get("missing_psnr")
            candidate_psnr = candidate.get("missing_psnr")
            baseline_score = math.inf if baseline_psnr is None else float(baseline_psnr)
            candidate_score = math.inf if candidate_psnr is None else float(candidate_psnr)
            psnr_delta = candidate_score - baseline_score
            if math.isinf(candidate_score) and math.isinf(baseline_score):
                psnr_delta_json = None
                ssim_delta = float(candidate["composite_ssim"]) - float(
                    baseline["composite_ssim"]
                )
                winner = "candidate" if ssim_delta > 0 else "baseline" if ssim_delta < 0 else "tie"
            elif math.isinf(candidate_score):
                psnr_delta_json = None
                ssim_delta = float(candidate["composite_ssim"]) - float(
                    baseline["composite_ssim"]
                )
                winner = "candidate"
            elif math.isinf(baseline_score):
                psnr_delta_json = None
                ssim_delta = float(candidate["composite_ssim"]) - float(
                    baseline["composite_ssim"]
                )
                winner = "baseline"
            else:
                psnr_delta_json = psnr_delta
                ssim_delta = float(candidate["composite_ssim"]) - float(
                    baseline["composite_ssim"]
                )
                winner = "candidate" if psnr_delta > 0 else "baseline" if psnr_delta < 0 else "tie"
            result = {
                "run_id": run_id,
                "winner": winner,
                "primary_metric": "missing_psnr",
                "baseline_algorithm": baseline.get("algorithm_name"),
                "candidate_algorithm": candidate.get("algorithm_name"),
                "candidate_psnr_delta": psnr_delta_json,
                "candidate_ssim_delta": ssim_delta,
                "promotion_allowed": winner == "candidate",
            }
            output_path = Path(_required_string(parameters, "output_path"))
            _write_json(output_path, result)
            return ToolResponse.success(
                text="实验比较完成：winner=%s。" % winner,
                data={
                    **result,
                    "artifacts": {"comparison": str(output_path)},
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return self.invalid(error)
        except Exception as error:
            return self.failed(error)
