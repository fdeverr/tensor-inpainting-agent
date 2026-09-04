from pathlib import Path

from inpainting_research_agent.candidate.generator import deterministic_candidate
from inpainting_research_agent.candidate.validator import CandidateValidator


def _validate(tmp_path: Path, source: str):
    model_path = tmp_path / "model.py"
    model_path.write_text(source, encoding="utf-8")
    return CandidateValidator(timeout_seconds=10).validate(str(model_path))


def test_deterministic_tv_candidate_passes_all_checks(tmp_path):
    proposal = deterministic_candidate("tucker")
    result = _validate(tmp_path, proposal.model_code)
    assert result["passed"] is True
    assert result["static_validation"]["passed"] is True
    assert result["smoke_test"]["passed"] is True
    assert "tv_regularization" in result["smoke_test"]["loss_terms"]


def test_forbidden_import_is_rejected_before_execution(tmp_path):
    source = '''import os
import torch

class CandidateTensorInpaintingModel(TuckerDecomposition):
    @classmethod
    def search_space(cls, image_shape):
        return {"rank_h": [4]}
'''
    result = _validate(tmp_path, source)
    assert result["passed"] is False
    assert result["smoke_test"]["skipped"] is True
    assert any(
        error["code"] == "FORBIDDEN_IMPORT"
        for error in result["static_validation"]["errors"]
    )


def test_wrong_forward_shape_fails_smoke_test(tmp_path):
    source = '''import torch

class CandidateTensorInpaintingModel(TuckerDecomposition):
    def forward(self):
        return super().forward()[:, :, 0]

    @classmethod
    def search_space(cls, image_shape):
        return {"rank_h": [4]}
'''
    result = _validate(tmp_path, source)
    assert result["static_validation"]["passed"] is True
    assert result["passed"] is False
    assert "forward shape mismatch" in result["smoke_test"]["message"]


def test_candidate_without_gradient_fails_smoke_test(tmp_path):
    source = '''import torch

class CandidateTensorInpaintingModel(BaseTensorInpaintingModel):
    def __init__(self, image_shape, initial_channel_mean):
        super().__init__(image_shape, initial_channel_mean)
        self.factor = torch.nn.Parameter(torch.ones(1))

    def forward(self):
        return torch.zeros(self.image_shape, dtype=torch.float32)

    @classmethod
    def search_space(cls, image_shape):
        return {"rank": [1]}
'''
    result = _validate(tmp_path, source)
    assert result["static_validation"]["passed"] is True
    assert result["passed"] is False
    assert result["smoke_test"]["error_type"] == "RuntimeError"
