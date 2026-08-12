import pytest

torch = pytest.importorskip("torch")
from torch import nn

from model.model_vlm import MMVisionProjector
from trainer.trainer_utils import build_vlm_optimizer, inference_state_dict


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.vision_proj = nn.Linear(4, 4)


def test_projector_variants_preserve_token_shape():
    inputs = torch.randn(2, 64, 8)
    linear = MMVisionProjector(8, 6, projector_type="linear")
    mlp = MMVisionProjector(8, 6, projector_type="mlp")
    assert linear(inputs).shape == (2, 64, 6)
    assert mlp(inputs).shape == (2, 64, 6)
    assert sum(p.numel() for p in mlp.parameters()) > sum(p.numel() for p in linear.parameters())


def test_inference_checkpoint_keeps_only_tuned_vision_parameters():
    model = TinyVLM()
    model.requires_grad_(False)
    model.vision_encoder[1].requires_grad_(True)
    model.vision_proj.requires_grad_(True)
    state = inference_state_dict(model)
    assert "vision_encoder.0.weight" not in state
    assert "vision_encoder.1.weight" in state
    assert "vision_proj.weight" in state


def test_optimizer_uses_lower_vision_learning_rate():
    model = TinyVLM()
    optimizer = build_vlm_optimizer(model, 4e-4, 1e-5)
    scales = sorted(group["lr_scale"] for group in optimizer.param_groups)
    assert scales == [0.025, 1.0]
