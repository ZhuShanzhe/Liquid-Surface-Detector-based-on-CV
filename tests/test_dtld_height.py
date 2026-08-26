import torch

from liquid_depth.training.dtld_height import (
    DTLDContactHeightLoss,
    DTLDContactHeightNet,
)


def test_dtld_contact_height_model_contract_and_loss():
    model = DTLDContactHeightNet(base_channels=8, max_height_mm=120.0)
    inputs = torch.randn(2, 5, 48, 80)
    object_index = torch.tensor([0, 3])
    pose = torch.randn(2, 12)
    prediction = model(inputs, object_index, pose)

    assert prediction["contact_logits"].shape == (2, 1, 48, 80)
    assert prediction["height_mm"].shape == (2,)
    assert torch.all((prediction["height_mm"] >= 0) & (prediction["height_mm"] <= 120))
    assert torch.all((prediction["height_confidence"] > 0) & (prediction["height_confidence"] < 1))

    target = {
        "contact": torch.zeros(2, 1, 48, 80),
        "height_mm": torch.tensor([25.0, 85.0]),
        "object_index": object_index,
        "pose": pose,
    }
    target["contact"][:, :, 20:23, 15:18] = 1.0
    losses = DTLDContactHeightLoss()(prediction, target)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
