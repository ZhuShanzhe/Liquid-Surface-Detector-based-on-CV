import torch

from liquid_depth.training.dtld_contact import (
    DTLDContactGeometryLoss,
    DTLDContactGeometryNet,
    sample_cubic_bezier,
)


def test_bezier_sampling_preserves_endpoints():
    control = torch.tensor(
        [[[0.1, 0.2], [0.3, 0.1], [0.7, 0.4], [0.9, 0.3]]],
        dtype=torch.float32,
    )
    curve = sample_cubic_bezier(control, samples=17)
    torch.testing.assert_close(curve[:, 0], control[:, 0])
    torch.testing.assert_close(curve[:, -1], control[:, -1])


def test_contact_geometry_model_and_loss_are_differentiable():
    model = DTLDContactGeometryNet(base_channels=8)
    inputs = torch.randn(2, 5, 48, 80)
    object_index = torch.tensor([0, 3])
    pose = torch.randn(2, 12)
    prediction = model(inputs, object_index, pose)

    assert prediction["contact_logits"].shape == (2, 1, 48, 80)
    assert prediction["color_residual"].shape == (2, 3, 48, 80)
    assert prediction["bezier_control_points"].shape == (2, 4, 2)
    assert prediction["contact_curve"].shape == (2, 64, 2)
    assert torch.all(
        (prediction["bezier_control_points"] >= 0)
        & (prediction["bezier_control_points"] <= 1)
    )

    target = {
        "contact": torch.zeros(2, 1, 48, 80),
        "bezier_control_points": torch.tensor(
            [
                [[0.1, 0.5], [0.35, 0.48], [0.65, 0.52], [0.9, 0.5]],
                [[0.2, 0.4], [0.4, 0.35], [0.6, 0.45], [0.8, 0.4]],
            ]
        ),
        "color_residual": torch.zeros(2, 3, 48, 80),
    }
    target["contact"][:, :, 20:23, 10:70] = 1.0
    losses = DTLDContactGeometryLoss()(prediction, target)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
