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


def test_pose_film_conditioning_changes_spatial_predictions_and_backpropagates():
    torch.manual_seed(11)
    model = DTLDContactGeometryNet(base_channels=8, geometry_conditioning=True)
    inputs = torch.randn(2, 5, 48, 80)
    object_index = torch.tensor([0, 3])
    pose = torch.randn(2, 12)

    prediction = model(inputs, object_index, pose)
    objective = prediction["contact_curve"].sum() + prediction["contact_logits"].mean()
    objective.backward()

    assert model.geometry_film[-1].weight.grad is not None
    assert torch.isfinite(model.geometry_film[-1].weight.grad).all()


def test_object_expert_heads_select_and_warm_start_from_shared_head(tmp_path):
    from liquid_depth.training.train_dtld_contact import _load_compatible

    shared = DTLDContactGeometryNet(base_channels=8)
    checkpoint = tmp_path / "shared.pth"
    torch.save({"model": shared.state_dict()}, checkpoint)
    expert = DTLDContactGeometryNet(base_channels=8, object_experts=True)

    loaded = _load_compatible(expert, checkpoint)
    prediction = expert(
        torch.randn(2, 5, 48, 80),
        torch.tensor([1, 3]),
        torch.randn(2, 12),
    )

    assert loaded == len(shared.state_dict())
    assert prediction["contact_logits"].shape == (2, 1, 48, 80)
    assert prediction["control_heatmap_logits"].shape == (2, 4, 48, 80)
    for object_index in range(4):
        torch.testing.assert_close(
            expert.contact_head.weight[object_index], shared.contact_head.weight[0]
        )
        torch.testing.assert_close(
            expert.control_heatmap_head.weight[object_index * 4 : (object_index + 1) * 4],
            shared.control_heatmap_head.weight,
        )
