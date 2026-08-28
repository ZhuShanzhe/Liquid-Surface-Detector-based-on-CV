import pytest

from liquid_depth.specialist_training import (
    assign_augmentation_profile,
    specialize_rows,
)


def test_profile_assignment_is_deterministic_and_validated():
    profiles = ("standard", "glare", "depth_failure", "low_light")
    assert assign_augmentation_profile(
        "frame-1",
        profiles,
        seed=7,
    ) == assign_augmentation_profile(
        "frame-1",
        profiles,
        seed=7,
    )
    with pytest.raises(ValueError):
        assign_augmentation_profile("frame", ())
    with pytest.raises(ValueError):
        assign_augmentation_profile("frame", ("unknown",))


def test_only_training_rows_receive_specialist_profiles():
    rows = [
        {
            "frame_id": "train-1",
            "split": "train",
            "difficulty_tags": "transparent",
        },
        {
            "frame_id": "val-1",
            "split": "val",
            "difficulty_tags": "transparent",
        },
    ]
    output = specialize_rows(
        rows,
        ("glare",),
        seed=7,
    )
    assert output[0]["augmentation_profile"] == "glare"
    assert "glare" in output[0]["difficulty_tags"]
    assert output[1]["augmentation_profile"] == "standard"
    assert output[1]["difficulty_tags"] == "transparent"
