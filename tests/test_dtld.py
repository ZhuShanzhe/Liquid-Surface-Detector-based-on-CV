import csv
import json

from liquid_depth.dtld import build_dtld_rows, write_dtld_manifest


def test_build_dtld_manifest_preserves_sequence_split_and_metric_height(tmp_path):
    scene = tmp_path / "train" / "000013"
    for name in ("rgb", "depth", "mask", "mask_visib"):
        (scene / name).mkdir(parents=True)
    (scene / "rgb" / "000007.png").write_bytes(b"rgb")
    (scene / "depth" / "000007.png").write_bytes(b"depth")
    (scene / "mask" / "000007_000000.png").write_bytes(b"mask")
    (scene / "mask_visib" / "000007_000000.png").write_bytes(b"visible")
    (scene / "liquid_label").mkdir()
    (scene / "liquid_label" / "scene_gt_liquid.json").write_text(
        json.dumps(
            {
                "7": [
                    {
                        "obj_id": 16,
                        "liquid_h": 55.0,
                        "liquid_label": {"A": [1, 2], "B": [3, 4]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (scene / "scene_camera.json").write_text(
        json.dumps({"7": {"depth_scale": 0.001, "cam_K": [1, 0, 0]}}),
        encoding="utf-8",
    )
    (scene / "scene_gt.json").write_text(
        json.dumps({"7": [{"obj_id": 16, "cam_t_m2c": [0, 0, 1000]}]}),
        encoding="utf-8",
    )
    (scene / "scene_gt_info.json").write_text(
        '{"7": [{"visib_fract": "truncated',
        encoding="utf-8",
    )

    rows = build_dtld_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["split"] == "train"
    assert row["sequence_id"] == "train/000013"
    assert row["object_id"] == 16
    assert row["liquid_height_mm"] == 55.0
    assert row["depth_cm"] == 5.5
    assert row["depth_scale_to_m"] == 0.001
    assert json.loads(row["liquid_label_json"])["A"] == [1, 2]
    assert json.loads(row["object_info_json"]) == {}

    output = tmp_path / "manifest.csv"
    counts = write_dtld_manifest(tmp_path, output)
    with output.open(encoding="utf-8", newline="") as stream:
        written = list(csv.DictReader(stream))
    assert counts == {"train": 1}
    assert written[0]["frame_id"] == row["frame_id"]

    overridden = build_dtld_rows(tmp_path, split_map={"train/000013": "test"})
    assert overridden[0]["split"] == "test"
