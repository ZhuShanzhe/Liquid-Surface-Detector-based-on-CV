import json

from liquid_depth.evaluation import evaluate


def test_evaluation_reports_coverage_and_scenarios(tmp_path):
    ground_truth = tmp_path / "truth.csv"
    ground_truth.write_text(
        "frame_id,depth,scenario\nfirst,10.0,clear\nsecond,20.0,reflection\n", encoding="utf-8"
    )
    prediction_dir = tmp_path / "predictions" / "first"
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "depth_result.json").write_text(
        json.dumps({"frame_id": "first", "liquid_depth": 11.0, "accepted": True}), encoding="utf-8"
    )

    report = evaluate(ground_truth, tmp_path / "predictions", tolerance=1.0)

    assert report["overall"]["prediction_coverage"] == 0.5
    assert report["overall"]["accepted_predictions"]["mae"] == 1.0
    assert report["missing_prediction_ids"] == ["second"]
    assert set(report["by_scenario"]) == {"clear", "reflection"}
