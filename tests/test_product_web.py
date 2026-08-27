from pathlib import Path

from liquid_depth import product_web


def _complete_frame(path: Path) -> Path:
    path.mkdir(parents=True)
    for name in ("rgb.png", "depth.npy", "depth_info.json"):
        (path / name).touch()
    return path


def test_source_from_payload_accepts_frame_or_collection(tmp_path: Path) -> None:
    older = _complete_frame(tmp_path / "20260101_000000")
    latest = _complete_frame(tmp_path / "20260101_000001")

    assert product_web._source_from_payload({}, tmp_path) == latest
    assert product_web._source_from_payload({"path": str(tmp_path)}, tmp_path) == latest
    assert product_web._source_from_payload({"frame_dir": str(older)}, tmp_path) == older


def test_panel_state_reuses_and_invalidates_product_system(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created: list[tuple[Path, str | None, bool]] = []

    def make_system(profile: Path, device: str | None, temporal: bool) -> object:
        created.append((profile, device, temporal))
        return object()

    monkeypatch.setattr(product_web, "make_product_system", make_system)
    state = product_web.PanelState(tmp_path / "capture", tmp_path / "output")
    profile = tmp_path / "site.yaml"

    first = state.system_for(profile, "cuda", False)
    second = state.system_for(profile, "cuda", False)
    temporal = state.system_for(profile, "cuda", True)

    assert first is second
    assert temporal is not first
    assert created == [
        (profile, "cuda", False),
        (profile, "cuda", True),
    ]

    state.invalidate_profile(profile)
    replacement = state.system_for(profile, "cuda", False)

    assert replacement is not first
    assert created[-1] == (profile, "cuda", False)


def test_panel_exposes_scene_adaptive_model_policy() -> None:
    html = (Path(product_web.__file__).with_name("ui") / "index.html").read_text(encoding="utf-8")
    assert 'id="complexSceneMode"' in html
    assert 'value="auto" selected' in html
    assert "complex_scene_mode:cfg.complex_scene_mode" in html
    assert "端到端 P95 ≤ 0.50 秒" in html
