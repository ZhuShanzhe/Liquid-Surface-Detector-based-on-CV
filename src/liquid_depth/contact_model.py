from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContactModelBundle:
    model: Any
    torch: Any
    device: Any
    input_size: tuple[int, int]
    max_depth_m: float
    checkpoint_path: Path


def load_contact_model_bundle(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
) -> ContactModelBundle:
    """Load one DTLD contact model without coupling it to a measurement geometry."""

    import torch

    from .training.dtld_contact import build_dtld_contact_model

    path = Path(checkpoint_path).expanduser().resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    requested = torch.device(device or "cuda")
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    model = build_dtld_contact_model(
        state.get("backbone", "unet"),
        int(state.get("base_channels", 24)),
        pretrained_backbone=False,
        geometry_conditioning=bool(state.get("geometry_conditioning", False)),
        object_experts=bool(state.get("object_experts", False)),
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(requested).eval()
    return ContactModelBundle(
        model=model,
        torch=torch,
        device=requested,
        input_size=tuple(int(item) for item in state.get("image_size", (320, 180))),
        max_depth_m=float(state.get("max_depth_m", 3.0)),
        checkpoint_path=path,
    )


def load_contact_specialists(
    profile: dict,
    *,
    resolve_path,
    device: str | None = None,
) -> dict[str, ContactModelBundle]:
    models = profile.get("complex_scene", {}).get("models", {})
    if not isinstance(models, dict):
        raise TypeError("complex_scene.models must be a mapping")
    output: dict[str, ContactModelBundle] = {}
    cache: dict[Path, ContactModelBundle] = {}
    for variant, options in models.items():
        if not isinstance(options, dict):
            raise TypeError(f"complex_scene.models.{variant} must be a mapping")
        if not bool(options.get("enabled", True)):
            continue
        checkpoint = options.get("checkpoint_path")
        if not checkpoint:
            raise ValueError(f"complex_scene.models.{variant}.checkpoint_path is required")
        path = resolve_path(profile, checkpoint).resolve()
        bundle = cache.get(path)
        if bundle is None:
            bundle = load_contact_model_bundle(path, device=device)
            cache[path] = bundle
        output[str(variant)] = bundle
    return output
