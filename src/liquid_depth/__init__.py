"""Core RGB-D liquid depth pipeline."""

from .geometry import Plane, PlaneFit
from .pipeline import fit_bottom, infer_frame

__all__ = ["Plane", "PlaneFit", "fit_bottom", "infer_frame"]
__version__ = "0.1.0"

