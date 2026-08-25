"""Core RGB-D liquid depth pipeline."""

from .geometry import Plane, PlaneFit
from .pipeline import fit_bottom, infer_frame
from .refinement import RefinedDepth

__all__ = ["Plane", "PlaneFit", "RefinedDepth", "fit_bottom", "infer_frame"]
__version__ = "0.2.0"
