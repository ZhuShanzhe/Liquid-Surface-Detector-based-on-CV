"""Trainable model definitions. PyTorch is only required when this package is imported."""

from .multitask import LiquidSurfaceMultiTaskNet, MultiTaskLoss

__all__ = ["LiquidSurfaceMultiTaskNet", "MultiTaskLoss"]
