"""Read-only packaged assets and explicit local execution-device selection."""
import os
import sys
from pathlib import Path


def resource_root():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parents[2]


def inference_device():
    import torch
    choice = os.environ.get('LIQUID_DEVICE', 'auto')
    if choice not in ('auto', 'cpu', 'cuda'):
        raise ValueError('LIQUID_DEVICE must be auto, cpu, or cuda')
    if choice == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return 'cuda' if choice != 'cpu' and torch.cuda.is_available() else 'cpu'


def user_data_root():
    base = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.local/share')))
    directory = base / 'LiquidDepthSimulation'
    directory.mkdir(parents=True, exist_ok=True)
    return directory
