"""Multi-device generalization benchmark."""

from .devices import DEVICE_PROFILES
from .run import build_device_arrays, run_benchmark, save_heatmap

__all__ = ["DEVICE_PROFILES", "build_device_arrays", "run_benchmark", "save_heatmap"]
