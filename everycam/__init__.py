"""EveryCam — every camera is a robot teacher.

The bottleneck in Physical AI is robot training data. EveryCam turns ordinary
cameras (webcam, phone, dashcam, fixed/CCTV-style, smart glasses, or recorded
files) into standardized, LeRobot-compatible embodied datasets — with privacy
enforced *before* anything is written to disk.

Quick start (no hardware, no GPU)::

    from everycam import Pipeline, PipelineConfig
    cfg = PipelineConfig.demo()          # synthetic clip, all defaults
    episode = Pipeline(cfg).run()
    print(episode.summary())
"""

from .types import (
    Frame,
    Detection,
    HandObs,
    PerceptionResult,
    EmbodiedStep,
    Episode,
    Provenance,
)
from .config import PipelineConfig, PrivacyConfig, PerceptionConfig, SourceConfig

__version__ = "0.1.0"
__slogan__ = "Every camera is a robot teacher."

__all__ = [
    "Frame",
    "Detection",
    "HandObs",
    "PerceptionResult",
    "EmbodiedStep",
    "Episode",
    "Provenance",
    "PipelineConfig",
    "PrivacyConfig",
    "PerceptionConfig",
    "SourceConfig",
    "Pipeline",
    "__version__",
]


def __getattr__(name):  # PEP 562 lazy import — avoids loading the full stack early
    if name == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline
    raise AttributeError(f"module 'everycam' has no attribute {name!r}")
