"""Frame-source registry: turn a ``SourceConfig`` into an iterable of frames."""

from __future__ import annotations

from ..config import SourceConfig
from .base import FrameSource
from .synthetic import SyntheticSource
from .video import CV2Source, ImageDirSource

__all__ = ["FrameSource", "SyntheticSource", "CV2Source", "ImageDirSource", "build_source"]


def build_source(cfg: SourceConfig) -> FrameSource:
    kind = cfg.kind.lower()
    if kind == "synthetic":
        return SyntheticSource(
            n_frames=cfg.max_frames or 60,
            seed=cfg.seed,
            fps=cfg.fps,
            resize=cfg.resize,
        )
    if kind == "webcam":
        return CV2Source(
            cfg.device_index,
            source_id=cfg.source_type,
            fps=cfg.fps,
            max_frames=cfg.max_frames,
            resize=cfg.resize,
        )
    if kind in ("file", "stream"):
        if not cfg.path:
            raise ValueError(f"source.kind='{kind}' requires source.path to be set.")
        return CV2Source(
            cfg.path,
            source_id=cfg.source_type,
            fps=cfg.fps,
            max_frames=cfg.max_frames,
            resize=cfg.resize,
        )
    if kind == "image_dir":
        if not cfg.path:
            raise ValueError("source.kind='image_dir' requires source.path (a folder).")
        return ImageDirSource(
            cfg.path,
            fps=cfg.fps,
            max_frames=cfg.max_frames,
            resize=cfg.resize,
        )
    raise ValueError(f"Unknown source.kind='{cfg.kind}'.")
