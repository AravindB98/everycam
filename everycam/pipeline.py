"""The EveryCam pipeline: frames -> privacy gate -> perception -> signals -> dataset.

This is the spine that makes "every camera a robot teacher": point any source at
it and get a privacy-clean, LeRobot-style embodied dataset out the other end.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import __version__
from .config import PipelineConfig
from .perception import Perceiver
from .privacy import PrivacyGate
from .signals import SignalExtractor
from .sources import build_source
from .types import Episode, Provenance


class Pipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None) -> None:
        self.cfg = cfg or PipelineConfig.demo()
        self.gate = PrivacyGate(self.cfg.privacy)
        self.perceiver = Perceiver(self.cfg.perception)
        self.extractor = SignalExtractor()
        self.last_images: List[np.ndarray] = []
        self.last_privacy: Dict[str, int] = {}

    def run(self, keep_images: bool = True) -> Episode:
        cfg = self.cfg
        source = build_source(cfg.source)
        self.perceiver.reset()

        frames, percs, images = [], [], []
        n_faces = n_plates = n_full = 0
        for frame in source:
            res = self.gate.process(frame.image)  # anonymize BEFORE anything else
            frame.image = res.image
            frame.meta["anonymized"] = True
            n_faces += res.n_faces
            n_plates += res.n_plates
            n_full += int(res.fully_blurred)
            percs.append(self.perceiver.process(frame))
            frames.append(frame)
            if keep_images:
                images.append(frame.image.copy())

        if not frames:
            raise RuntimeError("No frames read from source — check the camera/path.")

        h, w = frames[0].image.shape[:2]
        prov = Provenance(
            source_type=cfg.source.source_type,
            device=cfg.source.device,
            anonymized=True,
            consent=cfg.consent,
            license=cfg.license,
            tool_version=__version__,
            notes=f"faces_blurred={n_faces}, plates_blurred={n_plates}",
        )
        episode = self.extractor.extract(
            frames,
            percs,
            episode_id="episode_000000",
            task=cfg.export.task,
            fps=cfg.source.fps,
            provenance=prov,
            image_shape=(h, w),
        )
        self.last_images = images
        self.last_privacy = {
            "frames": len(frames),
            "faces": n_faces,
            "plates": n_plates,
            "fully_blurred": n_full,
        }
        return episode

    def run_and_export(self) -> Tuple[Episode, str]:
        from .export import DatasetWriter

        episode = self.run(keep_images=self.cfg.export.save_images)
        writer = DatasetWriter(
            self.cfg.export.out_dir,
            fps=self.cfg.source.fps,
            fmt=self.cfg.export.fmt,
            save_images=self.cfg.export.save_images,
            image_format=self.cfg.export.image_format,
        )
        writer.write_episode(
            episode, images=self.last_images if self.cfg.export.save_images else None
        )
        out_dir = writer.finalize()
        return episode, out_dir
