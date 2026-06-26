"""Perception: turn anonymized frames into hands, objects, and ego-motion."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import PerceptionConfig
from ..types import Frame, PerceptionResult
from .egomotion import compute_ego_motion
from .hands import build_hand_detector
from .objects import HeuristicObjectDetector, NullObjectDetector

__all__ = ["Perceiver", "build_hand_detector", "compute_ego_motion"]


class Perceiver:
    """Run the enabled perception modules over a frame stream (stateful: ego-motion)."""

    def __init__(self, cfg: Optional[PerceptionConfig] = None) -> None:
        self.cfg = cfg or PerceptionConfig()
        self.hands = build_hand_detector(self.cfg) if self.cfg.hands else None
        if self.cfg.objects:
            self.objects = (
                NullObjectDetector()
                if self.cfg.object_backend == "none"
                else HeuristicObjectDetector()
            )
        else:
            self.objects = None
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None

    def process(self, frame: Frame) -> PerceptionResult:
        hands = self.hands.detect(frame.image) if self.hands else []
        objects = self.objects.detect(frame.image) if self.objects else []
        ego = None
        if self.cfg.ego_motion:
            import cv2

            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            ego = compute_ego_motion(self._prev_gray, gray)
            self._prev_gray = gray
        return PerceptionResult(
            frame_index=frame.index,
            timestamp=frame.timestamp,
            hands=hands,
            objects=objects,
            depth=None,
            ego_motion=ego,
            anonymized=bool(frame.meta.get("anonymized", False)),
        )
