"""Lightweight, dependency-free object proposals.

The heuristic detector finds salient, saturated, non-skin blobs (the things a
hand tends to interact with). It is intentionally simple so the pipeline runs
anywhere; swap in YOLO/Detic for production by implementing ``.detect``.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..types import Detection

_SKIN_HUE_MAX = 25  # exclude warm skin tones so hands aren't reported as objects


class HeuristicObjectDetector:
    backend = "heuristic"

    def __init__(self, min_area_frac: float = 0.01, max_objects: int = 5) -> None:
        self.min_area_frac = min_area_frac
        self.max_objects = max_objects

    def detect(self, image: np.ndarray) -> List[Detection]:
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        mask = ((sat > 70) & (val > 50)).astype(np.uint8) * 255
        skin = ((hue < _SKIN_HUE_MAX) | (hue > 160)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image.shape[:2]
        min_area = self.min_area_frac * h * w
        dets: List[Detection] = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[: self.max_objects]:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            dets.append(
                Detection(
                    bbox=(float(x), float(y), float(x + bw), float(y + bh)),
                    score=float(min(1.0, area / (0.2 * h * w))),
                    label="object",
                )
            )
        return dets


class NullObjectDetector:
    backend = "none"

    def detect(self, image: np.ndarray) -> List[Detection]:
        return []
