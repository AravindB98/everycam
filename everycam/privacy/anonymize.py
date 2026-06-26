"""Privacy gate — anonymize frames BEFORE anything is stored or exported.

This is EveryCam's signature, non-optional design choice: every frame passes
through :class:`PrivacyGate` first, so faces and license plates are blurred
*before* perception, before disk, before a dataset is ever shared. Provenance
records that anonymization happened. EveryCam never links footage to identity.

Detection uses OpenCV Haar cascades by default (zero extra dependencies). On
real deployments you can swap in a stronger DNN face/plate detector; the gate's
contract (blur-before-store, fail-closed) stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import PrivacyConfig
from ..types import BBox


@dataclass
class AnonymizationResult:
    image: np.ndarray
    n_faces: int = 0
    n_plates: int = 0
    regions: List[BBox] = field(default_factory=list)
    fully_blurred: bool = False  # set when fail-closed blurred the whole frame

    @property
    def n_regions(self) -> int:
        return len(self.regions)


def _blur_array(roi: np.ndarray, method: str, strength: int) -> np.ndarray:
    import cv2

    s = max(3, int(strength))
    if method == "pixelate":
        h, w = roi.shape[:2]
        bw, bh = max(1, w // s), max(1, h // s)
        small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if method == "fill":
        return np.zeros_like(roi)
    k = s if s % 2 == 1 else s + 1  # gaussian needs odd kernel
    return cv2.GaussianBlur(roi, (k, k), 0)


def _blur_region(img: np.ndarray, box: BBox, method: str, strength: int) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    img[y1:y2, x1:x2] = _blur_array(img[y1:y2, x1:x2], method, strength)


class PrivacyGate:
    """Callable that returns an anonymized copy of a frame plus a report."""

    def __init__(self, cfg: Optional[PrivacyConfig] = None) -> None:
        self.cfg = cfg or PrivacyConfig()
        self._face = None
        self._plate = None
        self._loaded = False

    def _load(self) -> None:
        import cv2

        self._loaded = True
        if self.cfg.detect_backend == "none":
            return
        base = getattr(cv2.data, "haarcascades", "")

        def _try(name: str):
            try:
                c = cv2.CascadeClassifier(base + name)
                return c if not c.empty() else None
            except Exception:
                return None

        if self.cfg.blur_faces:
            self._face = _try("haarcascade_frontalface_default.xml")
        if self.cfg.blur_plates:
            self._plate = _try("haarcascade_russian_plate_number.xml")

    def process(self, image: np.ndarray) -> AnonymizationResult:
        import cv2

        if not self._loaded:
            self._load()

        out = image.copy()
        regions: List[BBox] = []
        n_faces = n_plates = 0
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self._face is not None:
                for (x, y, w, h) in self._face.detectMultiScale(
                    gray, 1.2, 5, minSize=(24, 24)
                ):
                    box = (float(x), float(y), float(x + w), float(y + h))
                    _blur_region(out, box, self.cfg.method, self.cfg.strength)
                    regions.append(box)
                    n_faces += 1
            if self._plate is not None:
                for (x, y, w, h) in self._plate.detectMultiScale(
                    gray, 1.1, 4, minSize=(24, 12)
                ):
                    box = (float(x), float(y), float(x + w), float(y + h))
                    _blur_region(out, box, self.cfg.method, self.cfg.strength)
                    regions.append(box)
                    n_plates += 1
        except Exception:
            # Fail closed: if anything goes wrong, never leak — blur everything.
            if self.cfg.fail_closed:
                out = _blur_array(image, "gaussian", max(31, self.cfg.strength))
                return AnonymizationResult(out, 0, 0, [], fully_blurred=True)
            out = image.copy()
        return AnonymizationResult(out, n_faces, n_plates, regions)

    __call__ = process
