"""Hand detection with a graceful dependency ladder.

- ``mediapipe`` backend: 21 high-quality 3D landmarks (install ``[hands]``).
- ``heuristic`` backend: dependency-free skin-segmentation fallback so the
  pipeline runs on a bare ``numpy + opencv`` install (used in CI and the demo).

Both return a list of :class:`~everycam.types.HandObs`.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..config import PerceptionConfig
from ..types import HandObs

# HSV skin ranges (OpenCV hue is 0..179). Two bands cover warm/reddish skin.
_SKIN_LO1, _SKIN_HI1 = np.array([0, 40, 80]), np.array([25, 190, 255])
_SKIN_LO2, _SKIN_HI2 = np.array([160, 40, 80]), np.array([179, 190, 255])


class HeuristicHands:
    """Skin-segmentation hand detector — no extra dependencies."""

    backend = "heuristic"

    def __init__(self, min_area_frac: float = 0.01) -> None:
        self.min_area_frac = min_area_frac

    def detect(self, image: np.ndarray) -> List[HandObs]:
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _SKIN_LO1, _SKIN_HI1) | cv2.inRange(
            hsv, _SKIN_LO2, _SKIN_HI2
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image.shape[:2]
        min_area = self.min_area_frac * h * w
        out: List[HandObs] = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            m = cv2.moments(c)
            cx = m["m10"] / m["m00"] if m["m00"] else x + bw / 2
            cy = m["m01"] / m["m00"] if m["m00"] else y + bh / 2
            top = c[c[:, :, 1].argmin()][0]  # fingertip proxy = topmost point
            kp = np.array(
                [[cx, cy], [x, y], [x + bw, y], [x, y + bh], [x + bw, y + bh],
                 [float(top[0]), float(top[1])]],
                dtype=np.float32,
            )
            score = float(min(1.0, area / (0.15 * h * w)))
            out.append(
                HandObs(
                    present=True,
                    keypoints_2d=kp,
                    score=score,
                    bbox=(float(x), float(y), float(x + bw), float(y + bh)),
                    backend=self.backend,
                )
            )
        return out


class MediaPipeHands:
    """21-landmark hand tracking via MediaPipe (optional dependency)."""

    backend = "mediapipe"

    def __init__(self, max_hands: int = 2) -> None:
        import mediapipe as mp  # raises if not installed

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=max_hands, min_detection_confidence=0.4
        )

    def detect(self, image: np.ndarray) -> List[HandObs]:
        import cv2

        h, w = image.shape[:2]
        res = self._hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        out: List[HandObs] = []
        if not res.multi_hand_landmarks:
            return out
        handed = res.multi_handedness or []
        for i, lms in enumerate(res.multi_hand_landmarks):
            kp = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark], dtype=np.float32)
            label, score = "unknown", 0.9
            if i < len(handed):
                cl = handed[i].classification[0]
                label, score = cl.label.lower(), float(cl.score)
            x1, y1 = kp[:, 0].min(), kp[:, 1].min()
            x2, y2 = kp[:, 0].max(), kp[:, 1].max()
            out.append(
                HandObs(True, kp, label, score, (x1, y1, x2, y2), self.backend)
            )
        return out


class NullHands:
    backend = "none"

    def detect(self, image: np.ndarray) -> List[HandObs]:
        return []


def build_hand_detector(cfg: PerceptionConfig):
    backend = cfg.hand_backend
    if backend in ("auto", "mediapipe"):
        try:
            return MediaPipeHands()
        except Exception:
            if backend == "mediapipe":
                raise
            return HeuristicHands()
    if backend == "heuristic":
        return HeuristicHands()
    return NullHands()
