"""A dependency-light synthetic 'hand manipulates an object' clip.

Each frame carries ground-truth annotations in ``frame.meta`` (hand position,
object bbox, contact flag, grasp point) so the entire pipeline — perception,
signal extraction, dataset export, and model training/eval — can be exercised
and unit-tested with **zero hardware, network, or GPU**. This is what makes
EveryCam "run completely" on any machine.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from .base import FrameSource

SKIN_BGR = (140, 175, 220)
OBJECT_BGR = (200, 120, 60)


class SyntheticSource(FrameSource):
    def __init__(
        self,
        n_frames: int = 60,
        width: int = 240,
        height: int = 180,
        seed: int = 0,
        fps: float = 30.0,
        resize: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            source_id="synthetic", fps=fps, max_frames=n_frames, resize=resize
        )
        self.n = int(n_frames)
        self.W = int(width)
        self.H = int(height)
        self.seed = int(seed)
        # Randomize the scene per seed so grasp locations span the frame. This makes
        # "predict the average" a poor baseline and forces a model to localize from
        # pixels — i.e. it has to actually learn.
        rng = np.random.default_rng(seed)
        W, H = self.W, self.H
        self.obj0 = np.array([rng.uniform(0.28, 0.55) * W, rng.uniform(0.32, 0.66) * H])
        self.objF = np.array([rng.uniform(0.60, 0.86) * W, rng.uniform(0.32, 0.66) * H])
        self.hand_start = np.array([rng.uniform(0.05, 0.20) * W, rng.uniform(0.70, 0.93) * H])
        self.hand_end = np.array([rng.uniform(0.80, 0.95) * W, rng.uniform(0.74, 0.95) * H])

    def _trajectory(self, i: int):
        """Return (hand_xy, obj_center_xy, contact) for frame i (pixel coords)."""
        N = self.n
        t1, t2 = int(0.4 * N), int(0.8 * N)
        obj0, objF = self.obj0, self.objF
        hand_start, hand_end = self.hand_start, self.hand_end
        if i < t1:  # reach toward the object
            a = i / max(1, t1)
            hand = hand_start * (1 - a) + obj0 * a
            obj = obj0.copy()
            contact = False
        elif i < t2:  # grasp + transport
            a = (i - t1) / max(1, (t2 - t1))
            obj = obj0 * (1 - a) + objF * a
            hand = obj + np.array([0.0, -2.0])
            contact = True
        else:  # release + retreat
            a = (i - t2) / max(1, (N - t2))
            obj = objF.copy()
            hand = objF * (1 - a) + hand_end * a
            contact = False
        return hand, obj, contact

    def _raw_frames(self) -> Iterator[tuple]:
        import cv2

        rng = np.random.default_rng(self.seed)
        W, H = self.W, self.H
        ow, oh = 0.18 * W, 0.24 * H
        ramp = np.linspace(180, 130, W, dtype=np.float32)
        base = np.stack([np.tile(ramp, (H, 1))] * 3, axis=-1)
        base[..., 2] += 8.0  # faint warm tint

        for i in range(self.n):
            img = (base + rng.normal(0, 3, (H, W, 3))).clip(0, 255).astype(np.uint8)
            hand, obj, contact = self._trajectory(i)

            x1, y1 = int(obj[0] - ow / 2), int(obj[1] - oh / 2)
            x2, y2 = int(obj[0] + ow / 2), int(obj[1] + oh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), OBJECT_BGR, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 40, 20), 2)

            hc = (int(hand[0]), int(hand[1]))
            cv2.ellipse(img, hc, (22, 16), 0, 0, 360, SKIN_BGR, -1)
            for fa in (-0.6, -0.3, 0.0, 0.3, 0.6):
                fx = int(hand[0] + 26 * np.cos(fa - 1.5708))
                fy = int(hand[1] + 26 * np.sin(fa - 1.5708))
                cv2.circle(img, (fx, fy), 5, SKIN_BGR, -1)

            meta = {
                "gt_hand_xy": (float(hand[0]), float(hand[1])),
                "gt_object_bbox": (float(x1), float(y1), float(x2), float(y2)),
                "gt_contact": bool(contact),
                "gt_grasp_xy": (float(obj[0]), float(obj[1])),
            }
            yield img, meta
