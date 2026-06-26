"""Camera ego-motion from dense optical flow (Farneback).

Returns a compact summary {dx, dy, rot, mag} per frame — enough to characterize
how a wearable / dashcam / handheld camera is moving, which matters for turning
egocentric video into world-model and navigation signals.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_ego_motion(
    prev_gray: Optional[np.ndarray], gray: np.ndarray
) -> Dict[str, float]:
    if prev_gray is None or prev_gray.shape != gray.shape:
        return {"dx": 0.0, "dy": 0.0, "rot": 0.0, "mag": 0.0}
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 2, 13, 3, 5, 1.2, 0
    )
    fx, fy = flow[..., 0], flow[..., 1]
    dx, dy = float(fx.mean()), float(fy.mean())
    mag = float(np.sqrt(fx**2 + fy**2).mean())
    # crude rotation proxy: curl of the flow field about the image center
    h, w = gray.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    rx, ry = xs - w / 2.0, ys - h / 2.0
    norm = (rx**2 + ry**2).mean() + 1e-6
    rot = float(((rx * fy - ry * fx).mean()) / norm)
    return {"dx": dx, "dy": dy, "rot": rot, "mag": mag}
