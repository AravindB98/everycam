"""Device-profile image transforms — simulate the domain shift of real cameras.

Each profile takes a clean frame and applies the characteristic distortions of a
device class (resolution, noise, color, blur, vignette, lens distortion). Training
on one device and testing on another then measures the *generalization gap* —
the "robust skills, brittle grounding" problem at the heart of egocentric robot
learning. Same idea, no hardware required.
"""

from __future__ import annotations

import numpy as np


def _noise(img, sigma, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _resample(img, scale):
    import cv2

    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _saturate(img, f):
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * f, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _contrast(img, f, bias=0.0):
    return np.clip((img.astype(np.float32) - 128) * f + 128 + bias, 0, 255).astype(np.uint8)


def _vignette(img, strength=0.5):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    r = r / r.max()
    mask = 1.0 - strength * (r ** 2)
    return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def _motion_blur(img, k=7):
    import cv2

    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def _barrel(img, strength=3e-5):
    import cv2

    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - w / 2, yy - h / 2
    factor = 1 + strength * (dx * dx + dy * dy)
    mapx = (w / 2 + dx * factor).astype(np.float32)
    mapy = (h / 2 + dy * factor).astype(np.float32)
    return cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def webcam(img, rng):
    return _noise(img, 3, rng)


def phone(img, rng):
    return _noise(_contrast(_saturate(img, 1.35), 1.1), 3, rng)


def dashcam(img, rng):
    return _vignette(_motion_blur(_contrast(_resample(img, 0.6), 0.85), 7), 0.4)


def fixed_cam(img, rng):
    return _noise(_resample(_saturate(_contrast(img, 0.9, -12), 0.4), 0.45), 8, rng)


def glasses(img, rng):
    return _noise(_barrel(_resample(img, 0.8)), 3, rng)


DEVICE_PROFILES = {
    "webcam": webcam,
    "phone": phone,
    "dashcam": dashcam,
    "fixed_cam": fixed_cam,
    "glasses": glasses,
}
