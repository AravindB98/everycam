"""Cross-device generalization benchmark.

Train the affordance model on each simulated device, evaluate it on every device,
and report the grasp-MAE transfer matrix plus the in-device vs. cross-device gap.
Uses the synthetic generator's ground-truth grasp labels so the matrix isolates the
*model's* visual-domain generalization (not perception drift).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from ..config import PrivacyConfig
from ..models.affordance import AffordanceModel
from ..privacy import PrivacyGate
from ..sources.synthetic import SyntheticSource
from .devices import DEVICE_PROFILES


def build_device_arrays(device, episodes=8, frames=40, seed0=0, anonymize=True):
    profile = DEVICE_PROFILES[device]
    gate = PrivacyGate(PrivacyConfig()) if anonymize else None
    rng = np.random.default_rng(1234)
    imgs, affs, cons = [], [], []
    for s in range(episodes):
        for f in SyntheticSource(n_frames=frames, seed=seed0 + s):
            img = profile(f.image, rng)
            if gate is not None:
                img = gate.process(img).image
            h, w = img.shape[:2]
            gx, gy = f.meta["gt_grasp_xy"]
            imgs.append(img)
            affs.append([gx / w, gy / h])
            cons.append(bool(f.meta["gt_contact"]))
    return np.array(imgs), np.array(affs, np.float32), np.array(cons, bool)


def run_benchmark(
    devices: Optional[List[str]] = None,
    episodes=8,
    frames=40,
    epochs=250,
    seed=0,
    out_dir: Optional[str] = None,
) -> Dict:
    devices = devices or list(DEVICE_PROFILES)
    rng = np.random.default_rng(seed)
    data, splits, models = {}, {}, {}
    for d in devices:
        imgs, aff, con = build_device_arrays(d, episodes, frames)
        idx = rng.permutation(len(imgs))
        nv = max(1, int(0.3 * len(imgs)))
        val, tr = idx[:nv], idx[nv:]
        data[d], splits[d] = (imgs, aff, con), (tr, val)
        models[d] = AffordanceModel(seed=seed).fit(imgs[tr], aff[tr], con[tr], epochs=epochs)

    n = len(devices)
    grasp = np.zeros((n, n))
    contact = np.zeros((n, n))
    for i, dtr in enumerate(devices):
        for j, dte in enumerate(devices):
            imgs, aff, con = data[dte]
            val = splits[dte][1]
            grasp[i, j] = float(np.abs(models[dtr].predict_grasp(imgs[val]) - aff[val]).mean())
            contact[i, j] = float((models[dtr].predict_contact(imgs[val]) == con[val]).mean())

    diag = np.diag(grasp)
    off = grasp[~np.eye(n, dtype=bool)]
    summary = {
        "devices": devices,
        "grasp_mae": np.round(grasp, 4).tolist(),
        "contact_acc": np.round(contact, 4).tolist(),
        "in_device_grasp_mae": round(float(diag.mean()), 4),
        "cross_device_grasp_mae": round(float(off.mean()), 4),
        "generalization_gap": round(float(off.mean() - diag.mean()), 4),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
            json.dump(summary, f, indent=2)
        save_heatmap(grasp, devices, os.path.join(out_dir, "benchmark_matrix.png"))
    return summary


def save_heatmap(mat, devices, path):
    import cv2

    n = len(devices)
    cell, pad_l, pad_t = 96, 96, 76
    H, W = pad_t + n * cell + 24, pad_l + n * cell + 20
    canvas = np.full((H, W, 3), 248, np.uint8)
    rng = mat.max() - mat.min() + 1e-9
    for i in range(n):
        for j in range(n):
            v = (mat[i, j] - mat.min()) / rng
            color = cv2.applyColorMap(np.array([[int(v * 255)]], np.uint8), cv2.COLORMAP_INFERNO)[0, 0].tolist()
            x0, y0 = pad_l + j * cell, pad_t + i * cell
            cv2.rectangle(canvas, (x0, y0), (x0 + cell - 2, y0 + cell - 2), color, -1)
            tc = (255, 255, 255) if v > 0.55 else (20, 20, 20)
            cv2.putText(canvas, f"{mat[i, j]:.3f}", (x0 + 14, y0 + cell // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tc, 1)
    for k, d in enumerate(devices):
        cv2.putText(canvas, d[:9], (pad_l + k * cell + 6, pad_t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(canvas, d[:9], (4, pad_t + k * cell + cell // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(canvas, "Grasp MAE  -  train (rows) to test (cols)", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, "lower = better; diagonal = in-device", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)
    cv2.imwrite(path, canvas)
    return path
