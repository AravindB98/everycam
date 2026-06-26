"""Analyze any EveryCam/LeRobot-style dataset — real or synthetic.

Contributions come with *analysis*, not just data: dataset stats plus, when the
anonymized frames are available, a quick affordance+contact model evaluation.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from ..export import load_dataset_arrays, read_info


def _provenance(dataset_dir: str) -> dict:
    p = os.path.join(dataset_dir, "meta", "everycam.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f).get("provenance", {})
    return {}


def analyze_dataset(dataset_dir: str, out_dir: Optional[str] = None, train: bool = True) -> Dict:
    info = read_info(dataset_dir)
    d = load_dataset_arrays(dataset_dir, with_images=True)
    prov = _provenance(dataset_dir)
    n = len(d["states"])
    act = d["actions"]
    stats: Dict = {
        "dataset": os.path.basename(os.path.abspath(dataset_dir)),
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_frames": int(info.get("total_frames", n)),
        "has_images": d["images"] is not None,
        "contact_ratio": round(float(np.mean(d["contacts"])), 3) if n else 0.0,
        "mean_action_magnitude": round(float(np.mean(np.linalg.norm(act[:, :2], axis=1))), 4) if n else 0.0,
        "device": prov.get("source_type"),
        "consent": prov.get("consent"),
        "anonymized": prov.get("anonymized"),
    }
    if train and d["images"] is not None and n >= 8:
        from ..models import train_from_dataset

        _, m = train_from_dataset(dataset_dir, epochs=300)
        stats["model"] = m
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "analysis.json"), "w") as f:
            json.dump(stats, f, indent=2)
    return stats
