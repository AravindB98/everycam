"""Dataset schema — a LeRobot-style feature spec plus EveryCam's privacy sidecar.

We mirror LeRobot's ``meta/info.json`` feature layout (observation.state, action,
observation.image, timestamps, indices) so the output drops into the LeRobot
ecosystem, and we add EveryCam-specific keys: ``observation.affordance_xy`` and
``contact``, plus a ``meta/everycam.json`` provenance/consent record.
"""

from __future__ import annotations

from typing import Any, Dict

SCHEMA_VERSION = "0.1.0"
CODEBASE_VERSION = "v2.0"  # LeRobot dataset codebase version we target
ROBOT_TYPE = "human_proxy_ee"  # the human hand acts as a proxy end-effector


def features_schema(h: int, w: int) -> Dict[str, Any]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [3],
            "names": ["ee_x_norm", "ee_y_norm", "contact"],
        },
        "action": {
            "dtype": "float32",
            "shape": [3],
            "names": ["d_ee_x", "d_ee_y", "d_contact"],
        },
        "observation.image": {
            "dtype": "image",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.affordance_xy": {
            "dtype": "float32",
            "shape": [2],
            "names": ["grasp_x_norm", "grasp_y_norm"],
        },
        "contact": {"dtype": "bool", "shape": [1], "names": ["in_contact"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def build_info(fps: float, robot_type: str, h: int, w: int) -> Dict[str, Any]:
    return {
        "codebase_version": CODEBASE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "robot_type": robot_type,
        "fps": float(fps),
        "chunks_size": 1000,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "features": features_schema(h, w),
    }
