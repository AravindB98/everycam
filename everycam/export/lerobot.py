"""Write/read EveryCam episodes in a LeRobot-style on-disk layout.

Layout::

    out_dir/
      meta/info.json        # feature schema, fps, totals (LeRobot-style)
      meta/episodes.jsonl    # one row per episode
      meta/tasks.jsonl       # task_index -> task string
      meta/everycam.json     # provenance + consent + anonymization record
      data/chunk-000/episode_000000.parquet   # (jsonl fallback if no pyarrow)
      images/episode_000000/frame_000000.png ...

Parquet is used when ``pandas`` + ``pyarrow`` are available (LeRobot-native);
otherwise we fall back to JSONL so the toolkit still runs on a bare install.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..types import Episode
from .schema import ROBOT_TYPE, SCHEMA_VERSION, build_info


def have_parquet() -> bool:
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


def _dump_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class DatasetWriter:
    def __init__(
        self,
        out_dir: str,
        fps: float = 30.0,
        robot_type: str = ROBOT_TYPE,
        fmt: str = "auto",
        save_images: bool = True,
        image_format: str = "png",
    ) -> None:
        self.out_dir = out_dir
        self.fps = float(fps)
        self.robot_type = robot_type
        self.save_images = save_images
        self.image_format = image_format
        self.fmt = ("parquet" if have_parquet() else "jsonl") if fmt == "auto" else fmt
        self.episodes: List[Dict[str, Any]] = []
        self._task_list: List[str] = []
        self._tasks: Dict[str, int] = {}
        self.total_frames = 0
        self._global_index = 0
        self.image_shape: Optional[tuple] = None
        self.provenance = None
        os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "data", "chunk-000"), exist_ok=True)

    def _task_index(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks[task] = len(self._task_list)
            self._task_list.append(task)
        return self._tasks[task]

    def write_episode(
        self, episode: Episode, images: Optional[Sequence[np.ndarray]] = None
    ) -> int:
        ep_idx = len(self.episodes)
        ti = self._task_index(episode.task)
        h, w = episode.image_shape
        self.image_shape = (h, w)
        self.provenance = episode.provenance

        if self.save_images and images is not None:
            os.makedirs(
                os.path.join(self.out_dir, "images", f"episode_{ep_idx:06d}"),
                exist_ok=True,
            )

        rows: List[Dict[str, Any]] = []
        for i, step in enumerate(episode.steps):
            img_path = ""
            if self.save_images and images is not None and i < len(images):
                import cv2

                rel = os.path.join(
                    "images", f"episode_{ep_idx:06d}", f"frame_{i:06d}.{self.image_format}"
                )
                cv2.imwrite(os.path.join(self.out_dir, rel), images[i])
                img_path = rel
            aff = step.affordance_xy if step.affordance_xy is not None else (np.nan, np.nan)
            rows.append(
                {
                    "observation.state": [float(x) for x in step.state],
                    "action": [float(x) for x in step.action],
                    "observation.affordance_xy": [float(aff[0]), float(aff[1])],
                    "contact": bool(step.contact),
                    "observation.image": img_path,
                    "timestamp": float(step.timestamp),
                    "frame_index": int(i),
                    "episode_index": int(ep_idx),
                    "index": int(self._global_index),
                    "task_index": int(ti),
                }
            )
            self._global_index += 1

        self._write_rows(ep_idx, rows)
        self.episodes.append(
            {"episode_index": ep_idx, "tasks": [episode.task], "length": len(episode.steps)}
        )
        self.total_frames += len(episode.steps)
        return ep_idx

    def _write_rows(self, ep_idx: int, rows: List[Dict[str, Any]]) -> None:
        base = os.path.join(self.out_dir, "data", "chunk-000")
        if self.fmt == "parquet":
            import pandas as pd

            pd.DataFrame(rows).to_parquet(
                os.path.join(base, f"episode_{ep_idx:06d}.parquet"), index=False
            )
        else:
            with open(os.path.join(base, f"episode_{ep_idx:06d}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

    def finalize(self) -> str:
        h, w = self.image_shape or (0, 0)
        info = build_info(self.fps, self.robot_type, h, w)
        info.update(
            {
                "total_episodes": len(self.episodes),
                "total_frames": self.total_frames,
                "total_tasks": len(self._task_list),
                "data_format": self.fmt,
            }
        )
        _dump_json(os.path.join(self.out_dir, "meta", "info.json"), info)
        with open(os.path.join(self.out_dir, "meta", "episodes.jsonl"), "w") as f:
            for e in self.episodes:
                f.write(json.dumps(e) + "\n")
        with open(os.path.join(self.out_dir, "meta", "tasks.jsonl"), "w") as f:
            for i, t in enumerate(self._task_list):
                f.write(json.dumps({"task_index": i, "task": t}) + "\n")
        prov = self.provenance.to_dict() if self.provenance else {}
        _dump_json(
            os.path.join(self.out_dir, "meta", "everycam.json"),
            {
                "schema_version": SCHEMA_VERSION,
                "generator": "everycam",
                "privacy": {
                    "anonymized": prov.get("anonymized", True),
                    "policy": "Faces and license plates are blurred before storage. "
                    "Identity is never extracted, matched, or linked.",
                },
                "provenance": prov,
            },
        )
        return self.out_dir


# ----- reader (round-trips the dataset back for training/eval) ----------------------
def read_info(out_dir: str) -> Dict[str, Any]:
    with open(os.path.join(out_dir, "meta", "info.json")) as f:
        return json.load(f)


def _read_rows(out_dir: str, ep_idx: int, fmt: str) -> List[Dict[str, Any]]:
    base = os.path.join(out_dir, "data", "chunk-000")
    if fmt == "parquet":
        import pandas as pd

        return pd.read_parquet(
            os.path.join(base, f"episode_{ep_idx:06d}.parquet")
        ).to_dict("records")
    rows = []
    with open(os.path.join(base, f"episode_{ep_idx:06d}.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_episode_arrays(
    out_dir: str, ep_idx: int = 0, with_images: bool = True
) -> Dict[str, Any]:
    """Load one episode back into numpy arrays (states, actions, affordances, images)."""
    info = read_info(out_dir)
    rows = _read_rows(out_dir, ep_idx, info.get("data_format", "jsonl"))
    states = np.array([r["observation.state"] for r in rows], dtype=np.float32)
    actions = np.array([r["action"] for r in rows], dtype=np.float32)
    affs = np.array([r["observation.affordance_xy"] for r in rows], dtype=np.float32)
    contacts = np.array([bool(r["contact"]) for r in rows], dtype=bool)
    images = None
    if with_images:
        import cv2

        loaded = []
        ok = True
        for r in rows:
            p = r.get("observation.image", "")
            img = cv2.imread(os.path.join(out_dir, p)) if p else None
            if img is None:
                ok = False
                break
            loaded.append(img)
        if ok and loaded:
            images = np.stack(loaded)
    return {
        "info": info,
        "states": states,
        "actions": actions,
        "affordances": affs,
        "contacts": contacts,
        "images": images,
    }


def load_dataset_arrays(out_dir: str, with_images: bool = True) -> Dict[str, Any]:
    """Concatenate every episode in the dataset into flat numpy arrays."""
    info = read_info(out_dir)
    n_ep = max(1, int(info.get("total_episodes", 1)))
    states, actions, affs, contacts, imgs = [], [], [], [], []
    have_imgs = with_images
    for e in range(n_ep):
        d = load_episode_arrays(out_dir, e, with_images)
        states.append(d["states"])
        actions.append(d["actions"])
        affs.append(d["affordances"])
        contacts.append(d["contacts"])
        if d["images"] is None:
            have_imgs = False
        else:
            imgs.append(d["images"])
    return {
        "info": info,
        "states": np.concatenate(states) if states else np.zeros((0, 3), np.float32),
        "actions": np.concatenate(actions) if actions else np.zeros((0, 3), np.float32),
        "affordances": np.concatenate(affs) if affs else np.zeros((0, 2), np.float32),
        "contacts": np.concatenate(contacts) if contacts else np.zeros((0,), bool),
        "images": np.concatenate(imgs) if (have_imgs and imgs) else None,
    }
