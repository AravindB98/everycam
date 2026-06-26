"""Core data types shared across the EveryCam pipeline.

These are deliberately lightweight dataclasses (numpy is the only heavy
dependency) so they are easy to serialize, test, and reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Frame:
    """A single RGB frame (stored BGR, OpenCV convention) plus light metadata."""

    index: int
    timestamp: float  # seconds since clip start
    image: np.ndarray  # (H, W, 3) uint8, BGR
    source_id: str = "cam0"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass
class Detection:
    bbox: BBox
    score: float = 1.0
    label: str = "object"

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class HandObs:
    """A detected hand. ``keypoints_2d`` is (K, 2) in pixel coordinates.

    K = 21 for the MediaPipe backend, or a small landmark set for the
    dependency-free heuristic backend.
    """

    present: bool = False
    keypoints_2d: Optional[np.ndarray] = None
    handedness: str = "unknown"  # left | right | unknown
    score: float = 0.0
    bbox: Optional[BBox] = None
    backend: str = "none"

    def palm_center(self) -> Optional[Tuple[float, float]]:
        if self.keypoints_2d is not None and len(self.keypoints_2d):
            c = self.keypoints_2d.mean(axis=0)
            return (float(c[0]), float(c[1]))
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        return None


@dataclass
class PerceptionResult:
    frame_index: int
    timestamp: float
    hands: List[HandObs] = field(default_factory=list)
    objects: List[Detection] = field(default_factory=list)
    depth: Optional[np.ndarray] = None
    ego_motion: Optional[Dict[str, float]] = None  # {dx, dy, rot, mag}
    anonymized: bool = False

    def primary_hand(self) -> Optional[HandObs]:
        present = [h for h in self.hands if h.present]
        return max(present, key=lambda h: h.score) if present else None


@dataclass
class EmbodiedStep:
    """One timestep of robot-ready signal derived from a frame."""

    frame_index: int
    timestamp: float
    state: np.ndarray  # proxy end-effector state, e.g. [x, y, openness]
    action: np.ndarray  # delta-pose action to reach the next state
    affordance_xy: Optional[Tuple[float, float]] = None  # grasp point, normalized 0..1
    contact: bool = False
    image_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Provenance:
    """Provenance + consent metadata — written into every exported dataset.

    Recording *where data came from and under what consent* is the core of
    EveryCam's privacy-by-design contract.
    """

    source_type: str  # webcam|phone|dashcam|fixed_cam|glasses|file|synthetic|stream
    device: str = "unknown"
    anonymized: bool = True
    consent: str = "unspecified"  # self|participant-consented|public-cc|synthetic|...
    license: str = "unspecified"
    tool_version: str = "0.1.0"
    created_at: str = field(default_factory=_now_iso)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "device": self.device,
            "anonymized": self.anonymized,
            "consent": self.consent,
            "license": self.license,
            "tool_version": self.tool_version,
            "created_at": self.created_at,
            "notes": self.notes,
        }


@dataclass
class Episode:
    episode_id: str
    task: str
    fps: float
    steps: List[EmbodiedStep]
    provenance: Provenance
    image_shape: Tuple[int, int]  # (H, W)

    def __len__(self) -> int:
        return len(self.steps)

    def states(self) -> np.ndarray:
        return np.stack([s.state for s in self.steps]) if self.steps else np.zeros((0, 0))

    def actions(self) -> np.ndarray:
        return np.stack([s.action for s in self.steps]) if self.steps else np.zeros((0, 0))

    def contact_ratio(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([s.contact for s in self.steps]))

    def summary(self) -> str:
        return (
            f"Episode '{self.episode_id}' | task='{self.task}' | "
            f"{len(self)} steps @ {self.fps:.1f} fps | "
            f"source={self.provenance.source_type} | "
            f"anonymized={self.provenance.anonymized} | "
            f"contact={self.contact_ratio():.0%}"
        )
