"""Turn per-frame perception into robot-ready embodied signals.

The key idea behind learning from everyday video: a human hand is a *proxy
end-effector*. We convert the hand trajectory into normalized states and
delta-pose **actions** (exactly the format an imitation-learning / VLA policy
consumes), and read **grasp affordances** from hand-object contact.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..types import BBox, EmbodiedStep, Episode, Frame, PerceptionResult, Provenance


def _pad_box(box: BBox, pad: float, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    return (x1 - pad * bw, y1 - pad * bh, x2 + pad * bw, y2 + pad * bh)


def _point_in_box(pt: Tuple[float, float], box: BBox) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


class SignalExtractor:
    def __init__(self, contact_pad: float = 0.2) -> None:
        self.contact_pad = contact_pad

    def _contact_and_grasp(
        self, perc: PerceptionResult, w: int, h: int
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        hand = perc.primary_hand()
        if hand is None or not perc.objects:
            return False, None
        palm = hand.palm_center()
        if palm is None:
            return False, None
        obj = min(
            perc.objects,
            key=lambda o: (o.center()[0] - palm[0]) ** 2 + (o.center()[1] - palm[1]) ** 2,
        )
        contact = _point_in_box(palm, _pad_box(obj.bbox, self.contact_pad, w, h))
        cx, cy = obj.center()
        return bool(contact), (cx / w, cy / h)

    def extract(
        self,
        frames: Sequence[Frame],
        percs: Sequence[PerceptionResult],
        *,
        episode_id: str,
        task: str,
        fps: float,
        provenance: Provenance,
        image_shape: Tuple[int, int],
    ) -> Episode:
        h, w = image_shape
        states: List[np.ndarray] = []
        affs: List[Optional[Tuple[float, float]]] = []
        contacts: List[bool] = []
        idxs: List[int] = []
        ts: List[float] = []

        last = np.array([0.5, 0.5], dtype=np.float32)
        for f, p in zip(frames, percs):
            hand = p.primary_hand()
            palm = hand.palm_center() if hand else None
            if palm is not None:
                last = np.array([palm[0] / w, palm[1] / h], dtype=np.float32)
            contact, grasp = self._contact_and_grasp(p, w, h)
            states.append(np.array([last[0], last[1], float(contact)], dtype=np.float32))
            affs.append(grasp)
            contacts.append(contact)
            idxs.append(f.index)
            ts.append(f.timestamp)

        n = len(states)
        steps: List[EmbodiedStep] = []
        for i in range(n):
            action = (states[i + 1] - states[i]) if i < n - 1 else np.zeros(3, np.float32)
            steps.append(
                EmbodiedStep(
                    frame_index=idxs[i],
                    timestamp=ts[i],
                    state=states[i],
                    action=action.astype(np.float32),
                    affordance_xy=affs[i],
                    contact=contacts[i],
                )
            )
        return Episode(episode_id, task, fps, steps, provenance, (h, w))
