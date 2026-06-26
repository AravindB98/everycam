"""Optional monocular depth.

Depth is an *optional* signal. By default EveryCam ships no heavy depth model,
so :class:`MonocularDepth` reports ``available == False`` and the pipeline simply
omits depth. To enable real metric-ish depth, install a backend such as
Depth-Anything and implement ``estimate`` — the schema already reserves a slot.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class MonocularDepth:
    def __init__(self, backend: str = "none") -> None:
        self.backend = backend
        self.available = backend not in ("none", "", None)

    def estimate(self, image: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
        raise NotImplementedError(
            "Install a depth backend (e.g. Depth-Anything) and implement estimate()."
        )
