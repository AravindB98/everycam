"""Common interface for everyday-camera frame sources.

Every input kind (synthetic, webcam, video file, IP/CCTV stream, image folder)
implements :class:`FrameSource` and yields :class:`~everycam.types.Frame`
objects through a single, uniform iterator. Resizing and frame-count caps are
handled once, here, so individual adapters stay tiny.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple

import numpy as np

from ..types import Frame


class FrameSource(ABC):
    def __init__(
        self,
        source_id: str = "cam0",
        fps: float = 30.0,
        max_frames: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,  # (W, H)
    ) -> None:
        self.source_id = source_id
        self.fps = float(fps)
        self.max_frames = max_frames
        self.resize = resize

    @abstractmethod
    def _raw_frames(self) -> Iterator[np.ndarray]:
        """Yield raw BGR uint8 images (H, W, 3). Subclasses implement this."""
        raise NotImplementedError

    # -- public iteration ------------------------------------------------------------
    def __iter__(self) -> Iterator[Frame]:
        count = 0
        for item in self._raw_frames():
            if self.max_frames is not None and count >= self.max_frames:
                break
            # Adapters may yield a bare image or an (image, meta) tuple.
            if isinstance(item, tuple):
                img, meta = item
            else:
                img, meta = item, {}
            img = self._postprocess(img)
            yield Frame(
                index=count,
                timestamp=count / self.fps if self.fps else float(count),
                image=img,
                source_id=self.source_id,
                meta=dict(meta),
            )
            count += 1
        self.close()

    def _postprocess(self, img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:  # grayscale -> BGR
            img = np.stack([img] * 3, axis=-1)
        if self.resize is not None:
            import cv2

            w, h = self.resize
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

    def close(self) -> None:  # overridden by adapters that hold OS handles
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
