"""Real-camera frame sources backed by OpenCV.

A single :class:`CV2Source` covers webcams (integer device index), recorded
files (phone / dashcam / glasses exports), and IP / RTSP / HTTP streams
(fixed and CCTV-style cameras) — OpenCV's ``VideoCapture`` accepts all of them.
:class:`ImageDirSource` reads a folder of image frames.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple, Union

import numpy as np

from .base import FrameSource


class CV2Source(FrameSource):
    def __init__(
        self,
        target: Union[int, str],
        source_id: str = "cam0",
        fps: float = 30.0,
        max_frames: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(source_id=source_id, fps=fps, max_frames=max_frames, resize=resize)
        self.target = target
        self._cap = None

    def _raw_frames(self) -> Iterator[np.ndarray]:
        import cv2

        cap = cv2.VideoCapture(self.target)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video source {self.target!r}. "
                "Check the path/URL, camera index, or codecs."
            )
        self._cap = cap
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield frame
        finally:
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageDirSource(FrameSource):
    def __init__(self, directory: str, **kwargs) -> None:
        super().__init__(source_id=kwargs.pop("source_id", "image_dir"), **kwargs)
        self.directory = directory

    def _raw_frames(self) -> Iterator[np.ndarray]:
        import glob
        import os

        import cv2

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files: list[str] = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.directory, e)))
        for f in sorted(files):
            img = cv2.imread(f)
            if img is not None:
                yield img
