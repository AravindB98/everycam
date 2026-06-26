import numpy as np
import pytest

from everycam.config import SourceConfig
from everycam.sources import build_source


def test_synthetic_yields_frames():
    frames = list(build_source(SourceConfig(kind="synthetic", max_frames=10, seed=0)))
    assert len(frames) == 10
    f = frames[0]
    assert f.image.shape == (180, 240, 3)
    assert f.image.dtype == np.uint8
    assert "gt_contact" in f.meta and "gt_grasp_xy" in f.meta


def test_synthetic_has_contact_phase():
    frames = list(build_source(SourceConfig(kind="synthetic", max_frames=20, seed=1)))
    contacts = [f.meta["gt_contact"] for f in frames]
    assert any(contacts) and not all(contacts)


def test_resize_applies():
    frames = list(
        build_source(SourceConfig(kind="synthetic", max_frames=3, resize=(120, 90)))
    )
    assert frames[0].image.shape == (90, 120, 3)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_source(SourceConfig(kind="nope"))


def test_file_kind_requires_path():
    with pytest.raises(ValueError):
        build_source(SourceConfig(kind="file", path=None))
