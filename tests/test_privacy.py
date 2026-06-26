import numpy as np

from everycam.config import PrivacyConfig
from everycam.privacy import PrivacyGate
from everycam.privacy.anonymize import _blur_region


def test_gate_preserves_shape_and_dtype():
    img = np.random.default_rng(0).integers(0, 255, (180, 240, 3)).astype(np.uint8)
    res = PrivacyGate(PrivacyConfig()).process(img)
    assert res.image.shape == img.shape
    assert res.image.dtype == np.uint8


def test_blur_reduces_local_variance():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3)).astype(np.uint8)
    before = float(img[10:40, 10:40].var())
    _blur_region(img, (10, 10, 40, 40), "gaussian", 31)
    after = float(img[10:40, 10:40].var())
    assert after < before


def test_fail_closed_blurs_everything():
    # A 2D image makes cvtColor(BGR2GRAY) raise -> fail-closed path engages.
    gate = PrivacyGate(PrivacyConfig(fail_closed=True))
    bad = np.zeros((50, 50), np.uint8)
    res = gate.process(bad)
    assert res.fully_blurred is True
