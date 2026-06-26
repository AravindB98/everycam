import numpy as np

from everycam.benchmark import DEVICE_PROFILES, build_device_arrays, run_benchmark
from everycam.sources.synthetic import SyntheticSource


def test_device_profiles_preserve_shape_and_dtype():
    frame = next(iter(SyntheticSource(n_frames=1, seed=0)))
    rng = np.random.default_rng(0)
    for fn in DEVICE_PROFILES.values():
        out = fn(frame.image, rng)
        assert out.shape == frame.image.shape
        assert out.dtype == np.uint8


def test_build_device_arrays():
    imgs, aff, con = build_device_arrays("phone", episodes=2, frames=15)
    assert len(imgs) == len(aff) == len(con) > 0
    assert aff.shape[1] == 2
    assert (aff >= 0).all() and (aff <= 1).all()


def test_benchmark_matrix_and_generalization_gap():
    s = run_benchmark(
        devices=["webcam", "fixed_cam", "glasses"], episodes=3, frames=20, epochs=80, seed=0
    )
    mat = np.array(s["grasp_mae"])
    assert mat.shape == (3, 3)
    assert np.isfinite(mat).all()
    cont = np.array(s["contact_acc"])
    assert ((cont >= 0) & (cont <= 1)).all()
    # In-device (diagonal) should be no worse than cross-device, modulo small-sample noise.
    assert s["in_device_grasp_mae"] <= s["cross_device_grasp_mae"] + 0.03
