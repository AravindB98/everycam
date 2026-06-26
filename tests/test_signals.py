import numpy as np

from everycam.config import PerceptionConfig, SourceConfig
from everycam.perception import Perceiver
from everycam.signals import SignalExtractor
from everycam.sources import build_source
from everycam.types import Provenance


def _episode(seed=0, n=20):
    perc = Perceiver(PerceptionConfig())
    frames = list(build_source(SourceConfig(kind="synthetic", max_frames=n, seed=seed)))
    percs = [perc.process(f) for f in frames]
    return SignalExtractor().extract(
        frames,
        percs,
        episode_id="e",
        task="t",
        fps=30.0,
        provenance=Provenance("synthetic"),
        image_shape=(180, 240),
    )


def test_episode_shapes():
    ep = _episode(n=20)
    assert len(ep) == 20
    assert ep.states().shape == (20, 3)
    assert ep.actions().shape == (20, 3)


def test_action_is_state_delta():
    ep = _episode(n=18)
    S, A = ep.states(), ep.actions()
    assert np.allclose(A[:-1], S[1:] - S[:-1], atol=1e-5)
    assert np.allclose(A[-1], 0.0)


def test_contact_detected():
    assert _episode(seed=1, n=25).contact_ratio() > 0.0
