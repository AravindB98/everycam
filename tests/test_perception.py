from everycam.config import PerceptionConfig, SourceConfig
from everycam.perception import Perceiver
from everycam.sources import build_source


def test_detects_hand_and_object_on_synthetic():
    perc = Perceiver(PerceptionConfig())
    frames = list(build_source(SourceConfig(kind="synthetic", max_frames=12, seed=2)))
    hands = objs = 0
    for f in frames:
        r = perc.process(f)
        hands += r.primary_hand() is not None
        objs += len(r.objects) > 0
    assert hands >= 8
    assert objs >= 8


def test_ego_motion_keys_present():
    perc = Perceiver(PerceptionConfig())
    frames = list(build_source(SourceConfig(kind="synthetic", max_frames=3)))
    results = [perc.process(f) for f in frames]
    assert set(results[-1].ego_motion) == {"dx", "dy", "rot", "mag"}


def test_hands_can_be_disabled():
    perc = Perceiver(PerceptionConfig(hands=False, objects=True))
    frames = list(build_source(SourceConfig(kind="synthetic", max_frames=2)))
    r = perc.process(frames[0])
    assert r.hands == []
