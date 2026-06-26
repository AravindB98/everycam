import json
import os

from everycam.export import load_dataset_arrays, read_info


def test_roundtrip(demo_dataset):
    info = read_info(demo_dataset)
    assert info["total_episodes"] == 4
    assert info["total_frames"] > 0
    d = load_dataset_arrays(demo_dataset, with_images=True)
    n = d["states"].shape[0]
    assert n == info["total_frames"]
    assert d["actions"].shape == (n, 3)
    assert d["affordances"].shape == (n, 2)
    assert d["images"] is not None and d["images"].shape[0] == n


def test_meta_files_and_privacy_record(demo_dataset):
    for f in ("info.json", "episodes.jsonl", "tasks.jsonl", "everycam.json"):
        assert os.path.exists(os.path.join(demo_dataset, "meta", f))
    with open(os.path.join(demo_dataset, "meta", "everycam.json")) as fh:
        ev = json.load(fh)
    assert ev["privacy"]["anonymized"] is True
    assert ev["provenance"]["anonymized"] is True


def test_schema_has_lerobot_keys(demo_dataset):
    info = read_info(demo_dataset)
    feats = info["features"]
    for key in ("observation.state", "action", "observation.image"):
        assert key in feats
    assert feats["observation.state"]["shape"] == [3]
