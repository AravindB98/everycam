import os

from everycam import Pipeline, PipelineConfig


def test_demo_run_is_anonymized():
    cfg = PipelineConfig.demo()
    cfg.source.max_frames = 15
    ep = Pipeline(cfg).run()
    assert len(ep) == 15
    assert ep.provenance.anonymized is True


def test_run_and_export(tmp_path):
    cfg = PipelineConfig.demo()
    cfg.source.max_frames = 15
    cfg.export.out_dir = str(tmp_path / "ds")
    ep, out = Pipeline(cfg).run_and_export()
    assert os.path.exists(os.path.join(out, "meta", "info.json"))
    assert len(ep) == 15


def test_presets_exist():
    from everycam.config import PRESETS

    for name in PRESETS:
        cfg = PipelineConfig.from_preset(name)
        assert cfg.privacy.blur_faces in (True, False)
        assert cfg.source.kind in ("webcam", "file", "stream", "image_dir", "synthetic")
