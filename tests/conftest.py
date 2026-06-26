import pytest


@pytest.fixture
def demo_dataset(tmp_path):
    """Build a small multi-episode synthetic dataset for tests."""
    from everycam import Pipeline, PipelineConfig
    from everycam.export import DatasetWriter

    out = str(tmp_path / "ds")
    writer = DatasetWriter(out, fps=30.0)
    for seed in range(4):
        cfg = PipelineConfig.demo()
        cfg.source.seed = seed
        cfg.source.max_frames = 25
        pipe = Pipeline(cfg)
        writer.write_episode(pipe.run(keep_images=True), images=pipe.last_images)
    writer.finalize()
    return out
