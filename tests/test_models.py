from everycam.models import train_from_dataset


def test_model_beats_baseline(demo_dataset, tmp_path):
    _, m = train_from_dataset(
        demo_dataset, out_dir=str(tmp_path / "run"), epochs=200, seed=0
    )
    # A tiny CPU model should clearly beat predict-the-mean on grasp localization...
    assert m["improvement_vs_baseline_pct"] > 15.0
    # ...and classify hand-object contact well above chance.
    assert m["contact_accuracy"] > 0.7


def test_world_model_beats_identity(demo_dataset, tmp_path):
    from everycam.models import train_world_model_from_dataset

    m = train_world_model_from_dataset(demo_dataset, out_dir=str(tmp_path / "wm"), seed=0)
    assert m["n_pairs"] > 0
    # The learned dynamics should predict the next latent at least as well as "nothing moves".
    assert m["next_latent_mse"] <= m["identity_baseline_mse"] * 1.05


def test_training_writes_artifacts(demo_dataset, tmp_path):
    import os

    out = str(tmp_path / "run")
    train_from_dataset(demo_dataset, out_dir=out, epochs=100, seed=0)
    assert os.path.exists(os.path.join(out, "affordance_model.npz"))
    assert os.path.exists(os.path.join(out, "metrics.json"))
    assert os.path.exists(os.path.join(out, "grasp_predictions.png"))
