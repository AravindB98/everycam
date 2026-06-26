"""A tiny latent **world model** — the AI's "imagination".

It learns one-step forward dynamics: given the current frame's latent (a small
image feature) and the action taken, predict how the latent *changes* — i.e. what
the next frame will look like, in feature space. Same idea as modern video world
models (V-JEPA 2, etc.), shrunk to a closed-form ridge model that trains in a blink
on CPU over an EveryCam dataset.

We report two things:
  - **1-step** next-latent error on a held-out split (the rigorous metric), and
  - a **k-step rollout** where the model imagines several frames ahead by feeding its
    own predictions back in — versus a "nothing moves" baseline, whose error grows
    with the horizon. The rollout makes the value of learned dynamics visible.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .affordance import RidgeRegressor, featurize


def _episode_pairs(lat: np.ndarray, actions: np.ndarray):
    x = np.hstack([lat[:-1], actions[:-1]])   # [latent_t, action_t]
    d_next = lat[1:] - lat[:-1]               # target: change in latent
    return x, d_next, lat[:-1], lat[1:]


def _rollout(model, eps: List[Tuple[np.ndarray, np.ndarray]], k: int):
    errs, idents = [], []
    for lat, acts in eps:
        if len(lat) <= k:
            continue
        for t in range(len(lat) - k):
            cur = lat[t].copy()
            for j in range(k):
                cur = cur + model.predict(np.hstack([cur, acts[t + j]])[None, :])[0]
            errs.append(float(np.mean((cur - lat[t + k]) ** 2)))
            idents.append(float(np.mean((lat[t] - lat[t + k]) ** 2)))
    if not errs:
        return None
    return float(np.mean(errs)), float(np.mean(idents))


def train_world_model_from_dataset(
    dataset_dir: str, out_dir: Optional[str] = None, feat_size: int = 8,
    l2: float = 20.0, val_frac: float = 0.3, seed: int = 0, rollout_k: int = 5,
) -> Dict:
    from ..export import load_episode_arrays, read_info

    info = read_info(dataset_dir)
    n_ep = max(1, int(info.get("total_episodes", 1)))
    Xs, dYs, LTs, LNs, eps = [], [], [], [], []
    for e in range(n_ep):
        d = load_episode_arrays(dataset_dir, e, with_images=True)
        if d["images"] is None or len(d["images"]) < 2:
            continue
        lat = featurize(d["images"], feat_size)
        x, dy, lt, ln = _episode_pairs(lat, d["actions"])
        Xs.append(x); dYs.append(dy); LTs.append(lt); LNs.append(ln)
        eps.append((lat, d["actions"]))
    if not Xs:
        raise RuntimeError("World model needs image episodes with >= 2 frames each.")

    X = np.vstack(Xs); dY = np.vstack(dYs); LT = np.vstack(LTs); LN = np.vstack(LNs)
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nv = max(1, int(val_frac * n))
    val, tr = idx[:nv], idx[nv:]

    model = RidgeRegressor(l2).fit(X[tr], dY[tr])
    pred_next = LT[val] + model.predict(X[val])
    model_mse = float(np.mean((pred_next - LN[val]) ** 2))
    identity_mse = float(np.mean((LT[val] - LN[val]) ** 2))

    metrics = {
        "n_pairs": int(n),
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "feat_size": feat_size,
        "latent_dim": int(feat_size * feat_size),
        "next_latent_mse": round(model_mse, 6),
        "identity_baseline_mse": round(identity_mse, 6),
        "improvement_vs_identity_pct": round(100.0 * (identity_mse - model_mse) / max(identity_mse, 1e-9), 1),
    }
    roll = _rollout(model, eps, rollout_k)
    if roll is not None:
        r_mse, r_ident = roll
        metrics.update({
            "rollout_k": rollout_k,
            "rollout_mse": round(r_mse, 6),
            "rollout_identity_mse": round(r_ident, 6),
            "rollout_improvement_pct": round(100.0 * (r_ident - r_mse) / max(r_ident, 1e-9), 1),
        })
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "world_model.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        np.savez(os.path.join(out_dir, "world_model.npz"), W=model.W, feat_size=feat_size, l2=l2)
    return metrics
