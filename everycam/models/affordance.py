"""A tiny, dependency-free model that *learns* from EveryCam datasets.

Two heads on a shared image featurizer:
  - grasp regressor:  image -> normalized grasp (x, y)
  - contact classifier: image -> in-contact?

It is a pure-numpy MLP with Adam, so it trains in seconds on a CPU and proves
the exported data is genuinely learnable end-to-end. Install the ``[torch]``
extra to swap in a CNN; the dataset format is unchanged.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np


def featurize(images: np.ndarray, size: int = 16) -> np.ndarray:
    """Downsample BGR frames to a flat, normalized grayscale feature vector."""
    import cv2

    feats = []
    for img in images:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        feats.append(g.astype(np.float32).ravel() / 255.0)
    return np.stack(feats) if feats else np.zeros((0, size * size), np.float32)


def _relu(x):
    return np.maximum(0.0, x)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyMLP:
    """One-hidden-layer MLP with Adam. task in {'regression','classification'}."""

    def __init__(self, in_dim, hidden, out_dim, task="regression", seed=0):
        rng = np.random.default_rng(seed)
        self.task = task
        self.W1 = (rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = (rng.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(out_dim, np.float32)
        self._m = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self._t = 0

    def _forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = _relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        out = _sigmoid(Z2) if self.task == "classification" else Z2
        return out, (X, Z1, A1)

    def predict(self, X):
        return self._forward(X)[0]

    def _backward(self, cache, out, Y):
        X, Z1, A1 = cache
        n = max(1, X.shape[0])
        dZ2 = (out - Y) / n  # works for MSE (linear) and BCE (sigmoid) alike
        if self.task == "regression":
            dZ2 = 2.0 * dZ2
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(0)
        dZ1 = (dZ2 @ self.W2.T) * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            mhat = self._m[k] / (1 - b1**self._t)
            vhat = self._v[k] / (1 - b2**self._t)
            setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))

    def fit(self, X, Y, epochs=400, lr=0.01, batch=32, seed=0):
        rng = np.random.default_rng(seed)
        Y = Y.reshape(X.shape[0], -1).astype(np.float32)
        n = X.shape[0]
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s : s + batch]
                out, cache = self._forward(X[b])
                self._adam(self._backward(cache, out, Y[b]), lr)
        return self


class RidgeRegressor:
    """Closed-form L2-regularized linear regression — a robust, tiny grasp head.

    A linear readout of standardized pixels approximates the object's centroid and,
    unlike a small MLP, does not overfit a few hundred frames.
    """

    def __init__(self, l2: float = 30.0):
        self.l2 = float(l2)
        self.W = None

    def fit(self, X, Y):
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1), np.float32)])
        A = Xb.T @ Xb + self.l2 * np.eye(d + 1, dtype=np.float32)
        self.W = np.linalg.solve(A, Xb.T @ Y.astype(np.float32))
        return self

    def predict(self, X):
        Xb = np.hstack([X, np.ones((X.shape[0], 1), np.float32)])
        return Xb @ self.W


class AffordanceModel:
    """Grasp head = ridge regression; contact head = numpy MLP classifier."""

    def __init__(self, feat_size=16, hidden=64, l2=10.0, seed=0):
        self.feat_size = feat_size
        self.hidden = hidden
        self.l2 = l2
        self.seed = seed
        self.grasp: Optional[RidgeRegressor] = None
        self.contact: Optional[NumpyMLP] = None
        self.mu = None  # feature mean/std (removes the constant background, aids scaling)
        self.sigma = None

    def _features(self, images, fit=False):
        # Per-feature centering removes the static background; a single global scale
        # keeps the ridge system well-conditioned (per-feature std would blow up on
        # near-constant pixels).
        X = featurize(images, self.feat_size)
        if fit or self.mu is None:
            self.mu = X.mean(axis=0)
            self.sigma = float(X.std()) + 1e-6
        return (X - self.mu) / self.sigma

    def fit(self, images, affordances, contacts, epochs=400, lr=0.01):
        X = self._features(images, fit=True)
        self.grasp = RidgeRegressor(self.l2).fit(X, affordances)
        self.contact = NumpyMLP(X.shape[1], self.hidden, 1, "classification", self.seed)
        self.contact.fit(
            X, contacts.astype(np.float32).reshape(-1, 1), epochs=epochs, lr=lr, seed=self.seed
        )
        return self

    def predict_grasp(self, images):
        return self.grasp.predict(self._features(images))

    def predict_contact(self, images):
        return (self.contact.predict(self._features(images)) > 0.5).ravel()

    def save(self, path):
        np.savez(
            path,
            feat_size=self.feat_size, hidden=self.hidden, l2=self.l2,
            mu=self.mu, sigma=self.sigma, grasp_W=self.grasp.W,
            cW1=self.contact.W1, cb1=self.contact.b1, cW2=self.contact.W2, cb2=self.contact.b2,
        )


def save_eval_viz(images, true_aff, pred_aff, path, n=6):
    """Montage: true grasp (green) vs predicted (red) on a few validation frames."""
    import cv2

    h, w = images[0].shape[:2]
    k = min(n, len(images))
    tiles = []
    for i in range(k):
        im = images[i].copy()
        tx, ty = int(true_aff[i][0] * w), int(true_aff[i][1] * h)
        px, py = int(pred_aff[i][0] * w), int(pred_aff[i][1] * h)
        cv2.circle(im, (tx, ty), 7, (0, 200, 0), 2)
        cv2.circle(im, (px, py), 5, (0, 0, 230), -1)
        tiles.append(im)
    montage = np.hstack(tiles)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, montage)
    return path


def train_from_dataset(
    dataset_dir, out_dir=None, epochs=400, lr=0.01, val_frac=0.3, seed=0
) -> Tuple[AffordanceModel, Dict]:
    from ..export import load_dataset_arrays

    d = load_dataset_arrays(dataset_dir, with_images=True)
    images, aff, contacts = d["images"], d["affordances"], d["contacts"]
    if images is None:
        raise RuntimeError("Dataset has no images. Export with save_images=True to train.")

    valid = ~np.isnan(aff).any(axis=1)
    images, aff, contacts = images[valid], aff[valid], contacts[valid]
    n = len(images)
    if n < 4:
        raise RuntimeError(f"Too few valid frames to train ({n}).")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    val, tr = idx[:n_val], idx[n_val:]

    model = AffordanceModel(seed=seed).fit(images[tr], aff[tr], contacts[tr], epochs=epochs, lr=lr)

    pg = model.predict_grasp(images[val])
    mae = float(np.abs(pg - aff[val]).mean())
    base = aff[tr].mean(axis=0)  # predict-the-mean baseline
    base_mae = float(np.abs(base - aff[val]).mean())
    shape = d["info"]["features"]["observation.image"]["shape"]
    H, W = shape[0], shape[1]
    mae_px = float(np.abs((pg - aff[val]) * np.array([W, H])).mean())
    pc = model.predict_contact(images[val])
    cacc = float((pc == contacts[val]).mean())

    metrics = {
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "grasp_mae_norm": round(mae, 4),
        "baseline_mae_norm": round(base_mae, 4),
        "grasp_mae_px": round(mae_px, 2),
        "contact_accuracy": round(cacc, 4),
        "improvement_vs_baseline_pct": round(100.0 * (base_mae - mae) / max(base_mae, 1e-9), 1),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        model.save(os.path.join(out_dir, "affordance_model.npz"))
        import json

        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        save_eval_viz(
            images[val], aff[val], pg, os.path.join(out_dir, "grasp_predictions.png")
        )
    return model, metrics
