"""Tiny, CPU-only models that learn from EveryCam datasets."""

from .affordance import AffordanceModel, NumpyMLP, featurize, train_from_dataset

__all__ = ["AffordanceModel", "NumpyMLP", "featurize", "train_from_dataset"]
