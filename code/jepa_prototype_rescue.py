#!/usr/bin/env python3
"""Prototype-memory rescuer for missed JEPA error proposals.

The module is intentionally narrow: it learns reliable positive and negative
JEPA proposal prototypes from a training fold, then scores unseen rescue
candidates by positive-vs-negative prototype similarity.
"""
from __future__ import annotations

import numpy as np


def _normalise_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    normed = ((values - mean) / std).astype(np.float32)
    lengths = np.linalg.norm(normed, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-6)
    return (normed / lengths).astype(np.float32)


def _select_diverse_prototypes(vectors: np.ndarray, max_items: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) <= int(max_items):
        return vectors.copy()
    centroid = vectors.mean(axis=0, keepdims=True)
    first = int(np.argmax((vectors @ centroid.T).reshape(-1)))
    selected = [first]
    min_distance = np.sum((vectors - vectors[first]) ** 2, axis=1)
    while len(selected) < int(max_items):
        next_idx = int(np.argmax(min_distance))
        if next_idx in selected:
            break
        selected.append(next_idx)
        distance = np.sum((vectors - vectors[next_idx]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return vectors[selected].copy()


class PrototypeRescuer:
    def __init__(
        self,
        positive_prototypes: np.ndarray,
        negative_prototypes: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        scale: float = 6.0,
    ):
        self.positive_prototypes = np.asarray(positive_prototypes, dtype=np.float32)
        self.negative_prototypes = np.asarray(negative_prototypes, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.scale = float(scale)

    def _max_similarity(self, vectors: np.ndarray, prototypes: np.ndarray, default: float) -> np.ndarray:
        if len(prototypes) == 0:
            return np.full(len(vectors), float(default), dtype=np.float32)
        return np.max(vectors @ prototypes.T, axis=1).astype(np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if len(self.positive_prototypes) == 0:
            positive = np.full(len(values), 0.01, dtype=np.float32)
            return np.stack([1.0 - positive, positive], axis=1)
        vectors = _normalise_features(values, self.mean, self.std)
        pos_sim = self._max_similarity(vectors, self.positive_prototypes, default=-1.0)
        neg_sim = self._max_similarity(vectors, self.negative_prototypes, default=0.0)
        logits = self.scale * (pos_sim - neg_sim)
        positive = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        return np.stack([1.0 - positive, positive], axis=1)


PrototypeRescuer.__module__ = "jepa_prototype_rescue"


def fit_prototype_rescuer(
    x: np.ndarray,
    y: np.ndarray,
    max_positive_prototypes: int = 96,
    max_negative_prototypes: int = 128,
    scale: float = 6.0,
) -> PrototypeRescuer:
    values = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"expected 2D feature matrix, got {values.shape}")
    if len(values) != len(labels):
        raise ValueError(f"feature/label count mismatch: {len(values)} vs {len(labels)}")
    if len(values) == 0:
        raise ValueError("cannot fit prototype rescuer without candidates")

    mean = values.mean(axis=0).astype(np.float32)
    std = np.maximum(values.std(axis=0).astype(np.float32), 1e-4)
    normed = _normalise_features(values, mean, std)
    positives = normed[labels > 0]
    negatives = normed[labels <= 0]
    positive_prototypes = _select_diverse_prototypes(positives, int(max_positive_prototypes)) if len(positives) else positives
    negative_prototypes = _select_diverse_prototypes(negatives, int(max_negative_prototypes)) if len(negatives) else negatives
    return PrototypeRescuer(
        positive_prototypes=positive_prototypes,
        negative_prototypes=negative_prototypes,
        mean=mean,
        std=std,
        scale=float(scale),
    )
