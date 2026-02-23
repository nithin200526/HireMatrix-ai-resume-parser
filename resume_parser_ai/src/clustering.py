"""KMeans clustering utilities for grouping resumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


@dataclass
class ClusteringConfig:
    """Configuration for KMeans clustering."""

    max_k: int = 10
    random_state: int = 42
    n_init: int = 10


class ResumeClusterer:
    """KMeans based resume clustering with elbow method support."""

    def __init__(self, config: ClusteringConfig | None = None) -> None:
        self.config = config or ClusteringConfig()
        self.model: KMeans | None = None

    def find_optimal_k(self, features: csr_matrix) -> tuple[int, List[float]]:
        """Determine best k using largest drop in inertia (elbow heuristic)."""
        max_k = min(self.config.max_k, features.shape[0])
        if max_k < 2:
            return 1, [0.0]

        inertias: List[float] = []
        for k in range(1, max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=self.config.n_init)
            km.fit(features)
            inertias.append(float(km.inertia_))

        if len(inertias) < 3:
            return 2 if len(inertias) > 1 else 1, inertias

        drops = np.diff(inertias)
        second_diff = np.diff(drops)
        elbow_idx = int(np.argmax(np.abs(second_diff))) + 2
        return elbow_idx, inertias

    def fit(self, features: csr_matrix, n_clusters: int) -> KMeans:
        """Fit KMeans model."""
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_init=self.config.n_init,
        )
        self.model.fit(features)
        return self.model

    def predict(self, features: csr_matrix) -> np.ndarray:
        """Predict cluster labels for features."""
        if self.model is None:
            raise RuntimeError("Clusterer is not fitted.")
        return self.model.predict(features)
