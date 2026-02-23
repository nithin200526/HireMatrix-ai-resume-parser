"""Feature engineering utilities for resume text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class VectorizerConfig:
    """Configuration for TF-IDF vectorizer."""

    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 0.95


class FeatureEngineer:
    """Reusable TF-IDF feature engineering component."""

    def __init__(self, config: VectorizerConfig | None = None) -> None:
        self.config = config or VectorizerConfig()
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            lowercase=False,
        )

    def fit_transform(self, texts: Iterable[str]) -> csr_matrix:
        """Fit TF-IDF and transform input texts."""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        """Transform texts using previously fitted vectorizer."""
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> np.ndarray:
        """Return learned feature names."""
        return self.vectorizer.get_feature_names_out()

    def extract_top_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top TF-IDF keywords from a single processed text."""
        row = self.transform([text])
        if row.nnz == 0:
            return []
        feature_names = self.get_feature_names()
        dense = row.toarray().ravel()
        top_idx = dense.argsort()[::-1][:top_n]
        return [feature_names[i] for i in top_idx if dense[i] > 0]
