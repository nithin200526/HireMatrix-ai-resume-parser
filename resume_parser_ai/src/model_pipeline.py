"""End-to-end resume parser model pipeline."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd
from sklearn.metrics import silhouette_score

from .clustering import ClusteringConfig, ResumeClusterer
from .feature_engineering import FeatureEngineer, VectorizerConfig
from .preprocessing import ResumePreprocessor
from .scoring import ResumeScorer, ScoringConfig
from .utils import load_pickle, save_pickle

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """High-level pipeline configuration."""

    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


class ResumeParserPipeline:
    """Trainable and deployable resume parser pipeline."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.preprocessor = ResumePreprocessor()
        self.engineer = FeatureEngineer(self.config.vectorizer)
        self.clusterer = ResumeClusterer(self.config.clustering)
        self.scorer = ResumeScorer(self.config.scoring)
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train pipeline from dataframe with Job Title and Resume columns."""
        required_cols = {"Job Title", "Resume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        df = df.dropna(subset=["Job Title", "Resume"]).copy()
        df["processed_resume"] = self.preprocessor.preprocess_many(df["Resume"].astype(str))

        resume_features = self.engineer.fit_transform(df["processed_resume"])
        optimal_k, inertias = self.clusterer.find_optimal_k(resume_features)
        self.clusterer.fit(resume_features, n_clusters=optimal_k)
        labels = self.clusterer.predict(resume_features)

        sil_score = silhouette_score(resume_features, labels) if optimal_k > 1 else 0.0
        self.metrics = {
            "samples": float(df.shape[0]),
            "optimal_k": float(optimal_k),
            "final_inertia": inertias[optimal_k - 1] if inertias else 0.0,
            "silhouette_score": float(sil_score),
        }
        self.is_fitted = True
        LOGGER.info("Pipeline trained successfully with %s samples and %s clusters", df.shape[0], optimal_k)
        return self.metrics

    def evaluate_resume(self, job_title: str, resume_text: str, top_n_keywords: int = 5) -> Dict[str, object]:
        """Score a new resume against target job title and return rich outputs."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be trained or loaded before evaluation.")

        processed_resume = self.preprocessor.preprocess_text(resume_text)
        processed_title = self.preprocessor.preprocess_text(job_title)

        resume_vec = self.engineer.transform([processed_resume])
        title_vec = self.engineer.transform([processed_title])

        scoring = self.scorer.score(resume_vec, title_vec, processed_resume)
        cluster = int(self.clusterer.predict(resume_vec)[0])
        top_keywords = self.engineer.extract_top_keywords(processed_resume, top_n=top_n_keywords)

        token_counts = Counter(processed_resume.split())
        repeated_skills = {
            skill: count
            for skill, count in token_counts.most_common(10)
            if count > 1
        }

        return {
            "score": scoring["score"],
            "cluster": cluster,
            "top_keywords": top_keywords,
            "details": scoring,
            "skill_frequency": repeated_skills,
        }

    def save(self, path: str) -> None:
        """Save pipeline to disk."""
        save_pickle(self, path)
        LOGGER.info("Saved pipeline to %s", path)

    @staticmethod
    def load(path: str) -> "ResumeParserPipeline":
        """Load pipeline from disk."""
        model = load_pickle(path)
        if not isinstance(model, ResumeParserPipeline):
            raise TypeError("Loaded object is not ResumeParserPipeline")
        return model
