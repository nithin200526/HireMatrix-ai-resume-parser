"""Resume scoring logic based on semantic relevance and quality penalties."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ScoringConfig:
    """Configuration knobs for scoring behavior."""

    stuffing_threshold: float = 0.2
    stuffing_penalty_max: float = 3.0


class ResumeScorer:
    """Scores resumes against a target job title from 0 to 10."""

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig()

    def keyword_stuffing_penalty(self, processed_resume: str) -> float:
        """Penalty based on excessive repeated token ratio."""
        tokens = processed_resume.split()
        if not tokens:
            return self.config.stuffing_penalty_max

        counts = Counter(tokens)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        repeat_ratio = repeated / len(tokens)

        if repeat_ratio <= self.config.stuffing_threshold:
            return 0.0

        normalized = min(1.0, (repeat_ratio - self.config.stuffing_threshold) / (1 - self.config.stuffing_threshold))
        return float(normalized * self.config.stuffing_penalty_max)

    def score(self, resume_vec: np.ndarray, title_vec: np.ndarray, processed_resume: str) -> Dict[str, float]:
        """Compute final score and components."""
        sim = float(cosine_similarity(resume_vec, title_vec)[0][0])
        base_score = max(0.0, min(10.0, sim * 10))
        penalty = self.keyword_stuffing_penalty(processed_resume)
        final_score = max(0.0, round(base_score - penalty, 2))
        return {
            "similarity": round(sim, 4),
            "base_score": round(base_score, 2),
            "penalty": round(penalty, 2),
            "score": final_score,
        }
