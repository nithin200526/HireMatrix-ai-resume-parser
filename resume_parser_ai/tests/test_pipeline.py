"""Unit tests for Resume Parser pipeline."""

from __future__ import annotations

import pandas as pd

from src.model_pipeline import ResumeParserPipeline


def test_pipeline_training_and_inference() -> None:
    df = pd.DataFrame(
        {
            "Job Title": ["Data Scientist", "Backend Engineer", "Data Analyst"],
            "Resume": [
                "Python machine learning NLP SQL statistics.",
                "FastAPI Docker AWS APIs microservices backend.",
                "SQL Tableau Excel reporting analytics dashboard.",
            ],
        }
    )

    pipeline = ResumeParserPipeline()
    metrics = pipeline.train(df)

    assert metrics["samples"] == 3.0
    result = pipeline.evaluate_resume(
        job_title="Data Scientist",
        resume_text="I have python, machine learning, NLP and model deployment skills.",
    )

    assert 0.0 <= result["score"] <= 10.0
    assert isinstance(result["cluster"], int)
    assert isinstance(result["top_keywords"], list)
