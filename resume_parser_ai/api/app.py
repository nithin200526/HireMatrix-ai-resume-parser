"""FastAPI application exposing resume scoring endpoint."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model_pipeline import ResumeParserPipeline
from src.utils import DEFAULT_MODEL_PATH, get_env, setup_logging

setup_logging(get_env("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Resume Parser AI", version="1.0.0")

MODEL_PATH = get_env("MODEL_PATH", DEFAULT_MODEL_PATH)
_pipeline: ResumeParserPipeline | None = None


class ScoreRequest(BaseModel):
    """Schema for resume scoring requests."""

    job_title: str = Field(..., min_length=2)
    resume_text: str = Field(..., min_length=20)


class ScoreResponse(BaseModel):
    """Schema for resume scoring responses."""

    score: float
    cluster: int
    top_keywords: list[str]


@app.on_event("startup")
def load_model() -> None:
    """Load serialized model on API startup."""
    global _pipeline
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        LOGGER.error("Model file not found at %s", model_file)
        return

    _pipeline = ResumeParserPipeline.load(str(model_file))
    LOGGER.info("Model loaded from %s", model_file)


@app.post("/score", response_model=ScoreResponse)
def score_resume(payload: ScoreRequest) -> ScoreResponse:
    """Evaluate resume relevance for a target job title."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model first.")

    try:
        result = _pipeline.evaluate_resume(payload.job_title, payload.resume_text)
        return ScoreResponse(
            score=float(result["score"]),
            cluster=int(result["cluster"]),
            top_keywords=list(result["top_keywords"]),
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Failed scoring resume")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
