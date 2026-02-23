"""Training entrypoint for Resume Parser AI."""

from __future__ import annotations

import logging

import pandas as pd

from src.model_pipeline import ResumeParserPipeline
from src.utils import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, get_env, setup_logging


def run_training() -> None:
    """Train pipeline and persist model."""
    setup_logging(get_env("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)

    data_path = get_env("DATA_PATH", DEFAULT_DATA_PATH)
    model_path = get_env("MODEL_PATH", DEFAULT_MODEL_PATH)

    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)

    pipeline = ResumeParserPipeline()
    metrics = pipeline.train(df)
    pipeline.save(model_path)

    logger.info("Training complete. Metrics: %s", metrics)


if __name__ == "__main__":
    run_training()
