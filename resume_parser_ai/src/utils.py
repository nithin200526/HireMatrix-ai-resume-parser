"""Utility helpers for configuration, IO, and logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "models/saved_model.pkl"
DEFAULT_DATA_PATH = "data/raw_resume_dataset.csv"


def setup_logging(level: str = "INFO") -> None:
    """Configure application-wide logging format and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    """Read an environment variable with optional default and required check."""
    value = os.getenv(name, default)
    if required and value is None:
        raise ValueError(f"Environment variable '{name}' is required.")
    if value is None:
        raise ValueError(f"Environment variable '{name}' was not set and no default provided.")
    return value


def ensure_parent(path: str | Path) -> Path:
    """Ensure parent directory exists for the given file path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(obj: Any, path: str | Path) -> Path:
    """Persist a Python object to disk using joblib."""
    from joblib import dump

    p = ensure_parent(path)
    dump(obj, p)
    return p


def load_pickle(path: str | Path) -> Any:
    """Load a serialized Python object from disk using joblib."""
    from joblib import load

    return load(path)
