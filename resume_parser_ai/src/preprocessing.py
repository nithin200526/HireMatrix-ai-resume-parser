"""Text preprocessing module built on top of NLTK."""

from __future__ import annotations

import logging
import re
import string
from functools import lru_cache
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

LOGGER = logging.getLogger(__name__)

NLTK_PACKAGES = ("punkt", "stopwords", "wordnet", "omw-1.4")


@lru_cache(maxsize=1)
def _download_nltk_resources() -> None:
    """Download and cache required NLTK resources."""
    for pkg in NLTK_PACKAGES:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed downloading NLTK package '%s': %s", pkg, exc)


class ResumePreprocessor:
    """Preprocess resume text with standard NLP cleaning operations."""

    def __init__(self, language: str = "english") -> None:
        _download_nltk_resources()
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self._punct_table = str.maketrans("", "", string.punctuation)

    def preprocess_text(self, text: str) -> str:
        """Apply full preprocessing chain to raw text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.lower()
        text = text.translate(self._punct_table)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = word_tokenize(text)
        tokens = [tok for tok in tokens if tok not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha()]

        return " ".join(tokens)

    def preprocess_many(self, texts: Iterable[str]) -> List[str]:
        """Preprocess a collection of texts."""
        return [self.preprocess_text(text) for text in texts]
