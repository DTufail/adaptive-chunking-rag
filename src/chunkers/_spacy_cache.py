"""
_spacy_cache.py
───────────────
Process-level cache for spaCy models.

Loading a spaCy model (especially en_core_web_sm with the full pipeline)
takes 1-3 seconds.  Without caching, every chunker instance that uses
spaCy pays this cost independently — catastrophic when tests or grid
searches instantiate dozens of chunkers.

This module provides a single get_spacy_model() function that loads
each (model_name, components) combination exactly once per process.
"""

from typing import Optional, Tuple

import spacy
from spacy.language import Language

# Process-level cache: (model_name, disabled_tuple) -> Language
_MODEL_CACHE: dict[Tuple[str, Optional[Tuple[str, ...]]], Language] = {}


def get_spacy_model(
    model_name: str = "en_core_web_sm",
    *,
    disable: Optional[list[str]] = None,
) -> Language:
    """Return a cached spaCy model, loading it only on first call.

    Args:
        model_name: Name of the spaCy model (e.g. "en_core_web_sm").
        disable: Pipeline components to disable (e.g. ["ner", "tagger"]).

    Returns:
        The loaded (and cached) spaCy Language object.

    Raises:
        OSError: If the model is not installed.
    """
    disable_key = tuple(sorted(disable)) if disable else None
    cache_key = (model_name, disable_key)

    if cache_key not in _MODEL_CACHE:
        try:
            nlp = spacy.load(model_name, disable=disable or [])
        except OSError:
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Install it with: python -m spacy download {model_name}"
            )
        _MODEL_CACHE[cache_key] = nlp

    return _MODEL_CACHE[cache_key]


def get_blank_sentencizer(lang: str = "en") -> Language:
    """Return a cached blank spaCy model with only the sentencizer pipe.

    This is ~100x faster than loading en_core_web_sm for sentence
    splitting, because it skips the tagger, parser, NER, and lemmatizer.

    Args:
        lang: Language code (default "en").

    Returns:
        A blank Language object with the "sentencizer" pipe attached.
    """
    cache_key = (f"blank_{lang}_sentencizer", None)

    if cache_key not in _MODEL_CACHE:
        nlp = spacy.blank(lang)
        nlp.add_pipe("sentencizer")
        _MODEL_CACHE[cache_key] = nlp

    return _MODEL_CACHE[cache_key]
