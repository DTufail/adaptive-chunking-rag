"""
config.py
─────────
Configuration loader for rag-chunk-eval.

Loads settings from a YAML file and builds chunker instances dynamically.
"""

import os
from typing import Dict, Any

import yaml

# Project root (two levels up from src/utils/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'default.yaml')


def load_config(config_path: str = None) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.
                     Defaults to configs/default.yaml relative to project root.

    Returns:
        Parsed config dictionary with resolved paths.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve relative data path against project root
    data_path = config.get('data', {}).get('path', '')
    if data_path and not os.path.isabs(data_path):
        config['data']['path'] = os.path.join(PROJECT_ROOT, data_path)

    # Resolve output dir
    output_dir = config.get('evaluation', {}).get('output_dir', 'results')
    if output_dir and not os.path.isabs(output_dir):
        config['evaluation']['output_dir'] = os.path.join(PROJECT_ROOT, output_dir)

    return config


def build_chunkers(config: dict) -> Dict[str, Any]:
    """Build chunker instances from config.

    Reads the 'chunkers' key from the config dict and instantiates
    each chunker class with the specified parameters.

    Args:
        config: Parsed config dictionary with a 'chunkers' key.
                Each sub-key is a chunker class name, and its value
                is a dict of constructor kwargs.

    Returns:
        Dict mapping chunker names to instantiated chunker objects.

    Raises:
        ValueError: If a chunker name is not recognized.
    """
    from chunkers import (
        FixedChunker,
        SentenceChunker,
        ParagraphChunker,
        RecursiveChunker,
        StructureAwareChunker,
        SemanticDensityChunker,
    )

    CHUNKER_CLASSES = {
        'FixedChunker': FixedChunker,
        'SentenceChunker': SentenceChunker,
        'ParagraphChunker': ParagraphChunker,
        'RecursiveChunker': RecursiveChunker,
        'StructureAwareChunker': StructureAwareChunker,
        'SemanticDensityChunker': SemanticDensityChunker,
    }

    chunkers = {}
    for name, params in config.get('chunkers', {}).items():
        # Support suffixed names like "FixedChunker_256": try the full
        # name first, then progressively strip trailing _segments to
        # find the class.  Keeps the full name as the dict key so
        # multiple instances of the same class can coexist.
        cls = CHUNKER_CLASSES.get(name)
        if cls is None:
            # Try stripping trailing _suffix segments (e.g. "FixedChunker_256" -> "FixedChunker")
            candidate = name
            while "_" in candidate:
                candidate = candidate.rsplit("_", 1)[0]
                cls = CHUNKER_CLASSES.get(candidate)
                if cls is not None:
                    break
        if cls is None:
            raise ValueError(
                f"Unknown chunker '{name}'. "
                f"Available: {list(CHUNKER_CLASSES.keys())}"
            )
        chunkers[name] = cls(**params)

    return chunkers
