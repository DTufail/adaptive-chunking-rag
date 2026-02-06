"""Embeddings module for text vectorization and similarity search.

This module provides wrappers for sentence-transformers embedding models
and FAISS vector indexing for efficient nearest-neighbor search in RAG pipelines.
"""

from .embedding_model import EmbeddingModel
from .faiss_index import FaissIndex
from .search import numpy_l2_search

__all__ = ["EmbeddingModel", "FaissIndex", "numpy_l2_search"]
