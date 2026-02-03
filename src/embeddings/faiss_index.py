"""FAISS vector index wrapper for similarity search."""

import logging
from typing import List

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissIndex:
    """Wrapper for FAISS vector storage and similarity search.

    Uses IndexFlatL2 for exact L2 distance search. Maintains a mapping
    from FAISS's internal integer IDs to application-level chunk IDs.

    NOTE: IDs are positional. The internal chunk_id list stays aligned with
    FAISS's auto-assigned integer IDs (0, 1, 2, ...) by insertion order.
    This breaks if vectors are ever deleted or indices are merged.
    IndexFlatL2 does not support deletion, so this is safe for Phase 3.
    If deletion or merging is needed later, switch to faiss.IndexIDMap.
    """

    def __init__(self, dimension: int):
        """Initialize the FAISS index.

        Args:
            dimension: Dimensionality of the vectors. Must be a positive integer.

        Raises:
            ValueError: If dimension is not a positive integer.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {dimension}")

        self.dimension = dimension
        self._index = faiss.IndexFlatL2(dimension)
        self._chunk_ids: List[str] = []

        logger.debug(f"FaissIndex initialized with dimension={dimension}")

    def add(self, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        """Add vectors to the index with corresponding chunk IDs.

        Args:
            vectors: 2D numpy array of shape (N, dimension) with dtype float32.
            chunk_ids: List of N chunk ID strings corresponding to each vector.

        Raises:
            ValueError: If shapes don't match, vectors is not 2D, or dimension
                doesn't match the index dimension.
            TypeError: If vectors is not a numpy array or chunk_ids is not a list.
        """
        if not isinstance(vectors, np.ndarray):
            raise TypeError(f"Vectors must be a numpy array, got {type(vectors).__name__}")

        if not isinstance(chunk_ids, list):
            raise TypeError(f"chunk_ids must be a list, got {type(chunk_ids).__name__}")

        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D, got shape {vectors.shape}")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        if vectors.shape[0] != len(chunk_ids):
            raise ValueError(
                f"Number of vectors ({vectors.shape[0]}) does not match "
                f"number of chunk_ids ({len(chunk_ids)})"
            )

        if vectors.dtype != np.float32:
            logger.debug(f"Converting vectors from {vectors.dtype} to float32")
            vectors = vectors.astype(np.float32)

        self._index.add(vectors)
        self._chunk_ids.extend(chunk_ids)

        logger.debug(f"Added {len(chunk_ids)} vectors to index (total size: {self.size})")

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[dict]:
        """Search for the k nearest neighbors of a query vector.

        Args:
            query_vector: 1D or 2D numpy array of shape (dimension,) or (1, dimension).
            k: Number of nearest neighbors to return. Defaults to 5.

        Returns:
            List of dictionaries, each containing:
                - "chunk_id": The chunk ID string
                - "score": L2 distance (lower is better)
                - "rank": 0-indexed rank (0 is best match)
            Sorted by score ascending (best first). May contain fewer than k
            results if the index has fewer than k vectors.

        Raises:
            ValueError: If query_vector has wrong shape or dimension.
            TypeError: If query_vector is not a numpy array or k is not an int.
        """
        if not isinstance(query_vector, np.ndarray):
            raise TypeError(f"query_vector must be a numpy array, got {type(query_vector).__name__}")

        if not isinstance(k, int) or k <= 0:
            raise TypeError(f"k must be a positive integer, got {k}")

        # Normalize to 2D shape (1, dimension)
        if query_vector.ndim == 1:
            if query_vector.shape[0] != self.dimension:
                raise ValueError(
                    f"Query vector dimension {query_vector.shape[0]} does not match "
                    f"index dimension {self.dimension}"
                )
            query_vector = query_vector.reshape(1, -1)
        elif query_vector.ndim == 2:
            if query_vector.shape != (1, self.dimension):
                raise ValueError(
                    f"Query vector shape {query_vector.shape} does not match "
                    f"expected shape (1, {self.dimension})"
                )
        else:
            raise ValueError(f"Query vector must be 1D or 2D, got shape {query_vector.shape}")

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        # Nothing to search
        if self.size == 0:
            logger.debug("Search called on empty index, returning []")
            return []

        distances, indices = self._index.search(query_vector, k)

        # Filter out invalid FAISS indices first, then assign ranks
        # on the clean list so ranks are always contiguous (0, 1, 2, ...)
        valid = [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx >= 0 and idx < len(self._chunk_ids)
        ]

        results = [
            {
                "chunk_id": self._chunk_ids[idx],
                "score": dist,
                "rank": rank,
            }
            for rank, (idx, dist) in enumerate(valid)
        ]

        logger.debug(f"Search returned {len(results)} results for k={k}")
        return results

    @property
    def size(self) -> int:
        """Get the number of vectors currently in the index.

        Returns:
            Number of vectors stored in the index.
        """
        return self._index.ntotal

    def clear(self) -> None:
        """Clear the index, removing all vectors and chunk IDs."""
        self._index.reset()
        self._chunk_ids.clear()
        logger.debug("Index cleared")