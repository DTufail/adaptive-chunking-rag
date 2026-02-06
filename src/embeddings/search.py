"""Shared numpy-based L2 nearest-neighbor search.

Replaces per-context FAISS indexes with direct numpy computation.
For the typical 3-30 vectors per context in this project, numpy is
faster than FAISS due to zero object-creation overhead.
"""

from typing import List

import numpy as np


def numpy_l2_search(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    texts: List[str],
    k: int,
) -> List[dict]:
    """Search pre-computed vectors using numpy L2 squared distance.

    Uses the same distance metric as FAISS IndexFlatL2.

    Args:
        vectors: (n, dim) float32 embedding matrix.
        query_vector: (dim,) float32 query vector.
        texts: List of n text strings corresponding to each row of vectors.
        k: Number of results to return.

    Returns:
        List of dicts with keys ``rank``, ``score``, ``text``,
        sorted by L2 distance ascending (best first).
        May return fewer than *k* results if *vectors* has fewer rows.
    """
    if vectors.shape[0] == 0:
        return []

    # L2 squared distance — row-wise dot of (vectors - query)
    diff = vectors - query_vector
    distances = np.einsum('ij,ij->i', diff, diff)

    n = len(distances)
    actual_k = min(k, n)

    if actual_k < n:
        # argpartition is O(n) vs O(n log n) for full sort
        top_idx = np.argpartition(distances, actual_k)[:actual_k]
        top_idx = top_idx[np.argsort(distances[top_idx])]
    else:
        top_idx = np.argsort(distances)

    return [
        {"rank": rank, "score": float(distances[idx]), "text": texts[idx]}
        for rank, idx in enumerate(top_idx)
    ]
