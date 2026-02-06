"""
test_retrieval.py
─────────────────
Pytest tests for the FAISS index wrapper and evaluation metric functions.

Run:  pytest tests/test_retrieval.py -v
"""

import numpy as np
import pytest

from embeddings.faiss_index import FaissIndex
from evaluation.eval_metrics import compute_hit_rank, derive_metrics


# ─── FaissIndex tests ────────────────────────────────────────────────────────

class TestFaissIndex:

    def test_add_and_search_finds_exact_match(self):
        dim = 8
        index = FaissIndex(dimension=dim)

        np.random.seed(42)
        vectors = np.random.rand(5, dim).astype(np.float32)
        chunk_ids = [f"chunk_{i}" for i in range(5)]

        index.add(vectors, chunk_ids)

        # Search for the first vector — should find itself
        results = index.search(vectors[0], k=3)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "chunk_0"
        assert results[0]["rank"] == 0

    def test_search_returns_correct_k(self):
        dim = 8
        index = FaissIndex(dimension=dim)

        np.random.seed(42)
        vectors = np.random.rand(10, dim).astype(np.float32)
        chunk_ids = [f"chunk_{i}" for i in range(10)]

        index.add(vectors, chunk_ids)

        results = index.search(np.random.rand(dim).astype(np.float32), k=5)
        assert len(results) == 5

    def test_k_exceeds_index_size_returns_all(self):
        dim = 8
        index = FaissIndex(dimension=dim)

        np.random.seed(42)
        vectors = np.random.rand(3, dim).astype(np.float32)
        chunk_ids = [f"chunk_{i}" for i in range(3)]

        index.add(vectors, chunk_ids)

        results = index.search(np.random.rand(dim).astype(np.float32), k=10)
        assert len(results) == 3

    def test_results_have_required_fields(self):
        dim = 8
        index = FaissIndex(dimension=dim)

        np.random.seed(42)
        vectors = np.random.rand(5, dim).astype(np.float32)
        chunk_ids = [f"chunk_{i}" for i in range(5)]

        index.add(vectors, chunk_ids)

        results = index.search(np.random.rand(dim).astype(np.float32), k=3)
        for r in results:
            assert "rank" in r
            assert "score" in r
            assert "chunk_id" in r
            assert isinstance(r["rank"], int)
            assert r["rank"] >= 0

    def test_results_are_rank_ordered(self):
        dim = 8
        index = FaissIndex(dimension=dim)

        np.random.seed(42)
        vectors = np.random.rand(5, dim).astype(np.float32)
        chunk_ids = [f"chunk_{i}" for i in range(5)]

        index.add(vectors, chunk_ids)

        results = index.search(np.random.rand(dim).astype(np.float32), k=5)
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(len(ranks)))

    def test_scores_are_nondecreasing(self):
        """L2 distances should be non-decreasing (best = lowest)."""
        dim = 8
        index = FaissIndex(dimension=dim)

        np.random.seed(42)
        vectors = np.random.rand(10, dim).astype(np.float32)
        chunk_ids = [f"chunk_{i}" for i in range(10)]

        index.add(vectors, chunk_ids)

        results = index.search(np.random.rand(dim).astype(np.float32), k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores)

    def test_empty_index_returns_empty(self):
        dim = 8
        index = FaissIndex(dimension=dim)
        query = np.random.rand(dim).astype(np.float32)
        results = index.search(query, k=3)
        assert results == []

    def test_size_property(self):
        dim = 8
        index = FaissIndex(dimension=dim)
        assert index.size == 0

        vectors = np.random.rand(5, dim).astype(np.float32)
        index.add(vectors, [f"c_{i}" for i in range(5)])
        assert index.size == 5

    def test_clear(self):
        dim = 8
        index = FaissIndex(dimension=dim)

        vectors = np.random.rand(5, dim).astype(np.float32)
        index.add(vectors, [f"c_{i}" for i in range(5)])
        assert index.size == 5

        index.clear()
        assert index.size == 0

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            FaissIndex(dimension=0)
        with pytest.raises(ValueError):
            FaissIndex(dimension=-1)

    def test_shape_mismatch_raises(self):
        index = FaissIndex(dimension=8)
        vectors = np.random.rand(3, 16).astype(np.float32)  # wrong dim
        with pytest.raises(ValueError):
            index.add(vectors, ["a", "b", "c"])

    def test_count_mismatch_raises(self):
        index = FaissIndex(dimension=8)
        vectors = np.random.rand(3, 8).astype(np.float32)
        with pytest.raises(ValueError):
            index.add(vectors, ["a", "b"])  # 3 vectors, 2 ids


# ─── Metric function tests ──────────────────────────────────────────────────

class TestComputeHitRank:

    def test_answer_in_first_result(self):
        results = [
            {"rank": 0, "score": 0.1, "text": "The capital of France is Paris."},
            {"rank": 1, "score": 0.2, "text": "London is in England."},
        ]
        assert compute_hit_rank(results, "Paris") == 0

    def test_answer_in_second_result(self):
        results = [
            {"rank": 0, "score": 0.1, "text": "London is in England."},
            {"rank": 1, "score": 0.2, "text": "The capital of France is Paris."},
        ]
        assert compute_hit_rank(results, "Paris") == 1

    def test_answer_not_found(self):
        results = [
            {"rank": 0, "score": 0.1, "text": "London is in England."},
            {"rank": 1, "score": 0.2, "text": "Berlin is in Germany."},
        ]
        assert compute_hit_rank(results, "Paris") is None

    def test_case_insensitive(self):
        results = [
            {"rank": 0, "score": 0.1, "text": "The answer is PARIS."},
        ]
        assert compute_hit_rank(results, "paris") == 0

    def test_empty_results(self):
        assert compute_hit_rank([], "anything") is None


class TestDeriveMetrics:

    def test_hit_at_rank_0(self):
        metrics = derive_metrics(0, [1, 3, 5])
        assert metrics["hit@1"] == 1
        assert metrics["hit@3"] == 1
        assert metrics["hit@5"] == 1
        assert metrics["rr"] == 1.0

    def test_hit_at_rank_2(self):
        metrics = derive_metrics(2, [1, 3, 5])
        assert metrics["hit@1"] == 0
        assert metrics["hit@3"] == 1
        assert metrics["hit@5"] == 1
        assert metrics["rr"] == pytest.approx(1 / 3)

    def test_hit_at_rank_4(self):
        metrics = derive_metrics(4, [1, 3, 5])
        assert metrics["hit@1"] == 0
        assert metrics["hit@3"] == 0
        assert metrics["hit@5"] == 1
        assert metrics["rr"] == pytest.approx(1 / 5)

    def test_no_hit(self):
        metrics = derive_metrics(None, [1, 3, 5])
        assert metrics["hit@1"] == 0
        assert metrics["hit@3"] == 0
        assert metrics["hit@5"] == 0
        assert metrics["rr"] == 0.0
