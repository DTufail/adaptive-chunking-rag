"""
eval_metrics.py
───────────────
Phase 4: quantitative evaluation of all chunkers.

Runs the full retrieval pipeline (chunk → embed → index → search)
across every valid example in the dataset and computes:

    Hit Rate @ K = 1, 3, 5
    MRR (Mean Reciprocal Rank)

PERFORMANCE ARCHITECTURE
─────────────────────────
On a low-clock dual-core CPU (e.g. 1.1 GHz i3) the only thing that
moves wall time is the number of model forward passes.  Everything
else — FAISS search on 3-4 vectors, dict lookups, list appends — is
sub-millisecond.  The script is structured around this fact.

The pipeline runs in three distinct phases:

    Phase A — CHUNK (CPU-bound text ops, no model calls)
        For every unique context in the dataset, run all 4 chunkers.
        Collect the raw chunk texts grouped by chunker.  No encoding
        happens here.  This phase is fast (< 1 second total).

    Phase B — EMBED (all model forward passes live here)
        For each chunker, take every chunk text produced in Phase A
        and encode them in a single batch call.  sentence-transformers
        internally splits this into mini-batches of batch_size and
        runs one forward pass per mini-batch.  Concentrating all texts
        into one call means the tokenizer and forward pass run at
        maximum throughput.  This replaces ~1200 tiny encode() calls
        (one per cache miss in the old code) with 4 large ones.

        Question vectors are also batch-encoded here — one call for
        all ~1000 questions.

    Phase C — SEARCH (zero model calls, everything pre-computed)
        The eval loop.  Every context is already chunked, embedded,
        and indexed.  Every question is already embedded.  Each
        iteration is: dict lookup → FAISS search on 3-4 vectors →
        substring check.  Sub-millisecond per chunker per example.

DESIGN DECISIONS (unchanged from original)
───────────────────────────────────────────
  • Answer filter: drop examples with answers < MIN_ANSWER_LENGTH chars.
  • Search depth: one search(k=5) call. All Hit@K derived from it.
  • Ranks are 0-indexed. hit@K checks hit_rank < K.
  • Saved JSON stores hit_rank (int or null) per example per chunker.

HOW TO RUN:
─────────────
    cd /path/to/adaptive-chunking-rag
    python eval_metrics.py

OUTPUT:
─────────────
  Live progress bar while running.
  Final comparison table to terminal.
  Raw results saved to: results/eval_metrics.json
"""

import json
import sys
import os
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np

# ─── Importable from project root ────────────────────────────────────────────
# Add src directory to sys.path so we can import chunkers and embeddings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from chunkers import FixedChunker, SentenceChunker, ParagraphChunker, RecursiveChunker
from embeddings.embedding_model import EmbeddingModel
from embeddings.faiss_index import FaissIndex


# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'train_1000.json')

K_MAX    = 5                # Single search depth. All Hit@K derived from it.
K_VALUES = [1, 3, 5]        # Which K values to report.
MIN_ANSWER_LENGTH = 4       # Answers shorter than this are filtered out.
OUTPUT_DIR = "results"      # Where to write results JSON.


# ─── Chunker configs — identical to test_retrieval.py ───────────────────────
CHUNKERS: Dict[str, object] = {
    "FixedChunker":     FixedChunker(chunk_size=512, overlap=50),
    "SentenceChunker":  SentenceChunker(max_chars=512, overlap_sentences=1),
    "ParagraphChunker": ParagraphChunker(max_chars=512, min_chars=100),
    "RecursiveChunker": RecursiveChunker(max_chars=512, overlap=50),
}


# ─── Data loading ────────────────────────────────────────────────────────────
def load_and_filter(path: str, min_answer_len: int) -> Tuple[List[dict], Dict[str, int]]:
    """Load examples and drop those whose answers are too short to eval safely.

    Args:
        path: Path to JSON dataset (SQuAD format with top-level "examples" key).
        min_answer_len: Minimum character length for an answer to be included.

    Returns:
        Tuple of (valid_examples, drop_counts).
    """
    with open(path, "r") as f:
        data = json.load(f)

    all_examples = data["examples"]
    valid: List[dict] = []
    drop_counts: Dict[str, int] = defaultdict(int)

    for ex in all_examples:
        answer = ex.get("answer", "")

        if not answer or not answer.strip():
            drop_counts["empty answer"] += 1
            continue

        if len(answer.strip()) < min_answer_len:
            drop_counts[f"answer < {min_answer_len} chars"] += 1
            continue

        valid.append(ex)

    return valid, dict(drop_counts)


# ─── Phase A: chunk all unique contexts ─────────────────────────────────────
def chunk_all_contexts(
    examples: List[dict],
) -> Dict[str, Dict[str, list]]:
    """Run every chunker on every unique context. No encoding here.

    Groups examples by context first so each unique context is chunked
    exactly once per chunker.  Chunking is cheap (text splitting); this
    phase exists to separate it cleanly from the expensive encode phase.

    Args:
        examples: Filtered example list.

    Returns:
        Nested dict:  { chunker_name: { context_text: [Chunk, ...] } }
    """
    # Collect unique contexts (preserve order for determinism)
    seen: Dict[str, bool] = {}
    unique_contexts: List[str] = []
    for ex in examples:
        ctx = ex["context"]
        if ctx not in seen:
            seen[ctx] = True
            unique_contexts.append(ctx)

    chunked: Dict[str, Dict[str, list]] = {}

    for chunker_name, chunker in CHUNKERS.items():
        context_chunks: Dict[str, list] = {}
        for ctx in unique_contexts:
            # doc_id is keyed on context content so it's stable across questions
            doc_id = f"ctx_{hash(ctx) % (10**8)}"
            chunks = chunker.chunk(ctx, document_id=doc_id)
            if chunks:
                context_chunks[ctx] = chunks
        chunked[chunker_name] = context_chunks

    return chunked


# ─── Phase B: batch-encode everything ────────────────────────────────────────
def encode_all(
    examples: List[dict],
    chunked: Dict[str, Dict[str, list]],
    embed_model: EmbeddingModel,
) -> Tuple[
    Dict[int, np.ndarray],                                          # question vectors
    Dict[str, Dict[str, Tuple[list, "FaissIndex", Dict[str, object]]]]  # per-chunker indexed cache
]:
    """Batch-encode questions and all chunk texts, build FAISS indexes.

    This is where every model forward pass in the entire script lives.
    Two batch encode() calls: one for all questions, one per chunker
    for all chunk texts.  sentence-transformers handles the internal
    mini-batching.

    Args:
        examples: Filtered example list (for questions).
        chunked: Output of chunk_all_contexts().
        embed_model: Shared EmbeddingModel instance.

    Returns:
        question_vectors: { example_index: (384,) float32 array }
        indexed_cache:    { chunker_name: { context: (chunks, FaissIndex, chunk_map) } }
    """
    # ── Questions: one batch call ────────────────────────────────────────────
    print("  Encoding questions...")
    questions = [ex["question"] for ex in examples]
    q_matrix  = embed_model.encode(questions)                        # (N, 384)
    question_vectors: Dict[int, np.ndarray] = {
        i: q_matrix[i] for i in range(len(questions))
    }
    print(f"    {len(question_vectors)} question vectors.")

    # ── Chunks: one batch call per chunker ───────────────────────────────────
    indexed_cache: Dict[str, Dict[str, Tuple[list, FaissIndex, dict]]] = {}

    for chunker_name, context_chunks in chunked.items():
        print(f"  Encoding chunks for {chunker_name}...")

        if not context_chunks:
            indexed_cache[chunker_name] = {}
            continue

        # Flatten: collect all chunk texts in order, track boundaries.
        # boundaries[i] = (context, start_index, end_index) into the flat list.
        contexts_ordered: List[str]   = []
        flat_texts:       List[str]   = []
        boundaries:       List[Tuple[str, int, int]] = []

        for ctx, chunks in context_chunks.items():
            start = len(flat_texts)
            flat_texts.extend(c.text for c in chunks)
            end = len(flat_texts)
            contexts_ordered.append(ctx)
            boundaries.append((ctx, start, end))

        # ONE encode call for all chunk texts of this chunker.
        all_vectors = embed_model.encode(flat_texts)                 # (M, 384)

        # Split vectors back into per-context groups, build index per context.
        chunker_cache: Dict[str, Tuple[list, FaissIndex, dict]] = {}

        for ctx, start, end in boundaries:
            chunks      = context_chunks[ctx]
            vectors     = all_vectors[start:end]                     # (n_chunks, 384)
            chunk_ids   = [c.chunk_id for c in chunks]

            index = FaissIndex(dimension=EmbeddingModel.DIMENSION)
            index.add(vectors, chunk_ids)

            # Build chunk_map once, cache it — not on every search call.
            chunk_map = {c.chunk_id: c for c in chunks}

            chunker_cache[ctx] = (chunks, index, chunk_map)

        indexed_cache[chunker_name] = chunker_cache
        print(f"    {len(flat_texts)} chunk texts → {len(boundaries)} indexes.")

    return question_vectors, indexed_cache


# ─── Phase C: eval loop (zero encode calls) ─────────────────────────────────
def search_cached(
    indexed_cache: Dict[str, Dict[str, Tuple[list, FaissIndex, dict]]],
    chunker_name: str,
    context: str,
    query_vector: np.ndarray,
    k: int,
) -> List[dict]:
    """Search a pre-built index. No encoding, no chunking, no cache miss path.

    Args:
        indexed_cache: Output of encode_all().
        chunker_name: Which chunker's index to search.
        context: The context text (used as key into the cache).
        query_vector: Pre-computed question vector.
        k: Number of results.

    Returns:
        List of dicts: rank, score, text.  Empty if context not indexed.
    """
    entry = indexed_cache[chunker_name].get(context)
    if entry is None:
        return []

    chunks, index, chunk_map = entry
    raw_results = index.search(query_vector, k=k)

    return [
        {"rank": r["rank"], "score": r["score"], "text": chunk_map[r["chunk_id"]].text}
        for r in raw_results
    ]


# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_hit_rank(results: List[dict], answer: str) -> Optional[int]:
    """Find the rank of the first result whose text contains the answer.

    Args:
        results: Search results sorted by rank ascending.
        answer: Ground-truth answer string.

    Returns:
        0-indexed rank of first hit, or None if answer not found.
    """
    answer_lower = answer.lower()
    for r in results:
        if answer_lower in r["text"].lower():
            return r["rank"]
    return None


def derive_metrics(hit_rank: Optional[int], k_values: List[int]) -> dict:
    """Derive all metrics from a single hit_rank value.

    Args:
        hit_rank: 0-indexed rank of first hit, or None.
        k_values: List of K values to compute Hit Rate for.

    Returns:
        Dict: {"hit@1": 0|1, "hit@3": 0|1, "hit@5": 0|1, "rr": float}
    """
    metrics: dict = {}
    for k in k_values:
        metrics[f"hit@{k}"] = 1 if (hit_rank is not None and hit_rank < k) else 0
    metrics["rr"] = (1.0 / (hit_rank + 1)) if hit_rank is not None else 0.0
    return metrics


# ─── Progress display ────────────────────────────────────────────────────────
def print_progress(current: int, total: int, start_time: float) -> None:
    """Overwrite-in-place progress bar.

    Args:
        current: Examples processed so far.
        total: Total examples.
        start_time: time.time() when the run started.
    """
    pct = current / total
    bar_width = 28
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    elapsed = time.time() - start_time
    if current > 0:
        eta_seconds = int((elapsed / current) * (total - current))
        eta_str = f"{eta_seconds}s"
    else:
        eta_str = "—"

    print(
        f"\r  [{current:>4}/{total}]  {bar}  {pct * 100:5.1f}%  "
        f"elapsed {int(elapsed):>4}s  ETA {eta_str:<6}",
        end="",
        flush=True,
    )


# ─── Results table ───────────────────────────────────────────────────────────
def print_results_table(
    aggregated: Dict[str, dict],
    num_examples: int,
    num_unique_contexts: int,
    elapsed: float,
) -> None:
    """Print the final comparison table, sorted by MRR descending.

    Args:
        aggregated: {chunker_name: {"hit@1": float, ..., "rr": float}}
        num_examples: Number of examples evaluated.
        num_unique_contexts: Number of unique contexts in the dataset.
        elapsed: Total wall-clock seconds.
    """
    print("\n" + "=" * 68)
    print("  📊  PHASE 4 RESULTS — Chunker Comparison")
    print("=" * 68)
    print(f"  Examples evaluated  : {num_examples}")
    print(f"  Unique contexts     : {num_unique_contexts}")
    print(f"  Answer filter       : min {MIN_ANSWER_LENGTH} chars")
    print(f"  Search depth (K)    : {K_MAX}")
    print(f"  Wall-clock time     : {elapsed:.1f}s")
    print()

    # Header
    k_header = "".join(f"{'Hit@' + str(k):>9}" for k in K_VALUES)
    print(f"  {'Chunker':<22} {k_header} {'MRR':>9}")
    print(f"  {'─' * 22} " + " ────────" * len(K_VALUES) + " ────────")

    # Rows sorted by MRR descending
    for name, agg in sorted(aggregated.items(), key=lambda x: x[1]["rr"], reverse=True):
        k_vals = "".join(f"{agg[f'hit@{k}']:>9.4f}" for k in K_VALUES)
        print(f"  {name:<22} {k_vals} {agg['rr']:>9.4f}")

    print("=" * 68)


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── 1. Load & filter ─────────────────────────────────────────────────────
    print("\n⏳  Loading and filtering dataset...")
    examples, drop_counts = load_and_filter(DATA_PATH, MIN_ANSWER_LENGTH)

    print(f"  Valid examples : {len(examples)}")
    for reason, count in drop_counts.items():
        print(f"  Dropped        : {count} ({reason})")

    if not examples:
        print("❌  No valid examples after filtering. Check DATA_PATH.")
        return

    # ── 2. Init embedding model ─────────────────────────────────────────────
    print("\n⏳  Loading embedding model (downloads on first use if needed)...")
    embed_model = EmbeddingModel()

    # ── 3. Phase A: chunk all unique contexts (no model calls) ──────────────
    print("\n⏳  Phase A — chunking all unique contexts...")
    chunked = chunk_all_contexts(examples)

    num_unique = len(set(ex["context"] for ex in examples))
    print(f"  {num_unique} unique contexts chunked across {len(CHUNKERS)} chunkers.")

    # ── 4. Phase B: batch-encode everything (all forward passes) ────────────
    print("\n⏳  Phase B — batch encoding (all model inference runs here)...")
    question_vectors, indexed_cache = encode_all(examples, chunked, embed_model)

    # ── 5. Phase C: eval loop (zero model calls) ───────────────────────────
    total = len(examples)
    print(f"\n⏳  Phase C — evaluating {total} examples × {len(CHUNKERS)} chunkers...")
    print(f"    (all indexes pre-built, zero encode calls in this loop)\n")

    per_example_hit_ranks: Dict[str, List[Optional[int]]] = {
        name: [] for name in CHUNKERS
    }

    start_time = time.time()

    for i, example in enumerate(examples):
        context = example["context"]
        answer  = example["answer"]
        query_vector = question_vectors[i]

        for chunker_name in CHUNKERS:
            results  = search_cached(indexed_cache, chunker_name, context, query_vector, k=K_MAX)
            hit_rank = compute_hit_rank(results, answer)
            per_example_hit_ranks[chunker_name].append(hit_rank)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print_progress(i + 1, total, start_time)

    elapsed = time.time() - start_time
    print("\n")

    # ── 6. Aggregate ────────────────────────────────────────────────────────
    aggregated: Dict[str, dict] = {}

    for chunker_name, hit_ranks in per_example_hit_ranks.items():
        all_metrics = [derive_metrics(hr, K_VALUES) for hr in hit_ranks]
        agg: dict = {}
        for key in all_metrics[0]:
            agg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        aggregated[chunker_name] = agg

    # ── 7. Print table ──────────────────────────────────────────────────────
    print_results_table(aggregated, total, num_unique, elapsed)

    # ── 8. Save to JSON ─────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "eval_metrics.json")

    save_payload = {
        "config": {
            "data_path":         DATA_PATH,
            "num_examples":      total,
            "num_unique_contexts": num_unique,
            "k_max":             K_MAX,
            "k_values":          K_VALUES,
            "min_answer_length": MIN_ANSWER_LENGTH,
            "chunkers":          {name: repr(c) for name, c in CHUNKERS.items()},
            "elapsed_seconds":   round(elapsed, 2),
        },
        "aggregated": aggregated,
        "per_example_hit_ranks": per_example_hit_ranks,
    }

    with open(output_path, "w") as f:
        json.dump(save_payload, f, indent=2)

    print(f"\n  💾  Full results saved to: {output_path}")


if __name__ == "__main__":
    main()