"""
eval_metrics.py
───────────────
Phase 4: quantitative evaluation of all chunkers.

Runs the full retrieval pipeline (chunk → embed → index → search)
across every valid example in the dataset and computes:

    Hit Rate @ K = 1, 3, 5
    MRR (Mean Reciprocal Rank)

ARCHITECTURE — WHY SUBPROCESSES
────────────────────────────────
PyTorch's CPU memory allocator (c10::Allocator) requests pages from
the OS via mmap and never returns them, even after tensors are freed.
On macOS there is no API to force a release.  In a single long-lived
process the mapped address space grows monotonically across every
encode() call until the OS OOM-killer steps in.

The fix: Phase B (all model inference) runs as a sequence of
short-lived subprocesses.  Each one loads the model, encodes one
batch of texts, writes the vectors to disk as .npy, and exits.
On exit the OS reclaims every page.  The next subprocess starts
with a clean address space.

The main process (this file) never imports sentence-transformers or
torch.  It does Phase A (chunking — pure text), orchestrates Phase B
(spawning workers), and runs Phase C (FAISS search + metrics — numpy
only).  Its peak memory is ~200 MB regardless of dataset size.

PHASES
──────
    Phase A — CHUNK   : chunk all unique contexts.  Text ops only.
    Phase B — ENCODE  : spawn one subprocess per encode job.
                        Jobs: questions, FixedChunker, SentenceChunker,
                              ParagraphChunker, RecursiveChunker.
                        Each subprocess: load model → encode → write .npy → exit.
    Phase C — SEARCH  : load vectors from .npy, build FAISS indexes,
                        run eval loop.  Zero model calls.

DESIGN DECISIONS (unchanged)
─────────────────────────────
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
import subprocess
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np

# ─── Importable from project root ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# NOTE: we do NOT import EmbeddingModel or sentence-transformers here.
# That's the whole point.  Only the subprocess (encode_worker.py) does.
from chunkers import FixedChunker, SentenceChunker, ParagraphChunker, RecursiveChunker
from embeddings.faiss_index import FaissIndex


# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/train_1000.json"

K_MAX    = 5
K_VALUES = [1, 3, 5]
MIN_ANSWER_LENGTH = 4
OUTPUT_DIR = "results"

# Temp directory for Phase B I/O between main process and workers.
# Created at the start, cleaned up at the end.
ENCODE_TMP_DIR = "results/_encode_tmp"

# The worker script, relative to this file.
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encode_worker.py")

# EmbeddingModel.DIMENSION — hardcoded here so we don't have to import it.
# If the model changes, update this too.
EMBEDDING_DIMENSION = 384


# ─── Chunker configs ─────────────────────────────────────────────────────────
CHUNKERS: Dict[str, object] = {
    "FixedChunker":     FixedChunker(chunk_size=512, overlap=50),
    "SentenceChunker":  SentenceChunker(max_chars=512, overlap_sentences=1),
    "ParagraphChunker": ParagraphChunker(max_chars=512, min_chars=100),
    "RecursiveChunker": RecursiveChunker(max_chars=512, overlap=50),
}


# ─── Data loading ────────────────────────────────────────────────────────────
def load_and_filter(path: str, min_answer_len: int) -> Tuple[List[dict], Dict[str, int]]:
    """Load examples and drop those whose answers are too short."""
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
    """Run every chunker on every unique context.  No encoding.

    Returns:
        { chunker_name: { context_text: [Chunk, ...] } }
    """
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
            doc_id = f"ctx_{hash(ctx) % (10**8)}"
            chunks = chunker.chunk(ctx, document_id=doc_id)
            if chunks:
                context_chunks[ctx] = chunks
        chunked[chunker_name] = context_chunks

    return chunked


# ─── Phase B: spawn workers ─────────────────────────────────────────────────
def write_texts_json(texts: List[str], path: str) -> None:
    """Write a list of texts to a JSON file for the worker to read."""
    with open(path, "w") as f:
        json.dump({"texts": texts}, f)


def run_encode_worker(job_name: str, input_json: str, output_npy: str) -> None:
    """Spawn encode_worker.py as a subprocess and wait for it to finish.

    The subprocess loads the model, encodes, writes .npy, and exits.
    All of PyTorch's memory is reclaimed by the OS on exit.

    Args:
        job_name: Label for progress output (e.g. "questions", "FixedChunker").
        input_json: Path to the JSON file with texts.
        output_npy: Path where the worker will write the .npy output.

    Raises:
        RuntimeError: If the subprocess exits with non-zero status.
    """
    cmd = [sys.executable, WORKER_SCRIPT, job_name, input_json, output_npy]
    print(f"  Spawning worker: {job_name}...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Always print worker stdout so progress is visible
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")

    if result.returncode != 0:
        print(f"    STDERR: {result.stderr}", flush=True)
        raise RuntimeError(
            f"encode_worker failed for {job_name} "
            f"(exit code {result.returncode})"
        )


def phase_b_encode(
    examples: List[dict],
    chunked: Dict[str, Dict[str, list]],
) -> Tuple[
    Dict[int, np.ndarray],                                          # question vectors
    Dict[str, np.ndarray],                                          # chunker name → flat vectors
    Dict[str, List[Tuple[str, int, int]]],                          # chunker name → boundaries
]:
    """Write texts to disk, spawn workers, read vectors back.

    No model is loaded in this process.  Each worker loads its own
    copy of the model, uses it, and exits.

    Args:
        examples: Filtered example list (for questions).
        chunked: Output of chunk_all_contexts().

    Returns:
        question_vectors: { index: (384,) float32 }
        chunker_vectors:  { chunker_name: (M, 384) float32 }
        chunker_boundaries: { chunker_name: [(context, start, end), ...] }
                            (needed to slice vectors back into per-context groups)
    """
    os.makedirs(ENCODE_TMP_DIR, exist_ok=True)

    # ── Questions ────────────────────────────────────────────────────────────
    questions      = [ex["question"] for ex in examples]
    q_input_path   = os.path.join(ENCODE_TMP_DIR, "questions_input.json")
    q_output_path  = os.path.join(ENCODE_TMP_DIR, "questions.npy")

    write_texts_json(questions, q_input_path)
    run_encode_worker("questions", q_input_path, q_output_path)

    q_matrix = np.load(q_output_path)                                # (N, 384)
    question_vectors: Dict[int, np.ndarray] = {
        i: q_matrix[i] for i in range(len(questions))
    }
    del q_matrix   # rows are views but that's fine — q_matrix is small

    # ── Chunks: one worker per chunker ───────────────────────────────────────
    chunker_vectors:    Dict[str, np.ndarray]                    = {}
    chunker_boundaries: Dict[str, List[Tuple[str, int, int]]]   = {}

    for chunker_name, context_chunks in chunked.items():
        if not context_chunks:
            chunker_vectors[chunker_name]    = np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
            chunker_boundaries[chunker_name] = []
            continue

        # Flatten chunk texts, track boundaries (same logic as before)
        flat_texts: List[str]               = []
        boundaries: List[Tuple[str, int, int]] = []

        for ctx, chunks in context_chunks.items():
            start = len(flat_texts)
            flat_texts.extend(c.text for c in chunks)
            end = len(flat_texts)
            boundaries.append((ctx, start, end))

        # Write texts, spawn worker, read vectors back
        input_path  = os.path.join(ENCODE_TMP_DIR, f"{chunker_name}_input.json")
        output_path = os.path.join(ENCODE_TMP_DIR, f"{chunker_name}.npy")

        write_texts_json(flat_texts, input_path)
        run_encode_worker(chunker_name, input_path, output_path)

        chunker_vectors[chunker_name]    = np.load(output_path)      # (M, 384)
        chunker_boundaries[chunker_name] = boundaries

    return question_vectors, chunker_vectors, chunker_boundaries


def cleanup_encode_tmp() -> None:
    """Remove the temp directory and all files in it."""
    import shutil
    if os.path.exists(ENCODE_TMP_DIR):
        shutil.rmtree(ENCODE_TMP_DIR)


# ─── Phase C: build indexes and search ──────────────────────────────────────
def build_indexed_cache(
    chunked:            Dict[str, Dict[str, list]],
    chunker_vectors:    Dict[str, np.ndarray],
    chunker_boundaries: Dict[str, List[Tuple[str, int, int]]],
) -> Dict[str, Dict[str, Tuple[list, "FaissIndex", dict]]]:
    """Build FAISS indexes from the vectors loaded off disk.

    This runs in the main process.  No model, no PyTorch.  Just numpy
    slicing and FAISS index construction.

    Args:
        chunked: The Chunk objects from Phase A (needed for chunk_ids and text).
        chunker_vectors: The encoded vectors from Phase B.
        chunker_boundaries: The (context, start, end) slicing info from Phase B.

    Returns:
        { chunker_name: { context: (chunks, FaissIndex, chunk_map) } }
    """
    indexed_cache: Dict[str, Dict[str, Tuple[list, FaissIndex, dict]]] = {}

    for chunker_name in CHUNKERS:
        all_vectors = chunker_vectors[chunker_name]
        boundaries  = chunker_boundaries[chunker_name]
        context_chunks = chunked[chunker_name]

        chunker_cache: Dict[str, Tuple[list, FaissIndex, dict]] = {}

        for ctx, start, end in boundaries:
            chunks    = context_chunks[ctx]
            vectors   = all_vectors[start:end].copy()
            chunk_ids = [c.chunk_id for c in chunks]

            index = FaissIndex(dimension=EMBEDDING_DIMENSION)
            index.add(vectors, chunk_ids)

            chunk_map = {c.chunk_id: c for c in chunks}
            chunker_cache[ctx] = (chunks, index, chunk_map)

        indexed_cache[chunker_name] = chunker_cache

    return indexed_cache


def search_cached(
    indexed_cache: Dict[str, Dict[str, Tuple[list, "FaissIndex", dict]]],
    chunker_name: str,
    context: str,
    query_vector: np.ndarray,
    k: int,
) -> List[dict]:
    """Search a pre-built index.  Zero model calls."""
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
    """First rank whose text contains the answer, or None."""
    answer_lower = answer.lower()
    for r in results:
        if answer_lower in r["text"].lower():
            return r["rank"]
    return None


def derive_metrics(hit_rank: Optional[int], k_values: List[int]) -> dict:
    """Derive Hit@K and RR from a single hit_rank."""
    metrics: dict = {}
    for k in k_values:
        metrics[f"hit@{k}"] = 1 if (hit_rank is not None and hit_rank < k) else 0
    metrics["rr"] = (1.0 / (hit_rank + 1)) if hit_rank is not None else 0.0
    return metrics


# ─── Progress display ────────────────────────────────────────────────────────
def print_progress(current: int, total: int, start_time: float) -> None:
    pct = current / total
    bar_width = 28
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    elapsed = time.time() - start_time
    eta_str = f"{int((elapsed / current) * (total - current))}s" if current > 0 else "—"

    print(
        f"\r  [{current:>4}/{total}]  {bar}  {pct * 100:5.1f}%  "
        f"elapsed {int(elapsed):>4}s  ETA {eta_str:<6}",
        end="", flush=True,
    )


# ─── Results table ───────────────────────────────────────────────────────────
def print_results_table(
    aggregated: Dict[str, dict],
    num_examples: int,
    num_unique_contexts: int,
    elapsed: float,
) -> None:
    print("\n" + "=" * 68)
    print("  📊  PHASE 4 RESULTS — Chunker Comparison")
    print("=" * 68)
    print(f"  Examples evaluated  : {num_examples}")
    print(f"  Unique contexts     : {num_unique_contexts}")
    print(f"  Answer filter       : min {MIN_ANSWER_LENGTH} chars")
    print(f"  Search depth (K)    : {K_MAX}")
    print(f"  Wall-clock time     : {elapsed:.1f}s")
    print()

    k_header = "".join(f"{'Hit@' + str(k):>9}" for k in K_VALUES)
    print(f"  {'Chunker':<22} {k_header} {'MRR':>9}")
    print(f"  {'─' * 22} " + " ────────" * len(K_VALUES) + " ────────")

    for name, agg in sorted(aggregated.items(), key=lambda x: x[1]["rr"], reverse=True):
        k_vals = "".join(f"{agg[f'hit@{k}']:>9.4f}" for k in K_VALUES)
        print(f"  {name:<22} {k_vals} {agg['rr']:>9.4f}")

    print("=" * 68)


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    total_start = time.time()

    # ── 1. Load & filter ─────────────────────────────────────────────────────
    print("\n⏳  Loading and filtering dataset...")
    examples, drop_counts = load_and_filter(DATA_PATH, MIN_ANSWER_LENGTH)

    print(f"  Valid examples : {len(examples)}")
    for reason, count in drop_counts.items():
        print(f"  Dropped        : {count} ({reason})")

    if not examples:
        print("❌  No valid examples after filtering. Check DATA_PATH.")
        return

    num_unique = len(set(ex["context"] for ex in examples))

    # ── 2. Phase A: chunk ────────────────────────────────────────────────────
    print("\n⏳  Phase A — chunking all unique contexts...")
    chunked = chunk_all_contexts(examples)
    print(f"  {num_unique} unique contexts × {len(CHUNKERS)} chunkers.")

    # ── 3. Phase B: encode via subprocesses ──────────────────────────────────
    print("\n⏳  Phase B — encoding (each job is a separate subprocess)...")
    print(f"  Each subprocess loads the model, encodes, writes .npy, and exits.")
    print(f"  PyTorch memory is fully reclaimed between jobs.\n")

    question_vectors, chunker_vectors, chunker_boundaries = phase_b_encode(
        examples, chunked
    )

    # ── 4. Build FAISS indexes (main process, no model) ─────────────────────
    print("\n⏳  Building FAISS indexes...")
    indexed_cache = build_indexed_cache(chunked, chunker_vectors, chunker_boundaries)

    # Release the flat vector arrays — they've been sliced+copied into indexes
    del chunker_vectors, chunker_boundaries

    # Clean up temp files
    cleanup_encode_tmp()
    print("  Done. Temp files cleaned up.")

    # ── 5. Phase C: eval loop ────────────────────────────────────────────────
    total = len(examples)
    print(f"\n⏳  Phase C — evaluating {total} examples × {len(CHUNKERS)} chunkers...")
    print(f"    (all indexes pre-built, zero encode calls)\n")

    per_example_hit_ranks: Dict[str, List[Optional[int]]] = {
        name: [] for name in CHUNKERS
    }

    loop_start = time.time()

    for i, example in enumerate(examples):
        context      = example["context"]
        answer       = example["answer"]
        query_vector = question_vectors[i]

        for chunker_name in CHUNKERS:
            results  = search_cached(indexed_cache, chunker_name, context, query_vector, k=K_MAX)
            hit_rank = compute_hit_rank(results, answer)
            per_example_hit_ranks[chunker_name].append(hit_rank)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print_progress(i + 1, total, loop_start)

    print("\n")

    # ── 6. Aggregate ────────────────────────────────────────────────────────
    aggregated: Dict[str, dict] = {}
    for chunker_name, hit_ranks in per_example_hit_ranks.items():
        all_metrics = [derive_metrics(hr, K_VALUES) for hr in hit_ranks]
        agg: dict = {}
        for key in all_metrics[0]:
            agg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        aggregated[chunker_name] = agg

    # ── 7. Print + save ─────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print_results_table(aggregated, total, num_unique, total_elapsed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "eval_metrics.json")

    save_payload = {
        "config": {
            "data_path":           DATA_PATH,
            "num_examples":        total,
            "num_unique_contexts": num_unique,
            "k_max":               K_MAX,
            "k_values":            K_VALUES,
            "min_answer_length":   MIN_ANSWER_LENGTH,
            "chunkers":            {name: repr(c) for name, c in CHUNKERS.items()},
            "elapsed_seconds":     round(total_elapsed, 2),
            "encoding_method":     "subprocess-per-job",
        },
        "aggregated": aggregated,
        "per_example_hit_ranks": per_example_hit_ranks,
    }

    with open(output_path, "w") as f:
        json.dump(save_payload, f, indent=2)

    print(f"\n  💾  Full results saved to: {output_path}")


if __name__ == "__main__":
    main()