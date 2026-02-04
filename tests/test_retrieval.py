"""
test_retrieval.py
─────────────────
Phase 3 manual test: wires chunkers → embeddings → FAISS → search.

Runs the full retrieval pipeline on 3 examples from your dataset.
For each one:
    1. Chunks the context with all 4 baseline chunkers
    2. Embeds every chunk
    3. Indexes them into FAISS
    4. Embeds the question
    5. Searches the index (k=3)
    6. Prints the retrieved chunks so you can eyeball whether
       the correct answer is inside them

HOW TO RUN:
─────────────
    cd /Users/daniyal/adaptive-chunking-rag
    python test_retrieval.py

REQUIREMENTS:
─────────────
- pip install sentence-transformers faiss-cpu numpy
- python -m spacy download en_core_web_sm
- chunkers/ and embeddings/ folders in project root
"""

import json
import sys
import os

# ─── Make sure packages are importable ───────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from chunkers import FixedChunker, SentenceChunker, ParagraphChunker, RecursiveChunker
from embeddings.embedding_model import EmbeddingModel
from embeddings.faiss_index import FaissIndex

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'train_100.json')
#DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'natural_questions_squad_100.json')
# These are the 3 unique contexts picked by find_best_contexts.py
# Update these if you ran find_best_contexts on a different dataset
EXAMPLE_INDICES = [63, 76, 99]

# How many results to retrieve per query
K = 3

# ─── Load data ───────────────────────────────────────────────────────────────
def load_examples(path: str, indices: list) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    examples = data["examples"]
    selected = []
    for i in indices:
        if i < len(examples):
            selected.append((i, examples[i]))
        else:
            print(f"⚠  Index {i} out of range (dataset has {len(examples)} examples). Skipping.")
    return selected


# ─── Chunker configs ─────────────────────────────────────────────────────────
CHUNKERS = {
    "FixedChunker":     FixedChunker(chunk_size=512, overlap=50),
    "SentenceChunker":  SentenceChunker(max_chars=512, overlap_sentences=1),
    "ParagraphChunker": ParagraphChunker(max_chars=512, min_chars=100),
    "RecursiveChunker": RecursiveChunker(max_chars=512, overlap=50),
}


# ─── Core pipeline ───────────────────────────────────────────────────────────
def run_pipeline(chunker_name, chunker, context, question, answer, doc_id, embed_model):
    """
    Full pipeline for one chunker on one example.
    Returns the search results with chunk text resolved.
    """
    # 1. Chunk
    chunks = chunker.chunk(context, document_id=doc_id)
    if not chunks:
        return None, chunks

    # 2. Build chunk_id → chunk lookup (this is the one-line lookup
    #    that keeps text OUT of FaissIndex, per the design decision)
    chunk_map = {c.chunk_id: c for c in chunks}

    # 3. Embed all chunk texts
    chunk_texts = [c.text for c in chunks]
    chunk_vectors = embed_model.encode(chunk_texts)   # (N, 384) float32

    # 4. Build a fresh FAISS index per chunker per example
    #    (each chunker's chunks are a different set — compare apples to apples)
    index = FaissIndex(dimension=EmbeddingModel.DIMENSION)
    index.add(chunk_vectors, [c.chunk_id for c in chunks])

    # 5. Embed the question
    query_vector = embed_model.encode_single(question)  # (384,) float32

    # 6. Search
    results = index.search(query_vector, k=K)

    # 7. Resolve chunk_ids back to text
    resolved = []
    for r in results:
        chunk = chunk_map[r["chunk_id"]]
        resolved.append({
            "rank":       r["rank"],
            "score":      r["score"],
            "chunk_id":   r["chunk_id"],
            "text":       chunk.text,
            "contains_answer": answer.lower() in chunk.text.lower(),
        })

    return resolved, chunks


# ─── Display ─────────────────────────────────────────────────────────────────
def print_separator(char="─", width=70):
    print(char * width)


def print_results(chunker_name, results, chunks):
    """Print one chunker's retrieval results."""
    print(f"\n  📦  {chunker_name}  ({len(chunks)} chunks indexed)")
    print_separator("─", 70)

    if results is None:
        print("    ⚠  No chunks produced — nothing indexed.")
        return

    for r in results:
        hit = "✓ HIT" if r["contains_answer"] else "✗ miss"
        print(f"\n    Rank {r['rank']} | Score {r['score']:.4f} | {hit}")
        print(f"    \"{r['text'][:200]}{'...' if len(r['text']) > 200 else ''}\"")


def print_summary(example_idx, chunker_name, results, answer):
    """One-line summary for the side-by-side table."""
    if results is None:
        return f"  {chunker_name:<22} {'—':>6} {'—':>8} {'—':>6}"

    top1_hit = "✓" if results[0]["contains_answer"] else "✗"
    any_hit  = "✓" if any(r["contains_answer"] for r in results) else "✗"
    return (f"  {chunker_name:<22} {top1_hit:>6} {results[0]['score']:>8.4f} {any_hit:>6}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    examples = load_examples(DATA_PATH, EXAMPLE_INDICES)
    if not examples:
        print("❌  No examples loaded. Check DATA_PATH and EXAMPLE_INDICES.")
        return

    # Load embedding model once (lazy — actual model loads on first .encode())
    print("\n⏳  Loading embedding model (first call downloads if needed)...")
    embed_model = EmbeddingModel()

    for ex_idx, example in examples:
        context  = example["context"]
        question = example["question"]
        answer   = example["answer"]
        doc_id   = f"example_{ex_idx}"

        # ── Header ──
        print("\n\n" + "=" * 70)
        print(f"  EXAMPLE {ex_idx}")
        print("=" * 70)
        print(f"  Question : {question}")
        print(f"  Answer   : {answer}")
        print(f"  Context  : {len(context)} chars")
        print_separator()

        # ── Run each chunker through the full pipeline ──
        summary_rows = []

        for name, chunker in CHUNKERS.items():
            results, chunks = run_pipeline(
                name, chunker, context, question, answer, doc_id, embed_model
            )
            print_results(name, results, chunks)
            summary_rows.append((name, results))

        # ── Side-by-side summary ──
        print("\n")
        print_separator("─", 70)
        print("  📊  RETRIEVAL SUMMARY  (did the answer land in top-K?)")
        print_separator("─", 70)
        print(f"  {'Chunker':<22} {'Top1':>6} {'Score':>8} {'Any@K':>6}")
        print(f"  {'─'*22} {'─'*6} {'─'*8} {'─'*6}")
        for name, results in summary_rows:
            print(print_summary(ex_idx, name, results, answer))


if __name__ == "__main__":
    main()