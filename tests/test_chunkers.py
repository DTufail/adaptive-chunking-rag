"""
test_chunkers.py
─────────────────
Manual comparison test: runs all 4 baseline chunkers on the same text
from your train_100.json so you can see exactly how each one behaves.

HOW TO RUN:
─────────────
    cd /Users/daniyal/adaptive-chunking-rag
    python test_chunkers.py

REQUIREMENTS:
─────────────
- Your chunkers/ folder must be in the same directory (or on sys.path)
- spaCy + en_core_web_sm installed:
      pip install spacy
      python -m spacy download en_core_web_sm
- train_100.json at: data/train_100.json
"""

import json
import sys
import os

# ─── Make sure the chunkers package is importable ───────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from chunkers import (
    FixedChunker,
    SentenceChunker,
    ParagraphChunker,
    RecursiveChunker,
)

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/train_100.json"

# Which example indices from your dataset to test on (0-indexed)
# Change these to look at different examples
EXAMPLE_INDICES = [63, 76, 99]

# ─── Load data ───────────────────────────────────────────────────────────────
def load_examples(path: str, indices: list) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    examples = data["examples"]
    selected = []
    for i in indices:
        if i < len(examples):
            selected.append(examples[i])
        else:
            print(f"⚠  Index {i} out of range (dataset has {len(examples)} examples). Skipping.")
    return selected


# ─── Define all chunkers with matching configs ──────────────────────────────
# All use ~512 char target so the comparison is apples-to-apples.
CHUNKERS = {
    "FixedChunker":     FixedChunker(chunk_size=512, overlap=50),
    "SentenceChunker":  SentenceChunker(max_chars=512, overlap_sentences=1),
    "ParagraphChunker": ParagraphChunker(max_chars=512, min_chars=100),
    "RecursiveChunker": RecursiveChunker(max_chars=512, overlap=50),
}


# ─── Display helpers ─────────────────────────────────────────────────────────
def print_separator(char="─", width=70):
    print(char * width)


def print_chunk_detail(chunk, show_text=True):
    """Print one chunk with its metadata."""
    print(f"    Chunk {chunk.metadata['chunk_index']:>2} | "
          f"pos [{chunk.start_pos:>4}:{chunk.end_pos:>4}] | "
          f"len {chunk.length:>4} chars")
    if show_text:
        # Indent and wrap the text for readability
        lines = chunk.text.strip().replace("\n", "\n      ")
        print(f"      \"{lines}\"")
        print()


def print_stats(chunker_name, chunks):
    """Print a quick stats summary for a chunker's output."""
    if not chunks:
        print(f"    ⚠  No chunks produced.")
        return

    lengths = [c.length for c in chunks]
    print(f"    Count: {len(chunks)} chunks | "
          f"Min: {min(lengths)} | "
          f"Max: {max(lengths)} | "
          f"Avg: {sum(lengths)/len(lengths):.0f} chars")


# ─── Main comparison loop ───────────────────────────────────────────────────
def run_comparison():
    examples = load_examples(DATA_PATH, EXAMPLE_INDICES)

    if not examples:
        print("❌  No examples loaded. Check DATA_PATH and EXAMPLE_INDICES.")
        return

    for ex_idx, example in enumerate(examples):
        context = example["context"]
        question = example["question"]
        answer   = example["answer"]

        # ── Header ──
        print("\n")
        print("=" * 70)
        print(f"  EXAMPLE {EXAMPLE_INDICES[ex_idx]}")
        print("=" * 70)
        print(f"  Question : {question}")
        print(f"  Answer   : {answer}")
        print(f"  Context  : {len(context)} chars")
        print_separator()

        # ── Run each chunker on the same context ──
        for name, chunker in CHUNKERS.items():
            print(f"\n  📦  {name}")
            print_separator("─", 70)

            chunks = chunker.chunk(context, document_id=f"example_{EXAMPLE_INDICES[ex_idx]}")

            # Stats line
            print_stats(name, chunks)
            print()

            # Show each chunk
            for chunk in chunks:
                print_chunk_detail(chunk, show_text=True)

        # ── Quick side-by-side stats comparison at the end of each example ──
        print()
        print_separator("─", 70)
        print("  📊  SIDE-BY-SIDE STATS")
        print_separator("─", 70)
        print(f"  {'Chunker':<22} {'Chunks':>6} {'Min':>6} {'Max':>6} {'Avg':>6}")
        print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

        for name, chunker in CHUNKERS.items():
            chunks = chunker.chunk(context, document_id=f"example_{EXAMPLE_INDICES[ex_idx]}")
            if chunks:
                lengths = [c.length for c in chunks]
                print(f"  {name:<22} {len(chunks):>6} {min(lengths):>6} {max(lengths):>6} {sum(lengths)/len(lengths):>6.0f}")
            else:
                print(f"  {name:<22}      0     —     —     —")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_comparison()