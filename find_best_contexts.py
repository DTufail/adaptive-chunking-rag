"""
find_best_contexts.py  (v2)
────────────────────────────
Scans your dataset and picks the best UNIQUE contexts for chunker testing.

Key fix over v1: SQuAD has many questions per context. This version
deduplicates first, then scores — so you get 3 genuinely different texts.

HOW TO RUN:
─────────────
    cd /Users/daniyal/adaptive-chunking-rag
    python find_best_contexts.py

Change this line at the bottom to switch datasets:
    DATA_PATH = "data/train_100.json"   ← or "data/train_1000.json"
"""

import json

DATA_PATH = "data/train_100.json"  # ← change to train_100.json if needed


def analyze_context(idx: int, context: str, question: str) -> dict:
    """Extract structural signals from a context string."""
    return {
        "index": idx,
        "length": len(context),
        "num_sentences": context.count(". ") + context.count(".\n"),
        "has_paragraphs": "\n\n" in context,
        "num_paragraphs": context.count("\n\n") + 1,
        "num_newlines": context.count("\n"),
        "has_bullets": any(line.strip().startswith(b) for line in context.split("\n") for b in ["- ", "* ", "• "]),
        "has_numbers": any(f"{i}." in context for i in range(1, 10)),
        "question": question,
        "context": context,
    }


def score_context(a: dict) -> int:
    """Score a context. Higher = better for testing chunkers."""
    score = 0
    score += a["length"]                      # longer is better
    score += a["num_sentences"] * 100         # more sentences = more boundary variety
    score += a["has_paragraphs"] * 800        # paragraph breaks are the most valuable signal
    score += a["num_paragraphs"] * 300        # more paragraphs = more structure
    score += a["num_newlines"] * 80           # newlines add structure
    score += a["has_bullets"] * 400           # bullet lists stress-test chunkers well
    score += a["has_numbers"] * 200           # numbered content adds variety
    return score


def main():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    examples = data["examples"]
    print(f"Loaded {len(examples)} examples from {DATA_PATH}\n")

    # ── Step 1: Deduplicate by context text ─────────────────────────────
    # Keep the first occurrence of each unique context.
    seen_contexts = set()
    unique = []

    for i, ex in enumerate(examples):
        ctx = ex["context"]
        if ctx not in seen_contexts:
            seen_contexts.add(ctx)
            a = analyze_context(i, ctx, ex["question"])
            a["score"] = score_context(a)
            unique.append(a)

    print(f"Unique contexts: {len(unique)}  (out of {len(examples)} examples)\n")

    # ── Step 2: Sort by score ────────────────────────────────────────────
    unique.sort(key=lambda x: x["score"], reverse=True)

    # ── Full ranked table ────────────────────────────────────────────────
    print("=" * 90)
    print("  UNIQUE CONTEXTS — RANKED BY SUITABILITY FOR CHUNKER TESTING")
    print("=" * 90)
    print(f"  {'Rank':<5} {'Idx':<5} {'Len':<6} {'Sents':<6} {'Paras':<6} "
          f"{'\\n':<4} {'Bul':<4} {'Score':<7} Question")
    print(f"  {'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*6} "
          f"{'─'*4} {'─'*4} {'─'*7} {'─'*45}")

    for rank, a in enumerate(unique):
        bul = "✓" if a["has_bullets"] else " "
        print(f"  {rank:<5} {a['index']:<5} {a['length']:<6} {a['num_sentences']:<6} "
              f"{a['num_paragraphs']:<6} {a['num_newlines']:<4} {bul:<4} {a['score']:<7} "
              f"{a['question'][:50]}")

    # ── Top 5 detailed picks ─────────────────────────────────────────────
    top5 = unique[:5]

    print("\n" + "=" * 90)
    print("  TOP 5 UNIQUE CONTEXTS — WHY THEY'RE GOOD")
    print("=" * 90)

    for rank, a in enumerate(top5):
        print(f"\n  #{rank + 1}  →  Example index {a['index']}  (score: {a['score']})")
        print(f"  {'─' * 65}")
        print(f"  Length:      {a['length']} chars")
        print(f"  Sentences:   {a['num_sentences']}")
        print(f"  Paragraphs:  {a['num_paragraphs']} {'✓ has \\n\\n breaks' if a['has_paragraphs'] else '✗ single block'}")
        print(f"  Newlines:    {a['num_newlines']}")
        print(f"  Bullets:     {'✓ yes' if a['has_bullets'] else '✗ no'}")
        print(f"  Numbers:     {'✓ yes' if a['has_numbers'] else '✗ no'}")
        print(f"  Question:    {a['question']}")
        print(f"  Preview:     {a['context'][:150]}...")

    # ── The line to paste ────────────────────────────────────────────────
    top3_indices = [a["index"] for a in unique[:3]]

    print("\n" + "=" * 90)
    print("  PASTE THIS INTO test_chunkers.py")
    print("=" * 90)
    print(f"\n  EXAMPLE_INDICES = {top3_indices}\n")


if __name__ == "__main__":
    main()