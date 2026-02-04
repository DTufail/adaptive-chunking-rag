"""
hyperparameter_search_cpu.py
─────────────────────────────
CPU-optimized grid search over chunk_size and overlap.

Designed for MacBook (8GB RAM, dual-core i3). Memory-efficient:
- No SentenceChunker (avoids spaCy RAM issue)
- Batch size 64 for CPU
- gc.collect() between configs
- Tests only FixedChunker and RecursiveChunker

Grid search space:
- chunk_size: [256, 384, 512, 768, 1024]
- overlap: [0, 25, 50, 75, 100]

For each valid config:
1. Run eval pipeline (~30-60s on CPU)
2. Record MRR, Hit@1, Hit@3
3. Save results to hyperparameter_results.json

WARNING: 20-25 configs × 45s = ~20 minutes total.
"""

import gc
import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(src_path, 'src'))

from chunkers import FixedChunker, ParagraphChunker, RecursiveChunker
from embeddings.embedding_model import EmbeddingModel
from embeddings.faiss_index import FaissIndex


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
DATA_PATH = "data/train_1000.json"
MIN_ANSWER_LENGTH = 4
K_MAX = 5
K_VALUES = [1, 3, 5]

OUTPUT_DIR = Path("results/hyperparameter_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid search space
CHUNK_SIZES = [256, 384, 512, 768, 1024]
OVERLAPS = [0, 25, 50, 75, 100]

# Which chunker to tune
# Options: "FixedChunker" or "RecursiveChunker"
# (ParagraphChunker doesn't have overlap param, so skip it)
CHUNKER_TYPE = "FixedChunker"


# ═══════════════════════════════════════════════════════════════════════════
# Eval Functions (from eval_metrics.py, CPU-optimized)
# ═══════════════════════════════════════════════════════════════════════════
def load_and_filter(path, min_answer_len):
    """Load and filter examples."""
    with open(path, "r") as f:
        data = json.load(f)
    all_examples = data["examples"]
    valid = []
    for ex in all_examples:
        answer = ex.get("answer", "")
        if not answer or not answer.strip():
            continue
        if len(answer.strip()) < min_answer_len:
            continue
        valid.append(ex)
    return valid


def chunk_all_contexts(examples, chunker):
    """Chunk all unique contexts with given chunker."""
    seen = {}
    unique_contexts = []
    for ex in examples:
        ctx = ex["context"]
        if ctx not in seen:
            seen[ctx] = True
            unique_contexts.append(ctx)
    
    context_chunks = {}
    for ctx in unique_contexts:
        doc_id = f"ctx_{hash(ctx) % (10**8)}"
        chunks = chunker.chunk(ctx, document_id=doc_id)
        if chunks:
            context_chunks[ctx] = chunks
    return context_chunks


def encode_and_index(examples, context_chunks, embed_model):
    """Encode questions and chunks, build FAISS indexes."""
    # Questions
    questions = [ex["question"] for ex in examples]
    q_matrix = embed_model.encode(questions)  # Uses batch_size=64 internally
    question_vectors = {i: q_matrix[i].copy() for i in range(len(questions))}
    
    # Release immediately
    del questions, q_matrix
    gc.collect()
    
    # Chunks
    flat_texts = []
    boundaries = []
    for ctx, chunks in context_chunks.items():
        start = len(flat_texts)
        flat_texts.extend(c.text for c in chunks)
        end = len(flat_texts)
        boundaries.append((ctx, start, end))
    
    if flat_texts:
        all_vectors = embed_model.encode(flat_texts)
        del flat_texts
    else:
        all_vectors = np.array([]).reshape(0, 384)
    
    # Build indexes
    indexed_cache = {}
    for ctx, start, end in boundaries:
        chunks = context_chunks[ctx]
        vectors = all_vectors[start:end].copy()
        chunk_ids = [c.chunk_id for c in chunks]
        
        index = FaissIndex(dimension=384)
        index.add(vectors, chunk_ids)
        chunk_map = {c.chunk_id: c for c in chunks}
        indexed_cache[ctx] = (chunks, index, chunk_map)
    
    del all_vectors
    gc.collect()
    
    return question_vectors, indexed_cache


def search_cached(indexed_cache, context, query_vector, k):
    """Search pre-built index."""
    entry = indexed_cache.get(context)
    if entry is None:
        return []
    chunks, index, chunk_map = entry
    raw_results = index.search(query_vector, k=k)
    return [
        {"rank": r["rank"], "score": r["score"], "text": chunk_map[r["chunk_id"]].text}
        for r in raw_results
    ]


def compute_hit_rank(results, answer):
    """Find rank of first result containing answer."""
    answer_lower = answer.lower()
    for r in results:
        if answer_lower in r["text"].lower():
            return r["rank"]
    return None


def derive_metrics(hit_rank, k_values):
    """Derive Hit@K and RR from hit_rank."""
    metrics = {}
    for k in k_values:
        metrics[f"hit@{k}"] = 1 if (hit_rank is not None and hit_rank < k) else 0
    metrics["rr"] = (1.0 / (hit_rank + 1)) if hit_rank is not None else 0.0
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Run Single Config
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_config(chunk_size, overlap, examples, embed_model):
    """Run eval for a single (chunk_size, overlap) config.
    
    Returns:
        dict with keys: mrr, hit@1, hit@3, hit@5, num_chunks
    """
    # Create chunker
    if CHUNKER_TYPE == "FixedChunker":
        chunker = FixedChunker(chunk_size=chunk_size, overlap=overlap)
    elif CHUNKER_TYPE == "RecursiveChunker":
        chunker = RecursiveChunker(max_chars=chunk_size, overlap=overlap)
    else:
        raise ValueError(f"Unknown chunker type: {CHUNKER_TYPE}")
    
    # Chunk
    context_chunks = chunk_all_contexts(examples, chunker)
    total_chunks = sum(len(chunks) for chunks in context_chunks.values())
    
    # Encode and index
    question_vectors, indexed_cache = encode_and_index(examples, context_chunks, embed_model)
    
    # Eval loop
    per_example_metrics = []
    for i, example in enumerate(examples):
        context = example["context"]
        answer = example["answer"]
        query_vector = question_vectors[i]
        
        results = search_cached(indexed_cache, context, query_vector, k=K_MAX)
        hit_rank = compute_hit_rank(results, answer)
        metrics = derive_metrics(hit_rank, K_VALUES)
        per_example_metrics.append(metrics)
    
    # Aggregate
    aggregated = {}
    for key in per_example_metrics[0]:
        aggregated[key] = sum(m[key] for m in per_example_metrics) / len(per_example_metrics)
    
    aggregated['num_chunks'] = total_chunks
    
    # Release memory before next config
    del context_chunks, question_vectors, indexed_cache, per_example_metrics
    gc.collect()
    
    return aggregated


# ═══════════════════════════════════════════════════════════════════════════
# Grid Search
# ═══════════════════════════════════════════════════════════════════════════
def run_grid_search():
    """Run grid search over all (chunk_size, overlap) combinations."""
    print("\n" + "="*70)
    print(f"  🔍  HYPERPARAMETER GRID SEARCH: {CHUNKER_TYPE}")
    print("="*70)
    print(f"\n  ⚙️  Running on CPU (MacBook optimized)")
    
    # Load data
    print(f"\n  📂 Loading data: {DATA_PATH}")
    examples = load_and_filter(DATA_PATH, MIN_ANSWER_LENGTH)
    print(f"     Examples: {len(examples)}")
    
    # Load model
    print(f"\n  🤖 Loading embedding model (this will take ~5-10s)...")
    embed_model = EmbeddingModel()
    _ = embed_model.model  # Trigger lazy load
    
    # Count valid configs
    all_configs = list(itertools.product(CHUNK_SIZES, OVERLAPS))
    valid_configs = [(cs, ol) for cs, ol in all_configs if ol < cs]
    
    print(f"\n  🔬 Testing {len(valid_configs)} configs (skipping {len(all_configs) - len(valid_configs)} invalid):")
    print(f"     chunk_sizes: {CHUNK_SIZES}")
    print(f"     overlaps: {OVERLAPS}")
    print(f"\n  ⏱️  Estimated time: {len(valid_configs) * 45 / 60:.1f} minutes (~45s per config)")
    print()
    
    results = []
    start_time = time.time()
    
    for i, (chunk_size, overlap) in enumerate(valid_configs):
        config_start = time.time()
        
        print(f"  [{i+1:>2}/{len(valid_configs)}] Testing chunk_size={chunk_size:>4} overlap={overlap:>3} ... ", 
              end='', flush=True)
        
        metrics = evaluate_config(chunk_size, overlap, examples, embed_model)
        config_elapsed = time.time() - config_start
        
        result = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            **metrics,
            "elapsed": round(config_elapsed, 1)
        }
        results.append(result)
        
        # Compute ETA
        elapsed_so_far = time.time() - start_time
        avg_time_per_config = elapsed_so_far / (i + 1)
        eta_seconds = avg_time_per_config * (len(valid_configs) - (i + 1))
        eta_minutes = eta_seconds / 60
        
        print(f"✓ MRR={metrics['rr']:.4f} Hit@1={metrics['hit@1']:.4f} "
              f"chunks={metrics['num_chunks']:>3} ({config_elapsed:.0f}s) "
              f"ETA: {eta_minutes:.1f}m")
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"  ✅ Grid search complete in {total_elapsed / 60:.1f} minutes")
    print("="*70)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Save Results
# ═══════════════════════════════════════════════════════════════════════════
def save_results(results):
    """Save grid search results to JSON."""
    output_path = OUTPUT_DIR / "hyperparameter_results.json"
    
    # Find best config by MRR
    best = max(results, key=lambda x: x['rr'])
    
    # Find best by Hit@1 (might differ from MRR)
    best_hit1 = max(results, key=lambda x: x['hit@1'])
    
    payload = {
        "chunker_type": CHUNKER_TYPE,
        "device": "CPU",
        "search_space": {
            "chunk_sizes": CHUNK_SIZES,
            "overlaps": OVERLAPS
        },
        "best_by_mrr": best,
        "best_by_hit1": best_hit1,
        "all_results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    
    print(f"\n  💾  Results saved to: {output_path}")
    
    print(f"\n  🏆  BEST CONFIG (by MRR):")
    print(f"      chunk_size: {best['chunk_size']}")
    print(f"      overlap: {best['overlap']}")
    print(f"      MRR: {best['rr']:.4f}")
    print(f"      Hit@1: {best['hit@1']:.4f}")
    print(f"      Hit@3: {best['hit@3']:.4f}")
    print(f"      Total chunks: {best['num_chunks']}")
    
    if best_hit1 != best:
        print(f"\n  📌  BEST CONFIG (by Hit@1):")
        print(f"      chunk_size: {best_hit1['chunk_size']}")
        print(f"      overlap: {best_hit1['overlap']}")
        print(f"      Hit@1: {best_hit1['hit@1']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def plot_heatmap(results):
    """Heatmap: MRR vs (chunk_size, overlap)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Pivot data
    chunk_sizes_unique = sorted(set(r['chunk_size'] for r in results))
    overlaps_unique = sorted(set(r['overlap'] for r in results))
    
    mrr_matrix = np.full((len(overlaps_unique), len(chunk_sizes_unique)), np.nan)
    
    for r in results:
        i = overlaps_unique.index(r['overlap'])
        j = chunk_sizes_unique.index(r['chunk_size'])
        mrr_matrix[i, j] = r['rr']
    
    # Plot
    im = ax.imshow(mrr_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=np.nanmin(mrr_matrix) - 0.01, 
                   vmax=np.nanmax(mrr_matrix) + 0.01)
    
    # Annotate cells
    for i in range(len(overlaps_unique)):
        for j in range(len(chunk_sizes_unique)):
            if not np.isnan(mrr_matrix[i, j]):
                text = ax.text(j, i, f'{mrr_matrix[i, j]:.3f}',
                              ha="center", va="center", 
                              color="black" if mrr_matrix[i, j] < np.nanmax(mrr_matrix) - 0.005 else "white",
                              fontsize=10, fontweight='bold')
    
    # Labels
    ax.set_xticks(range(len(chunk_sizes_unique)))
    ax.set_yticks(range(len(overlaps_unique)))
    ax.set_xticklabels(chunk_sizes_unique)
    ax.set_yticklabels(overlaps_unique)
    ax.set_xlabel('Chunk Size (characters)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Overlap (characters)', fontweight='bold', fontsize=12)
    ax.set_title(f'MRR Heatmap: {CHUNKER_TYPE} Hyperparameter Search\n(CPU, SQuAD train_1000)', 
                fontweight='bold', pad=20, fontsize=13)
    
    # Color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MRR', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heatmap_mrr.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: heatmap_mrr.png")


def plot_line_curves(results):
    """Line plot: MRR vs chunk_size (one line per overlap)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    overlaps_unique = sorted(set(r['overlap'] for r in results))
    
    for overlap in overlaps_unique:
        subset = [r for r in results if r['overlap'] == overlap]
        subset_sorted = sorted(subset, key=lambda x: x['chunk_size'])
        
        chunk_sizes = [r['chunk_size'] for r in subset_sorted]
        mrrs = [r['rr'] for r in subset_sorted]
        
        ax.plot(chunk_sizes, mrrs, marker='o', linewidth=2.5, markersize=9, 
               label=f'overlap={overlap}', alpha=0.85)
    
    # Styling
    ax.set_xlabel('Chunk Size (characters)', fontweight='bold', fontsize=12)
    ax.set_ylabel('MRR', fontweight='bold', fontsize=12)
    ax.set_title(f'MRR vs Chunk Size: {CHUNKER_TYPE}\n(CPU, SQuAD train_1000)', 
                fontweight='bold', pad=20, fontsize=13)
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'line_mrr_vs_chunksize.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: line_mrr_vs_chunksize.png")


def plot_top_configs(results):
    """Bar chart: top 5 configs by MRR."""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    sorted_results = sorted(results, key=lambda x: x['rr'], reverse=True)[:5]
    
    labels = [f"{r['chunk_size']}/{r['overlap']}" for r in sorted_results]
    mrrs = [r['rr'] for r in sorted_results]
    
    colors = plt.cm.RdYlGn(np.linspace(0.6, 0.9, 5))
    bars = ax.barh(labels, mrrs, color=colors, alpha=0.85)
    
    # Annotate
    for bar, value, r in zip(bars, mrrs, sorted_results):
        ax.text(value + 0.0005, bar.get_y() + bar.get_height()/2, 
               f'{value:.4f} ({r["num_chunks"]} chunks)', 
               va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('MRR', fontweight='bold', fontsize=12)
    ax.set_ylabel('Config (chunk_size/overlap)', fontweight='bold', fontsize=12)
    ax.set_title(f'Top 5 Configs: {CHUNKER_TYPE}\n(CPU, SQuAD train_1000)', 
                fontweight='bold', pad=20, fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top5_configs.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: top5_configs.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # Run grid search
    results = run_grid_search()
    
    # Save results
    save_results(results)
    
    # Generate plots
    print(f"\n  📊  Generating visualizations...")
    plot_heatmap(results)
    plot_line_curves(results)
    plot_top_configs(results)
    
    print("\n" + "="*70)
    print("  ✅  HYPERPARAMETER SEARCH COMPLETE")
    print("="*70)
    print(f"\n  📁  All results in: {OUTPUT_DIR.absolute()}")
    print(f"\n  💡  Next steps:")
    print(f"     1. Check heatmap to see optimal region")
    print(f"     2. Compare to your baseline (512, 50)")
    print(f"     3. If better config found, update eval_metrics.py and re-run")
    print()


if __name__ == "__main__":
    main()