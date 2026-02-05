"""
hyperparameter_search.py
────────────────────────────
Efficient grid search over chunk_size and overlap parameters.

Key optimizations:
1. Compute embeddings ONCE (configuration-invariant)
2. Only re-chunk and re-index per configuration
3. Reuse question embeddings across all configs
4. Minimize memory allocations

Grid search space:
- chunk_size: [256, 384, 512, 768, 1024]
- overlap: [0, 25, 50, 75, 100]

For each valid config:
1. Chunk documents (fast)
2. Encode chunks ONLY (reuse question embeddings)
3. Build FAISS index (fast)
4. Search and compute metrics
5. Record MRR, Hit@1, Hit@3, Hit@5
"""

import gc
import json
import sys
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import itertools

import numpy as np
import matplotlib.pyplot as plt

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(src_path, 'src'))

from chunkers import (
    FixedChunker, 
    RecursiveChunker, 
    SentenceChunker, 
    ParagraphChunker,
    StructureAwareChunker, 
    SemanticDensityChunker
)
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

# Available chunker types
AVAILABLE_CHUNKERS = {
    "FixedChunker": FixedChunker,
    "RecursiveChunker": RecursiveChunker,
    "SentenceChunker": SentenceChunker,
    "ParagraphChunker": ParagraphChunker,
    "StructureAwareChunker": StructureAwareChunker,
    "SemanticDensityChunker": SemanticDensityChunker,
}

# Global variables (will be set by command line args)
CHUNKER_TYPES = None  # Will be set to list of chunker names to test


# ═══════════════════════════════════════════════════════════════════════════
# Command Line Arguments
# ═══════════════════════════════════════════════════════════════════════════
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter grid search for text chunkers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all chunkers (default)
  python hyperparameter_search.py

  # Test specific chunker
  python hyperparameter_search.py --chunker FixedChunker

  # Test multiple chunkers
  python hyperparameter_search.py --chunker FixedChunker StructureAwareChunker

Available chunkers: FixedChunker, RecursiveChunker, SentenceChunker, 
                   ParagraphChunker, StructureAwareChunker, SemanticDensityChunker
        """
    )
    
    parser.add_argument(
        '--chunker', '-c',
        nargs='*',
        choices=list(AVAILABLE_CHUNKERS.keys()),
        help='Chunker type(s) to test. If not specified, all chunkers are tested.'
    )
    
    parser.add_argument(
        '--data', '-d',
        default=DATA_PATH,
        help=f'Path to dataset (default: {DATA_PATH})'
    )
    
    return parser.parse_args()


def create_chunker(chunker_type: str, chunk_size: int, overlap: int):
    """Create a chunker instance based on type and parameters."""
    chunker_class = AVAILABLE_CHUNKERS[chunker_type]
    
    if chunker_type == "FixedChunker":
        return chunker_class(chunk_size=chunk_size, overlap=overlap)
    
    elif chunker_type == "RecursiveChunker":
        return chunker_class(max_chars=chunk_size, overlap=overlap)
    
    elif chunker_type == "SentenceChunker":
        return chunker_class(max_chars=chunk_size, overlap=overlap)
    
    elif chunker_type == "ParagraphChunker":
        return chunker_class(max_chars=chunk_size, overlap=overlap)
    
    elif chunker_type == "StructureAwareChunker":
        return chunker_class(chunk_size=chunk_size, overlap=overlap)
    
    elif chunker_type == "SemanticDensityChunker":
        # SemanticDensityChunker uses different parameter names
        return chunker_class(
            chunk_size=chunk_size, 
            min_overlap=max(overlap - 25, 0),  # Convert overlap to min_overlap
            max_overlap=overlap + 25  # Set max_overlap slightly higher
        )
    
    else:
        raise ValueError(f"Unknown chunker type: {chunker_type}")


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════
def load_and_filter(path: str, min_answer_len: int) -> List[Dict]:
    """Load and filter examples."""
    with open(path, "r") as f:
        data = json.load(f)
    
    valid = []
    for ex in data["examples"]:
        answer = ex.get("answer", "")
        if not answer or not answer.strip():
            continue
        if len(answer.strip()) < min_answer_len:
            continue
        valid.append(ex)
    
    return valid


# ═══════════════════════════════════════════════════════════════════════════
# ONE-TIME: Embed Questions (Configuration-Invariant)
# ═══════════════════════════════════════════════════════════════════════════
def embed_questions_once(examples: List[Dict], embed_model: EmbeddingModel) -> np.ndarray:
    """Compute question embeddings ONCE for all configurations.
    
    Returns:
        question_vectors: (num_examples, embedding_dim) array
    """
    questions = [ex["question"] for ex in examples]
    question_vectors = embed_model.encode(questions)
    return question_vectors


# ═══════════════════════════════════════════════════════════════════════════
# PER-CONFIG: Chunk, Embed Chunks, Index
# ═══════════════════════════════════════════════════════════════════════════
def chunk_all_contexts(examples: List[Dict], chunker) -> Dict[str, List]:
    """Chunk all unique contexts."""
    unique_contexts = list(set(ex["context"] for ex in examples))
    
    context_chunks = {}
    for ctx in unique_contexts:
        doc_id = f"ctx_{hash(ctx) % (10**8)}"
        chunks = chunker.chunk(ctx, document_id=doc_id)
        if chunks:
            context_chunks[ctx] = chunks
    
    return context_chunks


def encode_chunks_and_index(context_chunks: Dict, embed_model: EmbeddingModel) -> Dict:
    """Encode chunks and build FAISS indexes for each context.
    
    Returns:
        indexed_cache: {context: (chunks, index, chunk_map)}
    """
    # Collect all chunk texts with boundaries
    flat_texts = []
    boundaries = []
    
    for ctx, chunks in context_chunks.items():
        start = len(flat_texts)
        flat_texts.extend(c.text for c in chunks)
        end = len(flat_texts)
        boundaries.append((ctx, start, end))
    
    # Encode all chunks in one batch
    if flat_texts:
        all_vectors = embed_model.encode(flat_texts)
    else:
        all_vectors = np.array([]).reshape(0, 384)
    
    # Build per-context indexes
    indexed_cache = {}
    for ctx, start, end in boundaries:
        chunks = context_chunks[ctx]
        vectors = all_vectors[start:end]
        chunk_ids = [c.chunk_id for c in chunks]
        
        # Build FAISS index
        index = FaissIndex(dimension=384)
        index.add(vectors, chunk_ids)
        
        chunk_map = {c.chunk_id: c for c in chunks}
        indexed_cache[ctx] = (chunks, index, chunk_map)
    
    return indexed_cache


# ═══════════════════════════════════════════════════════════════════════════
# Search and Metrics
# ═══════════════════════════════════════════════════════════════════════════
def search_and_evaluate(examples: List[Dict], 
                       question_vectors: np.ndarray,
                       indexed_cache: Dict,
                       k_max: int,
                       k_values: List[int]) -> Dict:
    """Search and compute metrics for all examples.
    
    Returns:
        Aggregated metrics dict
    """
    per_example_metrics = []
    
    for i, example in enumerate(examples):
        context = example["context"]
        answer = example["answer"]
        query_vector = question_vectors[i]
        
        # Search
        results = search_cached(indexed_cache, context, query_vector, k_max)
        
        # Compute hit rank
        hit_rank = compute_hit_rank(results, answer)
        
        # Derive metrics
        metrics = derive_metrics(hit_rank, k_values)
        per_example_metrics.append(metrics)
    
    # Aggregate
    aggregated = {}
    for key in per_example_metrics[0]:
        aggregated[key] = sum(m[key] for m in per_example_metrics) / len(per_example_metrics)
    
    return aggregated


def search_cached(indexed_cache: Dict, context: str, query_vector: np.ndarray, k: int) -> List[Dict]:
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


def compute_hit_rank(results: List[Dict], answer: str) -> int:
    """Find rank of first result containing answer."""
    answer_lower = answer.lower()
    for r in results:
        if answer_lower in r["text"].lower():
            return r["rank"]
    return None


def derive_metrics(hit_rank: int, k_values: List[int]) -> Dict:
    """Derive Hit@K and RR from hit_rank."""
    metrics = {}
    for k in k_values:
        metrics[f"hit@{k}"] = 1 if (hit_rank is not None and hit_rank < k) else 0
    metrics["rr"] = (1.0 / (hit_rank + 1)) if hit_rank is not None else 0.0
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Evaluate Single Config
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_config(chunker_type: str,
                   chunk_size: int, 
                   overlap: int, 
                   examples: List[Dict],
                   question_vectors: np.ndarray,
                   embed_model: EmbeddingModel) -> Dict:
    """Evaluate a single (chunker_type, chunk_size, overlap) configuration.
    
    Args:
        chunker_type: Type of chunker to use
        chunk_size: Chunk size in characters
        overlap: Overlap in characters
        examples: Dataset examples
        question_vectors: Pre-computed question embeddings (REUSED!)
        embed_model: Embedding model (only for chunks)
    
    Returns:
        Metrics dict with mrr, hit@k, num_chunks
    """
    # Create chunker
    chunker = create_chunker(chunker_type, chunk_size, overlap)
    
    # 1. Chunk contexts (fast)
    context_chunks = chunk_all_contexts(examples, chunker)
    total_chunks = sum(len(chunks) for chunks in context_chunks.values())
    
    # 2. Encode chunks and build indexes (only chunks, not questions!)
    indexed_cache = encode_chunks_and_index(context_chunks, embed_model)
    
    # 3. Search and evaluate (using pre-computed question embeddings)
    metrics = search_and_evaluate(examples, question_vectors, indexed_cache, K_MAX, K_VALUES)
    metrics['num_chunks'] = total_chunks
    metrics['chunker_type'] = chunker_type
    
    # Clean up
    del context_chunks, indexed_cache
    gc.collect()
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Grid Search
# ═══════════════════════════════════════════════════════════════════════════
def run_grid_search():
    """Run grid search over all (chunker_type, chunk_size, overlap) combinations."""
    print("\n" + "="*70)
    print(f"  🔍  HYPERPARAMETER GRID SEARCH")
    print(f"  📋  Testing chunkers: {', '.join(CHUNKER_TYPES)}")
    print("="*70)
    
    # Load data
    print(f"\n  📂 Loading data: {DATA_PATH}")
    examples = load_and_filter(DATA_PATH, MIN_ANSWER_LENGTH)
    print(f"     Examples: {len(examples)}")
    
    # Load embedding model
    print(f"\n  🤖 Loading embedding model...")
    embed_model = EmbeddingModel()
    _ = embed_model.model  # Trigger lazy load
    print(f"     ✓ Model loaded")
    
    # ═══════════════════════════════════════════════════════════════════════
    # CRITICAL OPTIMIZATION: Embed questions ONCE
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n  ⚡ Computing question embeddings (ONCE for all configs)...")
    question_start = time.time()
    question_vectors = embed_questions_once(examples, embed_model)
    question_time = time.time() - question_start
    print(f"     ✓ Encoded {len(examples)} questions in {question_time:.1f}s")
    
    # Count valid configs
    all_configs = list(itertools.product(CHUNKER_TYPES, CHUNK_SIZES, OVERLAPS))
    valid_configs = [(ct, cs, ol) for ct, cs, ol in all_configs if ol < cs]
    
    total_configs = len(valid_configs)
    print(f"     💡 Will be REUSED across all {total_configs} configs")
    
    print(f"\n  🔬 Testing {total_configs} configurations:")
    print(f"     chunkers: {CHUNKER_TYPES}")
    print(f"     chunk_sizes: {CHUNK_SIZES}")
    print(f"     overlaps: {OVERLAPS}")
    print()
    
    results = []
    start_time = time.time()
    
    for i, (chunker_type, chunk_size, overlap) in enumerate(valid_configs):
        config_start = time.time()
        
        print(f"  [{i+1:>2}/{total_configs}] {chunker_type:<22} size={chunk_size:>4} overlap={overlap:>3} ... ", 
              end='', flush=True)
        
        try:
            # Evaluate config (question embeddings are REUSED!)
            metrics = evaluate_config(chunker_type, chunk_size, overlap, examples, question_vectors, embed_model)
            config_elapsed = time.time() - config_start
            
            result = {
                "chunker_type": chunker_type,
                "chunk_size": chunk_size,
                "overlap": overlap,
                **metrics,
                "elapsed": round(config_elapsed, 1)
            }
            results.append(result)
            
            # Compute ETA
            elapsed_so_far = time.time() - start_time
            avg_time_per_config = elapsed_so_far / (i + 1)
            eta_seconds = avg_time_per_config * (total_configs - (i + 1))
            eta_minutes = eta_seconds / 60
            
            print(f"✓ MRR={metrics['rr']:.4f} Hit@1={metrics['hit@1']:.4f} "
                  f"chunks={metrics['num_chunks']:>4} ({config_elapsed:.0f}s) "
                  f"ETA: {eta_minutes:.1f}m")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            continue
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"  ✅ Grid search complete in {total_elapsed / 60:.1f} minutes")
    print("="*70)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Save Results
# ═══════════════════════════════════════════════════════════════════════════
def save_results(results: List[Dict]):
    """Save grid search results to JSON."""
    output_path = OUTPUT_DIR / "hyperparameter_results.json"
    
    # Find best configs overall and per chunker
    best_overall_mrr = max(results, key=lambda x: x['rr'])
    best_overall_hit1 = max(results, key=lambda x: x['hit@1'])
    
    # Group by chunker type
    chunker_results = {}
    for r in results:
        chunker_type = r['chunker_type']
        if chunker_type not in chunker_results:
            chunker_results[chunker_type] = []
        chunker_results[chunker_type].append(r)
    
    # Find best per chunker
    best_per_chunker = {}
    for chunker_type, chunker_res in chunker_results.items():
        best_per_chunker[chunker_type] = {
            'best_by_mrr': max(chunker_res, key=lambda x: x['rr']),
            'best_by_hit1': max(chunker_res, key=lambda x: x['hit@1'])
        }
    
    payload = {
        "tested_chunkers": CHUNKER_TYPES,
        "search_space": {
            "chunk_sizes": CHUNK_SIZES,
            "overlaps": OVERLAPS
        },
        "best_overall_by_mrr": best_overall_mrr,
        "best_overall_by_hit1": best_overall_hit1,
        "best_per_chunker": best_per_chunker,
        "all_results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    
    print(f"\n  💾 Results saved to: {output_path}")
    
    print(f"\n  🏆 BEST CONFIG OVERALL (by MRR):")
    print(f"      chunker: {best_overall_mrr['chunker_type']}")
    print(f"      chunk_size: {best_overall_mrr['chunk_size']}")
    print(f"      overlap: {best_overall_mrr['overlap']}")
    print(f"      MRR: {best_overall_mrr['rr']:.4f}")
    print(f"      Hit@1: {best_overall_mrr['hit@1']:.4f}")
    print(f"      Hit@3: {best_overall_mrr['hit@3']:.4f}")
    print(f"      Total chunks: {best_overall_mrr['num_chunks']}")
    
    print(f"\n  📊 BEST CONFIG PER CHUNKER (by MRR):")
    for chunker_type in CHUNKER_TYPES:
        if chunker_type in best_per_chunker:
            best = best_per_chunker[chunker_type]['best_by_mrr']
            print(f"      {chunker_type:<22}: size={best['chunk_size']:>4}, overlap={best['overlap']:>3}, "
                  f"MRR={best['rr']:.4f}, Hit@1={best['hit@1']:.4f}")
    
    if best_overall_hit1['chunker_type'] != best_overall_mrr['chunker_type'] or \
       best_overall_hit1['chunk_size'] != best_overall_mrr['chunk_size'] or \
       best_overall_hit1['overlap'] != best_overall_mrr['overlap']:
        print(f"\n  📌 BEST CONFIG OVERALL (by Hit@1):")
        print(f"      chunker: {best_overall_hit1['chunker_type']}")
        print(f"      chunk_size: {best_overall_hit1['chunk_size']}")
        print(f"      overlap: {best_overall_hit1['overlap']}")
        print(f"      Hit@1: {best_overall_hit1['hit@1']:.4f}")
        print(f"      MRR: {best_overall_hit1['rr']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def plot_heatmap(results: List[Dict]):
    """Generate MRR heatmap for each chunker type."""
    chunker_types = sorted(set(r['chunker_type'] for r in results))
    
    for chunker_type in chunker_types:
        chunker_results = [r for r in results if r['chunker_type'] == chunker_type]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        chunk_sizes_unique = sorted(set(r['chunk_size'] for r in chunker_results))
        overlaps_unique = sorted(set(r['overlap'] for r in chunker_results))
        
        mrr_matrix = np.full((len(overlaps_unique), len(chunk_sizes_unique)), np.nan)
        
        for r in chunker_results:
            i = overlaps_unique.index(r['overlap'])
            j = chunk_sizes_unique.index(r['chunk_size'])
            mrr_matrix[i, j] = r['rr']
        
        im = ax.imshow(mrr_matrix, cmap='RdYlGn', aspect='auto', 
                       vmin=np.nanmin(mrr_matrix) - 0.01, 
                       vmax=np.nanmax(mrr_matrix) + 0.01)
        
        # Annotate cells
        for i in range(len(overlaps_unique)):
            for j in range(len(chunk_sizes_unique)):
                if not np.isnan(mrr_matrix[i, j]):
                    color = "white" if mrr_matrix[i, j] > np.nanmax(mrr_matrix) - 0.01 else "black"
                    ax.text(j, i, f'{mrr_matrix[i, j]:.3f}',
                           ha="center", va="center", color=color,
                           fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(chunk_sizes_unique)))
        ax.set_yticks(range(len(overlaps_unique)))
        ax.set_xticklabels(chunk_sizes_unique)
        ax.set_yticklabels(overlaps_unique)
        ax.set_xlabel('Chunk Size (characters)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Overlap (characters)', fontweight='bold', fontsize=12)
        ax.set_title(f'MRR Heatmap: {chunker_type}', 
                    fontweight='bold', pad=20, fontsize=13)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('MRR', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'heatmap_mrr_{chunker_type}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    print(f"  ✓ Saved: heatmap_mrr_{{chunker_type}}.png for each chunker")


def plot_line_curves(results: List[Dict]):
    """Generate line plot: MRR vs chunk_size for each chunker type."""
    chunker_types = sorted(set(r['chunker_type'] for r in results))
    
    for chunker_type in chunker_types:
        chunker_results = [r for r in results if r['chunker_type'] == chunker_type]
        
        fig, ax = plt.subplots(figsize=(11, 6))
        
        overlaps_unique = sorted(set(r['overlap'] for r in chunker_results))
        
        for overlap in overlaps_unique:
            subset = sorted([r for r in chunker_results if r['overlap'] == overlap], 
                           key=lambda x: x['chunk_size'])
            
            chunk_sizes = [r['chunk_size'] for r in subset]
            mrrs = [r['rr'] for r in subset]
            
            ax.plot(chunk_sizes, mrrs, marker='o', linewidth=2.5, markersize=9, 
                   label=f'overlap={overlap}', alpha=0.85)
        
        ax.set_xlabel('Chunk Size (characters)', fontweight='bold', fontsize=12)
        ax.set_ylabel('MRR', fontweight='bold', fontsize=12)
        ax.set_title(f'MRR vs Chunk Size: {chunker_type}', 
                    fontweight='bold', pad=20, fontsize=13)
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'line_mrr_vs_chunksize_{chunker_type}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    print(f"  ✓ Saved: line_mrr_vs_chunksize_{{chunker_type}}.png for each chunker")


def plot_top_configs(results: List[Dict]):
    """Generate bar chart: top 5 configs overall and chunker comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 5 configs overall
    sorted_results = sorted(results, key=lambda x: x['rr'], reverse=True)[:5]
    
    labels = [f"{r['chunker_type'][:8]}_{r['chunk_size']}/{r['overlap']}" for r in sorted_results]
    mrrs = [r['rr'] for r in sorted_results]
    
    colors = plt.cm.RdYlGn(np.linspace(0.6, 0.9, 5))
    bars = ax1.barh(labels, mrrs, color=colors, alpha=0.85)
    
    for bar, value, r in zip(bars, mrrs, sorted_results):
        ax1.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', 
                va='center', fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('MRR', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Config (chunker_size/overlap)', fontweight='bold', fontsize=12)
    ax1.set_title('Top 5 Configs Overall', fontweight='bold', pad=20, fontsize=13)
    ax1.grid(axis='x', alpha=0.3)
    
    # Best config per chunker
    chunker_types = sorted(set(r['chunker_type'] for r in results))
    chunker_best = []
    for chunker_type in chunker_types:
        chunker_results = [r for r in results if r['chunker_type'] == chunker_type]
        best = max(chunker_results, key=lambda x: x['rr'])
        chunker_best.append(best)
    
    chunker_names = [r['chunker_type'] for r in chunker_best]
    chunker_mrrs = [r['rr'] for r in chunker_best]
    
    colors = plt.cm.Set3(range(len(chunker_names)))
    bars = ax2.bar(range(len(chunker_names)), chunker_mrrs, color=colors, alpha=0.85)
    
    for i, (bar, value, r) in enumerate(zip(bars, chunker_mrrs, chunker_best)):
        ax2.text(i, value + 0.001, 
                f'{value:.4f}\n{r["chunk_size"]}/{r["overlap"]}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_ylabel('Best MRR', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Chunker Type', fontweight='bold', fontsize=12)
    ax2.set_title('Best Config per Chunker', fontweight='bold', pad=20, fontsize=13)
    ax2.set_xticks(range(len(chunker_names)))
    ax2.set_xticklabels([name[:12] for name in chunker_names], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top5_configs_and_chunker_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: top5_configs_and_chunker_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """Run hyperparameter grid search."""
    global CHUNKER_TYPES, DATA_PATH
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set chunker types to test
    if args.chunker:
        CHUNKER_TYPES = args.chunker
    else:
        CHUNKER_TYPES = list(AVAILABLE_CHUNKERS.keys())  # Test all chunkers by default
    
    # Set data path
    DATA_PATH = args.data
    
    # Print configuration
    print("\n" + "="*70)
    print("  🚀  HYPERPARAMETER GRID SEARCH CONFIGURATION")
    print("="*70)
    print(f"  📋  Chunkers to test: {', '.join(CHUNKER_TYPES)}")
    print(f"  📂  Dataset: {DATA_PATH}")
    print(f"  🔢  Chunk sizes: {CHUNK_SIZES}")
    print(f"  ↔️   Overlaps: {OVERLAPS}")
    print("="*70)
    
    # Run grid search
    results = run_grid_search()
    
    if not results:
        print("\n  ❌ No valid results obtained!")
        return
    
    # Save results
    save_results(results)
    
    # Generate visualizations
    print(f"\n  📊 Generating visualizations...")
    plot_heatmap(results)
    plot_line_curves(results)
    plot_top_configs(results)
    
    print("\n" + "="*70)
    print("  ✅ HYPERPARAMETER SEARCH COMPLETE")
    print("="*70)
    print(f"\n  📁 All results in: {OUTPUT_DIR.absolute()}")
    print(f"\n  💡 Next steps:")
    print(f"     1. Review heatmaps to identify optimal regions per chunker")
    print(f"     2. Compare top configs across different chunker types")
    print(f"     3. Update your eval pipeline with the best overall config")
    print(f"     4. Consider chunker-specific strengths for your use case")
    print()


if __name__ == "__main__":
    main()