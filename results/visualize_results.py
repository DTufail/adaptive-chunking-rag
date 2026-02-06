"""
visualize_results.py
────────────────────
Comprehensive visualization suite for chunking evaluation results.

Generates 6 publication-quality plots:
1. Hit Rate Comparison (bar chart)
2. MRR Comparison (horizontal bar chart with values)
3. Per-Example Heatmap (which examples did each chunker miss?)
4. Rank Distribution (histogram of where answers were found)
5. Pairwise Agreement Matrix (do chunkers agree on which examples are hard?)
6. Cumulative Hit Rate Curve (Hit@K vs K)

Run this after eval_metrics.py to visualize the results.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Paths (will be set in main() based on dataset)
RESULTS_PATH = None
OUTPUT_DIR = None


def load_results():
    """Load the results JSON from eval_metrics.py."""
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: Hit Rate Comparison (Grouped Bar Chart)
# ═══════════════════════════════════════════════════════════════════════════
def plot_hit_rate_comparison(results):
    """Bar chart comparing Hit@1, Hit@3, Hit@5 across chunkers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    chunkers = list(results['aggregated'].keys())
    k_values = results['config']['k_values']
    
    # Data
    hit_data = {
        f'Hit@{k}': [results['aggregated'][c][f'hit@{k}'] for c in chunkers]
        for k in k_values
    }
    
    # Bar positions
    x = np.arange(len(chunkers))
    width = 0.25
    
    # Plot bars
    colors = ['#3498db', '#2ecc71', '#f39c12']
    for i, (label, values) in enumerate(hit_data.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=label, color=colors[i], alpha=0.85)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    # Styling
    ax.set_xlabel('Chunker', fontweight='bold')
    ax.set_ylabel('Hit Rate', fontweight='bold')
    ax.set_title('Hit Rate Comparison Across Chunkers', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(chunkers, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_hit_rate_comparison.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 1_hit_rate_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: MRR Comparison (Horizontal Bar with Values)
# ═══════════════════════════════════════════════════════════════════════════
def plot_mrr_comparison(results):
    """Horizontal bar chart showing MRR with exact values."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    chunkers = list(results['aggregated'].keys())
    mrr_values = [results['aggregated'][c]['rr'] for c in chunkers]
    
    # Sort by MRR descending
    sorted_pairs = sorted(zip(chunkers, mrr_values), key=lambda x: x[1], reverse=True)
    chunkers_sorted, mrr_sorted = zip(*sorted_pairs)
    
    # Color gradient from best to worst
    colors = plt.cm.RdYlGn(np.linspace(0.5, 0.9, len(chunkers)))
    
    bars = ax.barh(chunkers_sorted, mrr_sorted, color=colors, alpha=0.85)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, mrr_sorted)):
        ax.text(value + 0.002, i, f'{value:.4f}', 
               va='center', fontweight='bold', fontsize=11)
    
    # Styling
    ax.set_xlabel('Mean Reciprocal Rank (MRR)', fontweight='bold')
    ax.set_title('MRR Comparison: Which Chunker Ranks Answers Highest?', 
                fontweight='bold', pad=20)
    ax.set_xlim(0.88, 0.96)
    ax.axvline(x=0.95, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellence threshold (0.95)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_mrr_comparison.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 2_mrr_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: Per-Example Heatmap (Which Examples Are Hard?)
# ═══════════════════════════════════════════════════════════════════════════
def plot_per_example_heatmap(results):
    """Heatmap showing hit_rank for each example × chunker.
    
    Reveals: which examples are universally hard, which are chunker-specific failures.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    chunkers = list(results['per_example_hit_ranks'].keys())
    hit_ranks = np.array([results['per_example_hit_ranks'][c] for c in chunkers], dtype=object)
    
    # Convert None to 6 (worse than max rank 5) for visualization
    hit_ranks_viz = np.zeros_like(hit_ranks, dtype=float)
    for i in range(hit_ranks.shape[0]):
        for j in range(hit_ranks.shape[1]):
            hit_ranks_viz[i, j] = 6 if hit_ranks[i, j] is None else hit_ranks[i, j]
    
    # Only show first 200 examples (952 is too many for readability)
    hit_ranks_viz = hit_ranks_viz[:, :200]
    
    # Plot heatmap
    im = ax.imshow(hit_ranks_viz, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=6)
    
    # Color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Hit Rank (6 = Not Found)', rotation=270, labelpad=20, fontweight='bold')
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(['0 (Perfect)', '1', '2', '3', '4', '5', 'Miss'])
    
    # Labels
    ax.set_xlabel('Example Index (first 200 of 952)', fontweight='bold')
    ax.set_ylabel('Chunker', fontweight='bold')
    ax.set_title('Per-Example Performance Heatmap\n(Green = Answer at rank 0, Red = Missed)', 
                fontweight='bold', pad=20)
    ax.set_yticks(range(len(chunkers)))
    ax.set_yticklabels(chunkers)
    
    # Grid lines
    ax.set_xticks(np.arange(0, 200, 20))
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_per_example_heatmap.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 3_per_example_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4: Rank Distribution (Where Are Answers Found?)
# ═══════════════════════════════════════════════════════════════════════════
def plot_rank_distribution(results):
    """Stacked bar chart showing distribution of hit ranks."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    chunkers = list(results['per_example_hit_ranks'].keys())
    
    # Count occurrences of each rank (0-5, None)
    rank_counts = {}
    for chunker in chunkers:
        ranks = results['per_example_hit_ranks'][chunker]
        counts = {r: 0 for r in range(6)}
        counts['miss'] = 0
        
        for rank in ranks:
            if rank is None:
                counts['miss'] += 1
            else:
                counts[rank] += 1
        
        rank_counts[chunker] = counts
    
    # Prepare data for stacked bar
    x = np.arange(len(chunkers))
    width = 0.6
    
    # Colors for each rank
    colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b', '#95a5a6']
    rank_labels = ['Rank 0', 'Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5', 'Miss']
    
    # Stack bars
    bottom = np.zeros(len(chunkers))
    for i, rank_key in enumerate(list(range(6)) + ['miss']):
        values = [rank_counts[c][rank_key] for c in chunkers]
        ax.bar(x, values, width, bottom=bottom, label=rank_labels[i], 
              color=colors[i], alpha=0.85)
        bottom += values
    
    # Styling
    ax.set_xlabel('Chunker', fontweight='bold')
    ax.set_ylabel('Number of Examples', fontweight='bold')
    ax.set_title('Distribution of Answer Ranks Across Chunkers\n(Lower ranks = better)', 
                fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(chunkers, rotation=15, ha='right')
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_rank_distribution.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 4_rank_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5: Pairwise Agreement Matrix (Do Chunkers Agree on Hard Examples?)
# ═══════════════════════════════════════════════════════════════════════════
def plot_pairwise_agreement(results):
    """Heatmap showing agreement between chunkers on per-example success/failure."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    chunkers = list(results['per_example_hit_ranks'].keys())
    n = len(chunkers)
    
    # Agreement matrix: agreement[i][j] = fraction of examples where chunkers i and j
    # both succeeded (rank < 3) or both failed (rank >= 3 or None)
    agreement = np.zeros((n, n))
    
    for i, c1 in enumerate(chunkers):
        for j, c2 in enumerate(chunkers):
            ranks1 = results['per_example_hit_ranks'][c1]
            ranks2 = results['per_example_hit_ranks'][c2]
            
            agree = 0
            for r1, r2 in zip(ranks1, ranks2):
                # Both succeeded at top-3
                if (r1 is not None and r1 < 3) and (r2 is not None and r2 < 3):
                    agree += 1
                # Both failed at top-3
                elif (r1 is None or r1 >= 3) and (r2 is None or r2 >= 3):
                    agree += 1
            
            agreement[i][j] = agree / len(ranks1)
    
    # Plot heatmap
    im = ax.imshow(agreement, cmap='YlGnBu', vmin=0.95, vmax=1.0)
    
    # Annotate cells
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{agreement[i, j]:.3f}',
                          ha="center", va="center", color="black" if agreement[i,j] > 0.975 else "white",
                          fontsize=11, fontweight='bold')
    
    # Color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Agreement Rate', rotation=270, labelpad=20, fontweight='bold')
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(chunkers, rotation=45, ha='right')
    ax.set_yticklabels(chunkers)
    ax.set_title('Pairwise Chunker Agreement Matrix\n(Agreement = both succeed or both fail at top-3)', 
                fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_pairwise_agreement.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 5_pairwise_agreement.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 6: Cumulative Hit Rate Curve (Hit@K vs K)
# ═══════════════════════════════════════════════════════════════════════════
def plot_cumulative_hit_curve(results):
    """Line plot showing Hit@K as K increases from 0 to 5."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    chunkers = list(results['per_example_hit_ranks'].keys())
    
    # Compute Hit@K for K = 0, 1, 2, 3, 4, 5
    k_range = list(range(6))
    
    for chunker in chunkers:
        ranks = results['per_example_hit_ranks'][chunker]
        hit_at_k = []
        
        for k in k_range:
            hits = sum(1 for r in ranks if r is not None and r < k) if k > 0 else 0
            hit_at_k.append(hits / len(ranks))
        
        ax.plot(k_range, hit_at_k, marker='o', linewidth=2.5, markersize=8, 
               label=chunker, alpha=0.85)
    
    # Styling
    ax.set_xlabel('K (number of top chunks retrieved)', fontweight='bold')
    ax.set_ylabel('Hit@K (fraction of questions answered)', fontweight='bold')
    ax.set_title('Cumulative Hit Rate: How Many Top Chunks Do We Need?', 
                fontweight='bold', pad=20)
    ax.set_xticks(k_range)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (100%)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '6_cumulative_hit_curve.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 6_cumulative_hit_curve.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Generate All Plots
# ═══════════════════════════════════════════════════════════════════════════
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize results from specific dataset
  python visualize_results.py --dataset train_100
  python visualize_results.py --dataset natural_questions_squad_1000

  # If no dataset specified, looks for results/eval_metrics.json
  python visualize_results.py
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        default=None,
        help='Dataset name (e.g., train_100, natural_questions_squad_1000). Reads from results/<dataset>/eval_metrics.json'
    )
    
    return parser.parse_args()


def main():
    global RESULTS_PATH, OUTPUT_DIR
    
    args = parse_arguments()
    
    # Set paths based on dataset name
    if args.dataset:
        dataset_name = args.dataset
        results_dir = Path("results") / dataset_name
        RESULTS_PATH = results_dir / "eval_metrics.json"
        OUTPUT_DIR = results_dir / "visualizations"
    else:
        # Default to old location for backward compatibility
        RESULTS_PATH = "results/eval_metrics.json"
        OUTPUT_DIR = Path("results/visualizations")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("  📊  GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Load results
    print(f"\n  Loading results from: {RESULTS_PATH}")
    if not Path(RESULTS_PATH).exists():
        print(f"\n  ❌ ERROR: Results file not found: {RESULTS_PATH}")
        print(f"\n  Please run eval_metrics.py first to generate results.")
        if args.dataset:
            print(f"  Example: python src/evaluation/eval_metrics.py --data data/{args.dataset}.json")
        return
    
    results = load_results()
    
    num_examples = results['config']['num_examples']
    num_chunkers = len(results['aggregated'])
    print(f"  Examples: {num_examples}")
    print(f"  Chunkers: {num_chunkers}")
    
    # Generate all plots
    print(f"\n  Generating plots → {OUTPUT_DIR}/")
    plot_hit_rate_comparison(results)
    plot_mrr_comparison(results)
    plot_per_example_heatmap(results)
    plot_rank_distribution(results)
    plot_pairwise_agreement(results)
    plot_cumulative_hit_curve(results)
    
    print("\n" + "="*70)
    print("  ✅  ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print(f"\n  📁  Output directory: {OUTPUT_DIR.absolute()}")
    print("\n  Generated files:")
    for i in range(1, 7):
        png_files = list(OUTPUT_DIR.glob(f'{i}_*.png'))
        if png_files:
            print(f"    {i}. {png_files[0].name}")
    print()


if __name__ == "__main__":
    main()