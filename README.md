# rag-chunk-eval

Benchmarking harness that measures how well different text chunking strategies preserve retrievable answers in a RAG pipeline. Given a QA dataset, it chunks each context with multiple strategies, embeds everything with `all-MiniLM-L6-v2`, runs nearest-neighbor search, and reports **Hit Rate @ K** and **MRR**.

This is an evaluation tool, not a production RAG system.

## What it actually measures

For each (question, context) pair in the dataset:

1. The context is split into chunks by each chunker
2. Chunks are embedded alongside the question
3. Top-K chunks are retrieved by L2 distance
4. A hit is recorded if any top-K chunk contains the answer as a substring

The key metric is whether chunking destroyed the answer or left it retrievable. Evaluation is **per-context** (each question searches only its own context's chunks, not a shared corpus).

## Chunking strategies

| Chunker | Strategy | Key params |
|---------|----------|------------|
| **FixedChunker** | Fixed-size character windows | `chunk_size`, `overlap` |
| **RecursiveChunker** | Hierarchical separator fallback (`\n\n` > `\n` > ` ` > char) | `max_chars`, `overlap` |
| **ParagraphChunker** | Paragraph boundaries with merge/split | `max_chars`, `min_chars` |
| **StructureAwareChunker** | Respects Markdown/HTML headings, lists, HRs | `chunk_size`, `min_chunk_size`, `overlap` |
| **SemanticDensityChunker** | Fixed chunks, adaptive overlap based on lexical density (TTR, NER, vocab richness) | `chunk_size`, `min_overlap`, `max_overlap` |

## Setup

**Requirements:** Python 3.12+, ~2 GB disk for models.

```bash
git clone https://github.com/DTufail/adaptive-chunking-rag.git
cd adaptive-chunking-rag

python3 -m venv myenv
source myenv/bin/activate

# Install as editable package (resolves all imports without sys.path hacks)
pip install -r requirements.txt
pip install -e .

# Required by SemanticDensityChunker (NER pipeline)
python -m spacy download en_core_web_sm
```

The sentence-transformers model (`all-MiniLM-L6-v2`, ~80 MB) downloads automatically on first run.

## Usage

### Run evaluation

```bash
# Default config (configs/default.yaml)
eval-metrics

# Or directly
python src/evaluation/eval_metrics.py

# Custom config or dataset
python src/evaluation/eval_metrics.py --config configs/my_config.yaml
python src/evaluation/eval_metrics.py --data data/train_1000.json
```

Output: terminal comparison table + `results/<dataset>/eval_metrics.json`.

### Hyperparameter grid search

Sweeps `chunk_size` x `overlap` across all chunker types. Question embeddings are computed once and reused across all configurations.

```bash
# All chunkers (~150 configs)
hyperparameter-search

# Single chunker (faster)
python src/utils/hyperparameter_search.py --chunker FixedChunker

# Custom dataset
python src/utils/hyperparameter_search.py --data data/train_100.json
```

Output: JSON results + heatmaps + line plots in `results/<dataset>/hyperparameter_search/`.

### Tests

```bash
python -m pytest tests/ -v
```

## Data

The evaluation uses SQuAD-format JSON (`{examples: [{question, context, answer}]}`). Two loaders are provided:

```bash
# SQuAD v2 (official dataset)
python src/utils/squad_v2_loader.py

# Natural Questions -> SQuAD format
python src/utils/natural_questions_to_squad.py --sample 1000
```

NQ contexts contain raw Wikipedia HTML. The pipeline auto-detects this and converts to Markdown before chunking (configurable via `--preprocess-html` / `--no-preprocess-html`).

## Pipeline architecture

The evaluation runs in three phases, optimized around the fact that embedding is the only expensive operation:

```
Phase A — CHUNK (< 1 sec)
  Deduplicate contexts, run all chunkers. No model calls.

Phase B — EMBED (all model inference here)
  Batch-encode all questions (once).
  Deduplicate chunk texts globally across chunkers, encode unique texts (once).

Phase C — SEARCH (sub-millisecond per query)
  Numpy L2 distance on pre-computed vectors. No model calls.
  Record hit rank per (question, chunker) pair.
```

## Project structure

```
pyproject.toml                   # Package config, entry points, pytest settings
configs/default.yaml             # All experiment parameters
src/
  chunkers/                      # 5 chunking strategies + ABC base class
    base_chunker.py              #   Chunk dataclass + BaseChunker ABC
    fixed_chunker.py             #   Character-level fixed windows
    recursive_chunker.py         #   Hierarchical separator fallback
    paragraph_chunker.py         #   Paragraph-aware merge/split
    structure_aware_chunker.py   #   Markdown/HTML structure respecting
    semantic_density_chunker.py  #   Adaptive overlap via NLP density metrics
    _spacy_cache.py              #   Process-level spaCy model cache
  embeddings/
    embedding_model.py           #   sentence-transformers wrapper (manual batching)
    faiss_index.py               #   FAISS IndexFlatL2 wrapper
    search.py                    #   Shared numpy L2 nearest-neighbor search
  evaluation/
    eval_metrics.py              #   Main 3-phase evaluation pipeline
  utils/
    config.py                    #   YAML loader + dynamic chunker instantiation
    text_preprocessor.py         #   HTML -> Markdown converter (for NQ dataset)
    hyperparameter_search.py     #   Grid search over chunk_size x overlap
    squad_v2_loader.py           #   SQuAD v2 dataset loader
    natural_questions_to_squad.py  # NQ -> SQuAD format converter
tests/
  test_chunkers.py               # Contract tests (all chunkers) + per-chunker tests
  test_retrieval.py              # FAISS index + metric function tests
results/                         # Evaluation outputs + visualizations
```

## Configuration

All settings live in `configs/default.yaml`:

```yaml
seed: 42

embedding:
  model_name: all-MiniLM-L6-v2
  dimension: 384

data:
  path: data/natural_questions_squad_1000.json
  min_answer_length: 8       # Drop answers shorter than this (reduces false positives)
  preprocess_html: true      # Auto-convert NQ HTML to Markdown

evaluation:
  k_max: 5
  k_values: [1, 3, 5]
  output_dir: results

chunkers:                    # Each key instantiates a chunker class with these kwargs
  FixedChunker_256:
    chunk_size: 256
    overlap: 25
  RecursiveChunker:
    max_chars: 1024
    overlap: 50
  # ... (see file for full config)
```

Chunker names in the config map directly to class names in `src/chunkers/`. Add a suffix (e.g. `FixedChunker_256`) to run the same class with different parameters.

## Reproducibility

- All random seeds set to `42` by default (`numpy`, `torch`, `random`, `PYTHONHASHSEED`)
- Dependencies pinned to exact versions in `requirements.txt`
- Document IDs use `hashlib.sha256` (not `hash()`) for cross-run determinism
- Configurable via `seed:` in the YAML config
