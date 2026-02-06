# Adaptive Chunking RAG — Retrieval Evaluation Harness

Benchmarks 5 text-chunking strategies for retrieval in a RAG pipeline.
Measures **Hit Rate @ K** and **MRR** to determine which chunker +
hyperparameter combo best retrieves answer-containing chunks.

## Chunkers

| Chunker | Strategy |
|---------|----------|
| **FixedChunker** | Character-level fixed-size windows |
| **RecursiveChunker** | Splits on `\n\n` → `\n` → ` ` → char |
| **ParagraphChunker** | Paragraph boundaries with merge/split |
| **StructureAwareChunker** | Respects Markdown/HTML headings, lists |
| **SemanticDensityChunker** | Adaptive overlap based on lexical density (spaCy NER, TTR) |

## Setup

**Requirements:** Python 3.12+, ~2 GB disk for models.

```bash
# 1. Clone
git clone https://github.com/DTufail/adaptive-chunking-rag.git
cd adaptive-chunking-rag

# 2. Create virtual environment
python3 -m venv myenv
source myenv/bin/activate

# 3. Install dependencies (pinned versions)
pip install -r requirements.txt

# 4. Download spaCy model (required by SentenceChunker + SemanticDensityChunker)
python -m spacy download en_core_web_sm
```

The sentence-transformers model (`all-MiniLM-L6-v2`) downloads
automatically on first run (~80 MB).

## Data

The evaluation uses SQuAD-format JSON files in `data/`.

Two data-generation paths are supported:

**1) Native SQuAD v2 loader**
```bash
python src/utils/squad_v2_loader.py
```
Creates `data/train_100.json`, `data/train_1000.json`, etc.

**2) Natural Questions → SQuAD conversion**
```bash
python src/utils/natural_questions_to_squad.py --sample 1000
```
Creates SQuAD-format output from Natural Questions.

## Reproducing Results

### Single evaluation (uses `configs/default.yaml`)
```bash
python src/evaluation/eval_metrics.py
```

### Custom config
```bash
python src/evaluation/eval_metrics.py --config configs/my_config.yaml
```

### Hyperparameter grid search
```bash
# All chunkers (slow — ~150 configs)
python src/utils/hyperparameter_search.py

# Single chunker
python src/utils/hyperparameter_search.py --chunker FixedChunker

# Custom data
python src/utils/hyperparameter_search.py --data data/train_100.json
```

Outputs are written to `analysis/hyperparameter_search/`.

### Tests
```bash
python -m pytest tests/ -v
```

## Reproducibility

All random seeds are set to `42` by default (configurable in
`configs/default.yaml` under `seed:`). This controls:

- `PYTHONHASHSEED`
- `numpy.random.seed`
- `torch.manual_seed`
- `random.seed`

Dependencies are pinned to exact versions in `requirements.txt`.
Document IDs use `hashlib.sha256` (not `hash()`) for cross-run
determinism.

## Project Structure

```
configs/default.yaml          # All experiment configuration
src/
  chunkers/                   # 5 chunking strategies + base class
  embeddings/                 # sentence-transformers wrapper + FAISS index
  evaluation/eval_metrics.py  # Main evaluation pipeline
  utils/
    config.py                 # YAML config loader
    squad_v2_loader.py        # SQuAD v2 → SQuAD-format JSON
    natural_questions_to_squad.py  # Natural Questions → SQuAD-format JSON
    hyperparameter_search.py  # Grid search over chunk_size × overlap
    text_preprocessor.py      # HTML → Markdown cleaning
tests/                        # pytest test suite
results/                      # Evaluation outputs (gitignored)
analysis/                     # Visualization / analysis outputs (gitignored)
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
  min_answer_length: 4
  preprocess_html: true

evaluation:
  k_max: 5
  k_values: [1, 3, 5]
  output_dir: results

chunkers:
  FixedChunker:
    chunk_size: 1024
    overlap: 50
  # ... (see file for all chunker configs)
```
