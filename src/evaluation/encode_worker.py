"""
encode_worker.py
────────────────
Subprocess worker for Phase B encoding.

This script is NOT imported.  It is spawned as a standalone process
by eval_metrics.py, does one encode job, writes the output, and exits.
The exit is the point:  when this process terminates, the OS reclaims
ALL of PyTorch's mapped pages.  The parent process never loads PyTorch.

USAGE (called by eval_metrics.py, not by hand):
    python encode_worker.py <job_name> <input_json> <output_npy>

    job_name   : "questions" or a chunker name (e.g. "FixedChunker")
    input_json : path to JSON file containing the texts to encode.
                 Format: { "texts": ["text0", "text1", ...] }
    output_npy : path to write the output vectors as a .npy file.
                 Shape: (N, 384) float32.
"""

import json
import sys
import os
import numpy as np

# ─── project root on the path ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings.embedding_model import EmbeddingModel


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <job_name> <input_json> <output_npy>",
              file=sys.stderr)
        sys.exit(1)

    job_name   = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    # ── Load texts from JSON ─────────────────────────────────────────────────
    with open(input_path, "r") as f:
        data = json.load(f)
    texts = data["texts"]

    if not texts:
        print(f"  [worker:{job_name}] ERROR: empty texts list in {input_path}",
              file=sys.stderr)
        sys.exit(1)

    print(f"  [worker:{job_name}] Loaded {len(texts)} texts from {input_path}")

    # ── Load model and encode ────────────────────────────────────────────────
    # Model loads here.  This is the only process that holds it.
    # It will be gone when we exit.
    print(f"  [worker:{job_name}] Loading model...")
    embed_model = EmbeddingModel()

    print(f"  [worker:{job_name}] Encoding...")
    vectors = embed_model.encode(texts)                  # (N, 384) float32
    # embed_model used batching internally (DEFAULT_BATCH_SIZE=64).

    # ── Write output ─────────────────────────────────────────────────────────
    np.save(output_path, vectors)
    print(f"  [worker:{job_name}] Wrote {vectors.shape} to {output_path}")

    # Process exits here.  PyTorch, the model, all mapped pages — gone.


if __name__ == "__main__":
    main()