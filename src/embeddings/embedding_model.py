"""Embedding model wrapper for sentence-transformers."""

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding models.

    Lazy-loads the model on first use to avoid initialization overhead.
    Returns raw (unnormalized) float32 embeddings. Distance metric decisions
    are left to the index layer (FaissIndex uses L2).
    """

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, dimension: Optional[int] = DEFAULT_DIMENSION):
        """Initialize the embedding model wrapper.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Defaults to "all-MiniLM-L6-v2".
            dimension: Expected output embedding dimension for this model.
                If None, the dimension will be inferred from the first encode.
        """
        self.model_name = model_name
        self.dimension = dimension
        self._model: Optional[SentenceTransformer] = None
        logger.debug(
            f"EmbeddingModel initialized with model_name={model_name}, dimension={dimension}"
        )

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformers model.

        Returns:
            The loaded SentenceTransformer model.
        """
        if self._model is None:
            logger.debug(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.debug(f"Model loaded successfully")
        return self._model

    # Default batch size for encode().  Controls how many texts are
    # tokenized into memory simultaneously inside sentence-transformers.
    # 256 keeps peak tokenizer memory at (256, 256) int64 tensors — about
    # 4 MB — well within reach of 8 GB machines.  all-MiniLM-L6-v2 is
    # only 22 M params, so larger batches give better CPU throughput.
    # Lower this if OOM persists on < 8 GB machines.
    DEFAULT_BATCH_SIZE = 256

    def encode(self, texts: List[str], batch_size: Optional[int] = None, show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts into raw embeddings.

        Args:
            texts: List of text strings to encode. Must be non-empty.
            batch_size: How many texts to tokenize and run through the model
                at once.  Directly controls peak memory during inference.
                Defaults to DEFAULT_BATCH_SIZE (256).  Pass a smaller value
                on memory-constrained machines.
            show_progress: If True, print a batch-level progress line to
                stdout (overwritten in-place).  Useful for large encodes
                where the caller wants visibility into progress.

        Returns:
            2D numpy array of shape (len(texts), DIMENSION) as float32.

        Raises:
            ValueError: If texts is an empty list, or if the model returns
                vectors with a dimension that does not match DIMENSION.
            TypeError: If any element in texts is not a string.
        """
        if not texts:
            raise ValueError("Cannot encode an empty list of texts")

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("All elements in texts must be strings")

        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        logger.debug(f"Encoding {len(texts)} texts (batch_size={batch_size})")

        # ── Manual batching loop ─────────────────────────────────────────
        # sentence-transformers' batch_size kwarg only controls how the
        # forward pass is mini-batched.  The tokenizer still runs on the
        # FULL input list before the first mini-batch starts — meaning
        # all N tokenized sequences are in memory at once regardless of
        # batch_size.  On 8 GB machines with 1000+ texts this is what
        # triggers the OOM kill.
        #
        # Fix: slice the input ourselves.  Each call to model.encode()
        # sees at most batch_size texts.  The tokenizer's peak memory
        # is capped at (batch_size, max_seq_len) instead of (N, max_seq_len).
        #
        # Pre-allocate the output array after the first slice so we know
        # the exact shape, then fill it row-by-row.  No vstack, no
        # intermediate list of arrays.

        n = len(texts)

        # First slice: encode it, use the result to allocate the output
        # array and run the dimension/dtype checks.
        first_end   = min(batch_size, n)
        first_batch = self.model.encode(
            texts[:first_end],
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if first_batch.dtype != np.float32:
            first_batch = first_batch.astype(np.float32)

        inferred_dim = first_batch.shape[1]
        if self.dimension is None:
            self.dimension = inferred_dim
        elif inferred_dim != self.dimension:
            raise ValueError(
                f"Model '{self.model_name}' returned vectors of dimension "
                f"{inferred_dim}, expected {self.dimension}. "
                f"Update the configured dimension to match the model."
            )

        # Pre-allocate the full output array.  Only allocation for the
        # entire encode call — no reallocs, no list-of-arrays.
        embeddings = np.empty((n, self.dimension), dtype=np.float32)
        embeddings[:first_end] = first_batch
        del first_batch   # release immediately — data is copied into embeddings

        n_batches = (n + batch_size - 1) // batch_size
        if show_progress:
            print(f"\r    Encoding: batch 1/{n_batches} "
                  f"({min(first_end, n)}/{n} texts)", end="", flush=True)

        # Remaining slices.  Each one is tokenized, encoded, written into
        # the pre-allocated array, then discarded.  Peak memory per
        # iteration is (batch_size, 384) — nothing accumulates.
        for start in range(first_end, n, batch_size):
            end   = min(start + batch_size, n)
            batch = self.model.encode(
                texts[start:end],
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            if batch.dtype != np.float32:
                batch = batch.astype(np.float32)
            embeddings[start:end] = batch
            # batch goes out of scope here — the slice of texts[start:end]
            # and the tokenized tensors are all eligible for GC before
            # the next iteration allocates.

            if show_progress:
                batch_num = (start // batch_size) + 1
                print(f"\r    Encoding: batch {batch_num}/{n_batches} "
                      f"({end}/{n} texts)", end="", flush=True)

        if show_progress:
            print(f"\r    Encoding: {n}/{n} texts — done.{' ' * 20}")

        logger.debug(f"Encoded to shape {embeddings.shape}")
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string into a raw embedding.

        Args:
            text: Text string to encode.

        Returns:
            1D numpy array of shape (DIMENSION,) as float32.

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"Text must be a string, got {type(text).__name__}")

        return self.encode([text])[0]