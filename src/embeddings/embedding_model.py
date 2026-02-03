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

    DIMENSION = 384  # Output dimension for all-MiniLM-L6-v2

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model wrapper.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Defaults to "all-MiniLM-L6-v2" (384-dimensional embeddings).
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        logger.debug(f"EmbeddingModel initialized with model_name={model_name}")

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
    # 64 keeps peak tokenizer memory at (64, 256) int64 tensors — about
    # 1 MB — regardless of how many texts are passed in total.
    # Lower this if OOM persists on < 8 GB machines.
    DEFAULT_BATCH_SIZE = 64

    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Encode a list of texts into raw embeddings.

        Args:
            texts: List of text strings to encode. Must be non-empty.
            batch_size: How many texts to tokenize and run through the model
                at once.  Directly controls peak memory during inference.
                Defaults to DEFAULT_BATCH_SIZE (64).  Pass a smaller value
                on memory-constrained machines; a larger value on machines
                with headroom (marginal speed gain from larger batches
                is small on CPU).

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

        if first_batch.shape[1] != self.DIMENSION:
            raise ValueError(
                f"Model '{self.model_name}' returned vectors of dimension "
                f"{first_batch.shape[1]}, expected {self.DIMENSION}. "
                f"Update DIMENSION if you changed the model."
            )

        # Pre-allocate the full output array.  Only allocation for the
        # entire encode call — no reallocs, no list-of-arrays.
        embeddings = np.empty((n, self.DIMENSION), dtype=np.float32)
        embeddings[:first_end] = first_batch
        del first_batch   # release immediately — data is copied into embeddings

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