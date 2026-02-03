from typing import List, Optional

from .base_chunker import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """
    Recursive character-based chunker with hierarchical separator fallback.

    Attempts to split text using separators in priority order:
        1. Double newline  (paragraph boundary)
        2. Single newline  (line boundary)
        3. Space           (word boundary)
        4. Empty string    (character-level hard split — last resort)

    At each level, if a resulting piece still exceeds max_chars, the chunker
    recurses with the next separator in the list. This finds the most
    semantically natural break point available while staying within size limits.

    Overlap is applied at the final chunk level by repeating trailing
    characters from the previous chunk.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]

    def __init__(
        self,
        max_chars: int = 512,
        overlap: int = 0,
        separators: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the recursive chunker.

        Args:
            max_chars: Maximum characters per chunk.
            overlap: Number of characters to overlap between consecutive chunks.
                    Must be less than max_chars.
            separators: Ordered list of separator strings to try, from strongest
                       to weakest. Defaults to ["\\n\\n", "\\n", " ", ""].
            name: Optional name for this chunker instance.

        Raises:
            ValueError: If max_chars <= 0, overlap < 0, or overlap >= max_chars.
        """
        super().__init__(name=name)

        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= max_chars:
            raise ValueError("overlap must be less than max_chars")

        self.max_chars = max_chars
        self.overlap = overlap
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

    # ─── Core recursive split ────────────────────────────────────────────

    def _split_text(self, text: str, separator_index: int = 0) -> List[str]:
        """
        Recursively split text using the separator hierarchy.

        Args:
            text: The text to split.
            separator_index: Current position in the separator priority list.

        Returns:
            List of text fragments. Each fragment (except possibly the first)
            is sized to (max_chars - overlap) so that after overlap is
            prepended in _apply_overlap the final chunk stays <= max_chars.
        """
        # The first chunk has no overlap prepended, so it can use the full
        # max_chars. Every subsequent chunk will have overlap prepended, so
        # its raw size must leave room. We handle this by using a reduced
        # target for the recursive split and then letting the caller decide.
        # For simplicity and correctness we use the reduced target everywhere
        # — the first chunk may end up slightly smaller than it could be, but
        # it will never exceed max_chars after overlap is applied.
        effective_max = self.max_chars - self.overlap if self.overlap > 0 else self.max_chars

        # Base case: text already fits
        if len(text) <= effective_max:
            return [text] if text.strip() else []

        # Base case: no more separators to try — hard character split
        if separator_index >= len(self.separators):
            return self._hard_split(text, effective_max)

        separator = self.separators[separator_index]

        # Empty string separator == character-level hard split (terminal case)
        if separator == "":
            return self._hard_split(text, effective_max)

        # Split on current separator
        parts = text.split(separator)

        # If this separator didn't actually split anything, try the next one
        if len(parts) == 1:
            return self._split_text(text, separator_index + 1)

        # Greedily merge consecutive parts that fit within effective_max,
        # recursing into the next separator level for parts that don't.
        # Note: .split(sep) consumes the separator, so we must re-add it
        # when joining parts. Each part after the first was preceded by
        # the separator in the original text.
        chunks: List[str] = []
        current = ""

        for i, part in enumerate(parts):
            # Re-attach the separator that .split() consumed, except for
            # the very first part (nothing preceded it).
            part_with_sep = separator + part if i > 0 else part

            candidate = current + part_with_sep

            if len(candidate) <= effective_max:
                current = candidate
            else:
                # Flush the current accumulator
                if current.strip():
                    chunks.append(current)

                # If this single part exceeds effective_max, recurse deeper
                if len(part) > effective_max:
                    chunks.extend(self._split_text(part, separator_index + 1))
                    current = ""
                else:
                    # Start fresh with the separator included so the next
                    # chunk's text begins cleanly (e.g. " Dangerously...")
                    current = part_with_sep.lstrip() if not chunks else part_with_sep

        # Flush remainder
        if current.strip():
            chunks.append(current)

        return chunks

    def _hard_split(self, text: str, effective_max: Optional[int] = None) -> List[str]:
        """
        Character-level hard split when no semantic separator is available.

        Args:
            text: Text to split.
            effective_max: Max chars per fragment. Defaults to self.max_chars.

        Returns:
            List of fixed-size character fragments.
        """
        size = effective_max if effective_max is not None else self.max_chars
        return [
            text[i:i + size]
            for i in range(0, len(text), size)
            if text[i:i + size].strip()
        ]

    # ─── Overlap application ─────────────────────────────────────────────

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Prepend overlap characters from the previous chunk to each chunk.

        The overlap boundary is snapped forward to the nearest word boundary
        so we never start a chunk mid-word.

        Args:
            chunks: List of chunk strings without overlap.

        Returns:
            List of chunk strings with overlap applied.
        """
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            slice_start = len(prev) - self.overlap

            # Snap forward to the next space so we don't cut mid-word.
            # If no space is found after slice_start, fall back to slice_start.
            space_idx = prev.find(" ", slice_start)
            if space_idx != -1 and space_idx < len(prev):
                slice_start = space_idx + 1  # start AFTER the space

            prev_tail = prev[slice_start:]
            result.append(prev_tail + chunks[i])

        return result

    # ─── Public interface ────────────────────────────────────────────────

    def chunk(self, text: str, document_id: Optional[str] = None) -> List[Chunk]:
        """
        Split text into chunks using recursive separator fallback.

        Args:
            text: The input text to chunk.
            document_id: Optional document identifier.

        Returns:
            List of Chunk objects.
        """
        if not text:
            return []

        doc_id = document_id or self._generate_document_id()

        # Step 1: Recursive split
        raw_chunks = self._split_text(text)

        if not raw_chunks:
            return []

        # Step 2: Apply overlap
        overlapped_chunks = self._apply_overlap(raw_chunks)

        # Step 3: Locate each chunk in the original text and build Chunk objects.
        # We search forward from the last found position to handle duplicates.
        chunks: List[Chunk] = []
        search_start = 0

        for idx, chunk_text in enumerate(overlapped_chunks):
            # For position tracking we use the non-overlapped text to find
            # the true start in the original document.
            core_text = raw_chunks[idx]
            pos = text.find(core_text, search_start)

            if pos == -1:
                # Fallback: shouldn't happen, but be safe
                pos = search_start

            start_pos = pos
            end_pos = pos + len(core_text)
            search_start = end_pos  # advance for next iteration

            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(doc_id, idx),
                document_id=doc_id,
                text=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata={
                    "chunker": self.name,
                    "max_chars": self.max_chars,
                    "overlap": self.overlap,
                    "chunk_index": idx,
                    "separators_used": self.separators,
                }
            ))

        return chunks

    def __repr__(self) -> str:
        return (
            f"RecursiveChunker(max_chars={self.max_chars}, "
            f"overlap={self.overlap}, separators={self.separators}, "
            f"name='{self.name}')"
        )