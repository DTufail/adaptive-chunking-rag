import re
from typing import List, Optional, Tuple

from .base_chunker import BaseChunker, Chunk


class ParagraphChunker(BaseChunker):
    """
    Paragraph-based chunker that splits on paragraph boundaries.

    Splits text on double newlines, then merges small paragraphs
    and splits large ones to stay within size limits.
    """

    def __init__(
        self,
        max_chars: int = 1024,
        min_chars: int = 100,
        paragraph_separator: str = r"\n\s*\n",
        name: Optional[str] = None
    ):
        """
        Initialize the paragraph-based chunker.

        Args:
            max_chars: Maximum characters per chunk. Paragraphs exceeding
                      this will be split.
            min_chars: Minimum characters per chunk. Small paragraphs will
                      be merged with neighbors.
            paragraph_separator: Regex pattern for paragraph boundaries.
            name: Optional name for this chunker instance.

        Raises:
            ValueError: If min_chars >= max_chars or values are invalid.
        """
        super().__init__(name=name)

        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if min_chars < 0:
            raise ValueError("min_chars must be non-negative")
        if min_chars >= max_chars:
            raise ValueError("min_chars must be less than max_chars")

        self.max_chars = max_chars
        self.min_chars = min_chars
        self.paragraph_separator = paragraph_separator

    def _split_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into paragraphs with position tracking.

        Args:
            text: Input text.

        Returns:
            List of tuples (paragraph_text, start_pos, end_pos).
        """
        pattern = re.compile(self.paragraph_separator)
        paragraphs = []
        last_end = 0

        for match in pattern.finditer(text):
            para_text = text[last_end:match.start()]
            if para_text.strip():
                paragraphs.append((para_text, last_end, match.start()))
            last_end = match.end()

        # Handle the last paragraph
        if last_end < len(text):
            para_text = text[last_end:]
            if para_text.strip():
                paragraphs.append((para_text, last_end, len(text)))

        return paragraphs

    def _split_large_paragraph(
        self,
        text: str,
        start_pos: int
    ) -> List[Tuple[str, int, int]]:
        """
        Split a large paragraph into smaller chunks.

        Attempts to split at sentence boundaries (periods, etc.),
        falling back to word boundaries if necessary.

        Args:
            text: The paragraph text to split.
            start_pos: Starting position in original document.

        Returns:
            List of tuples (chunk_text, start_pos, end_pos).
        """
        if len(text) <= self.max_chars:
            return [(text, start_pos, start_pos + len(text))]

        chunks = []
        current_start = 0

        while current_start < len(text):
            # Determine the end of this chunk
            if current_start + self.max_chars >= len(text):
                # Last chunk takes the rest
                chunk_text = text[current_start:]
                chunks.append((
                    chunk_text,
                    start_pos + current_start,
                    start_pos + len(text)
                ))
                break

            # Try to find a good break point
            search_end = current_start + self.max_chars
            search_text = text[current_start:search_end]

            # Look for sentence endings in priority order (strongest → weakest).
            # Early-exit: take the first pattern that matches.
            best_break = None
            for pattern in [r'\.\s', r'\?\s', r'!\s', r';\s', r',\s']:
                matches = list(re.finditer(pattern, search_text))
                if matches:
                    best_break = current_start + matches[-1].end()
                    break  # strongest available break found, stop searching

            if best_break is None:
                # Fall back to word boundary
                space_pos = search_text.rfind(' ')
                if space_pos > 0:
                    best_break = current_start + space_pos + 1
                else:
                    # No good break point, hard split
                    best_break = search_end

            chunk_text = text[current_start:best_break].strip()
            if chunk_text:
                chunks.append((
                    chunk_text,
                    start_pos + current_start,
                    start_pos + best_break
                ))

            current_start = best_break

        return chunks

    def _merge_small_paragraphs(
        self,
        paragraphs: List[Tuple[str, int, int]]
    ) -> List[Tuple[str, int, int]]:
        """
        Merge consecutive small paragraphs.

        Args:
            paragraphs: List of (text, start, end) tuples.

        Returns:
            List of merged paragraph tuples.
        """
        if not paragraphs:
            return []

        merged = []
        current_text = ""
        current_start = paragraphs[0][1]
        current_end = paragraphs[0][2]

        for text, start, end in paragraphs:
            potential_text = current_text + ("\n\n" if current_text else "") + text

            # Only merge if the current chunk is still below min_chars threshold
            # AND the merged result stays within max_chars
            if len(current_text) < self.min_chars and len(potential_text) <= self.max_chars:
                current_text = potential_text
                current_end = end
                if not current_text.strip():
                    current_start = start
            else:
                # Current chunk is big enough — save it and start fresh
                if current_text.strip():
                    merged.append((current_text, current_start, current_end))
                current_text = text
                current_start = start
                current_end = end

        # Don't forget the last one
        if current_text.strip():
            merged.append((current_text, current_start, current_end))

        return merged

    def chunk(self, text: str, document_id: Optional[str] = None) -> List[Chunk]:
        """
        Split text into chunks based on paragraph boundaries.

        Process:
        1. Split on paragraph boundaries (double newlines)
        2. Split paragraphs that exceed max_chars
        3. Merge consecutive paragraphs below min_chars

        Args:
            text: The input text to chunk.
            document_id: Optional document identifier.

        Returns:
            List of Chunk objects.
        """
        if not text:
            return []

        doc_id = document_id or self._generate_document_id()

        # Step 1: Split into paragraphs
        paragraphs = self._split_paragraphs(text)

        if not paragraphs:
            return []

        # Step 2: Split large paragraphs
        split_paragraphs = []
        for para_text, start, end in paragraphs:
            if len(para_text) > self.max_chars:
                split_paragraphs.extend(
                    self._split_large_paragraph(para_text, start)
                )
            else:
                split_paragraphs.append((para_text, start, end))

        # Step 3: Merge small paragraphs
        merged_paragraphs = self._merge_small_paragraphs(split_paragraphs)

        # Create chunks
        chunks = []
        for idx, (para_text, start, end) in enumerate(merged_paragraphs):
            chunk = Chunk(
                chunk_id=self._generate_chunk_id(doc_id, idx),
                document_id=doc_id,
                text=para_text,
                start_pos=start,
                end_pos=end,
                metadata={
                    "chunker": self.name,
                    "max_chars": self.max_chars,
                    "min_chars": self.min_chars,
                    "chunk_index": idx,
                }
            )
            chunks.append(chunk)

        return chunks

    def __repr__(self) -> str:
        return (
            f"ParagraphChunker(max_chars={self.max_chars}, "
            f"min_chars={self.min_chars}, name='{self.name}')"
        )
