from typing import List, Optional

from .base_chunker import BaseChunker, Chunk


class FixedChunker(BaseChunker):
    """
    Fixed-size character-based chunker with configurable overlap.

    Splits text into chunks of a fixed character size, with optional
    overlap between consecutive chunks to preserve context.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 0,
        name: Optional[str] = None
    ):
        """
        Initialize the fixed-size chunker.

        Args:
            chunk_size: Target size of each chunk in characters.
            overlap: Number of overlapping characters between chunks.
                    Must be less than chunk_size.
            name: Optional name for this chunker instance.

        Raises:
            ValueError: If overlap >= chunk_size or if values are negative.
        """
        super().__init__(name=name)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, document_id: Optional[str] = None) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text: The input text to chunk.
            document_id: Optional document identifier.

        Returns:
            List of Chunk objects.
        """
        if not text:
            return []

        doc_id = document_id or self._generate_document_id()
        chunks = []
        step = self.chunk_size - self.overlap
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunk = Chunk(
                chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                document_id=doc_id,
                text=chunk_text,
                start_pos=start,
                end_pos=end,
                metadata={
                    "chunker": self.name,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "chunk_index": chunk_index,
                }
            )
            chunks.append(chunk)

            start += step
            chunk_index += 1

        return chunks

    def __repr__(self) -> str:
        return (
            f"FixedChunker(chunk_size={self.chunk_size}, "
            f"overlap={self.overlap}, name='{self.name}')"
        )
