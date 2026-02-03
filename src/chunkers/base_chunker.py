from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import uuid


@dataclass
class Chunk:
    """Represents a single chunk of text with metadata."""
    chunk_id: str
    document_id: str
    text: str
    start_pos: int
    end_pos: int
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Return the character length of the chunk."""
        return len(self.text)

    def to_dict(self) -> dict:
        """Convert chunk to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "length": self.length,
            "metadata": self.metadata,
        }


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.

    Subclasses must implement the chunk() method to define
    their specific chunking logic.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the chunker.

        Args:
            name: Optional name for this chunker instance.
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def chunk(self, text: str, document_id: Optional[str] = None) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The input text to chunk.
            document_id: Optional document identifier. If not provided,
                        a UUID will be generated.

        Returns:
            List of Chunk objects.
        """
        pass

    def _generate_document_id(self) -> str:
        """Generate a unique document ID."""
        return str(uuid.uuid4())

    def _generate_chunk_id(self, document_id: str, index: int) -> str:
        """Generate a chunk ID based on document ID and chunk index."""
        return f"{document_id}_chunk_{index:04d}"

    def chunk_multiple(
        self,
        documents: List[dict],
        text_key: str = "text",
        id_key: str = "id"
    ) -> List[List[Chunk]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries.
            text_key: Key for the text field in each document.
            id_key: Key for the document ID field.

        Returns:
            List of chunk lists, one per document.
        """
        results = []
        for doc in documents:
            text = doc.get(text_key, "")
            doc_id = doc.get(id_key, self._generate_document_id())
            chunks = self.chunk(text, document_id=doc_id)
            results.append(chunks)
        return results

    def get_stats(self, chunks: List[Chunk]) -> dict:
        """
        Calculate statistics for a list of chunks.

        Args:
            chunks: List of Chunk objects.

        Returns:
            Dictionary with chunk statistics.
        """
        if not chunks:
            return {
                "num_chunks": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
                "total_length": 0,
            }

        lengths = [c.length for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_length": sum(lengths),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
