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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
