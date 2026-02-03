from typing import List, Optional

import spacy
from spacy.language import Language

from .base_chunker import BaseChunker, Chunk


class SentenceChunker(BaseChunker):
    """
    Sentence-based chunker using spaCy for sentence detection.

    Accumulates sentences until the maximum character limit is reached,
    preserving sentence boundaries to maintain semantic coherence.
    """

    def __init__(
        self,
        max_chars: int = 512,
        overlap_sentences: int = 0,
        spacy_model: str = "en_core_web_sm",
        name: Optional[str] = None
    ):
        """
        Initialize the sentence-based chunker.

        Args:
            max_chars: Maximum characters per chunk. Sentences are accumulated
                      until this limit is reached.
            overlap_sentences: Number of sentences to overlap between chunks.
            spacy_model: Name of the spaCy model to use for sentence detection.
            name: Optional name for this chunker instance.

        Raises:
            ValueError: If max_chars is not positive.
        """
        super().__init__(name=name)

        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be non-negative")

        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences
        self.spacy_model = spacy_model
        self._nlp: Optional[Language] = None

    @property
    def nlp(self) -> Language:
        """Lazy-load the spaCy model."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.spacy_model)
            except OSError:
                raise OSError(
                    f"spaCy model '{self.spacy_model}' not found. "
                    f"Install it with: python -m spacy download {self.spacy_model}"
                )
        return self._nlp

    def _get_sentences(self, text: str) -> List[dict]:
        """
        Extract sentences from text using spaCy.

        Args:
            text: Input text.

        Returns:
            List of dicts with 'text', 'start', and 'end' keys.

        Raises:
            ValueError: If text exceeds spaCy's max_length limit.
        """
        if len(text) > self.nlp.max_length:
            raise ValueError(
                f"Text length ({len(text)} chars) exceeds spaCy model max_length "
                f"({self.nlp.max_length}). Consider splitting the document first or "
                f"increasing nlp.max_length."
            )
        doc = self.nlp(text)
        return [
            {
                "text": sent.text,
                "start": sent.start_char,
                "end": sent.end_char
            }
            for sent in doc.sents
        ]

    def chunk(self, text: str, document_id: Optional[str] = None) -> List[Chunk]:
        """
        Split text into chunks at sentence boundaries.

        Sentences are accumulated until max_chars is reached. Each chunk
        contains complete sentences only.

        Args:
            text: The input text to chunk.
            document_id: Optional document identifier.

        Returns:
            List of Chunk objects.
        """
        if not text:
            return []

        doc_id = document_id or self._generate_document_id()
        sentences = self._get_sentences(text)

        if not sentences:
            return []

        chunks = []
        chunk_index = 0
        i = 0

        while i < len(sentences):
            current_sentences = []
            start_pos = sentences[i]["start"]

            # Accumulate sentences until max_chars is reached
            while i < len(sentences):
                sent = sentences[i]
                # Use positional range to account for inter-sentence whitespace
                projected_length = sent["end"] - start_pos

                # If adding this sentence would exceed limit and we have content
                if projected_length > self.max_chars and current_sentences:
                    break

                current_sentences.append(sent)
                i += 1

            if not current_sentences:
                continue

            # Build chunk text preserving original spacing
            end_pos = current_sentences[-1]["end"]
            chunk_text = text[start_pos:end_pos]

            chunk = Chunk(
                chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                document_id=doc_id,
                text=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata={
                    "chunker": self.name,
                    "max_chars": self.max_chars,
                    "num_sentences": len(current_sentences),
                    "chunk_index": chunk_index,
                }
            )
            chunks.append(chunk)
            chunk_index += 1

            # Apply overlap by stepping back
            if self.overlap_sentences > 0 and i < len(sentences):
                overlap_count = min(self.overlap_sentences, len(current_sentences))
                i = i - overlap_count

        return chunks

    def __repr__(self) -> str:
        return (
            f"SentenceChunker(max_chars={self.max_chars}, "
            f"overlap_sentences={self.overlap_sentences}, "
            f"spacy_model='{self.spacy_model}', name='{self.name}')"
        )
