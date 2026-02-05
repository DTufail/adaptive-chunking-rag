"""
structure_aware_chunker.py
──────────────────────────
Structure-aware text chunker that respects document structure.

Features:
1. Proper greedy paragraph packing
2. Enforces minimum chunk size (default 200 chars)
3. Handles both Markdown and HTML headings
4. Smart list detection and chunking
5. Supports Wikipedia-style HTML content

The chunker recognizes:
- Markdown headings: # H1, ## H2, etc.
- HTML headings: <H1>, <H2>, etc.
- Horizontal rules: ---, <hr>
- Lists: -, *, •, numbered items, <Li>
"""

import re
from typing import List, Optional, Tuple

from .base_chunker import BaseChunker, Chunk


class StructureAwareChunker(BaseChunker):
    """
    Structure-aware chunker with empirically-tuned parameters.
    
    Respects document structure (headings, lists, paragraphs) while
    maintaining optimal chunk sizes. Handles both Markdown and HTML content.
    """

    # Structural patterns - handles both Markdown and HTML
    HEADING_PATTERN = re.compile(
        r'^\s*(#{1,6}\s.+|<[Hh][1-6][^>]*>.*?</[Hh][1-6]>)',
        re.MULTILINE
    )
    HORIZONTAL_RULE_PATTERN = re.compile(
        r'^(\s*[-*_]{3,}\s*|<[Hh][Rr]\s*/?>)$',
        re.MULTILINE | re.IGNORECASE
    )

    def __init__(
        self,
        chunk_size: int = 1024,
        min_chunk_size: int = 200,
        overlap: int = 50,
        name: Optional[str] = None
    ):
        """
        Initialize structure-aware chunker.

        Args:
            chunk_size: Target chunk size (default 1024 from tuning).
            min_chunk_size: Minimum chunk size when splitting (default 200).
                           Lowered from 512 to preserve small but important sections
                           (e.g., Wikipedia infoboxes, short paragraphs with answers).
            overlap: Overlap between chunks (default 50 from tuning).
            name: Optional chunker name.
        """
        super().__init__(name=name)

        if chunk_size <= 0 or min_chunk_size <= 0:
            raise ValueError("Chunk sizes must be positive")
        if min_chunk_size > chunk_size:
            raise ValueError("min_chunk_size must be <= chunk_size")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid overlap")

        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    # ─── Structure detection ─────────────────────────────────────────────

    def _detect_headings(self, text: str) -> List[int]:
        """Find positions of all headings."""
        return [match.start() for match in self.HEADING_PATTERN.finditer(text)]

    def _detect_horizontal_rules(self, text: str) -> List[int]:
        """Find positions of horizontal rules."""
        return [match.start() for match in self.HORIZONTAL_RULE_PATTERN.finditer(text)]

    def _detect_paragraph_boundaries(self, text: str) -> List[int]:
        """
        Find paragraph boundaries (double newlines).
        
        Returns positions where new paragraphs start.
        """
        boundaries = [0]  # Start of document
        for match in re.finditer(r'\n\s*\n', text):
            boundaries.append(match.end())
        return boundaries

    def _extract_sections(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Split text into sections at major structural boundaries.
        
        Returns:
            List of (start, end, type) tuples where type is one of:
            'heading', 'hr_section', 'paragraph_block'
        """
        headings = self._detect_headings(text)
        hrs = self._detect_horizontal_rules(text)
        
        # Combine and sort all major boundaries
        boundaries = sorted(set([0] + headings + hrs + [len(text)]))
        
        sections = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Determine section type
            if start in headings:
                section_type = 'heading'
            elif start in hrs:
                section_type = 'hr_section'
            else:
                section_type = 'paragraph_block'
            
            sections.append((start, end, section_type))
        
        return sections

    # ─── List detection ──────────────────────────────────────────────────

    def _is_list_section(self, text: str) -> bool:
        """Check if text is primarily a list (Markdown or HTML)."""
        lines = text.split('\n')
        # Match markdown lists: -, *, •, +, 1., 1)
        # Also match HTML list markers: <Li>, • from preprocessed HTML
        list_pattern = r'^\s*([-*•+]|\d+[.)]|<[Ll][Ii]>)\s*'
        list_lines = sum(
            1 for line in lines 
            if re.match(list_pattern, line.strip())
        )
        # Also count lines that start with bullet character (from HTML preprocessing)
        bullet_lines = sum(1 for line in lines if line.strip().startswith('•'))
        total_list_lines = max(list_lines, bullet_lines)
        return total_list_lines >= len(lines) * 0.5  # ≥50% list items

    def _chunk_list(
        self, 
        text: str, 
        start_pos: int, 
        doc_id: str, 
        chunk_index: int
    ) -> Tuple[List[Chunk], int]:
        """
        Chunk a list section.
        
        Strategy:
        - If list ≤ chunk_size: keep together
        - If list > chunk_size: split at item boundaries, min_chunk_size per chunk
        """
        # FIX #2: Add size validation for small lists  
        if len(text) <= self.chunk_size:
            if len(text.strip()) >= self.min_chunk_size:
                return [Chunk(
                    chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                    document_id=doc_id,
                    text=text,
                    start_pos=start_pos,
                    end_pos=start_pos + len(text),
                    metadata={
                        "chunker": self.name,
                        "structure_type": "list",
                        "chunk_index": chunk_index,
                    }
                )], chunk_index + 1
            else:
                return [], chunk_index  # Skip tiny lists

        # Split at list item boundaries
        item_pattern = re.compile(r'^(\s*[-*•+\d]+[.)]\s+)', re.MULTILINE)
        items = []
        last_end = 0

        for match in item_pattern.finditer(text):
            if match.start() > last_end:
                items.append(text[last_end:match.start()])
            last_end = match.start()

        if last_end < len(text):
            items.append(text[last_end:])

        # Greedily combine items into chunks
        chunks = []
        current_text = ""
        current_start = start_pos

        for item in items:
            if len(current_text + item) <= self.chunk_size:
                current_text += item
            else:
                # Emit current chunk if it meets min size
                if len(current_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                        document_id=doc_id,
                        text=current_text,
                        start_pos=current_start,
                        end_pos=current_start + len(current_text),
                        metadata={
                            "chunker": self.name,
                            "structure_type": "list",
                            "chunk_index": chunk_index,
                        }
                    ))
                    chunk_index += 1
                    current_start += len(current_text)
                    current_text = ""
                
                # Handle oversized list item
                if len(item) > self.chunk_size:
                    # Split oversized item using paragraph splitting
                    item_chunks, item_chunk_count = self._split_at_paragraphs(
                        item, start_pos=current_start, doc_id=doc_id, 
                        chunk_index=chunk_index
                    )
                    for chunk in item_chunks:
                        chunk.metadata["structure_type"] = "list_oversized"
                        chunks.append(chunk)
                    chunk_index += item_chunk_count
                    current_start += len(item)
                else:
                    current_text = item

        # Final chunk - with size validation
        if current_text.strip() and len(current_text) >= self.min_chunk_size:
            # If final chunk is oversized, split it
            if len(current_text) > self.chunk_size:
                final_chunks, final_chunk_count = self._split_at_paragraphs(
                    current_text, start_pos=current_start, doc_id=doc_id, 
                    chunk_index=chunk_index
                )
                for chunk in final_chunks:
                    chunk.metadata["structure_type"] = "list_oversized"
                    chunks.append(chunk)
                chunk_index += final_chunk_count
            else:
                chunks.append(Chunk(
                    chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                    document_id=doc_id,
                    text=current_text,
                    start_pos=current_start,
                    end_pos=current_start + len(current_text),
                    metadata={
                        "chunker": self.name,
                        "structure_type": "list",
                        "chunk_index": chunk_index,
                    }
                ))
                chunk_index += 1

        return chunks, chunk_index

    # ─── Paragraph-based splitting ───────────────────────────────────────

    def _split_at_paragraphs(
        self,
        text: str,
        start_pos: int,
        doc_id: str,
        chunk_index: int,
        section_type: str = 'paragraph_block'
    ) -> Tuple[List[Chunk], int]:
        """Split text at paragraph boundaries into chunks."""

        # Get paragraph boundaries
        para_boundaries = [0]
        for match in re.finditer(r'\n\s*\n', text):
            para_boundaries.append(match.end())
        para_boundaries.append(len(text))

        chunks = []
        current_chunk_start = 0
        current_chunk_text = ""

        # Greedily combine paragraphs up to chunk_size
        for i in range(len(para_boundaries) - 1):
            para_start = para_boundaries[i]
            para_end = para_boundaries[i + 1]
            paragraph = text[para_start:para_end]

            # CRITICAL FIX: If paragraph itself is too large, split it forcibly
            if len(paragraph) > self.chunk_size:
                # Emit current chunk first (if non-empty)
                if current_chunk_text.strip() and len(current_chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                        document_id=doc_id,
                        text=current_chunk_text,
                        start_pos=start_pos + current_chunk_start,
                        end_pos=start_pos + current_chunk_start + len(current_chunk_text),
                        metadata={
                            "chunker": self.name,
                            "structure_type": section_type,
                            "chunk_index": chunk_index,
                        }
                    ))
                    chunk_index += 1
                    current_chunk_start += len(current_chunk_text)
                    current_chunk_text = ""

                # Split this oversized paragraph into multiple chunks
                para_pos = 0
                while para_pos < len(paragraph):
                    chunk_end = min(para_pos + self.chunk_size, len(paragraph))
                    chunk_text = paragraph[para_pos:chunk_end]

                    # FIX #3: Add size validation
                    if chunk_text.strip() and len(chunk_text.strip()) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                            document_id=doc_id,
                            text=chunk_text,
                            start_pos=start_pos + current_chunk_start + para_start + para_pos,
                            end_pos=start_pos + current_chunk_start + para_start + chunk_end,
                            metadata={
                                "chunker": self.name,
                                "structure_type": section_type + "_split",
                                "chunk_index": chunk_index,
                            }
                        ))
                        chunk_index += 1

                    para_pos += self.chunk_size

                # Update position tracker
                current_chunk_start += len(paragraph)
                continue  # Skip normal processing for this paragraph

            # Normal case: Try adding this paragraph to current chunk
            candidate = current_chunk_text + paragraph

            if len(candidate) <= self.chunk_size:
                # Fits! Add to current chunk
                current_chunk_text = candidate
            else:
                # Doesn't fit. Emit current chunk, start new one
                if current_chunk_text.strip() and len(current_chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                        document_id=doc_id,
                        text=current_chunk_text,
                        start_pos=start_pos + current_chunk_start,
                        end_pos=start_pos + current_chunk_start + len(current_chunk_text),
                        metadata={
                            "chunker": self.name,
                            "structure_type": section_type,
                            "chunk_index": chunk_index,
                        }
                    ))
                    chunk_index += 1
                    current_chunk_start += len(current_chunk_text)

                # Start new chunk with current paragraph
                current_chunk_text = paragraph

        # Emit final chunk
        if current_chunk_text.strip() and len(current_chunk_text) >= self.min_chunk_size:
            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                document_id=doc_id,
                text=current_chunk_text,
                start_pos=start_pos + current_chunk_start,
                end_pos=start_pos + current_chunk_start + len(current_chunk_text),
                metadata={
                    "chunker": self.name,
                    "structure_type": section_type,
                    "chunk_index": chunk_index,
                }
            ))
            chunk_index += 1

        return chunks, chunk_index

    # ─── Main chunking logic ─────────────────────────────────────────────

    def chunk(self, text: str, document_id: Optional[str] = None) -> List[Chunk]:
        """
        Chunk text respecting structure with empirically-tuned sizes.
        
        Algorithm:
        1. Detect major sections (headings, HRs, paragraph blocks)
        2. For each section:
           - Check if it's a list → special list handling
           - If ≤ chunk_size → keep as one chunk
           - If > chunk_size → split at paragraph boundaries (FIXED)
        """
        if not text or not text.strip():
            return []

        doc_id = document_id or self._generate_document_id()
        sections = self._extract_sections(text)

        all_chunks: List[Chunk] = []
        chunk_index = 0

        for start, end, section_type in sections:
            section_text = text[start:end]
            
            if not section_text.strip():
                continue

            # Check if this is a list
            if self._is_list_section(section_text):
                section_chunks, chunk_index = self._chunk_list(
                    section_text, start, doc_id, chunk_index
                )
                all_chunks.extend(section_chunks)
            
            # Small section: keep together IF it meets minimum size
            elif len(section_text) <= self.chunk_size:
                # FIX #1: Add size validation
                if len(section_text.strip()) >= self.min_chunk_size:
                    all_chunks.append(Chunk(
                        chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                        document_id=doc_id,
                        text=section_text,
                        start_pos=start,
                        end_pos=end,
                        metadata={
                            "chunker": self.name,
                            "structure_type": section_type,
                            "chunk_index": chunk_index,
                        }
                    ))
                    chunk_index += 1
            
            # Large section: split at paragraphs (FIXED)
            else:
                # HARD CAP: If section is massive (>5x chunk_size), force-split without structure
                MAX_SECTION_SIZE = 5 * self.chunk_size  # 5120 chars for 1024 chunk_size
                
                if len(section_text) > MAX_SECTION_SIZE:
                    # Force-split monster section into fixed-size chunks
                    pos = 0
                    while pos < len(section_text):
                        chunk_end = min(pos + self.chunk_size, len(section_text))
                        chunk_text = section_text[pos:chunk_end]
                        
                        if chunk_text.strip() and len(chunk_text.strip()) >= self.min_chunk_size:
                            all_chunks.append(Chunk(
                                chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                                document_id=doc_id,
                                text=chunk_text,
                                start_pos=start + pos,
                                end_pos=start + chunk_end,
                                metadata={
                                    "chunker": self.name,
                                    "structure_type": section_type + "_force_split",
                                    "chunk_index": chunk_index,
                                }
                            ))
                            chunk_index += 1
                        pos += self.chunk_size
                else:
                    # Normal path: respect structure with paragraph splitting
                    section_chunks, chunk_index = self._split_at_paragraphs(
                        section_text, start, doc_id, chunk_index, section_type
                    )
                    all_chunks.extend(section_chunks)

        return all_chunks

    def __repr__(self) -> str:
        return (
            f"StructureAwareChunker(chunk_size={self.chunk_size}, "
            f"overlap={self.overlap}, name='{self.name}')"
        )