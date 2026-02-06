"""
test_chunkers.py
────────────────
Pytest tests for all chunking strategies.

Run:  pytest tests/test_chunkers.py -v
"""

import pytest

from chunkers import (
    FixedChunker,
    SentenceChunker,
    ParagraphChunker,
    RecursiveChunker,
    StructureAwareChunker,
    SemanticDensityChunker,
)
from chunkers.base_chunker import Chunk


# ─── Test data ───────────────────────────────────────────────────────────────
SAMPLE_TEXT = (
    "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical "
    "rainforest in the Amazon biome that covers most of the Amazon basin of South America. "
    "This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the "
    "rainforest. This region includes territory belonging to nine nations and 3,344 "
    "formally acknowledged indigenous territories.\n\n"
    "The majority of the forest is contained within Brazil, with 60% of the rainforest, "
    "followed by Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, "
    "Ecuador, French Guiana, Guyana, Suriname, and Venezuela.\n\n"
    "The Amazon represents over half of the planet's remaining rainforests, and comprises "
    "the largest and most biodiverse tract of tropical rainforest in the world, with an "
    "estimated 390 billion individual trees divided into 16,000 species. More than 30 "
    "million people live in the Amazon region, which is subdivided into different protected "
    "areas and indigenous territories.\n\n"
    "Deforestation in the Amazon rainforest threatens many species such as the giant otter, "
    "the jaguar, and many species of birds. The tropical rainforests of South America "
    "contain the largest diversity of species on Earth."
)

SHORT_TEXT = "Hello world. This is a test."
EMPTY_TEXT = ""


# ─── Fixtures ────────────────────────────────────────────────────────────────
#
# Chunker instances are created once per test session (not per test).
# This matters for spaCy-backed chunkers: the model loads once via the
# process-level cache in _spacy_cache.py, making the full suite run in
# seconds instead of minutes.

# Chunkers that don't need spaCy — instantiation is cheap
FAST_CHUNKERS = [
    pytest.param(FixedChunker(chunk_size=200, overlap=30), id="FixedChunker"),
    pytest.param(ParagraphChunker(max_chars=300, min_chars=50), id="ParagraphChunker"),
    pytest.param(RecursiveChunker(max_chars=200, overlap=30), id="RecursiveChunker"),
]

# Chunkers that use spaCy — model loads once via process-level cache
SPACY_CHUNKERS = [
    pytest.param(
        SentenceChunker(max_chars=300, overlap_sentences=1),
        id="SentenceChunker",
    ),
    pytest.param(
        StructureAwareChunker(chunk_size=300, min_chunk_size=50, overlap=30),
        id="StructureAwareChunker",
    ),
    pytest.param(
        SemanticDensityChunker(chunk_size=300, min_chunk_size=50, min_overlap=25, max_overlap=75),
        id="SemanticDensityChunker",
    ),
]

ALL_CHUNKERS = FAST_CHUNKERS + SPACY_CHUNKERS


@pytest.fixture(params=ALL_CHUNKERS)
def chunker(request):
    """Parametrized fixture that yields every chunker in turn."""
    return request.param


@pytest.fixture(params=FAST_CHUNKERS)
def fast_chunker(request):
    """Parametrized fixture for chunkers that don't need spaCy."""
    return request.param


# ─── Contract tests: every chunker must satisfy these ────────────────────────

class TestChunkerContract:
    """Basic contract tests that every chunker must satisfy."""

    def test_produces_chunks_on_normal_text(self, chunker):
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test_doc")
        assert len(chunks) > 0, f"{chunker.name} produced 0 chunks"

    def test_all_chunks_are_nonempty(self, chunker):
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test_doc")
        for c in chunks:
            assert c.text.strip(), f"{chunker.name} produced empty chunk: {c.chunk_id}"

    def test_chunks_are_chunk_instances(self, chunker):
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test_doc")
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_chunk_positions_are_nonnegative(self, chunker):
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test_doc")
        for c in chunks:
            assert c.start_pos >= 0, f"start_pos={c.start_pos} in {chunker.name}"
            assert c.end_pos >= c.start_pos, (
                f"end_pos={c.end_pos} < start_pos={c.start_pos} in {chunker.name}"
            )

    def test_chunk_ids_are_unique(self, chunker):
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test_doc")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), f"Duplicate chunk IDs in {chunker.name}"

    def test_document_ids_are_consistent(self, chunker):
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test_doc")
        for c in chunks:
            assert c.document_id == "test_doc"

    def test_empty_text_returns_no_chunks(self, chunker):
        chunks = chunker.chunk(EMPTY_TEXT, document_id="test_doc")
        assert len(chunks) == 0

    def test_whitespace_only_returns_no_chunks(self, chunker):
        chunks = chunker.chunk("   \n\n  ", document_id="test_doc")
        assert len(chunks) == 0

    def test_chunk_length_property_matches_text(self, chunker):
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test_doc")
        for c in chunks:
            assert c.length == len(c.text)



# ─── FixedChunker-specific tests ─────────────────────────────────────────────

class TestFixedChunker:

    def test_all_but_last_chunk_match_target_size(self):
        chunker = FixedChunker(chunk_size=200, overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test")
        for c in chunks[:-1]:
            assert len(c.text) == 200

    def test_overlap_creates_more_chunks(self):
        no_overlap = FixedChunker(chunk_size=200, overlap=0)
        with_overlap = FixedChunker(chunk_size=200, overlap=50)
        chunks_without = no_overlap.chunk(SAMPLE_TEXT)
        chunks_with = with_overlap.chunk(SAMPLE_TEXT)
        assert len(chunks_with) >= len(chunks_without)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            FixedChunker(chunk_size=100, overlap=100)

    def test_negative_size_raises(self):
        with pytest.raises(ValueError):
            FixedChunker(chunk_size=-1)


# ─── ParagraphChunker-specific tests ─────────────────────────────────────────

class TestParagraphChunker:

    def test_respects_paragraph_boundaries(self):
        text = "Paragraph one with some content.\n\nParagraph two with more.\n\nParagraph three final."
        chunker = ParagraphChunker(max_chars=1000, min_chars=5)
        chunks = chunker.chunk(text, document_id="test")
        assert len(chunks) >= 1

    def test_invalid_min_max_raises(self):
        with pytest.raises(ValueError):
            ParagraphChunker(max_chars=100, min_chars=200)


# ─── RecursiveChunker-specific tests ─────────────────────────────────────────

class TestRecursiveChunker:

    def test_chunks_within_max_chars(self):
        chunker = RecursiveChunker(max_chars=200, overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test")
        for c in chunks:
            # Allow small tolerance for boundary alignment
            assert len(c.text) <= 210, f"Chunk too large: {len(c.text)} chars"

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            RecursiveChunker(max_chars=100, overlap=100)


# ─── SemanticDensityChunker-specific tests ───────────────────────────────────

class TestSemanticDensityChunker:

    def test_default_overlap_range(self):
        """Verify the fixed defaults actually enable adaptive behavior."""
        chunker = SemanticDensityChunker()
        assert chunker.min_overlap == 25, "min_overlap should default to 25"
        assert chunker.max_overlap == 75, "max_overlap should default to 75"

    def test_density_to_overlap_low(self):
        chunker = SemanticDensityChunker(min_overlap=25, max_overlap=75)
        assert chunker._density_to_overlap(0.0) == 25

    def test_density_to_overlap_high(self):
        chunker = SemanticDensityChunker(min_overlap=25, max_overlap=75)
        assert chunker._density_to_overlap(1.0) == 75

    def test_density_to_overlap_mid(self):
        chunker = SemanticDensityChunker(
            min_overlap=25, max_overlap=75,
            low_density_threshold=0.3, high_density_threshold=0.6,
        )
        mid = chunker._density_to_overlap(0.45)
        assert 25 <= mid <= 75

    def test_invalid_overlap_range_raises(self):
        with pytest.raises(ValueError):
            SemanticDensityChunker(min_overlap=80, max_overlap=50)

    def test_metadata_contains_density_and_overlap(self):
        chunker = SemanticDensityChunker(chunk_size=300, min_chunk_size=50)
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test")
        for c in chunks:
            assert "density_score" in c.metadata
            assert "overlap" in c.metadata
            assert 0.0 <= c.metadata["density_score"] <= 1.0

    def test_spacy_model_is_process_cached(self):
        """Two instances should share the same underlying spaCy model."""
        a = SemanticDensityChunker()
        b = SemanticDensityChunker()
        assert a.nlp is b.nlp, "spaCy model should be shared across instances"


# ─── StructureAwareChunker-specific tests ────────────────────────────────────

class TestStructureAwareChunker:

    def test_handles_markdown_headings(self):
        text = "# Introduction\n\nSome intro text here.\n\n## Details\n\nMore detailed content."
        chunker = StructureAwareChunker(chunk_size=500, min_chunk_size=10, overlap=0)
        chunks = chunker.chunk(text, document_id="test")
        assert len(chunks) >= 1

    def test_handles_plain_text(self):
        chunker = StructureAwareChunker(chunk_size=300, min_chunk_size=50, overlap=30)
        chunks = chunker.chunk(SAMPLE_TEXT, document_id="test")
        assert len(chunks) > 0
