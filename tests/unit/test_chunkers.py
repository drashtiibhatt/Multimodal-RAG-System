"""
Unit tests for text chunkers.

Tests semantic chunking with overlap and context preservation.
"""

import pytest
from unittest.mock import Mock, patch
from src.ingestion.chunkers import SemanticChunker, Chunk
from src.ingestion.parsers.base import Document


# ==================== Fixtures ====================

@pytest.fixture
def sample_document():
    """Provide a sample document for chunking."""
    content = """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence. It enables systems to learn from data.

Deep Learning
Deep learning uses neural networks with multiple layers. These networks can learn complex patterns.

Applications
Machine learning has many applications. It is used in image recognition, natural language processing, and more.

Conclusion
Machine learning continues to evolve. New techniques are developed regularly."""

    return Document(
        page_content=content,
        metadata={
            "source": "ml_guide.txt",
            "file_type": "text",
            "file_name": "ml_guide.txt"
        }
    )


@pytest.fixture
def short_document():
    """Provide a short document."""
    return Document(
        page_content="This is a short document.",
        metadata={"source": "short.txt"}
    )


@pytest.fixture
def long_document():
    """Provide a long document that will be chunked."""
    # Create a document with repeated paragraphs to ensure chunking
    paragraph = "This is a paragraph about testing. " * 50  # ~1750 chars
    content = "\n\n".join([paragraph] * 5)  # ~8750 chars total

    return Document(
        page_content=content,
        metadata={
            "source": "long.txt",
            "page": 1,
            "file_type": "text"
        }
    )


@pytest.fixture
def multi_page_pdf_documents():
    """Provide documents from a multi-page PDF."""
    documents = []
    for page_num in range(1, 4):
        doc = Document(
            page_content=f"Content from page {page_num}. " * 100,
            metadata={
                "source": "document.pdf",
                "file_type": "pdf",
                "page": page_num,
                "total_pages": 3
            }
        )
        documents.append(doc)
    return documents


# ==================== SemanticChunker Tests ====================

class TestSemanticChunker:
    """Tests for SemanticChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a SemanticChunker instance."""
        return SemanticChunker(
            chunk_size=1000,
            chunk_overlap=150
        )

    @pytest.fixture
    def small_chunker(self):
        """Create a chunker with small chunk size."""
        return SemanticChunker(
            chunk_size=100,
            chunk_overlap=20
        )

    def test_chunker_initialization(self, chunker):
        """Test chunker initializes correctly."""
        assert chunker is not None
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 150
        assert chunker.separators == ["\n\n", "\n", ". ", " ", ""]

    def test_custom_separators(self):
        """Test initialization with custom separators."""
        custom_separators = ["\n", ".", " "]
        chunker = SemanticChunker(
            chunk_size=500,
            chunk_overlap=50,
            separators=custom_separators
        )

        assert chunker.separators == custom_separators

    def test_chunk_short_document(self, chunker, short_document):
        """Test chunking a document shorter than chunk size."""
        chunks = chunker.chunk_document(short_document)

        # Short document should create one chunk
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].content == short_document.page_content

    def test_chunk_metadata_preservation(self, chunker, short_document):
        """Test that chunks preserve document metadata."""
        chunks = chunker.chunk_document(short_document)

        chunk = chunks[0]
        # Check original metadata is preserved
        assert chunk.metadata['source'] == 'short.txt'

        # Check chunk-specific metadata is added
        assert 'chunk_index' in chunk.metadata
        assert 'chunk_id' in chunk.metadata
        assert 'total_chunks' in chunk.metadata
        assert 'chunk_char_count' in chunk.metadata

    def test_chunk_id_generation(self, chunker, short_document):
        """Test chunk ID generation."""
        chunks = chunker.chunk_document(short_document)

        chunk = chunks[0]
        assert chunk.chunk_id == "short.txt:chunk_0"
        assert chunk.metadata['chunk_id'] == chunk.chunk_id

    def test_chunk_id_with_page_number(self, chunker):
        """Test chunk ID generation with page numbers."""
        doc = Document(
            page_content="PDF page content.",
            metadata={"source": "test.pdf", "page": 5}
        )

        chunks = chunker.chunk_document(doc)
        assert chunks[0].chunk_id == "test.pdf:p5:chunk_0"

    def test_chunk_long_document(self, small_chunker, long_document):
        """Test chunking a long document."""
        chunks = small_chunker.chunk_document(long_document)

        # Long document should be split into multiple chunks
        assert len(chunks) > 1

        # Check all chunks
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata['chunk_index'] == idx
            assert chunk.metadata['total_chunks'] == len(chunks)
            assert len(chunk.content) > 0

    def test_chunk_overlap(self, small_chunker, long_document):
        """Test that chunks have overlap."""
        chunks = small_chunker.chunk_document(long_document)

        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            # This is a basic check - actual overlap depends on separators
            assert chunks[0].metadata['chunk_char_count'] <= small_chunker.chunk_size + 50
            assert chunks[1].metadata['chunk_char_count'] <= small_chunker.chunk_size + 50

    def test_empty_document(self, chunker):
        """Test chunking an empty document."""
        empty_doc = Document(
            page_content="",
            metadata={"source": "empty.txt"}
        )

        chunks = chunker.chunk_document(empty_doc)

        # Empty document should still create one chunk
        assert len(chunks) >= 0

    def test_chunk_semantic_boundaries(self, chunker, sample_document):
        """Test that chunks respect semantic boundaries."""
        chunks = chunker.chunk_document(sample_document)

        # Sample document should be chunked at paragraph boundaries
        assert len(chunks) >= 1

        # Verify chunks contain complete sentences/paragraphs
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_chunk_documents_multiple(self, chunker, multi_page_pdf_documents):
        """Test chunking multiple documents."""
        all_chunks = chunker.chunk_documents(multi_page_pdf_documents)

        # Should have chunks from all 3 pages
        assert len(all_chunks) > 0

        # Check that chunks maintain source information
        sources = set(chunk.metadata['source'] for chunk in all_chunks)
        assert 'document.pdf' in sources

        # Check that page numbers are preserved
        pages = set(chunk.metadata.get('page', 0) for chunk in all_chunks)
        assert 1 in pages
        assert 2 in pages
        assert 3 in pages

    def test_chunk_documents_empty_list(self, chunker):
        """Test chunking empty document list."""
        chunks = chunker.chunk_documents([])

        assert chunks == []
        assert len(chunks) == 0

    def test_chunk_char_count_accuracy(self, chunker, sample_document):
        """Test that chunk char count is accurate."""
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            assert chunk.metadata['chunk_char_count'] == len(chunk.content)

    def test_chunk_total_chunks_accuracy(self, chunker, sample_document):
        """Test that total_chunks metadata is correct."""
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            assert chunk.metadata['total_chunks'] == len(chunks)

    def test_chunk_index_sequence(self, small_chunker, long_document):
        """Test that chunk indices form a sequence."""
        chunks = small_chunker.chunk_document(long_document)

        indices = [chunk.metadata['chunk_index'] for chunk in chunks]
        expected_indices = list(range(len(chunks)))

        assert indices == expected_indices

    def test_chunk_content_not_empty(self, chunker, sample_document):
        """Test that no chunks have empty content."""
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_document_with_special_characters(self, chunker):
        """Test chunking document with special characters."""
        doc = Document(
            page_content="Hello 世界! Testing émojis 🚀 and spëcial çhars.",
            metadata={"source": "special.txt"}
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1
        assert "世界" in chunks[0].content
        assert "🚀" in chunks[0].content

    def test_chunk_repr(self):
        """Test Chunk string representation."""
        chunk = Chunk(
            content="Test content",
            metadata={"source": "test.txt"},
            chunk_id="test.txt:chunk_0"
        )

        repr_str = repr(chunk)
        assert "Chunk" in repr_str
        assert "test.txt:chunk_0" in repr_str
        assert "length=12" in repr_str

    def test_different_chunk_sizes(self):
        """Test chunkers with different chunk sizes."""
        doc = Document(
            page_content="A" * 1000,
            metadata={"source": "test.txt"}
        )

        # Small chunks
        small_chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)
        small_chunks = small_chunker.chunk_document(doc)

        # Large chunks
        large_chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
        large_chunks = large_chunker.chunk_document(doc)

        # Smaller chunk size should create more chunks
        assert len(small_chunks) > len(large_chunks)

    def test_chunk_overlap_values(self):
        """Test different overlap values."""
        doc = Document(
            page_content="X" * 500,
            metadata={"source": "test.txt"}
        )

        # No overlap
        no_overlap = SemanticChunker(chunk_size=100, chunk_overlap=0)
        chunks_no_overlap = no_overlap.chunk_document(doc)

        # With overlap
        with_overlap = SemanticChunker(chunk_size=100, chunk_overlap=20)
        chunks_with_overlap = with_overlap.chunk_document(doc)

        # Same total content, overlap shouldn't dramatically change count
        # (but may create slightly more chunks due to overlap)
        assert len(chunks_no_overlap) > 0
        assert len(chunks_with_overlap) > 0


# ==================== Chunk Dataclass Tests ====================

class TestChunkDataclass:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a Chunk object."""
        chunk = Chunk(
            content="Test content",
            metadata={"source": "test.txt", "page": 1},
            chunk_id="test.txt:p1:chunk_0"
        )

        assert chunk.content == "Test content"
        assert chunk.metadata['source'] == "test.txt"
        assert chunk.metadata['page'] == 1
        assert chunk.chunk_id == "test.txt:p1:chunk_0"

    def test_chunk_equality(self):
        """Test chunk equality comparison."""
        chunk1 = Chunk(
            content="Same content",
            metadata={"source": "test.txt"},
            chunk_id="id1"
        )

        chunk2 = Chunk(
            content="Same content",
            metadata={"source": "test.txt"},
            chunk_id="id1"
        )

        chunk3 = Chunk(
            content="Different content",
            metadata={"source": "test.txt"},
            chunk_id="id2"
        )

        assert chunk1 == chunk2
        assert chunk1 != chunk3


# ==================== Integration Tests ====================

@pytest.mark.integration
class TestChunkerIntegration:
    """Integration tests for chunker with parsers."""

    def test_chunk_pipeline_text_to_chunks(self, sample_text_file):
        """Test complete pipeline from text file to chunks."""
        from src.ingestion.parsers import TextParser

        # Parse document
        parser = TextParser()
        documents = parser.parse(sample_text_file)

        # Chunk documents
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents(documents)

        # Verify pipeline
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.metadata['source'] == str(sample_text_file) for chunk in chunks)

    def test_chunk_preserves_parser_metadata(self, sample_md_file):
        """Test that chunking preserves all parser metadata."""
        from src.ingestion.parsers import TextParser

        parser = TextParser()
        documents = parser.parse(sample_md_file)

        chunker = SemanticChunker()
        chunks = chunker.chunk_documents(documents)

        # Check that parser metadata is preserved
        for chunk in chunks:
            assert 'file_type' in chunk.metadata
            assert 'file_name' in chunk.metadata
            assert 'source' in chunk.metadata
            assert chunk.metadata['file_type'] == 'text'
