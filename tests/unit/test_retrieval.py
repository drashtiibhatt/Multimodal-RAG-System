"""
Unit tests for retrieval components.

Tests BM25 keyword search and retrieval functionality.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from src.retrieval.keyword_search import KeywordSearch, BM25Result


# ==================== Fixtures ====================

@pytest.fixture
def sample_corpus():
    """Provide sample corpus for indexing."""
    contents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks to learn patterns.",
        "Natural language processing enables machines to understand text.",
        "Computer vision allows machines to interpret images.",
        "Supervised learning requires labeled training data."
    ]

    chunk_ids = [
        "ml.txt:chunk_0",
        "dl.txt:chunk_0",
        "nlp.txt:chunk_0",
        "cv.txt:chunk_0",
        "sl.txt:chunk_0"
    ]

    metadata = [
        {"source": "ml.txt", "file_type": "text"},
        {"source": "dl.txt", "file_type": "text"},
        {"source": "nlp.txt", "file_type": "text"},
        {"source": "cv.txt", "file_type": "text"},
        {"source": "sl.txt", "file_type": "text"}
    ]

    return contents, chunk_ids, metadata


# ==================== KeywordSearch Tests ====================

@patch('src.retrieval.keyword_search.BM25_AVAILABLE', True)
class TestKeywordSearch:
    """Tests for KeywordSearch (BM25)."""

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_search_initialization(self, mock_bm25):
        """Test keyword search initializes correctly."""
        search = KeywordSearch(k1=1.5, b=0.75, epsilon=0.25)

        assert search is not None
        assert search.k1 == 1.5
        assert search.b == 0.75
        assert search.epsilon == 0.25
        assert search.bm25_index is None
        assert search.is_indexed == False

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_default_parameters(self, mock_bm25):
        """Test default parameter values."""
        search = KeywordSearch()

        assert search.k1 == 1.5
        assert search.b == 0.75
        assert search.epsilon == 0.25

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_build_index(self, mock_bm25, sample_corpus):
        """Test building BM25 index."""
        contents, chunk_ids, metadata = sample_corpus

        search = KeywordSearch()
        search.build_index(contents, chunk_ids, metadata)

        # Check index was built
        assert search.is_indexed == True
        assert len(search.chunk_contents) == 5
        assert len(search.chunk_ids) == 5
        assert len(search.chunk_metadata) == 5
        assert len(search.tokenized_corpus) == 5

        # Check BM25Okapi was called
        mock_bm25.assert_called_once()

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_build_index_mismatched_lengths(self, mock_bm25):
        """Test error when input lengths don't match."""
        search = KeywordSearch()

        with pytest.raises(ValueError, match="Mismatched input lengths"):
            search.build_index(
                contents=["text1", "text2"],
                chunk_ids=["id1"],  # Wrong length
                metadata=[{"key": "val"}]  # Wrong length
            )

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_tokenize(self, mock_bm25):
        """Test text tokenization."""
        search = KeywordSearch()

        tokens = search._tokenize("Hello, World! This is a test.")

        assert "hello" in tokens
        assert "world" in tokens
        assert "this" in tokens
        assert "test" in tokens
        # Punctuation should be removed
        assert "," not in tokens
        assert "!" not in tokens

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_tokenize_handles_punctuation(self, mock_bm25):
        """Test that tokenization removes punctuation."""
        search = KeywordSearch()

        text = "machine-learning (ML) is great; it's amazing!"
        tokens = search._tokenize(text)

        # Check lowercase
        assert all(token.islower() or token.isdigit() for token in tokens)
        # Check no standalone punctuation
        assert "(" not in tokens
        assert ")" not in tokens
        assert ";" not in tokens

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_search_without_index(self, mock_bm25):
        """Test that search fails if index not built."""
        search = KeywordSearch()

        with pytest.raises(ValueError, match="BM25 index not built"):
            search.search("machine learning")

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_search_returns_results(self, mock_bm25, sample_corpus):
        """Test search returns correctly formatted results."""
        contents, chunk_ids, metadata = sample_corpus

        # Mock BM25 scores
        mock_index = Mock()
        mock_index.get_scores.return_value = [5.0, 3.0, 4.0, 1.0, 2.0]
        mock_bm25.return_value = mock_index

        search = KeywordSearch()
        search.build_index(contents, chunk_ids, metadata)

        results = search.search("machine learning", top_k=3)

        # Check results
        assert len(results) == 3
        assert all(isinstance(r, BM25Result) for r in results)

        # Check results are sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Check ranks
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_search_with_min_score_filter(self, mock_bm25, sample_corpus):
        """Test filtering results by minimum score."""
        contents, chunk_ids, metadata = sample_corpus

        # Mock BM25 scores
        mock_index = Mock()
        mock_index.get_scores.return_value = [5.0, 3.0, 4.0, 0.5, 0.3]
        mock_bm25.return_value = mock_index

        search = KeywordSearch()
        search.build_index(contents, chunk_ids, metadata)

        # Search with min_score filter
        results = search.search("machine learning", top_k=5, min_score=2.0)

        # Only scores >= 2.0 should be included
        assert all(r.score >= 2.0 for r in results)
        assert len(results) == 3  # 5.0, 4.0, 3.0

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_search_result_structure(self, mock_bm25, sample_corpus):
        """Test BM25Result contains all required fields."""
        contents, chunk_ids, metadata = sample_corpus

        mock_index = Mock()
        mock_index.get_scores.return_value = [5.0, 3.0, 4.0, 1.0, 2.0]
        mock_bm25.return_value = mock_index

        search = KeywordSearch()
        search.build_index(contents, chunk_ids, metadata)

        results = search.search("machine learning", top_k=1)

        result = results[0]
        assert hasattr(result, 'score')
        assert hasattr(result, 'chunk_id')
        assert hasattr(result, 'content')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'rank')

        assert isinstance(result.score, float)
        assert isinstance(result.chunk_id, str)
        assert isinstance(result.content, str)
        assert isinstance(result.metadata, dict)
        assert isinstance(result.rank, int)

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_get_stats_not_indexed(self, mock_bm25):
        """Test get_stats when index not built."""
        search = KeywordSearch()

        stats = search.get_stats()

        assert stats['indexed'] == False

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_get_stats_indexed(self, mock_bm25, sample_corpus):
        """Test get_stats when index is built."""
        contents, chunk_ids, metadata = sample_corpus

        search = KeywordSearch(k1=1.2, b=0.8, epsilon=0.3)
        search.build_index(contents, chunk_ids, metadata)

        stats = search.get_stats()

        assert stats['indexed'] == True
        assert stats['num_documents'] == 5
        assert stats['parameters']['k1'] == 1.2
        assert stats['parameters']['b'] == 0.8
        assert stats['parameters']['epsilon'] == 0.3

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_save_and_load(self, mock_bm25, sample_corpus, temp_dir):
        """Test saving and loading BM25 index."""
        contents, chunk_ids, metadata = sample_corpus

        # Build and save index
        search1 = KeywordSearch(k1=1.3, b=0.7, epsilon=0.2)
        search1.build_index(contents, chunk_ids, metadata)
        search1.save(str(temp_dir))

        # Check files were created
        assert (temp_dir / "bm25_index.pkl").exists()
        assert (temp_dir / "bm25_metadata.pkl").exists()

        # Load index into new instance
        search2 = KeywordSearch()
        search2.load(str(temp_dir))

        # Check loaded data
        assert search2.is_indexed == True
        assert len(search2.chunk_contents) == 5
        assert search2.k1 == 1.3
        assert search2.b == 0.7
        assert search2.epsilon == 0.2

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_load_nonexistent_index(self, mock_bm25, temp_dir):
        """Test loading from nonexistent directory."""
        search = KeywordSearch()

        nonexistent_dir = temp_dir / "nonexistent"

        with pytest.raises(FileNotFoundError):
            search.load(str(nonexistent_dir))

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_empty_query(self, mock_bm25, sample_corpus):
        """Test handling of empty query."""
        contents, chunk_ids, metadata = sample_corpus

        mock_index = Mock()
        mock_index.get_scores.return_value = [0.0] * 5
        mock_bm25.return_value = mock_index

        search = KeywordSearch()
        search.build_index(contents, chunk_ids, metadata)

        results = search.search("", top_k=5, min_score=0.0)

        # Empty query should still work (may return low scores)
        assert isinstance(results, list)

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_unicode_query(self, mock_bm25, sample_corpus):
        """Test handling of unicode in queries."""
        contents, chunk_ids, metadata = sample_corpus

        mock_index = Mock()
        mock_index.get_scores.return_value = [1.0, 2.0, 3.0, 0.5, 0.3]
        mock_bm25.return_value = mock_index

        search = KeywordSearch()
        search.build_index(contents, chunk_ids, metadata)

        # Should handle unicode without errors
        results = search.search("Hello 世界 🚀", top_k=3)

        assert isinstance(results, list)

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_is_indexed_property(self, mock_bm25, sample_corpus):
        """Test is_indexed property."""
        contents, chunk_ids, metadata = sample_corpus

        search = KeywordSearch()

        # Before building index
        assert search.is_indexed == False

        # After building index
        search.build_index(contents, chunk_ids, metadata)
        assert search.is_indexed == True


@patch('src.retrieval.keyword_search.BM25_AVAILABLE', False)
class TestKeywordSearchWithoutBM25:
    """Test KeywordSearch when BM25 library is not available."""

    def test_initialization_fails_without_bm25(self):
        """Test that initialization fails when BM25 not available."""
        with pytest.raises(ImportError, match="rank-bm25 not available"):
            KeywordSearch()


# ==================== BM25Result Dataclass Tests ====================

class TestBM25Result:
    """Tests for BM25Result dataclass."""

    def test_result_creation(self):
        """Test creating BM25Result."""
        result = BM25Result(
            score=5.2,
            chunk_id="test.txt:chunk_0",
            content="Test content",
            metadata={"source": "test.txt"},
            rank=1
        )

        assert result.score == 5.2
        assert result.chunk_id == "test.txt:chunk_0"
        assert result.content == "Test content"
        assert result.metadata["source"] == "test.txt"
        assert result.rank == 1

    def test_result_comparison(self):
        """Test comparing results by score."""
        result1 = BM25Result(
            score=5.0,
            chunk_id="id1",
            content="content1",
            metadata={},
            rank=1
        )

        result2 = BM25Result(
            score=3.0,
            chunk_id="id2",
            content="content2",
            metadata={},
            rank=2
        )

        # Can compare by score
        assert result1.score > result2.score


# ==================== Integration Tests ====================

@pytest.mark.integration
@patch('src.retrieval.keyword_search.BM25_AVAILABLE', True)
class TestRetrievalIntegration:
    """Integration tests for retrieval with chunking."""

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_full_retrieval_pipeline(self, mock_bm25, sample_text):
        """Test complete retrieval pipeline."""
        from src.ingestion.parsers.base import Document
        from src.ingestion.chunkers import SemanticChunker

        # Create and chunk document
        doc = Document(
            page_content=sample_text,
            metadata={"source": "test.txt"}
        )

        chunker = SemanticChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)

        # Build search index
        contents = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]

        # Mock BM25 scores
        mock_index = Mock()
        num_chunks = len(chunks)
        mock_index.get_scores.return_value = [float(i) for i in range(num_chunks, 0, -1)]
        mock_bm25.return_value = mock_index

        search = KeywordSearch()
        search.build_index(contents, chunk_ids, metadata)

        # Perform search
        results = search.search("user", top_k=3)

        # Verify pipeline
        assert len(results) > 0
        assert all(isinstance(r, BM25Result) for r in results)
        assert all(r.metadata['source'] == 'test.txt' for r in results)

    @patch('src.retrieval.keyword_search.BM25Okapi')
    def test_search_maintains_chunk_metadata(self, mock_bm25, sample_text):
        """Test that search results maintain chunk metadata."""
        from src.ingestion.parsers.base import Document
        from src.ingestion.chunkers import SemanticChunker

        doc = Document(
            page_content=sample_text,
            metadata={"source": "test.txt", "custom_field": "custom_value"}
        )

        chunker = SemanticChunker()
        chunks = chunker.chunk_document(doc)

        # Build index
        contents = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]

        mock_index = Mock()
        mock_index.get_scores.return_value = [1.0] * len(chunks)
        mock_bm25.return_value = mock_index

        search = KeywordSearch()
        search.build_index(contents, chunk_ids, metadata)

        results = search.search("test", top_k=1)

        # Check metadata preserved
        result = results[0]
        assert 'custom_field' in result.metadata
        assert result.metadata['custom_field'] == "custom_value"
        assert 'chunk_index' in result.metadata
