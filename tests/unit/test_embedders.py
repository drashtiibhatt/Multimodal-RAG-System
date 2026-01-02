"""
Unit tests for embedding generation.

Tests OpenAI-based embedding generation for text chunks.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.ingestion.embedders import EmbeddingGenerator
from src.ingestion.chunkers import Chunk


# ==================== Fixtures ====================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()

    # Mock embedding response
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1] * 1536),
        Mock(embedding=[0.2] * 1536),
        Mock(embedding=[0.3] * 1536)
    ]

    mock_client.embeddings.create.return_value = mock_response

    return mock_client


@pytest.fixture
def sample_chunks():
    """Provide sample chunks for embedding."""
    return [
        Chunk(
            content="Machine learning is a subset of AI.",
            metadata={"source": "ml.txt"},
            chunk_id="ml.txt:chunk_0"
        ),
        Chunk(
            content="Deep learning uses neural networks.",
            metadata={"source": "ml.txt"},
            chunk_id="ml.txt:chunk_1"
        ),
        Chunk(
            content="Natural language processing enables text understanding.",
            metadata={"source": "nlp.txt"},
            chunk_id="nlp.txt:chunk_0"
        )
    ]


# ==================== EmbeddingGenerator Tests ====================

class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator."""

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_generator_initialization(self, mock_openai, mock_settings):
        """Test embedding generator initializes correctly."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"

        generator = EmbeddingGenerator()

        assert generator is not None
        assert generator.model == "text-embedding-3-small"
        mock_openai.assert_called_once_with(api_key="test_key")

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_generator_custom_model(self, mock_openai, mock_settings):
        """Test initialization with custom model."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"

        generator = EmbeddingGenerator(model="text-embedding-3-large")

        assert generator.model == "text-embedding-3-large"

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_generate_embeddings(self, mock_openai, mock_settings, mock_openai_client):
        """Test generating embeddings for text list."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"
        mock_openai.return_value = mock_openai_client

        generator = EmbeddingGenerator()
        texts = ["Text 1", "Text 2", "Text 3"]

        embeddings = generator.generate(texts)

        # Check embeddings shape
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 1536)

        # Check API was called correctly
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts
        )

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_generate_empty_list(self, mock_openai, mock_settings):
        """Test generating embeddings for empty list."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"

        generator = EmbeddingGenerator()
        embeddings = generator.generate([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0,)

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_generate_for_single_chunk(self, mock_openai, mock_settings, mock_openai_client, sample_chunks):
        """Test generating embedding for a single chunk."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"
        mock_openai.return_value = mock_openai_client

        # Mock response for single chunk
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.5] * 1536)]
        mock_openai_client.embeddings.create.return_value = mock_response

        generator = EmbeddingGenerator()
        chunk = sample_chunks[0]

        embedding = generator.generate_for_chunk(chunk)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1536,)
        assert np.allclose(embedding, [0.5] * 1536)

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_generate_for_chunks(self, mock_openai, mock_settings, mock_openai_client, sample_chunks):
        """Test generating embeddings for multiple chunks."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"
        mock_openai.return_value = mock_openai_client

        generator = EmbeddingGenerator()
        embeddings = generator.generate_for_chunks(sample_chunks)

        # Check embeddings shape
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 1536)

        # Check API was called with correct texts
        called_texts = mock_openai_client.embeddings.create.call_args[1]['input']
        expected_texts = [chunk.content for chunk in sample_chunks]
        assert called_texts == expected_texts

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_api_error_handling(self, mock_openai, mock_settings):
        """Test handling of API errors."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator()

        with pytest.raises(Exception, match="Failed to generate embeddings"):
            generator.generate(["Test text"])

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_get_embedding_dimension(self, mock_openai, mock_settings, mock_openai_client):
        """Test getting embedding dimension."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"

        # Mock response for dimension check
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.0] * 1536)]
        mock_openai_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_openai_client

        generator = EmbeddingGenerator()
        dimension = generator.get_embedding_dimension()

        assert dimension == 1536

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_embeddings_are_normalized(self, mock_openai, mock_settings, mock_openai_client):
        """Test that embeddings maintain expected properties."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"

        # Create realistic embedding values
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=list(np.random.randn(1536))),
            Mock(embedding=list(np.random.randn(1536)))
        ]
        mock_openai_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_openai_client

        generator = EmbeddingGenerator()
        embeddings = generator.generate(["Text 1", "Text 2"])

        # Check embeddings are 2D array
        assert len(embeddings.shape) == 2
        assert embeddings.shape[1] == 1536

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_unicode_text_handling(self, mock_openai, mock_settings, mock_openai_client):
        """Test handling of unicode text."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"
        mock_openai.return_value = mock_openai_client

        generator = EmbeddingGenerator()
        unicode_texts = [
            "Hello 世界",
            "Bonjour émoji 🚀",
            "Привет мир"
        ]

        embeddings = generator.generate(unicode_texts)

        assert embeddings.shape == (3, 1536)

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_long_text_handling(self, mock_openai, mock_settings, mock_openai_client):
        """Test handling of very long text."""
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"
        mock_openai.return_value = mock_openai_client

        generator = EmbeddingGenerator()
        long_text = "This is a test. " * 1000  # ~16000 chars

        embeddings = generator.generate([long_text])

        # Should still generate embeddings (OpenAI handles truncation)
        assert embeddings.shape == (1, 1536)


# ==================== Integration Tests ====================

@pytest.mark.integration
class TestEmbeddingIntegration:
    """Integration tests for embeddings with chunkers."""

    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    def test_full_chunk_to_embedding_pipeline(
        self,
        mock_openai,
        mock_settings,
        mock_openai_client,
        sample_text
    ):
        """Test complete pipeline from text to chunks to embeddings."""
        from src.ingestion.parsers.base import Document
        from src.ingestion.chunkers import SemanticChunker

        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"
        mock_openai.return_value = mock_openai_client

        # Create document
        doc = Document(
            page_content=sample_text,
            metadata={"source": "test.txt"}
        )

        # Chunk document
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)

        # Generate embeddings
        generator = EmbeddingGenerator()
        embeddings = generator.generate_for_chunks(chunks)

        # Verify pipeline
        assert len(chunks) > 0
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == 1536
