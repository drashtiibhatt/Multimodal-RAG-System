"""Generate embeddings for text chunks using OpenAI."""

from typing import List, Optional
import numpy as np
from openai import OpenAI
from ..chunkers.semantic_chunker import Chunk
from ...config import get_settings
from ...caching import EmbeddingCache


class EmbeddingGenerator:
    """Generates embeddings for text chunks using OpenAI embedding models."""

    def __init__(self, model: str = None, use_cache: bool = True):
        """
        Initialize embedding generator.

        Args:
            model: OpenAI embedding model name (defaults to config setting)
            use_cache: Whether to use caching for embeddings
        """
        settings = get_settings()
        self.model = model or settings.embedding_model
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.use_cache = use_cache
        self.cache = EmbeddingCache() if use_cache else None

    def generate(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts with caching support.

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)

        Raises:
            Exception: If API call fails
        """
        if not texts:
            return np.array([])

        # Check cache if enabled
        if self.use_cache and self.cache:
            embeddings_list = []
            texts_to_generate = []
            text_indices = []

            for idx, text in enumerate(texts):
                cached_embedding = self.cache.get(text)
                if cached_embedding is not None:
                    embeddings_list.append((idx, cached_embedding))
                else:
                    texts_to_generate.append(text)
                    text_indices.append(idx)

            # Generate embeddings for uncached texts
            if texts_to_generate:
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=texts_to_generate
                    )

                    for i, data in enumerate(response.data):
                        embedding = data.embedding
                        # Cache the new embedding
                        self.cache.set(texts_to_generate[i], embedding)
                        embeddings_list.append((text_indices[i], embedding))

                except Exception as e:
                    raise Exception(f"Failed to generate embeddings: {str(e)}")

            # Sort by original index and extract embeddings
            embeddings_list.sort(key=lambda x: x[0])
            embeddings = np.array([emb for _, emb in embeddings_list])

        else:
            # No cache - generate all embeddings
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )

                embeddings = np.array([
                    data.embedding for data in response.data
                ])

            except Exception as e:
                raise Exception(f"Failed to generate embeddings: {str(e)}")

        return embeddings

    def generate_for_chunk(self, chunk: Chunk) -> np.ndarray:
        """
        Generate embedding for a single chunk.

        Args:
            chunk: Chunk object to embed

        Returns:
            Numpy array with single embedding
        """
        embeddings = self.generate([chunk.content])
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def generate_for_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Generate embeddings for multiple chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            Numpy array of embeddings
        """
        texts = [chunk.content for chunk in chunks]
        return self.generate(texts)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for this model.

        Returns:
            Embedding dimension (1536 for text-embedding-3-small)
        """
        # Generate a test embedding to get dimension
        test_embedding = self.generate(["test"])
        return test_embedding.shape[1] if len(test_embedding.shape) > 1 else test_embedding.shape[0]

    def get_cache_stats(self) -> Optional[dict]:
        """
        Get embedding cache statistics.

        Returns:
            Cache statistics dict or None if cache disabled
        """
        if self.cache:
            return self.cache.get_stats()
        return None

    def clear_cache(self) -> int:
        """
        Clear embedding cache.

        Returns:
            Number of cache entries cleared
        """
        if self.cache:
            return self.cache.clear()
        return 0
