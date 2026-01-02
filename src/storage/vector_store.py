"""FAISS-based vector storage for embeddings."""

from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class SearchResult:
    """Represents a search result with score and metadata."""

    score: float
    chunk_id: str
    content: str
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        """String representation."""
        return f"SearchResult(score={self.score:.4f}, chunk_id={self.chunk_id})"


class VectorStore:
    """
    FAISS-based vector store for storing and retrieving embeddings.

    Provides methods to add vectors, search for similar vectors,
    and persist/load the index from disk.
    """

    def __init__(self, dimension: int = 1536):
        """
        Initialize vector store.

        Args:
            dimension: Dimension of embedding vectors (1536 for OpenAI text-embedding-3-small)
        """
        self.dimension = dimension

        # Create FAISS index (using Inner Product for cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)

        # Store metadata for each chunk
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.chunk_contents: List[str] = []
        self.chunk_ids: List[str] = []

    def add_vectors(
        self,
        embeddings: np.ndarray,
        contents: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Add vectors to the index.

        Args:
            embeddings: Numpy array of embeddings (n_vectors, dimension)
            contents: List of text contents corresponding to embeddings
            metadata: List of metadata dicts for each embedding

        Raises:
            ValueError: If lengths don't match or embeddings have wrong dimension
        """
        if len(embeddings) != len(contents) or len(embeddings) != len(metadata):
            raise ValueError("Embeddings, contents, and metadata must have same length")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embeddings dimension {embeddings.shape[1]} doesn't match expected {self.dimension}")

        # Convert to float32 and ensure C-contiguous for FAISS compatibility
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata and contents
        self.chunk_contents.extend(contents)
        self.chunk_metadata.extend(metadata)

        # Extract and store chunk IDs
        for meta in metadata:
            chunk_id = meta.get("chunk_id", f"chunk_{len(self.chunk_ids)}")
            self.chunk_ids.append(chunk_id)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query embedding vector (1, dimension)
            top_k: Number of results to return

        Returns:
            List of SearchResult objects sorted by similarity score

        Raises:
            ValueError: If query embedding has wrong shape
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} doesn't match expected {self.dimension}")

        if self.index.ntotal == 0:
            return []

        # Convert to float32 and ensure C-contiguous for FAISS compatibility
        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))

        # Normalize query for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append(SearchResult(
                    score=float(score),
                    chunk_id=self.chunk_ids[idx],
                    content=self.chunk_contents[idx],
                    metadata=self.chunk_metadata[idx]
                ))

        return results

    def save(self, save_dir: str) -> None:
        """
        Save index and metadata to disk.

        Args:
            save_dir: Directory to save files
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = save_path / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_path = save_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunk_metadata': self.chunk_metadata,
                'chunk_contents': self.chunk_contents,
                'chunk_ids': self.chunk_ids,
                'dimension': self.dimension
            }, f)

        print(f"[OK] Vector store saved to {save_dir}")
        print(f"  - Total vectors: {self.index.ntotal}")
        print(f"  - Dimension: {self.dimension}")

    def load(self, load_dir: str) -> None:
        """
        Load index and metadata from disk.

        Args:
            load_dir: Directory to load files from

        Raises:
            FileNotFoundError: If files don't exist
        """
        load_path = Path(load_dir)

        # Load FAISS index
        index_path = load_path / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = load_path / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)

        self.chunk_metadata = data['chunk_metadata']
        self.chunk_contents = data['chunk_contents']
        self.chunk_ids = data['chunk_ids']
        self.dimension = data['dimension']

        print(f"[OK] Vector store loaded from {load_dir}")
        print(f"  - Total vectors: {self.index.ntotal}")
        print(f"  - Dimension: {self.dimension}")

    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_metadata = []
        self.chunk_contents = []
        self.chunk_ids = []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_chunks": len(self.chunk_ids),
            "unique_sources": len(set(meta.get("source", "") for meta in self.chunk_metadata))
        }

    def find_by_source(self, source: str) -> List[int]:
        """
        Find indices of all chunks from a specific source.

        Args:
            source: Source identifier to search for

        Returns:
            List of indices matching the source
        """
        indices = []
        for idx, meta in enumerate(self.chunk_metadata):
            if meta.get("source") == source:
                indices.append(idx)
        return indices

    def find_by_chunk_id(self, chunk_id: str) -> int:
        """
        Find index of a chunk by its ID.

        Args:
            chunk_id: Chunk ID to search for

        Returns:
            Index of the chunk, or -1 if not found
        """
        try:
            return self.chunk_ids.index(chunk_id)
        except ValueError:
            return -1

    def delete_by_indices(self, indices_to_delete: List[int]) -> int:
        """
        Delete vectors by their indices.

        Note: This rebuilds the FAISS index without the deleted vectors.

        Args:
            indices_to_delete: List of indices to delete

        Returns:
            Number of vectors deleted
        """
        if not indices_to_delete:
            return 0

        # Sort in descending order to avoid index shifting issues
        indices_to_delete = sorted(set(indices_to_delete), reverse=True)

        # Validate indices
        max_idx = len(self.chunk_ids) - 1
        valid_indices = [idx for idx in indices_to_delete if 0 <= idx <= max_idx]

        if not valid_indices:
            return 0

        # Get all vectors from current index
        all_vectors = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
        for i in range(self.index.ntotal):
            all_vectors[i] = self.index.reconstruct(int(i))

        # Create new lists without deleted items
        new_vectors = []
        new_contents = []
        new_metadata = []
        new_ids = []

        for idx in range(len(self.chunk_ids)):
            if idx not in valid_indices:
                new_vectors.append(all_vectors[idx])
                new_contents.append(self.chunk_contents[idx])
                new_metadata.append(self.chunk_metadata[idx])
                new_ids.append(self.chunk_ids[idx])

        # Rebuild index
        self.index = faiss.IndexFlatIP(self.dimension)
        if new_vectors:
            vectors_array = np.array(new_vectors, dtype=np.float32)
            self.index.add(vectors_array)

        # Update metadata
        self.chunk_contents = new_contents
        self.chunk_metadata = new_metadata
        self.chunk_ids = new_ids

        return len(valid_indices)

    def delete_by_source(self, source: str) -> int:
        """
        Delete all vectors from a specific source.

        Args:
            source: Source identifier

        Returns:
            Number of vectors deleted
        """
        indices = self.find_by_source(source)
        if not indices:
            print(f"[WARN] No vectors found for source: {source}")
            return 0

        count = self.delete_by_indices(indices)
        print(f"[OK] Deleted {count} vectors from source: {source}")
        return count

    def delete_by_chunk_id(self, chunk_id: str) -> bool:
        """
        Delete a single vector by chunk ID.

        Args:
            chunk_id: Chunk ID to delete

        Returns:
            True if deleted, False if not found
        """
        idx = self.find_by_chunk_id(chunk_id)
        if idx == -1:
            print(f"[WARN] Chunk ID not found: {chunk_id}")
            return False

        count = self.delete_by_indices([idx])
        if count > 0:
            print(f"[OK] Deleted chunk: {chunk_id}")
            return True
        return False

    def update_by_source(
        self,
        source: str,
        new_embeddings: np.ndarray,
        new_contents: List[str],
        new_metadata: List[Dict[str, Any]]
    ) -> int:
        """
        Update all vectors from a specific source.

        This deletes existing vectors from the source and adds new ones.

        Args:
            source: Source identifier to update
            new_embeddings: New embeddings for the source
            new_contents: New contents
            new_metadata: New metadata

        Returns:
            Number of vectors updated
        """
        # Delete existing vectors
        deleted_count = self.delete_by_source(source)

        # Add new vectors
        self.add_vectors(new_embeddings, new_contents, new_metadata)

        print(f"[OK] Updated source '{source}': deleted {deleted_count}, added {len(new_contents)} vectors")
        return len(new_contents)

    def update_by_chunk_id(
        self,
        chunk_id: str,
        new_embedding: np.ndarray,
        new_content: str,
        new_metadata: Dict[str, Any]
    ) -> bool:
        """
        Update a single chunk by its ID.

        Args:
            chunk_id: Chunk ID to update
            new_embedding: New embedding vector
            new_content: New content
            new_metadata: New metadata

        Returns:
            True if updated, False if not found
        """
        # Delete existing chunk
        if not self.delete_by_chunk_id(chunk_id):
            return False

        # Add new version
        if new_embedding.ndim == 1:
            new_embedding = new_embedding.reshape(1, -1)

        self.add_vectors(new_embedding, [new_content], [new_metadata])

        print(f"[OK] Updated chunk: {chunk_id}")
        return True

    def list_sources(self) -> List[str]:
        """
        Get list of all unique sources in the vector store.

        Returns:
            List of source identifiers
        """
        sources = set()
        for meta in self.chunk_metadata:
            source = meta.get("source")
            if source:
                sources.add(source)
        return sorted(list(sources))

    def get_source_stats(self, source: str) -> Dict[str, Any]:
        """
        Get statistics for a specific source.

        Args:
            source: Source identifier

        Returns:
            Dictionary with source statistics
        """
        indices = self.find_by_source(source)
        chunks = [self.chunk_metadata[idx] for idx in indices]

        return {
            "source": source,
            "total_chunks": len(indices),
            "chunk_ids": [self.chunk_ids[idx] for idx in indices]
        }
