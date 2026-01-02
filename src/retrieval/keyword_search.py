"""
BM25 Keyword Search - Traditional keyword-based retrieval using BM25 algorithm.

This provides a complement to vector similarity search, capturing exact
keyword matches and traditional IR ranking.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """Result from BM25 keyword search."""
    score: float
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    rank: int


class KeywordSearch:
    """BM25-based keyword search for document retrieval."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25 keyword search.

        Args:
            k1: Controls term frequency saturation (default 1.5)
            b: Controls document length normalization (default 0.75)
            epsilon: Floor value for IDF (default 0.25)
        """
        if not BM25_AVAILABLE:
            raise ImportError(
                "rank-bm25 not available. Install with: pip install rank-bm25"
            )

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Storage
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunk_contents: List[str] = []
        self.chunk_ids: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

        logger.info(
            f"KeywordSearch initialized (k1={k1}, b={b}, epsilon={epsilon})"
        )

    def build_index(
        self,
        contents: List[str],
        chunk_ids: List[str],
        metadata: List[Dict[str, Any]]
    ):
        """
        Build BM25 index from document chunks.

        Args:
            contents: List of chunk text content
            chunk_ids: List of chunk IDs
            metadata: List of chunk metadata dicts

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        if not (len(contents) == len(chunk_ids) == len(metadata)):
            raise ValueError(
                f"Mismatched input lengths: {len(contents)} contents, "
                f"{len(chunk_ids)} IDs, {len(metadata)} metadata"
            )

        logger.info(f"Building BM25 index for {len(contents)} chunks")

        # Store references
        self.chunk_contents = contents
        self.chunk_ids = chunk_ids
        self.chunk_metadata = metadata

        # Tokenize corpus
        self.tokenized_corpus = [
            self._tokenize(content) for content in contents
        ]

        # Build BM25 index
        self.bm25_index = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )

        logger.info("BM25 index built successfully")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[BM25Result]:
        """
        Search for documents matching the query.

        Args:
            query: Search query text
            top_k: Number of top results to return
            min_score: Minimum BM25 score threshold

        Returns:
            List of BM25Result objects, sorted by score descending

        Raises:
            ValueError: If index not built yet
        """
        if self.bm25_index is None:
            raise ValueError("BM25 index not built. Call build_index() first.")

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = scores[idx]

            # Filter by minimum score
            if score < min_score:
                continue

            result = BM25Result(
                score=float(score),
                chunk_id=self.chunk_ids[idx],
                content=self.chunk_contents[idx],
                metadata=self.chunk_metadata[idx],
                rank=rank
            )

            results.append(result)

        logger.debug(
            f"BM25 search: query='{query[:50]}...', "
            f"found {len(results)}/{top_k} results above threshold"
        )

        return results

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Simple tokenization: lowercase, split on whitespace, remove punctuation.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Remove common punctuation
        for char in '.,;:!?()[]{}"\'-_':
            text = text.replace(char, ' ')

        # Split and filter empty
        tokens = [token for token in text.split() if token]

        return tokens

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the BM25 index.

        Returns:
            Dictionary with index statistics
        """
        if self.bm25_index is None:
            return {"indexed": False}

        return {
            "indexed": True,
            "num_documents": len(self.chunk_contents),
            "avg_doc_length": self.bm25_index.avgdl if hasattr(self.bm25_index, 'avgdl') else None,
            "parameters": {
                "k1": self.k1,
                "b": self.b,
                "epsilon": self.epsilon
            }
        }

    def save(self, save_dir: str):
        """
        Save BM25 index and data to disk.

        Args:
            save_dir: Directory to save to
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save BM25 index
        with open(save_path / "bm25_index.pkl", 'wb') as f:
            pickle.dump(self.bm25_index, f)

        # Save metadata
        with open(save_path / "bm25_metadata.pkl", 'wb') as f:
            pickle.dump({
                'chunk_contents': self.chunk_contents,
                'chunk_ids': self.chunk_ids,
                'chunk_metadata': self.chunk_metadata,
                'tokenized_corpus': self.tokenized_corpus,
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon
            }, f)

        logger.info(f"BM25 index saved to {save_dir}")

    def load(self, load_dir: str):
        """
        Load BM25 index and data from disk.

        Args:
            load_dir: Directory to load from

        Raises:
            FileNotFoundError: If index files not found
        """
        load_path = Path(load_dir)

        # Load BM25 index
        index_file = load_path / "bm25_index.pkl"
        if not index_file.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_file}")

        with open(index_file, 'rb') as f:
            self.bm25_index = pickle.load(f)

        # Load metadata
        metadata_file = load_path / "bm25_metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"BM25 metadata not found: {metadata_file}")

        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.chunk_contents = data['chunk_contents']
            self.chunk_ids = data['chunk_ids']
            self.chunk_metadata = data['chunk_metadata']
            self.tokenized_corpus = data['tokenized_corpus']
            self.k1 = data['k1']
            self.b = data['b']
            self.epsilon = data['epsilon']

        logger.info(
            f"BM25 index loaded from {load_dir} "
            f"({len(self.chunk_contents)} documents)"
        )

    @property
    def is_indexed(self) -> bool:
        """Check if index has been built."""
        return self.bm25_index is not None
