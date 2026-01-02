"""
Hybrid Retrieval Fusion - Combines vector and keyword search using RRF.

Implements Reciprocal Rank Fusion (RRF) algorithm to merge results from
multiple retrieval methods, improving both recall and precision.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .engine import SearchResult
from .keyword_search import BM25Result

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid retrieval with fusion scores."""
    fusion_score: float
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    sources: List[str] = None  # Which retrievers found this result

    def __post_init__(self):
        if self.sources is None:
            self.sources = []


class HybridRetrieval:
    """
    Hybrid retrieval combining vector similarity and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from multiple
    retrieval methods with configurable weights.
    """

    def __init__(
        self,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retrieval.

        Args:
            vector_weight: Weight for vector search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            rrf_k: RRF constant (default 60 from literature)

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        if abs(vector_weight + keyword_weight - 1.0) > 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got {vector_weight + keyword_weight}"
            )

        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

        logger.info(
            f"HybridRetrieval initialized: "
            f"vector={vector_weight:.2f}, keyword={keyword_weight:.2f}, k={rrf_k}"
        )

    def fuse(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[BM25Result],
        top_k: int = 5
    ) -> List[HybridResult]:
        """
        Fuse vector and keyword search results using RRF.

        Args:
            vector_results: Results from vector similarity search
            keyword_results: Results from BM25 keyword search
            top_k: Number of top results to return

        Returns:
            List of HybridResult objects, sorted by fusion score
        """
        logger.debug(
            f"Fusing {len(vector_results)} vector + {len(keyword_results)} keyword results"
        )

        # Calculate RRF scores for each unique chunk
        fusion_scores: Dict[str, Dict[str, Any]] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.chunk_id

            if chunk_id not in fusion_scores:
                fusion_scores[chunk_id] = {
                    'content': result.content,
                    'metadata': result.metadata,
                    'vector_score': result.score,
                    'keyword_score': None,
                    'vector_rank': rank,
                    'keyword_rank': None,
                    'sources': ['vector'],
                    'rrf_score': 0.0
                }

            # RRF formula: weight / (k + rank)
            rrf_contribution = self.vector_weight / (self.rrf_k + rank)
            fusion_scores[chunk_id]['rrf_score'] += rrf_contribution

        # Process keyword results
        for rank, result in enumerate(keyword_results, start=1):
            chunk_id = result.chunk_id

            if chunk_id not in fusion_scores:
                fusion_scores[chunk_id] = {
                    'content': result.content,
                    'metadata': result.metadata,
                    'vector_score': None,
                    'keyword_score': result.score,
                    'vector_rank': None,
                    'keyword_rank': rank,
                    'sources': ['keyword'],
                    'rrf_score': 0.0
                }
            else:
                # Update if chunk found by both methods
                fusion_scores[chunk_id]['keyword_score'] = result.score
                fusion_scores[chunk_id]['keyword_rank'] = rank
                fusion_scores[chunk_id]['sources'].append('keyword')

            # RRF formula: weight / (k + rank)
            rrf_contribution = self.keyword_weight / (self.rrf_k + rank)
            fusion_scores[chunk_id]['rrf_score'] += rrf_contribution

        # Sort by fusion score and take top-k
        sorted_results = sorted(
            fusion_scores.items(),
            key=lambda x: x[1]['rrf_score'],
            reverse=True
        )[:top_k]

        # Convert to HybridResult objects
        hybrid_results = []
        for chunk_id, data in sorted_results:
            result = HybridResult(
                fusion_score=data['rrf_score'],
                chunk_id=chunk_id,
                content=data['content'],
                metadata=data['metadata'],
                vector_score=data['vector_score'],
                keyword_score=data['keyword_score'],
                vector_rank=data['vector_rank'],
                keyword_rank=data['keyword_rank'],
                sources=data['sources']
            )
            hybrid_results.append(result)

        logger.debug(
            f"Fusion complete: {len(hybrid_results)} results, "
            f"{sum(1 for r in hybrid_results if len(r.sources) > 1)} found by both methods"
        )

        return hybrid_results

    def get_stats(self, results: List[HybridResult]) -> Dict[str, Any]:
        """
        Get statistics about fusion results.

        Args:
            results: List of HybridResult objects

        Returns:
            Dictionary with fusion statistics
        """
        if not results:
            return {
                "total_results": 0,
                "found_by_both": 0,
                "vector_only": 0,
                "keyword_only": 0
            }

        found_by_both = sum(1 for r in results if len(r.sources) > 1)
        vector_only = sum(1 for r in results if r.sources == ['vector'])
        keyword_only = sum(1 for r in results if r.sources == ['keyword'])

        return {
            "total_results": len(results),
            "found_by_both": found_by_both,
            "vector_only": vector_only,
            "keyword_only": keyword_only,
            "avg_fusion_score": sum(r.fusion_score for r in results) / len(results),
            "weights": {
                "vector": self.vector_weight,
                "keyword": self.keyword_weight
            },
            "rrf_k": self.rrf_k
        }

    def explain_result(self, result: HybridResult) -> str:
        """
        Generate human-readable explanation of a hybrid result.

        Args:
            result: HybridResult to explain

        Returns:
            Formatted explanation string
        """
        explanation = [
            f"Fusion Score: {result.fusion_score:.4f}",
            f"Sources: {', '.join(result.sources)}"
        ]

        if result.vector_score is not None:
            explanation.append(
                f"Vector: rank #{result.vector_rank}, "
                f"score {result.vector_score:.4f}"
            )

        if result.keyword_score is not None:
            explanation.append(
                f"Keyword: rank #{result.keyword_rank}, "
                f"score {result.keyword_score:.4f}"
            )

        return " | ".join(explanation)

    def adjust_weights(self, vector_weight: float, keyword_weight: float):
        """
        Adjust retrieval weights.

        Args:
            vector_weight: New weight for vector search
            keyword_weight: New weight for keyword search

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        if abs(vector_weight + keyword_weight - 1.0) > 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got {vector_weight + keyword_weight}"
            )

        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

        logger.info(
            f"Weights adjusted: vector={vector_weight:.2f}, "
            f"keyword={keyword_weight:.2f}"
        )
