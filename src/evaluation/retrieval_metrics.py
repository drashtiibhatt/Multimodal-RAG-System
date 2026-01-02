"""
Retrieval Metrics - Evaluate quality of document retrieval.

Implements standard IR metrics:
- Recall@k: Proportion of relevant docs in top-k
- Precision@k: Proportion of top-k that are relevant
- MRR (Mean Reciprocal Rank): Reciprocal of rank of first relevant doc
- NDCG@k (Normalized Discounted Cumulative Gain): Relevance-weighted ranking quality
"""

from typing import List, Dict, Any, Set
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval evaluation."""
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    num_queries: int
    avg_retrieved: float


class RetrievalMetrics:
    """
    Evaluate retrieval quality using standard IR metrics.

    Metrics implemented:
    - Recall@k: Coverage of relevant documents
    - Precision@k: Accuracy of retrieved documents
    - MRR: Quality of ranking
    - NDCG@k: Graded relevance ranking quality
    """

    def __init__(self, k_values: List[int] = None):
        """
        Initialize retrieval metrics.

        Args:
            k_values: List of k values for @k metrics (default: [1, 3, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]
        logger.info(f"RetrievalMetrics initialized with k_values={self.k_values}")

    def evaluate(
        self,
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        relevance_scores: List[List[float]] = None
    ) -> RetrievalResult:
        """
        Evaluate retrieval performance across multiple queries.

        Args:
            retrieved_docs: List of retrieved document IDs for each query
            relevant_docs: List of relevant document IDs for each query
            relevance_scores: Optional graded relevance scores (0-1) for NDCG

        Returns:
            RetrievalResult with all computed metrics

        Raises:
            ValueError: If input lists have different lengths
        """
        if len(retrieved_docs) != len(relevant_docs):
            raise ValueError(
                f"Mismatch: {len(retrieved_docs)} retrieved, "
                f"{len(relevant_docs)} relevant"
            )

        num_queries = len(retrieved_docs)

        # Calculate metrics for each query
        recall_scores = {k: [] for k in self.k_values}
        precision_scores = {k: [] for k in self.k_values}
        mrr_scores = []
        ndcg_scores = {k: [] for k in self.k_values}

        for idx, (retrieved, relevant) in enumerate(zip(retrieved_docs, relevant_docs)):
            # Get relevance scores for this query (if provided)
            query_relevance = relevance_scores[idx] if relevance_scores else None

            # Calculate metrics
            for k in self.k_values:
                recall_scores[k].append(
                    self._recall_at_k(retrieved, relevant, k)
                )
                precision_scores[k].append(
                    self._precision_at_k(retrieved, relevant, k)
                )
                ndcg_scores[k].append(
                    self._ndcg_at_k(retrieved, relevant, k, query_relevance)
                )

            mrr_scores.append(self._mrr(retrieved, relevant))

        # Aggregate across queries
        recall_at_k = {k: np.mean(scores) for k, scores in recall_scores.items()}
        precision_at_k = {k: np.mean(scores) for k, scores in precision_scores.items()}
        mrr = np.mean(mrr_scores)
        ndcg_at_k = {k: np.mean(scores) for k, scores in ndcg_scores.items()}

        avg_retrieved = np.mean([len(docs) for docs in retrieved_docs])

        logger.info(
            f"Retrieval evaluation complete: "
            f"{num_queries} queries, "
            f"Recall@5={recall_at_k.get(5, 0):.3f}, "
            f"MRR={mrr:.3f}"
        )

        return RetrievalResult(
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            num_queries=num_queries,
            avg_retrieved=avg_retrieved
        )

    def _recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@k.

        Recall@k = (# relevant docs in top-k) / (total # relevant docs)

        Args:
            retrieved: Retrieved document IDs
            relevant: Relevant document IDs
            k: Number of top documents to consider

        Returns:
            Recall@k score (0-1)
        """
        if not relevant:
            return 0.0

        top_k = set(retrieved[:k])
        relevant_set = set(relevant)

        num_relevant_retrieved = len(top_k.intersection(relevant_set))

        return num_relevant_retrieved / len(relevant_set)

    def _precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@k.

        Precision@k = (# relevant docs in top-k) / k

        Args:
            retrieved: Retrieved document IDs
            relevant: Relevant document IDs
            k: Number of top documents to consider

        Returns:
            Precision@k score (0-1)
        """
        if k == 0:
            return 0.0

        top_k = set(retrieved[:k])
        relevant_set = set(relevant)

        num_relevant_retrieved = len(top_k.intersection(relevant_set))

        return num_relevant_retrieved / min(k, len(retrieved))

    def _mrr(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.

        MRR = 1 / (rank of first relevant document)

        Args:
            retrieved: Retrieved document IDs (in ranked order)
            relevant: Relevant document IDs

        Returns:
            MRR score (0-1)
        """
        relevant_set = set(relevant)

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    def _ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int,
        relevance_scores: List[float] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@k.

        NDCG@k = DCG@k / IDCG@k

        DCG@k = sum(relevance[i] / log2(i+1) for i in 1..k)
        IDCG@k = DCG@k for perfect ranking

        Args:
            retrieved: Retrieved document IDs
            relevant: Relevant document IDs
            k: Number of top documents to consider
            relevance_scores: Graded relevance scores (0-1), if available

        Returns:
            NDCG@k score (0-1)
        """
        if not relevant:
            return 0.0

        # Create relevance mapping
        if relevance_scores is not None:
            relevance_map = {doc_id: score
                           for doc_id, score in zip(retrieved, relevance_scores)}
        else:
            # Binary relevance: 1 if relevant, 0 otherwise
            relevant_set = set(relevant)
            relevance_map = {doc_id: 1.0 if doc_id in relevant_set else 0.0
                           for doc_id in retrieved}

        # Calculate DCG@k
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            relevance = relevance_map.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 1)

        # Calculate IDCG@k (perfect ranking)
        # Sort by relevance descending
        if relevance_scores is not None:
            ideal_relevance = sorted(
                [relevance_map.get(doc_id, 0.0) for doc_id in retrieved],
                reverse=True
            )[:k]
        else:
            # Binary case: all relevant docs first
            num_relevant = min(len(relevant), k)
            ideal_relevance = [1.0] * num_relevant + [0.0] * (k - num_relevant)

        idcg = 0.0
        for i, relevance in enumerate(ideal_relevance, start=1):
            idcg += relevance / np.log2(i + 1)

        # Normalize
        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def evaluate_single_query(
        self,
        retrieved: List[str],
        relevant: List[str],
        relevance_scores: List[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single query.

        Args:
            retrieved: Retrieved document IDs
            relevant: Relevant document IDs
            relevance_scores: Optional graded relevance scores

        Returns:
            Dictionary with all metrics for this query
        """
        metrics = {}

        for k in self.k_values:
            metrics[f"recall@{k}"] = self._recall_at_k(retrieved, relevant, k)
            metrics[f"precision@{k}"] = self._precision_at_k(retrieved, relevant, k)
            metrics[f"ndcg@{k}"] = self._ndcg_at_k(
                retrieved, relevant, k, relevance_scores
            )

        metrics["mrr"] = self._mrr(retrieved, relevant)

        return metrics

    def get_summary(self, result: RetrievalResult) -> str:
        """
        Get human-readable summary of results.

        Args:
            result: RetrievalResult to summarize

        Returns:
            Formatted summary string
        """
        lines = [
            f"Retrieval Evaluation Summary ({result.num_queries} queries)",
            "=" * 60,
            f"Average documents retrieved: {result.avg_retrieved:.1f}",
            "",
            "Recall@k:",
        ]

        for k in sorted(result.recall_at_k.keys()):
            lines.append(f"  @{k:2d}: {result.recall_at_k[k]:.4f}")

        lines.append("")
        lines.append("Precision@k:")

        for k in sorted(result.precision_at_k.keys()):
            lines.append(f"  @{k:2d}: {result.precision_at_k[k]:.4f}")

        lines.append("")
        lines.append(f"MRR: {result.mrr:.4f}")
        lines.append("")
        lines.append("NDCG@k:")

        for k in sorted(result.ndcg_at_k.keys()):
            lines.append(f"  @{k:2d}: {result.ndcg_at_k[k]:.4f}")

        return "\n".join(lines)
