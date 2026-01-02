"""Retrieval engine for finding relevant chunks."""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

from ..storage import VectorStore, SearchResult
from ..ingestion.embedders import EmbeddingGenerator
from ..config import get_settings
from .keyword_search import KeywordSearch
from .hybrid_fusion import HybridRetrieval, HybridResult

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Retrieval engine for finding relevant context chunks.

    Supports:
    - Vector similarity search (Phase 1)
    - BM25 keyword search (Phase 2)
    - Hybrid fusion using RRF (Phase 2)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingGenerator = None,
        keyword_search: Optional[KeywordSearch] = None,
        enable_hybrid: bool = False
    ):
        """
        Initialize retrieval engine.

        Args:
            vector_store: Vector store containing indexed chunks
            embedder: Embedding generator (creates new one if not provided)
            keyword_search: Optional BM25 keyword search engine
            enable_hybrid: Whether to use hybrid retrieval by default
        """
        self.vector_store = vector_store
        self.embedder = embedder or EmbeddingGenerator()
        self.keyword_search = keyword_search
        self.enable_hybrid = enable_hybrid and keyword_search is not None
        self.settings = get_settings()

        # Initialize hybrid retrieval if enabled
        if self.enable_hybrid:
            vector_weight = getattr(self.settings, 'vector_weight', 0.6)
            keyword_weight = getattr(self.settings, 'keyword_weight', 0.4)
            rrf_k = getattr(self.settings, 'rrf_constant', 60)

            self.hybrid_retrieval = HybridRetrieval(
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                rrf_k=rrf_k
            )

            logger.info("RetrievalEngine initialized with hybrid mode enabled")
        else:
            self.hybrid_retrieval = None
            logger.info("RetrievalEngine initialized with vector-only mode")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None,
        debug: bool = False,
        use_hybrid: bool = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query string
            top_k: Number of results to return (defaults to config)
            min_score: Minimum similarity score (defaults to config)
            debug: Whether to print debug information
            use_hybrid: Override hybrid mode setting (None uses default)

        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Use config defaults if not specified
        top_k = top_k or self.settings.top_k
        min_score = min_score or self.settings.min_confidence

        # Determine if using hybrid mode
        use_hybrid = use_hybrid if use_hybrid is not None else self.enable_hybrid

        if debug:
            mode = "hybrid" if use_hybrid else "vector-only"
            print(f"\n[QUERY] Retrieving context for query: '{query}'")
            print(f"Mode: {mode}, top_k={top_k}, min_score={min_score}")

        # Hybrid retrieval
        if use_hybrid and self.keyword_search and self.keyword_search.is_indexed:
            return self._retrieve_hybrid(query, top_k, min_score, debug)

        # Vector-only retrieval (fallback or default)
        else:
            if use_hybrid and debug:
                print("[WARN] Hybrid mode requested but BM25 index not available, using vector-only")

            return self._retrieve_vector_only(query, top_k, min_score, debug)

    def _retrieve_vector_only(
        self,
        query: str,
        top_k: int,
        min_score: float,
        debug: bool
    ) -> List[SearchResult]:
        """Vector similarity search only."""

        # Step 1: Generate query embedding
        query_embedding = self.embedder.generate([query])

        if debug:
            print(f"[OK] Generated query embedding (shape: {query_embedding.shape})")

        # Step 2: Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        if debug:
            print(f"[OK] Found {len(results)} initial results from vector search")

        # Step 3: Filter by minimum score
        filtered_results = [
            result for result in results
            if result.score >= min_score
        ]

        if debug:
            print(f"[OK] {len(filtered_results)} results after filtering (min_score={min_score})")
            if filtered_results:
                print("\nTop results:")
                for i, result in enumerate(filtered_results[:3], 1):
                    print(f"  {i}. Score: {result.score:.4f}")
                    print(f"     Source: {result.metadata.get('source', 'unknown')}")
                    print(f"     Preview: {result.content[:100]}...")
                    print()

        return filtered_results

    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        min_score: float,
        debug: bool
    ) -> List[SearchResult]:
        """Hybrid retrieval using vector + keyword search with RRF fusion."""

        # Step 1: Vector search
        query_embedding = self.embedder.generate([query])
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2  # Get more candidates for fusion
        )

        if debug:
            print(f"[OK] Vector search: {len(vector_results)} results")

        # Step 2: Keyword search
        keyword_results = self.keyword_search.search(
            query=query,
            top_k=top_k * 2,  # Get more candidates for fusion
            min_score=0.0
        )

        if debug:
            print(f"[OK] Keyword search: {len(keyword_results)} results")

        # Step 3: Fuse results
        hybrid_results = self.hybrid_retrieval.fuse(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=top_k
        )

        if debug:
            print(f"[OK] Hybrid fusion: {len(hybrid_results)} results")
            stats = self.hybrid_retrieval.get_stats(hybrid_results)
            print(f"  - Found by both: {stats['found_by_both']}")
            print(f"  - Vector only: {stats['vector_only']}")
            print(f"  - Keyword only: {stats['keyword_only']}")

            if hybrid_results:
                print("\nTop hybrid results:")
                for i, result in enumerate(hybrid_results[:3], 1):
                    print(f"  {i}. {self.hybrid_retrieval.explain_result(result)}")
                    print(f"     Source: {result.metadata.get('source', 'unknown')}")
                    print(f"     Preview: {result.content[:100]}...")
                    print()

        # Convert HybridResults to SearchResults for compatibility
        search_results = [
            SearchResult(
                score=result.fusion_score,
                chunk_id=result.chunk_id,
                content=result.content,
                metadata=result.metadata
            )
            for result in hybrid_results
        ]

        return search_results

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None,
        debug: bool = False,
        use_hybrid: bool = None
    ) -> Dict[str, Any]:
        """
        Retrieve chunks and format as context for LLM.

        Args:
            query: User query string
            top_k: Number of results to return
            min_score: Minimum similarity score
            debug: Whether to print debug information
            use_hybrid: Override hybrid mode setting

        Returns:
            Dictionary with results and formatted context
        """
        # Retrieve chunks
        results = self.retrieve(query, top_k, min_score, debug, use_hybrid)

        # Format context for LLM
        context = self._format_context(results)

        # Calculate confidence
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0

        return {
            "results": results,
            "context": context,
            "num_chunks": len(results),
            "avg_score": avg_score,
            "has_sufficient_context": avg_score >= min_score if min_score else True
        }

    def _format_context(self, results: List[SearchResult]) -> str:
        """
        Format search results as context string for LLM.

        Args:
            results: List of search results

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = []

        for idx, result in enumerate(results, 1):
            source = result.metadata.get("source", "unknown")
            page = result.metadata.get("page", "")
            page_ref = f" (page {page})" if page else ""

            context_parts.append(
                f"[Document {idx}] {source}{page_ref}\n"
                f"{result.content}\n"
            )

        return "\n---\n\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = {
            "vector_store_stats": self.vector_store.get_stats(),
            "top_k": self.settings.top_k,
            "min_confidence": self.settings.min_confidence,
            "hybrid_mode_enabled": self.enable_hybrid
        }

        # Add keyword search stats if available
        if self.keyword_search and self.keyword_search.is_indexed:
            stats["keyword_search_stats"] = self.keyword_search.get_stats()

        # Add hybrid retrieval stats if available
        if self.hybrid_retrieval:
            stats["hybrid_weights"] = {
                "vector": self.hybrid_retrieval.vector_weight,
                "keyword": self.hybrid_retrieval.keyword_weight,
                "rrf_k": self.hybrid_retrieval.rrf_k
            }

        return stats
