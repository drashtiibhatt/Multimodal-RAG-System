"""Main CLI interface for the RAG system."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime

from .ingestion import IngestionPipeline
from .retrieval import RetrievalEngine, KeywordSearch
from .generation import UseCaseGenerator, SafeGenerator
from .storage import VectorStore
from .config import get_settings


class RAGSystem:
    """Complete RAG system for test case generation."""

    def __init__(self, enable_hybrid: bool = None, enable_guards: bool = False):
        """
        Initialize RAG system.

        Args:
            enable_hybrid: Override hybrid mode setting (None uses config)
            enable_guards: Enable safety guardrails (default: False)
        """
        self.settings = get_settings()

        # Determine hybrid mode
        if enable_hybrid is None:
            enable_hybrid = self.settings.enable_hybrid_retrieval

        # Initialize vector store
        self.vector_store = VectorStore()

        # Phase 2: Initialize keyword search if hybrid enabled
        self.keyword_search = None
        if enable_hybrid:
            try:
                self.keyword_search = KeywordSearch()
            except ImportError:
                print("[WARN]  BM25 dependencies not available, hybrid mode disabled")
                enable_hybrid = False

        # Initialize pipeline
        self.pipeline = IngestionPipeline(
            vector_store=self.vector_store,
            keyword_search=self.keyword_search,
            build_bm25=enable_hybrid
        )

        # Initialize retrieval engine
        self.retrieval = RetrievalEngine(
            vector_store=self.vector_store,
            keyword_search=self.keyword_search,
            enable_hybrid=enable_hybrid
        )

        # Phase 3: Initialize generator with or without guards
        self.enable_guards = enable_guards
        if enable_guards:
            try:
                self.generator = SafeGenerator(use_nli=False)  # Use fast mode by default
                print("[OK] Safety guardrails enabled")
            except Exception as e:
                print(f"[WARN]  Failed to enable guards: {e}")
                print("   Falling back to standard generator")
                self.generator = UseCaseGenerator()
                self.enable_guards = False
        else:
            self.generator = UseCaseGenerator()

        # Track modes
        self.hybrid_mode = enable_hybrid

        # Try to load existing vector store
        vector_db_path = Path(self.settings.vector_db_path)
        if vector_db_path.exists() and (vector_db_path / "faiss.index").exists():
            try:
                self.vector_store.load(str(vector_db_path))

                # Load BM25 index if available
                if self.keyword_search:
                    try:
                        self.keyword_search.load(str(vector_db_path))
                        print(f"[OK] Loaded vector store + BM25 index from {vector_db_path}")
                    except FileNotFoundError:
                        print(f"[OK] Loaded vector store from {vector_db_path} (BM25 index not found)")
                else:
                    print(f"[OK] Loaded vector store from {vector_db_path}")

            except Exception as e:
                print(f"[WARN]  Could not load vector store: {str(e)}")

    def ingest(self, folder_path: str, save: bool = True) -> None:
        """
        Ingest documents from folder.

        Args:
            folder_path: Path to folder containing documents
            save: Whether to save vector store after ingestion
        """
        print(f"\n{'='*60}")
        print("[INGEST] INGESTION MODE")
        print(f"{'='*60}")

        # Ingest folder
        stats = self.pipeline.ingest_folder(folder_path)

        # Save vector store
        if save and stats.get("chunks_created", 0) > 0:
            self.pipeline.save_vector_store()

    def query(self, query_text: str, debug: bool = False, use_hybrid: bool = None) -> Dict:
        """
        Query the RAG system.

        Args:
            query_text: User query
            debug: Whether to show debug information
            use_hybrid: Override hybrid mode (None uses default)

        Returns:
            Dictionary with generated use cases
        """
        # Determine hybrid mode for this query
        if use_hybrid is None:
            use_hybrid = self.hybrid_mode

        mode = "HYBRID" if use_hybrid else "VECTOR-ONLY"

        print(f"\n{'='*60}")
        print(f"[QUERY] QUERY MODE ({mode})")
        print(f"{'='*60}")
        print(f"Query: {query_text}\n")

        # Check if vector store has data
        stats = self.vector_store.get_stats()
        if stats["total_vectors"] == 0:
            print("[ERROR] No documents in vector store. Please ingest documents first.")
            print("   Run: python src/main.py --ingest data/sample_dataset/user-signup/")
            return {
                "error": "No documents in vector store",
                "use_cases": []
            }

        # Step 1: Retrieve relevant chunks
        retrieval_result = self.retrieval.retrieve_with_context(
            query=query_text,
            debug=debug,
            use_hybrid=use_hybrid
        )

        # Step 2: Generate use cases
        safety_report = None
        if self.enable_guards:
            output, safety_report = self.generator.generate(
                query=query_text,
                context=retrieval_result["context"],
                debug=debug
            )
        else:
            output = self.generator.generate(
                query=query_text,
                context=retrieval_result["context"],
                debug=debug
            )

        # Display results
        self._display_results(output, retrieval_result, debug, safety_report)

        return output

    def _display_results(
        self,
        output: Dict,
        retrieval_result: Dict,
        debug: bool,
        safety_report = None
    ) -> None:
        """Display query results."""

        print(f"\n{'='*60}")
        print("[RESULTS] RESULTS")
        print(f"{'='*60}")

        # Show safety report if guards enabled
        if safety_report is not None:
            print(f"\n[SAFETY] SAFETY CHECK")
            if safety_report.passed:
                print("[OK] All safety checks passed")
            else:
                print("[WARN]  Safety checks failed:")
                for reason in safety_report.blocked_reasons:
                    print(f"  - {reason}")

            if safety_report.warnings:
                print("\nWarnings:")
                for warning in safety_report.warnings:
                    print(f"  - {warning}")

        # Show retrieval stats
        print(f"\nRetrieved chunks: {retrieval_result['num_chunks']}")
        print(f"Average score: {retrieval_result['avg_score']:.4f}")

        # Check for insufficient context
        if output.get("insufficient_context"):
            print("\n[WARN]  INSUFFICIENT CONTEXT")
            print("\nClarifying Questions:")
            for q in output.get("clarifying_questions", []):
                print(f"  - {q}")

            print("\nMissing Information:")
            for info in output.get("missing_information", []):
                print(f"  - {info}")
            return

        # Display use cases
        use_cases = output.get("use_cases", [])
        print(f"\nGenerated use cases: {len(use_cases)}")

        if output.get("assumptions"):
            print("\nAssumptions:")
            for assumption in output["assumptions"]:
                print(f"  - {assumption}")

        print(f"\n{'='*60}")
        print("[SUMMARY] USE CASES (Summary)")
        print(f"{'='*60}")

        for idx, uc in enumerate(use_cases, 1):
            print(f"\n{idx}. {uc.get('title', 'Untitled')}")
            print(f"   Goal: {uc.get('goal', 'N/A')}")
            print(f"   Steps: {len(uc.get('steps', []))}")
            print(f"   Negative cases: {len(uc.get('negative_cases', []))}")
            print(f"   Boundary cases: {len(uc.get('boundary_cases', []))}")

        # Show full JSON if debug
        if debug:
            print(f"\n{'='*60}")
            print("[DEBUG] FULL OUTPUT (JSON)")
            print(f"{'='*60}")
            print(json.dumps(output, indent=2, default=str))


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Multimodal RAG System for Test Case Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest documents
  python src/main.py --ingest data/sample_dataset/user-signup/

  # Query the system (vector-only)
  python src/main.py --query "Create use cases for user signup"

  # Query with hybrid retrieval (vector + keyword)
  python src/main.py --query "Create use cases for user signup" --hybrid

  # Query with safety guardrails enabled
  python src/main.py --query "Create use cases for user signup" --enable-guards

  # Query with all features (hybrid + guards + debug)
  python src/main.py --query "Generate negative test cases" --hybrid --enable-guards --debug

  # Show statistics
  python src/main.py --stats
        """
    )

    parser.add_argument(
        "--ingest",
        type=str,
        metavar="FOLDER",
        help="Ingest documents from folder"
    )

    parser.add_argument(
        "--query",
        type=str,
        metavar="QUERY",
        help="Query the RAG system"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (show retrieved chunks and full output)"
    )

    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid retrieval (vector + keyword search)"
    )

    parser.add_argument(
        "--enable-guards",
        action="store_true",
        help="Enable safety guardrails (injection detection, hallucination checking, output validation)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show vector store statistics"
    )

    parser.add_argument(
        "--output",
        type=str,
        metavar="FILE",
        help="Save output to JSON file"
    )

    args = parser.parse_args()

    # Initialize RAG system
    try:
        # Use hybrid mode if flag is set
        enable_hybrid = args.hybrid if hasattr(args, 'hybrid') else None
        # Use guards if flag is set
        enable_guards = getattr(args, 'enable_guards', False)
        rag = RAGSystem(enable_hybrid=enable_hybrid, enable_guards=enable_guards)
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAG system: {str(e)}")
        print("\nMake sure you have:")
        print("  1. Created .env file with OPENAI_API_KEY")
        print("  2. Installed dependencies: pip install -r requirements.txt")
        sys.exit(1)

    # Execute commands
    if args.ingest:
        rag.ingest(args.ingest)

    elif args.query:
        output = rag.query(args.query, debug=args.debug, use_hybrid=args.hybrid if hasattr(args, 'hybrid') else None)

        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            print(f"\n[OK] Output saved to {args.output}")

    elif args.stats:
        stats = rag.vector_store.get_stats()
        print(f"\n{'='*60}")
        print("[RESULTS] VECTOR STORE STATISTICS")
        print(f"{'='*60}")
        for key, value in stats.items():
            print(f"{key}: {value}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
