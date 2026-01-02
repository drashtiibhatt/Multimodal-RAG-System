"""Simple script to query the RAG system."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import RAGSystem


def main():
    """Query RAG system from command line argument."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/query.py <query>")
        print('Example: python scripts/query.py "Create use cases for user signup"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print("Initializing RAG system...")
    rag = RAGSystem()

    print(f"Querying: {query}")
    rag.query(query, debug=False)


if __name__ == "__main__":
    main()
