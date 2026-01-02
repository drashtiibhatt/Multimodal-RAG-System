"""Simple script to ingest documents."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import RAGSystem


def main():
    """Ingest documents from command line argument."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <folder_path>")
        print("Example: python scripts/ingest.py data/sample_dataset/user-signup/")
        sys.exit(1)

    folder_path = sys.argv[1]

    print("Initializing RAG system...")
    rag = RAGSystem()

    print(f"Ingesting documents from: {folder_path}")
    rag.ingest(folder_path)


if __name__ == "__main__":
    main()
