"""Build BM25 index from existing vector store for hybrid retrieval."""

from pathlib import Path
import pickle
import faiss
from rank_bm25 import BM25Okapi

def tokenize(text: str):
    """Simple tokenization."""
    return text.lower().split()

def main():
    """Build BM25 index from existing vector store."""

    print("\n" + "="*60)
    print("[BM25] Building BM25 Index for Hybrid Retrieval")
    print("="*60)

    # Path to vector store
    vector_db_path = Path("data/storage/vector_db")

    # Step 1: Load metadata from vector store
    print(f"\n[1/3] Loading vector store metadata from {vector_db_path}...")
    metadata_path = vector_db_path / "metadata.pkl"

    if not metadata_path.exists():
        print("[ERROR] Vector store metadata not found. Please run ingestion first.")
        return

    with open(metadata_path, 'rb') as f:
        data = pickle.load(f)

    chunk_contents = data['chunk_contents']
    chunk_ids = data['chunk_ids']
    chunk_metadata = data['chunk_metadata']

    print(f"[OK] Loaded {len(chunk_contents)} chunks")

    # Step 2: Build BM25 index
    print(f"\n[2/3] Building BM25 index...")

    # Tokenize corpus
    tokenized_corpus = [tokenize(content) for content in chunk_contents]

    # Create BM25 index
    bm25_index = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75, epsilon=0.25)

    print(f"[OK] BM25 index built with {len(tokenized_corpus)} documents")

    # Step 3: Save BM25 index (in the format expected by KeywordSearch class)
    print(f"\n[3/3] Saving BM25 index to {vector_db_path}...")

    # Save BM25 index
    with open(vector_db_path / "bm25_index.pkl", 'wb') as f:
        pickle.dump(bm25_index, f)

    # Save metadata
    with open(vector_db_path / "bm25_metadata.pkl", 'wb') as f:
        pickle.dump({
            'chunk_contents': chunk_contents,
            'chunk_ids': chunk_ids,
            'chunk_metadata': chunk_metadata,
            'tokenized_corpus': tokenized_corpus,
            'k1': 1.5,
            'b': 0.75,
            'epsilon': 0.25
        }, f)

    print(f"[OK] BM25 index saved to {vector_db_path}")

    # Test search
    print("\n" + "="*60)
    print("[TEST] Testing BM25 Search")
    print("="*60)

    test_query = "user signup authentication"
    tokenized_query = tokenize(test_query)
    scores = bm25_index.get_scores(tokenized_query)

    # Get top 3 results
    top_k = 3
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    print(f"\nQuery: '{test_query}'")
    print(f"Results: {len(top_indices)}")

    for rank, idx in enumerate(top_indices, 1):
        print(f"\n{rank}. Score: {scores[idx]:.4f}")
        print(f"   Chunk ID: {chunk_ids[idx]}")
        print(f"   Source: {chunk_metadata[idx].get('source', 'unknown')}")
        print(f"   Preview: {chunk_contents[idx][:100]}...")

    print("\n" + "="*60)
    print("[SUCCESS] BM25 index ready for hybrid retrieval!")
    print("="*60)
    print("\nYou can now use --hybrid flag in queries:")
    print("  python -m src.main --query 'your query' --hybrid")
    print()

if __name__ == "__main__":
    main()
