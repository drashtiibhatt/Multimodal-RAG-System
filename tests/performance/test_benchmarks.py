"""
Performance benchmarks for RAG system.

Measures throughput, latency, and resource usage for key operations:
- Document parsing and chunking
- Embedding generation (batched)
- Vector search (various k values)
- End-to-end query processing
"""

import pytest
import time
import psutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.ingestion.parsers import TextParser
from src.ingestion.parsers.base import Document
from src.ingestion.chunkers import SemanticChunker
from src.storage import VectorStore
from src.retrieval import RetrievalEngine


# ==================== Fixtures ====================

@pytest.fixture
def large_document():
    """Create large document for performance testing."""
    # 10,000 words (~50KB)
    content = " ".join([f"Word{i}" for i in range(10000)])
    content = f"""
Authentication System Specification

{content}

Requirements:
Email validation, password strength, session management, two-factor authentication.

{content}

Security features include bcrypt hashing, account lockout, HTTPS enforcement.
"""
    return Document(
        page_content=content,
        metadata={"source": "large_spec.txt", "file_type": "text"}
    )


@pytest.fixture
def batch_documents():
    """Create batch of documents for throughput testing."""
    documents = []
    for i in range(100):
        content = f"""
Document {i}: This is a test document with some content about feature {i}.
It contains multiple sentences and paragraphs for realistic testing.
The content is varied enough to generate different embeddings.
"""
        documents.append(Document(
            page_content=content,
            metadata={"source": f"doc_{i}.txt", "file_type": "text"}
        ))
    return documents


@pytest.fixture
def populated_vector_store():
    """Create vector store with 1000 vectors for search benchmarking."""
    vector_store = VectorStore()

    # Generate 1000 random normalized vectors
    num_vectors = 1000
    dimension = 1536

    chunks = []
    embeddings = []

    for i in range(num_vectors):
        chunk = Document(
            page_content=f"This is chunk {i} with some content",
            metadata={"chunk_index": i, "source": f"doc_{i//10}.txt"}
        )
        chunk.chunk_id = f"chunk_{i}"
        chunks.append(chunk)

        # Random normalized embedding
        embedding = np.random.randn(dimension)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)

    embeddings_array = np.array(embeddings, dtype=np.float32)
    contents = [chunk.page_content for chunk in chunks]
    metadata_list = [chunk.metadata for chunk in chunks]
    vector_store.add_vectors(embeddings_array, contents, metadata_list)

    return vector_store


# ==================== Helper Functions ====================

class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def measure_throughput(operation, items: List, batch_size: int = 10) -> Dict[str, float]:
    """
    Measure throughput of batch operation.

    Args:
        operation: Callable that processes items
        items: List of items to process
        batch_size: Batch size for processing

    Returns:
        Dict with throughput metrics
    """
    total_items = len(items)
    start_time = time.perf_counter()
    start_memory = get_memory_usage()

    # Process in batches
    for i in range(0, total_items, batch_size):
        batch = items[i:i+batch_size]
        operation(batch)

    end_time = time.perf_counter()
    end_memory = get_memory_usage()

    elapsed = end_time - start_time
    throughput = total_items / elapsed if elapsed > 0 else 0

    return {
        "total_items": total_items,
        "elapsed_seconds": elapsed,
        "throughput_per_second": throughput,
        "memory_delta_mb": end_memory - start_memory,
        "avg_latency_ms": (elapsed / total_items) * 1000 if total_items > 0 else 0
    }


# ==================== Parsing Benchmarks ====================

@pytest.mark.benchmark
class TestParsingBenchmarks:
    """Benchmark document parsing operations."""

    def test_text_parsing_latency(self, tmp_path, large_document):
        """Measure latency of parsing large text file."""
        # Create test file
        test_file = tmp_path / "large.txt"
        test_file.write_text(large_document.page_content)

        parser = TextParser()

        # Warmup
        parser.parse(str(test_file))

        # Benchmark
        with Timer() as timer:
            documents = parser.parse(str(test_file))

        # Assertions
        assert len(documents) > 0
        assert timer.elapsed < 0.5  # Should parse in < 500ms

        print(f"\nText parsing latency: {timer.elapsed*1000:.2f}ms")

    def test_batch_parsing_throughput(self, tmp_path, batch_documents):
        """Measure throughput of parsing multiple files."""
        parser = TextParser()

        # Create test files
        files = []
        for i, doc in enumerate(batch_documents[:50]):  # Use 50 docs for speed
            file_path = tmp_path / f"doc_{i}.txt"
            file_path.write_text(doc.page_content)
            files.append(str(file_path))

        # Benchmark
        start_time = time.perf_counter()
        total_docs = 0

        for file_path in files:
            docs = parser.parse(file_path)
            total_docs += len(docs)

        elapsed = time.perf_counter() - start_time
        throughput = len(files) / elapsed

        # Assertions
        assert total_docs == len(files)
        assert throughput > 10  # Should process > 10 files/sec

        print(f"\nParsing throughput: {throughput:.2f} files/sec")


# ==================== Chunking Benchmarks ====================

@pytest.mark.benchmark
class TestChunkingBenchmarks:
    """Benchmark document chunking operations."""

    def test_chunking_latency(self, large_document):
        """Measure latency of chunking large document."""
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)

        # Warmup
        chunker.chunk_document(large_document)

        # Benchmark
        with Timer() as timer:
            chunks = chunker.chunk_document(large_document)

        # Assertions
        assert len(chunks) > 0
        assert timer.elapsed < 1.0  # Should chunk in < 1 second

        print(f"\nChunking latency: {timer.elapsed*1000:.2f}ms ({len(chunks)} chunks)")

    def test_batch_chunking_throughput(self, batch_documents):
        """Measure throughput of chunking multiple documents."""
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)

        # Benchmark
        start_time = time.perf_counter()
        total_chunks = 0

        for doc in batch_documents[:50]:
            chunks = chunker.chunk_document(doc)
            total_chunks += len(chunks)

        elapsed = time.perf_counter() - start_time
        throughput = len(batch_documents[:50]) / elapsed

        # Assertions
        assert total_chunks > 0
        assert throughput > 5  # Should process > 5 docs/sec

        print(f"\nChunking throughput: {throughput:.2f} docs/sec ({total_chunks} total chunks)")


# ==================== Embedding Benchmarks ====================

@pytest.mark.benchmark
class TestEmbeddingBenchmarks:
    """Benchmark embedding generation (mocked)."""

    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    @patch('src.ingestion.embedders.embedding_generator.get_settings')
    def test_embedding_generation_batched(
        self,
        mock_settings,
        mock_openai,
        batch_documents
    ):
        """Measure throughput of batched embedding generation."""
        # Mock settings
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        mock_settings.return_value.openai_api_key = "test_key"

        # Prepare chunks
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)
        all_chunks = []
        for doc in batch_documents[:50]:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        # Mock embeddings (simulate API call time)
        mock_client = Mock()

        def create_embeddings(*args, **kwargs):
            time.sleep(0.01)  # Simulate 10ms API latency
            input_texts = kwargs.get('input', [])
            num_inputs = len(input_texts) if isinstance(input_texts, list) else 1

            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=list(np.random.rand(1536)))
                for _ in range(num_inputs)
            ]
            return mock_response

        mock_client.embeddings.create = create_embeddings
        mock_openai.return_value = mock_client

        # Benchmark
        from src.ingestion.embedders import EmbeddingGenerator
        embedder = EmbeddingGenerator()

        start_time = time.perf_counter()
        embeddings = embedder.generate_for_chunks(all_chunks)
        elapsed = time.perf_counter() - start_time

        throughput = len(all_chunks) / elapsed

        # Assertions
        assert embeddings.shape[0] == len(all_chunks)

        print(f"\nEmbedding generation: {throughput:.2f} chunks/sec ({len(all_chunks)} chunks in {elapsed:.2f}s)")


# ==================== Vector Search Benchmarks ====================

@pytest.mark.benchmark
class TestVectorSearchBenchmarks:
    """Benchmark vector search operations."""

    def test_search_latency_small_k(self, populated_vector_store):
        """Measure search latency with k=5."""
        query_embedding = np.random.rand(1536).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Warmup
        populated_vector_store.search(query_embedding, top_k=5)

        # Benchmark
        latencies = []
        for _ in range(100):
            with Timer() as timer:
                results = populated_vector_store.search(query_embedding, top_k=5)
            latencies.append(timer.elapsed * 1000)  # Convert to ms

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        # Assertions
        assert len(results) == 5
        assert avg_latency < 10  # Should search in < 10ms

        print(f"\nSearch latency (k=5): avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms")

    def test_search_latency_large_k(self, populated_vector_store):
        """Measure search latency with k=100."""
        query_embedding = np.random.rand(1536).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Warmup
        populated_vector_store.search(query_embedding, top_k=100)

        # Benchmark
        latencies = []
        for _ in range(100):
            with Timer() as timer:
                results = populated_vector_store.search(query_embedding, top_k=100)
            latencies.append(timer.elapsed * 1000)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        # Assertions
        assert len(results) == 100
        assert avg_latency < 20  # Should search in < 20ms even with large k

        print(f"\nSearch latency (k=100): avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms")

    def test_concurrent_search_throughput(self, populated_vector_store):
        """Measure throughput of concurrent searches."""
        # Generate 100 random queries
        queries = [
            np.random.rand(1536).astype(np.float32) / np.linalg.norm(np.random.rand(1536))
            for _ in range(100)
        ]

        # Benchmark
        start_time = time.perf_counter()

        for query in queries:
            populated_vector_store.search(query, top_k=10)

        elapsed = time.perf_counter() - start_time
        throughput = len(queries) / elapsed

        # Assertions
        assert throughput > 50  # Should handle > 50 queries/sec

        print(f"\nSearch throughput: {throughput:.2f} queries/sec")


# ==================== End-to-End Benchmarks ====================

@pytest.mark.benchmark
class TestE2EBenchmarks:
    """Benchmark end-to-end query processing."""

    @patch('src.ingestion.embedders.embedding_generator.OpenAI')
    @patch('src.generation.generator.OpenAI')
    @patch('src.config.get_settings')
    def test_e2e_query_latency(
        self,
        mock_settings_module,
        mock_gen_openai,
        mock_embed_openai,
        populated_vector_store
    ):
        """Measure end-to-end query processing latency."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_settings.llm_model = "gpt-4"
        mock_settings.openai_api_key = "test_key"
        mock_settings.temperature = 0.7
        mock_settings_module.return_value = mock_settings

        # Mock embeddings
        mock_embed_client = Mock()

        def create_embedding(*args, **kwargs):
            time.sleep(0.01)  # 10ms latency
            mock_response = Mock()
            embedding = np.random.rand(1536)
            embedding = embedding / np.linalg.norm(embedding)
            mock_response.data = [Mock(embedding=list(embedding))]
            return mock_response

        mock_embed_client.embeddings.create = create_embedding
        mock_embed_openai.return_value = mock_embed_client

        # Mock generation
        mock_gen_client = Mock()

        def create_completion(*args, **kwargs):
            time.sleep(0.5)  # 500ms latency (realistic for GPT-4)
            mock_response = Mock()
            import json
            output = {
                "use_cases": [{"title": "Test", "goal": "Test goal"}],
                "assumptions": [],
                "confidence_score": 0.9
            }
            mock_response.choices = [
                Mock(message=Mock(content=json.dumps(output)))
            ]
            return mock_response

        mock_gen_client.chat.completions.create = create_completion
        mock_gen_openai.return_value = mock_gen_client

        # Setup pipeline
        from src.retrieval import RetrievalEngine
        from src.generation import UseCaseGenerator

        retrieval = RetrievalEngine(vector_store=populated_vector_store)
        generator = UseCaseGenerator()

        # Benchmark
        query = "Create test cases for user login"

        with Timer() as timer:
            # Retrieval
            results = retrieval.retrieve(query, top_k=5)

            # Generation
            context = "\n\n".join([r.content for r in results])
            output = generator.generate(query=query, context=context)

        # Assertions
        assert "use_cases" in output
        assert timer.elapsed < 2.0  # E2E should complete in < 2 seconds

        print(f"\nE2E query latency: {timer.elapsed*1000:.2f}ms")


# ==================== Memory Benchmarks ====================

@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Benchmark memory usage."""

    def test_vector_store_memory_scaling(self):
        """Measure memory usage as vector store grows."""
        measurements = []

        for num_vectors in [100, 500, 1000, 5000]:
            vector_store = VectorStore()

            # Generate vectors
            chunks = []
            embeddings = []

            for i in range(num_vectors):
                chunk = Document(
                    page_content=f"Chunk {i}",
                    metadata={"chunk_index": i}
                )
                chunk.chunk_id = f"chunk_{i}"
                chunks.append(chunk)

                embedding = np.random.rand(1536).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)

            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Measure memory before
            mem_before = get_memory_usage()

            # Add vectors
            contents = [chunk.page_content for chunk in chunks]
            metadata_list = [chunk.metadata for chunk in chunks]
            vector_store.add_vectors(embeddings_array, contents, metadata_list)

            # Measure memory after
            mem_after = get_memory_usage()
            mem_delta = mem_after - mem_before

            measurements.append({
                "num_vectors": num_vectors,
                "memory_mb": mem_delta,
                "memory_per_vector_kb": (mem_delta * 1024) / num_vectors
            })

        # Print results
        print("\nMemory scaling:")
        for m in measurements:
            print(f"  {m['num_vectors']:5d} vectors: {m['memory_mb']:6.2f} MB "
                  f"({m['memory_per_vector_kb']:.2f} KB/vector)")

        # Should scale roughly linearly
        assert measurements[-1]['memory_per_vector_kb'] < 50  # < 50KB per vector
