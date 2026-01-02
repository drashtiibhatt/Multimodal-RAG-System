# Architecture Decisions & Tradeoffs

## Overview

This document captures key architectural decisions for the Multimodal RAG system, including rationale and tradeoffs.

---

## ADR-001: Python as Primary Language

**Status**: Accepted

**Context**:
Need to choose a programming language for the RAG system.

**Decision**:
Use Python 3.10+ as the primary implementation language.

**Rationale**:
- Best ecosystem for ML/AI tasks (LangChain, transformers, FAISS)
- Rich library support for document processing
- Quick prototyping and iteration
- Familiar to most ML engineers
- Strong typing support with type hints

**Alternatives Considered**:
- TypeScript: Better for web apps, weaker ML ecosystem
- Go: Fast but limited ML libraries
- Rust: Excellent performance but steeper learning curve

**Tradeoffs**:
- **Pros**: Fast development, rich libraries, easy debugging
- **Cons**: Slower than compiled languages, GIL limitations
- **Mitigation**: Use NumPy/FAISS for performance-critical operations

---

## ADR-002: FAISS for Vector Storage

**Status**: Accepted

**Context**:
Need a local, file-based vector database for embeddings storage and similarity search.

**Decision**:
Use FAISS (Facebook AI Similarity Search) as the primary vector database.

**Rationale**:
- Extremely fast similarity search (optimized C++)
- No external dependencies (fully local)
- Mature and well-tested
- Excellent documentation
- Supports multiple index types

**Alternatives Considered**:
- ChromaDB: Easier API but adds overhead
- LanceDB: Modern but less mature
- Pinecone: Cloud-based (violates local requirement)
- Qdrant: Requires Docker/server

**Tradeoffs**:
- **Pros**: Fast, local, battle-tested, no server required
- **Cons**: Lower-level API, no built-in persistence helper
- **Mitigation**: Wrap FAISS with helper class for save/load

**Implementation Notes**:
```python
# Use IndexFlatIP for exact search with cosine similarity
index = faiss.IndexFlatIP(dimension)

# For larger datasets (>100k vectors), consider IndexIVFFlat
# quantizer = faiss.IndexFlatIP(dimension)
# index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
```

---

## ADR-003: Hybrid Retrieval (Vector + BM25)

**Status**: Accepted

**Context**:
Vector search alone may miss exact keyword matches; keyword search alone may miss semantic similarity.

**Decision**:
Implement hybrid retrieval combining vector similarity (FAISS) and keyword matching (BM25) using Reciprocal Rank Fusion.

**Rationale**:
- Vector search: Captures semantic similarity
- BM25: Captures exact term matches
- RRF: Proven fusion method, no parameter tuning needed
- Complementary strengths

**Alternatives Considered**:
- Vector-only: Misses exact matches (e.g., error codes, IDs)
- Keyword-only: Misses semantic similarity
- Linear combination: Requires weight tuning
- Learn-to-rank: Overkill for MVP, needs training data

**Tradeoffs**:
- **Pros**: Better recall and precision, robust to query types
- **Cons**: 2x retrieval operations, slight latency increase
- **Mitigation**: Run searches in parallel, cache results

**Implementation**:
```python
# Reciprocal Rank Fusion formula
score(d) = Σ [w_v / (k + rank_v(d))] + [w_k / (k + rank_k(d))]

# Default weights
w_v = 0.6  # vector weight
w_k = 0.4  # keyword weight
k = 60     # RRF constant
```

---

## ADR-004: OpenAI for LLM and Embeddings

**Status**: Accepted

**Context**:
Need reliable LLM for generation and embedding model.

**Decision**:
Use OpenAI as primary provider:
- LLM: GPT-4-turbo (or GPT-3.5-turbo for cost optimization)
- Embeddings: text-embedding-3-small

**Rationale**:
- High-quality outputs
- JSON mode for structured generation
- Fast and reliable API
- Good cost/performance ratio
- Well-documented

**Alternatives Considered**:
- Anthropic Claude: Excellent quality, higher cost
- Local models (Ollama): Free but slower, lower quality
- AWS Bedrock: More complex setup

**Tradeoffs**:
- **Pros**: Best quality, fast, reliable, JSON mode
- **Cons**: API costs, requires internet, data privacy concerns
- **Mitigation**: Implement caching, allow provider swapping

**Cost Optimization**:
```python
# Estimated costs (per 1000 queries)
# GPT-4-turbo: ~$5-10 (high quality)
# GPT-3.5-turbo: ~$0.50-1 (good quality, 10x cheaper)
# text-embedding-3-small: ~$0.02

# Mitigation strategies:
# 1. Cache embeddings
# 2. Cache LLM responses for identical queries
# 3. Use GPT-3.5 for non-critical paths
# 4. Limit max_tokens
```

---

## ADR-005: Semantic Chunking with Overlap

**Status**: Accepted

**Context**:
Need to split long documents into chunks for embedding and retrieval.

**Decision**:
Use recursive character splitting with:
- Chunk size: 1000 tokens (~750 words)
- Overlap: 150 tokens (~15%)
- Separators: ["\n\n", "\n", ". ", " "]

**Rationale**:
- Preserves semantic boundaries (paragraphs, sentences)
- Overlap ensures context continuity
- 1000 tokens fits well in embedding window
- Proven approach in production RAG systems

**Alternatives Considered**:
- Fixed-size chunks: Breaks mid-sentence
- Sentence-level: Too granular, loses context
- Paragraph-level: Too coarse, variable size
- Semantic splitting (LLM-based): Expensive, slow

**Tradeoffs**:
- **Pros**: Good balance of granularity and context
- **Cons**: Overlap creates redundancy (15% more storage)
- **Mitigation**: Deduplication checks during retrieval

**Implementation**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "]
)
```

---

## ADR-006: Multimodal Processing Strategy

**Status**: Accepted

**Context**:
Need to handle text documents (PDF, DOCX), images, and mixed content.

**Decision**:
Use dual-path processing:
1. **Text Path**: Direct text extraction from PDFs/DOCX
2. **Image Path**: OCR (Tesseract) + optional Vision API

For PDFs with images:
- Extract text with pdfplumber
- Extract images with pdf2image
- Run OCR on images
- Store both as separate chunks with metadata

**Rationale**:
- Text extraction is fast and accurate for digital PDFs
- OCR handles scanned documents and images
- Vision API (optional) for complex diagrams
- Separate indexing allows source tracking

**Alternatives Considered**:
- Vision API for all PDFs: Expensive, slower
- OCR only: Lower quality for digital PDFs
- Ignore images: Loses important context

**Tradeoffs**:
- **Pros**: Comprehensive coverage, accurate text
- **Cons**: Slower ingestion, more storage
- **Mitigation**: Parallel processing, configurable image extraction

**Implementation**:
```python
class PDFParser:
    def parse(self, pdf_path):
        docs = []

        # Path 1: Text extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                docs.append(Document(text, metadata={...}))

        # Path 2: Image extraction (if enabled)
        if self.extract_images:
            images = convert_from_path(pdf_path)
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                docs.append(Document(ocr_text, metadata={'type': 'image'}))

        return docs
```

---

## ADR-007: JSON Schema for Structured Output

**Status**: Accepted

**Context**:
Need reliable, parseable output format for use cases.

**Decision**:
Use strict JSON schema with OpenAI's JSON mode and Pydantic validation.

**Rationale**:
- Programmatically parseable
- Type-safe with Pydantic
- OpenAI JSON mode ensures valid JSON
- Easy to extend and validate

**Alternatives Considered**:
- Markdown: Human-readable but hard to parse
- YAML: Good for humans but LLM less reliable
- XML: Verbose
- Free-form text: Impossible to parse reliably

**Tradeoffs**:
- **Pros**: Reliable parsing, type safety, extensible
- **Cons**: Less human-readable in raw form
- **Mitigation**: Provide pretty-print option for display

**Schema**:
```python
class UseCaseSchema(BaseModel):
    title: str
    goal: str
    preconditions: List[str]
    test_data: Dict[str, Any] = {}
    steps: List[str]
    expected_results: List[str]
    negative_cases: List[str] = []
    boundary_cases: List[str] = []
    confidence_score: float
    sources: List[str] = []
```

---

## ADR-008: Guardrails Implementation

**Status**: Accepted

**Context**:
Need to prevent hallucinations, prompt injection, and low-quality outputs.

**Decision**:
Implement multi-layer guardrails:

1. **Pre-Generation Guards**:
   - Prompt injection detection in query and documents
   - Minimum retrieval confidence check
   - Context sufficiency validation

2. **Post-Generation Guards**:
   - Hallucination checking (verify claims against context)
   - JSON schema validation
   - Completeness checks

3. **Runtime Guards**:
   - Token limits
   - Timeout protection
   - Error handling

**Rationale**:
- Defense in depth
- Catches issues at multiple stages
- Provides clear feedback to users

**Alternatives Considered**:
- LLM-based fact checking: Too slow, expensive
- Manual review: Not scalable
- No guards: Unreliable outputs

**Tradeoffs**:
- **Pros**: Reliable outputs, user trust, safety
- **Cons**: Increased latency (~10-15%), complexity
- **Mitigation**: Make guards configurable, optimize checks

**Implementation**:
```python
class SafetyPipeline:
    def validate(self, query, context, output):
        # Pre-checks
        if self.detect_injection(query):
            raise SecurityError("Injection detected")

        if context.confidence < self.min_threshold:
            return {"insufficient_context": True}

        # Post-checks
        hallucination_report = self.check_hallucinations(output, context)
        if not hallucination_report['passed']:
            output['warnings'] = hallucination_report['warnings']

        return output
```

---

## ADR-009: File-Based Storage (No External DB)

**Status**: Accepted

**Context**:
Assignment requires local, file-based storage.

**Decision**:
Use file-based storage:
- Vector index: FAISS .index files
- Metadata: JSON files or SQLite
- Documents: Local file system
- Cache: Pickle files

**Rationale**:
- Meets assignment requirement
- Simple setup (no server required)
- Portable (can copy entire data/ folder)
- Version controllable

**Alternatives Considered**:
- PostgreSQL + pgvector: Requires server
- MongoDB: Requires server
- Redis: Requires server, not persistent

**Tradeoffs**:
- **Pros**: Simple, portable, no dependencies
- **Cons**: No concurrent access, no ACID guarantees
- **Mitigation**: Use file locking for writes

**Directory Structure**:
```
data/
├── storage/
│   ├── vector_db/
│   │   ├── faiss.index
│   │   └── metadata.json
│   └── cache/
│       └── embeddings_cache.pkl
└── uploads/
    └── [user files]
```

---

## ADR-010: Modular Architecture

**Status**: Accepted

**Context**:
Need clean, maintainable codebase for evaluation.

**Decision**:
Organize code into independent modules with clear interfaces:
- Ingestion (parsing, chunking, embedding)
- Retrieval (vector, keyword, fusion)
- Generation (LLM, prompts, schemas)
- Guards (safety checks)
- Storage (vector DB, metadata)
- Observability (logging, metrics)

**Rationale**:
- Single Responsibility Principle
- Easy to test independently
- Easy to swap implementations
- Clear documentation
- Meets "code quality" requirement

**Implementation Pattern**:
```python
# Each module has:
# 1. Interface/Protocol definition
# 2. Concrete implementation(s)
# 3. Configuration
# 4. Tests

# Example:
class ParserProtocol(Protocol):
    def parse(self, file_path: str) -> List[Document]:
        ...

class PDFParser(ParserProtocol):
    def parse(self, file_path: str) -> List[Document]:
        # Implementation
        ...
```

---

## ADR-011: Logging and Observability

**Status**: Accepted

**Context**:
Need debugging capability and performance metrics.

**Decision**:
Implement structured logging with Loguru and custom metrics:
- Log all pipeline stages (ingestion, retrieval, generation)
- Track latency, token usage, chunk counts
- Debug mode to show retrieved chunks
- Export logs to files

**Rationale**:
- Required for debugging
- Helps identify bottlenecks
- Demonstrates system understanding
- Professional best practice

**Implementation**:
```python
from loguru import logger

logger.add(
    "logs/app_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)

# Usage
@logger.catch
def ingest_document(file_path):
    logger.info(f"Ingesting {file_path}")
    start = time.time()

    # ... processing ...

    logger.info(f"Ingested in {time.time() - start:.2f}s")
```

---

## ADR-012: Testing Strategy

**Status**: Accepted

**Context**:
Need to ensure code quality and correctness.

**Decision**:
Implement three testing layers:
1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test end-to-end flows
3. **Evaluation Tests**: Test output quality with sample data

**Coverage Target**: 80%+

**Rationale**:
- Catches bugs early
- Enables refactoring
- Documents expected behavior
- Meets code quality requirement

**Implementation**:
```python
# Unit test example
def test_chunk_size():
    chunker = SemanticChunker(chunk_size=100)
    chunks = chunker.chunk("..." * 1000)
    assert all(len(chunk) <= 120 for chunk in chunks)  # +20% tolerance

# Integration test example
def test_end_to_end_pipeline():
    # Ingest sample document
    pipeline.ingest("sample.pdf")

    # Query
    result = pipeline.query("Create signup test cases")

    # Validate
    assert result['use_cases']
    assert all('title' in uc for uc in result['use_cases'])
```

---

## Summary of Key Tradeoffs

| Decision | Pros | Cons | Mitigation |
|----------|------|------|------------|
| Python | Fast development, rich ecosystem | Slower than compiled languages | Use NumPy/C++ libraries |
| FAISS | Very fast, local | Low-level API | Wrapper class |
| Hybrid Retrieval | Better accuracy | 2x operations | Parallel execution |
| OpenAI | High quality, JSON mode | Cost, API dependency | Caching, provider abstraction |
| Semantic Chunking | Preserves context | 15% redundancy | Deduplication |
| OCR + Vision | Comprehensive | Slower, more expensive | Configurable, parallel |
| JSON Output | Parseable, type-safe | Less human-readable | Pretty-print option |
| Guardrails | Reliable, safe | 10-15% latency | Optimized checks, configurable |
| File-based Storage | Simple, portable | No concurrency | File locking |
| Modular Architecture | Maintainable, testable | More upfront design | Clear interfaces |

---

## Future Improvements (Post-MVP)

1. **Performance**:
   - Implement async I/O for parallel document processing
   - Use FAISS GPU for faster similarity search
   - Add response caching layer

2. **Quality**:
   - Fine-tune reranking model on domain data
   - Implement feedback loop for user corrections
   - Add A/B testing framework

3. **Features**:
   - Support more file types (CSV, Excel, HTML)
   - Add incremental updates (don't re-index all files)
   - Implement document versioning

4. **Safety**:
   - Add semantic similarity-based fact checking
   - Implement adversarial prompt detection
   - Add output diversity checks

5. **Deployment**:
   - Add Docker support
   - Implement REST API
   - Add authentication layer

---

**Document Version**: 1.0
**Last Updated**: 2025-12-28
**Status**: Approved for Implementation
