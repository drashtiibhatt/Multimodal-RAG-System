# Multimodal RAG System - Implementation Plan

## Project Overview

**Goal**: Build a file-based Retrieval-Augmented Generation (RAG) application that generates high-quality test cases and use cases from multimodal input sources (text, PDFs, images).

**Primary Evaluation Criteria** (High Weight):
- RAG implementation quality and accuracy
- Grounded retrieval (relevant chunks)
- Guardrails and evidence-based answering
- Code quality, modularization, clean abstractions
- Sensible library/framework choices

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│              (CLI / Minimal Web Interface)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                   ORCHESTRATION LAYER                        │
│           (Query Processing & Response Generation)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────┴────────┐ ┌─────┴──────┐ ┌────────┴─────────┐
│   INGESTION     │ │ RETRIEVAL  │ │   GENERATION     │
│     MODULE      │ │   MODULE   │ │     MODULE       │
└────────┬────────┘ └─────┬──────┘ └────────┬─────────┘
         │                │                  │
         │                │                  │
┌────────┴────────────────┴──────────────────┴─────────┐
│              GUARDS & SAFETY MODULE                   │
│  (Hallucination Check, Prompt Injection, Thresholds) │
└───────────────────────────┬───────────────────────────┘
                            │
┌───────────────────────────┴───────────────────────────┐
│              STORAGE & PERSISTENCE LAYER              │
│    (Vector DB + Metadata Store + File Cache)         │
└───────────────────────────────────────────────────────┘
```

---

## Detailed Component Breakdown

### 1. INGESTION MODULE (File-Based Knowledge Base)

**Responsibilities**:
- Parse and extract content from multiple file types
- Chunk content with intelligent strategies
- Generate embeddings for chunks
- Store indexed chunks for retrieval

**File Type Support**:
- Text/Markdown: Direct text extraction
- PDF: Text + image extraction
- DOCX: Document parsing (at least PDF required)
- Images (PNG/JPG): OCR + Vision models
- Links: Optional web scraping

**Implementation Approach**:

```python
# Pseudo-structure
class IngestionPipeline:
    def __init__(self):
        self.parsers = {
            '.txt': TextParser(),
            '.md': MarkdownParser(),
            '.pdf': PDFParser(),
            '.docx': DOCXParser(),
            '.png': ImageParser(),
            '.jpg': ImageParser()
        }
        self.chunker = SmartChunker()
        self.embedder = EmbeddingGenerator()

    def ingest_folder(self, folder_path):
        # 1. Scan folder for supported files
        # 2. Parse each file based on extension
        # 3. Extract content (text + metadata)
        # 4. Chunk content intelligently
        # 5. Generate embeddings
        # 6. Store in vector DB with metadata
        # 7. Create deduplication checks
```

**Key Technologies**:
- PDF Parsing: `pdfplumber` or `PyPDF2` + `pdf2image`
- DOCX: `python-docx`
- OCR: `pytesseract` or `EasyOCR`
- Vision: OpenAI Vision API or similar
- Text Processing: `langchain` TextSplitter

**Chunking Strategy**:
- Semantic chunking (preserve context)
- Overlap between chunks (10-15%)
- Metadata preservation (file name, page number, chunk index)
- Configurable chunk size (default: 500-1000 tokens)

### 2. RETRIEVAL MODULE

**Responsibilities**:
- Accept user query
- Perform hybrid retrieval (vector + keyword)
- Rank and filter results
- Return top-k relevant chunks with metadata

**Implementation Approach**:

```python
class RetrievalEngine:
    def __init__(self, vector_db, bm25_index):
        self.vector_db = vector_db
        self.bm25_index = bm25_index
        self.reranker = CrossEncoderReranker()  # Optional

    def retrieve(self, query, top_k=5, threshold=0.7):
        # 1. Generate query embedding
        # 2. Vector similarity search (top_k * 2)
        # 3. BM25 keyword search (top_k * 2)
        # 4. Combine results with weighted scoring
        # 5. Optional: Rerank with cross-encoder
        # 6. Filter by confidence threshold
        # 7. Return top_k with metadata
```

**Hybrid Retrieval**:
- Vector Search: Cosine similarity on embeddings
- Keyword Search: BM25 algorithm
- Fusion Strategy: Reciprocal Rank Fusion (RRF)
- Configurable weights: vector_weight=0.6, keyword_weight=0.4

**Key Technologies**:
- Vector DB: `FAISS` (simple, local) or `ChromaDB` (persistent)
- BM25: `rank_bm25` library
- Reranking: `sentence-transformers/cross-encoder` (optional)

### 3. GENERATION MODULE

**Responsibilities**:
- Take query + retrieved chunks
- Generate structured JSON output
- Format use cases/test cases
- Handle insufficient context

**Implementation Approach**:

```python
class GenerationEngine:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_templates = PromptTemplates()

    def generate_use_cases(self, query, retrieved_chunks, context_threshold=0.7):
        # 1. Check if retrieved context meets threshold
        # 2. If insufficient: ask clarifying questions
        # 3. Build context-aware prompt
        # 4. Call LLM with strict JSON schema
        # 5. Parse and validate JSON output
        # 6. Add citations (optional)
```

**Output Structure**:
```json
{
  "use_cases": [
    {
      "title": "Use Case Title",
      "goal": "What this test achieves",
      "preconditions": ["condition1", "condition2"],
      "test_data": {"key": "value"},
      "steps": ["step1", "step2"],
      "expected_results": ["result1", "result2"],
      "negative_cases": ["negative1"],
      "boundary_cases": ["boundary1"],
      "confidence_score": 0.85,
      "sources": ["file1.pdf:p3", "file2.md"]
    }
  ],
  "assumptions": ["assumption1"],
  "missing_information": ["missing1"]
}
```

**Key Technologies**:
- LLM: OpenAI GPT-4 / Anthropic Claude / Ollama (local)
- Prompt Engineering: System prompts with strict JSON schema
- JSON Validation: `pydantic` models

### 4. GUARDS & SAFETY MODULE (HIGH PRIORITY)

**Responsibilities**:
- Prevent hallucinations
- Detect prompt injection attempts
- Enforce evidence-based answering
- Quality checks

**Implementation Approach**:

```python
class SafetyGuards:
    def __init__(self):
        self.hallucination_detector = HallucinationChecker()
        self.injection_detector = PromptInjectionDetector()

    def validate_generation(self, query, context, generated_output):
        # 1. Check context sufficiency (min threshold)
        # 2. Detect prompt injection in query/documents
        # 3. Verify claims against retrieved context
        # 4. Check for contradictions
        # 5. Deduplication check
        # 6. Return validation status + warnings
```

**Key Safeguards**:

1. **Hallucination Reduction**:
   - Strict prompt: "Only use information from provided context"
   - Confidence scoring on each use case
   - Citation linking (chunk_id references)
   - Fact verification against source chunks

2. **Minimum Evidence Threshold**:
   - If retrieval score < 0.6: Ask clarifying questions
   - If no relevant chunks: "Insufficient information to generate"
   - Explicit assumption documentation

3. **Prompt Injection Resilience**:
   - Scan documents for suspicious instructions
   - Pattern detection: "Ignore previous", "New instructions", etc.
   - Sanitize user queries
   - Separate system/user contexts clearly

4. **Deduplication**:
   - Hash-based chunk deduplication during ingestion
   - Near-duplicate detection (fuzzy matching)
   - Remove redundant test cases

5. **Quality Checks**:
   - Validate JSON structure
   - Check completeness (all required fields)
   - Logical consistency checks

**Key Technologies**:
- Pattern Matching: Regex + keyword lists
- Fact Checking: NLI models (optional)
- Deduplication: MinHash / SimHash

### 5. OBSERVABILITY & DEBUGGING MODULE

**Responsibilities**:
- Log ingestion, retrieval, generation steps
- Debug mode to show retrieved chunks
- Track metrics (latency, tokens, chunks)

**Implementation Approach**:

```python
class ObservabilityService:
    def __init__(self):
        self.logger = StructuredLogger()
        self.metrics = MetricsCollector()

    def log_ingestion(self, file_path, status, chunks_created):
        # Log file processing details

    def log_retrieval(self, query, chunks, scores, latency):
        # Log retrieval results

    def log_generation(self, query, output, tokens, latency):
        # Log generation details

    def export_metrics(self):
        # Export to JSON/CSV
```

**Metrics to Track**:
- Ingestion: files processed, chunks created, errors
- Retrieval: query count, avg chunks returned, avg score, latency
- Generation: token usage, latency, success rate
- Safety: hallucination warnings, injection attempts blocked

---

## Technology Stack Recommendation

### Core Language
- **Python 3.10+** (Best ecosystem for RAG/ML)

### LLM Provider
- **Primary**: OpenAI (GPT-4 or GPT-3.5-turbo)
- **Alternative**: Anthropic Claude, AWS Bedrock
- **Local Option**: Ollama (llama2, mistral)

### Embeddings
- **OpenAI**: text-embedding-3-small (fast, cheap)
- **Open Source**: sentence-transformers (all-MiniLM-L6-v2)
- **Local**: Ollama embeddings

### Vector Database
- **FAISS**: Lightweight, local, fast (recommended for MVP)
- **ChromaDB**: Persistent, easy API
- **LanceDB**: Modern, good for multimodal

### Document Processing
- PDF: `pdfplumber` + `pdf2image`
- DOCX: `python-docx`
- OCR: `pytesseract` + `Pillow`
- Vision: OpenAI Vision API

### Retrieval
- BM25: `rank-bm25`
- Chunking: `langchain` or custom
- Reranking: `sentence-transformers/cross-encoder` (optional)

### Framework
- **LangChain**: RAG orchestration, chains, document loaders
- **LlamaIndex**: Alternative RAG framework
- **Custom**: For maximum control and learning

### Development Tools
- Environment: `python-venv`
- Dependencies: `pip` + `requirements.txt`
- Linting: `black`, `flake8`, `mypy`
- Testing: `pytest`
- Logging: `loguru` or `structlog`

### Storage
- Vector Store: FAISS index files (.index)
- Metadata: SQLite or JSON files
- Document Cache: Local file system

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
**Priority: HIGH**

**Tasks**:
1. Project setup
   - Initialize repository structure
   - Create virtual environment
   - Set up requirements.txt
   - Configure .env.example
   - Create README skeleton

2. Ingestion Module - Basic
   - Text/Markdown parser
   - PDF parser (text only)
   - Basic chunking (fixed size)
   - Embedding generation
   - FAISS vector store

3. Retrieval Module - Basic
   - Vector search only
   - Top-k retrieval
   - Basic scoring

4. Generation Module - Basic
   - Simple prompt template
   - LLM integration
   - JSON output parsing

**Deliverable**: End-to-end pipeline with text files

### Phase 2: Multimodal Support (Week 2)
**Priority: HIGH**

**Tasks**:
1. Enhanced Ingestion
   - PDF image extraction
   - OCR implementation
   - DOCX support
   - Image file processing
   - Metadata tracking

2. Hybrid Retrieval
   - BM25 implementation
   - Fusion scoring
   - Configurable parameters

3. Improved Chunking
   - Semantic chunking
   - Overlap strategy
   - Context preservation

**Deliverable**: Full multimodal ingestion + hybrid retrieval

### Phase 3: Guards & Safety (Week 2-3)
**Priority: CRITICAL**

**Tasks**:
1. Hallucination Detection
   - Context-grounded prompts
   - Confidence scoring
   - Citation linking

2. Prompt Injection Protection
   - Pattern detection
   - Input sanitization
   - Document scanning

3. Evidence Thresholds
   - Minimum confidence checks
   - Clarifying question generation
   - Assumption documentation

4. Quality Checks
   - Deduplication
   - JSON validation
   - Completeness checks

**Deliverable**: Production-ready safety module

### Phase 4: Observability & Polish (Week 3)
**Priority: MEDIUM**

**Tasks**:
1. Logging System
   - Structured logging
   - Debug mode
   - Error tracking

2. Metrics Collection
   - Latency tracking
   - Token usage
   - Success rates

3. Testing
   - Unit tests for each module
   - Integration tests
   - Evaluation suite

4. Documentation
   - Architecture docs
   - API documentation
   - Design notes

**Deliverable**: Observable, testable system

### Phase 5: Interface & Deployment (Week 3-4)
**Priority: LOW**

**Tasks**:
1. CLI Interface
   - Command-line tool
   - Interactive mode
   - Batch processing

2. Optional Web UI
   - Minimal Streamlit/Gradio UI
   - Upload interface
   - Query interface

3. Deployment Prep
   - Docker setup (optional)
   - One-command run script
   - Sample dataset

**Deliverable**: User-friendly interface

---

## Project Structure

```
multimodal-rag-system/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── setup.py
├── Dockerfile (optional)
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── parsers/
│   │   │   ├── __init__.py
│   │   │   ├── text_parser.py
│   │   │   ├── pdf_parser.py
│   │   │   ├── docx_parser.py
│   │   │   └── image_parser.py
│   │   ├── chunkers/
│   │   │   ├── __init__.py
│   │   │   └── semantic_chunker.py
│   │   └── embedders/
│   │       ├── __init__.py
│   │       └── embedding_generator.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── vector_search.py
│   │   ├── keyword_search.py
│   │   └── hybrid_fusion.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── prompt_templates.py
│   │   └── output_schemas.py
│   │
│   ├── guards/
│   │   ├── __init__.py
│   │   ├── hallucination_checker.py
│   │   ├── injection_detector.py
│   │   ├── threshold_validator.py
│   │   └── quality_checker.py
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   └── metadata_store.py
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── helpers.py
│
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_guards.py
│
├── data/
│   ├── sample_dataset/
│   │   ├── PRD_Signup.md
│   │   ├── API_Spec.yaml
│   │   └── Signup_UI.png
│   └── storage/
│       ├── vector_db/
│       └── metadata/
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DESIGN_DECISIONS.md
│   └── TRADEOFFS.md
│
├── scripts/
│   ├── run.sh
│   └── ingest.py
│
└── ui/ (optional)
    ├── app.py
    └── components/
```

---

## Development Approach

### Modular Design Principles

1. **Separation of Concerns**
   - Each module has single responsibility
   - Clear interfaces between modules
   - Easy to test and maintain

2. **Dependency Injection**
   - Pass dependencies explicitly
   - Easy to swap implementations
   - Better testability

3. **Configuration Management**
   - Centralized config (config.yaml or .env)
   - Environment-specific settings
   - Runtime configurability

4. **Error Handling**
   - Graceful degradation
   - Clear error messages
   - Comprehensive logging

### Code Quality Standards

1. **Type Hints**
   - Use Python type annotations
   - Validate with `mypy`

2. **Documentation**
   - Docstrings for all functions/classes
   - Inline comments for complex logic
   - README with examples

3. **Testing**
   - Unit tests (80%+ coverage)
   - Integration tests
   - Evaluation suite

4. **Linting**
   - Black for formatting
   - Flake8 for style
   - Pre-commit hooks

---

## Testing Strategy

### Unit Tests
- Parser modules: validate extraction
- Chunker: verify chunk quality
- Retrieval: test ranking algorithms
- Guards: test edge cases

### Integration Tests
- End-to-end pipeline
- Multimodal file processing
- Query → Response flow

### Evaluation Suite
- Test with sample queries
- Measure accuracy (manual review)
- Benchmark latency
- Token usage tracking

### Example Test Cases
```python
def test_pdf_parser():
    # Test PDF text extraction

def test_hallucination_detection():
    # Test with known hallucination cases

def test_prompt_injection_resilience():
    # Test with injection attempts

def test_hybrid_retrieval():
    # Compare with baseline (vector-only)
```

---

## Risk Mitigation

### Technical Risks

1. **OCR Quality**
   - Risk: Poor image text extraction
   - Mitigation: Use multiple OCR engines, confidence scoring

2. **Hallucinations**
   - Risk: LLM inventing facts
   - Mitigation: Strict prompts, fact verification, citations

3. **Performance**
   - Risk: Slow retrieval/generation
   - Mitigation: Caching, batch processing, smaller models

4. **Cost**
   - Risk: High API costs
   - Mitigation: Use GPT-3.5, limit tokens, cache responses

### Project Risks

1. **Scope Creep**
   - Mitigation: Stick to MVP, prioritize high-weight items

2. **Time Constraints**
   - Mitigation: Phased approach, focus on core features first

---

## Deliverables Checklist

### A) GitHub Repository
- [ ] Full source code
- [ ] README.md with setup instructions
- [ ] .env.example file
- [ ] requirements.txt
- [ ] Sample dataset folder
- [ ] docs/ folder with design notes
- [ ] Grant access to specified emails

### B) Walkthrough Video
- [ ] Architecture overview
- [ ] Demo with 2+ queries
- [ ] Multimodal file handling demonstration
- [ ] Safeguards explanation
- [ ] Local setup walkthrough
- [ ] Upload to accessible platform

### C) Submission
- [ ] Email repo link to santhosh@devassure.io
- [ ] Include video link
- [ ] Submit before deadline

---

## Recommended Development Timeline

**Total Time: 3-4 weeks**

- **Week 1**: Core infrastructure + basic pipeline
- **Week 2**: Multimodal support + hybrid retrieval + guards
- **Week 3**: Safety hardening + observability + testing
- **Week 4**: Polish + documentation + video

---

## Key Success Factors

1. **Focus on High-Weight Items**
   - RAG quality > UI polish
   - Guardrails > Features
   - Code quality > Speed

2. **Demonstrate Understanding**
   - Document design decisions
   - Explain tradeoffs
   - Show depth of knowledge

3. **Make it Easy to Run**
   - One-command setup
   - Clear documentation
   - Sample data included

4. **Robust Error Handling**
   - Graceful failures
   - Helpful error messages
   - Edge case handling

5. **Measurable Quality**
   - Metrics and logging
   - Evaluation results
   - Performance benchmarks

---

## Next Steps

1. Set up development environment
2. Create project structure
3. Implement Phase 1 (core infrastructure)
4. Iterate based on testing
5. Record walkthrough video
6. Submit before deadline

---

**Document Version**: 1.0
**Last Updated**: 2025-12-28
**Status**: Ready for Implementation
