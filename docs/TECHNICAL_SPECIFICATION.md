# Technical Specification - Multimodal RAG System

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Technology Stack Details](#technology-stack-details)
3. [Module Specifications](#module-specifications)
4. [API Interfaces](#api-interfaces)
5. [Data Models](#data-models)
6. [Configuration](#configuration)
7. [Performance Targets](#performance-targets)

---

## System Requirements

### Functional Requirements

**FR1: Multimodal File Ingestion**
- Support text, markdown, PDF, DOCX, PNG, JPG files
- Extract text and images from PDFs
- Perform OCR on images
- Handle folders with mixed file types
- Store metadata (filename, page, timestamp)

**FR2: Hybrid Retrieval**
- Combine vector similarity and keyword matching
- Configurable top_k (default: 5)
- Adjustable confidence thresholds (default: 0.7)
- Optional reranking
- Optional citations in output

**FR3: Structured Generation**
- Generate JSON formatted use cases/test cases
- Include: title, goal, preconditions, steps, expected results, negative/boundary cases
- Ground output in retrieved context
- Handle insufficient context gracefully

**FR4: Safety Guardrails**
- Prevent hallucinations through context verification
- Detect and block prompt injection attempts
- Enforce minimum evidence thresholds
- Deduplicate chunks and outputs
- Validate output completeness

**FR5: Observability**
- Debug mode to show retrieved chunks
- Log all pipeline stages
- Track metrics (latency, tokens, chunk count)
- Export logs and metrics

**FR6: Web Interface & API (Phase 4 - NEW)**
- Modern responsive web UI
- RESTful API with automatic documentation
- File upload via drag-and-drop interface
- Image upload with Vision API processing
- Real-time query interface
- Statistics dashboard
- Interactive API documentation (Swagger/OpenAPI)
- Multi-format file support (including images)
- Visual feedback for all operations

### Non-Functional Requirements

**NFR1: Performance**
- Ingestion: < 2 seconds per page (PDF)
- Retrieval: < 1 second per query
- Generation: < 10 seconds per query
- Total end-to-end: < 15 seconds

**NFR2: Scalability**
- Handle up to 1000 documents
- Support up to 50,000 chunks
- Process images up to 10MB

**NFR3: Reliability**
- 99% successful parsing rate
- Graceful failure handling
- Automatic retry for transient errors

**NFR4: Maintainability**
- Modular architecture
- 80%+ test coverage
- Type hints throughout
- Comprehensive documentation

---

## Technology Stack Details

### Core Dependencies

```txt
# Web Framework (Phase 4)
fastapi                         # REST API framework
uvicorn                         # ASGI server
python-multipart                # File upload support

# LLM & Embeddings
openai==1.12.0                  # Primary LLM provider (+ GPT-4o-mini Vision)
anthropic==0.18.0               # Alternative LLM
sentence-transformers==2.3.1    # Local embeddings

# Vector Search
faiss-cpu==1.7.4               # Vector similarity search
rank-bm25==0.2.2               # BM25 keyword search
chromadb==0.4.22               # Alternative vector DB

# Document Processing
pdfplumber==0.10.3             # PDF text extraction
pdf2image==1.16.3              # PDF to images
python-docx==1.1.0             # DOCX parsing
Pillow==10.2.0                 # Image processing
pytesseract==0.3.10            # OCR

# RAG Framework
langchain==0.1.6               # RAG orchestration
langchain-community==0.0.19    # Community integrations
langchain-openai==0.0.5        # OpenAI integration

# NLP & Text Processing
tiktoken==0.5.2                # Token counting
spacy==3.7.2                   # NLP processing

# Data & Storage
pydantic==2.6.0                # Data validation
sqlalchemy==2.0.25             # Metadata storage (optional)

# Utilities
python-dotenv==1.0.0           # Environment management
loguru==0.7.2                  # Logging
pyyaml==6.0.1                  # Config management
tqdm==4.66.1                   # Progress bars

# Development
pytest==7.4.4                  # Testing
black==24.1.1                  # Code formatting
flake8==7.0.0                  # Linting
mypy==1.8.0                    # Type checking

# Optional UI
streamlit==1.30.0              # Web UI (optional)
gradio==4.16.0                 # Alternative UI (optional)
```

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils

# macOS
brew install tesseract
brew install poppler

# Windows
# Download and install Tesseract from GitHub
# Download and install Poppler from oschwartz10612
```

---

## Module Specifications

### 1. Ingestion Module

#### 1.1 File Parsers

**TextParser**
```python
class TextParser(BaseParser):
    """Parse plain text and markdown files."""

    def parse(self, file_path: str) -> Document:
        """
        Extract text content from file.

        Args:
            file_path: Path to text/markdown file

        Returns:
            Document with text content and metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = {
            'source': file_path,
            'file_type': 'text',
            'char_count': len(content)
        }

        return Document(page_content=content, metadata=metadata)
```

**PDFParser**
```python
class PDFParser(BaseParser):
    """Parse PDF files with text and image extraction."""

    def __init__(self, extract_images: bool = True):
        self.extract_images = extract_images
        self.ocr_engine = pytesseract

    def parse(self, file_path: str) -> List[Document]:
        """
        Extract text and images from PDF.

        Returns:
            List of Document objects (one per page)
        """
        documents = []

        # Extract text using pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""

                metadata = {
                    'source': file_path,
                    'page': page_num,
                    'file_type': 'pdf'
                }

                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))

        # Extract images if enabled
        if self.extract_images:
            images = convert_from_path(file_path)
            for page_num, image in enumerate(images, 1):
                # Run OCR on image
                ocr_text = self.ocr_engine.image_to_string(image)

                # Store as separate document
                metadata = {
                    'source': file_path,
                    'page': page_num,
                    'file_type': 'pdf_image',
                    'extraction_method': 'ocr'
                }

                documents.append(Document(
                    page_content=ocr_text,
                    metadata=metadata
                ))

        return documents
```

**ImageParser**
```python
class ImageParser(BaseParser):
    """Parse image files with OCR."""

    def __init__(self, use_vision_api: bool = False):
        self.use_vision_api = use_vision_api
        self.ocr_engine = pytesseract

    def parse(self, file_path: str) -> Document:
        """
        Extract text from image using OCR or Vision API.
        """
        image = Image.open(file_path)

        if self.use_vision_api:
            # Use OpenAI Vision API for better accuracy
            text = self._vision_api_extract(file_path)
        else:
            # Use Tesseract OCR
            text = self.ocr_engine.image_to_string(image)

        metadata = {
            'source': file_path,
            'file_type': 'image',
            'extraction_method': 'vision_api' if self.use_vision_api else 'ocr'
        }

        return Document(page_content=text, metadata=metadata)
```

#### 1.2 Chunking Strategy

**SemanticChunker**
```python
class SemanticChunker:
    """Intelligent chunking with context preservation."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split document into semantic chunks.

        Strategy:
        1. Try to split on paragraph boundaries
        2. If paragraph too long, split on sentences
        3. Preserve overlap for context continuity
        4. Maintain metadata inheritance
        """
        text = document.page_content
        chunks = []

        # Use recursive character splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )

        chunk_texts = splitter.split_text(text)

        for idx, chunk_text in enumerate(chunk_texts):
            metadata = {
                **document.metadata,
                'chunk_index': idx,
                'chunk_id': f"{document.metadata['source']}:chunk_{idx}"
            }

            chunks.append(Chunk(
                content=chunk_text,
                metadata=metadata
            ))

        return chunks
```

#### 1.3 Embedding Generation

**EmbeddingGenerator**
```python
class EmbeddingGenerator:
    """Generate embeddings for text chunks."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI()

    def generate(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings in batch.

        Args:
            texts: List of text chunks

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        embeddings = np.array([
            data.embedding for data in response.data
        ])

        return embeddings
```

### 2. Retrieval Module

#### 2.1 Vector Search

**VectorSearch**
```python
class VectorSearch:
    """FAISS-based vector similarity search."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)
        self.chunk_metadata = []

    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """Add embeddings to index."""
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunk_metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Returns:
            List of SearchResult with score and metadata
        """
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append(SearchResult(
                    score=float(score),
                    metadata=self.chunk_metadata[idx]
                ))

        return results
```

#### 2.2 Keyword Search

**BM25Search**
```python
class BM25Search:
    """BM25 keyword-based search."""

    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []

    def build_index(self, chunks: List[str], metadata: List[Dict]):
        """Build BM25 index from chunks."""
        self.chunks = chunks
        self.metadata = metadata

        # Tokenize corpus
        self.tokenized_corpus = [
            chunk.lower().split() for chunk in chunks
        ]

        # Create BM25 object
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search using BM25.

        Returns:
            List of SearchResult with BM25 scores
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                score=float(scores[idx]),
                metadata=self.metadata[idx]
            ))

        return results
```

#### 2.3 Hybrid Fusion

**HybridRetrieval**
```python
class HybridRetrieval:
    """Combine vector and keyword search with RRF."""

    def __init__(
        self,
        vector_search: VectorSearch,
        bm25_search: BM25Search,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ):
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Hybrid retrieval with reciprocal rank fusion.

        Strategy:
        1. Get top 2*k from vector search
        2. Get top 2*k from BM25 search
        3. Apply RRF to combine rankings
        4. Return top k results
        """
        # Get candidates
        vector_results = self.vector_search.search(
            query_embedding, top_k=top_k*2
        )
        bm25_results = self.bm25_search.search(
            query, top_k=top_k*2
        )

        # Apply Reciprocal Rank Fusion
        combined_scores = {}
        k_rrf = 60  # RRF constant

        for rank, result in enumerate(vector_results, 1):
            chunk_id = result.metadata['chunk_id']
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + \
                self.vector_weight / (k_rrf + rank)

        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.metadata['chunk_id']
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + \
                self.keyword_weight / (k_rrf + rank)

        # Sort by combined score
        sorted_chunks = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Reconstruct results
        final_results = []
        for chunk_id, score in sorted_chunks:
            # Find metadata from original results
            metadata = self._find_metadata(chunk_id, vector_results, bm25_results)
            final_results.append(SearchResult(
                score=score,
                metadata=metadata
            ))

        return final_results
```

### 3. Generation Module

#### 3.1 Prompt Templates

**PromptTemplates**
```python
SYSTEM_PROMPT = """You are an expert test case generator. Your task is to create comprehensive, structured test cases based ONLY on the provided context.

CRITICAL RULES:
1. Use ONLY information from the provided context documents
2. Do NOT invent features, behaviors, or details not mentioned in the context
3. If information is insufficient, state assumptions explicitly
4. Generate output in strict JSON format
5. Include negative and boundary test cases
6. Be specific and actionable in test steps

If you cannot create accurate test cases due to insufficient context, respond with:
{{"insufficient_context": true, "clarifying_questions": ["question1", "question2"]}}
"""

USER_PROMPT_TEMPLATE = """Context Documents:
{context}

User Query: {query}

Generate comprehensive test cases in the following JSON format:
{{
  "use_cases": [
    {{
      "title": "Test Case Title",
      "goal": "What this test achieves",
      "preconditions": ["condition1", "condition2"],
      "test_data": {{"key": "value"}},
      "steps": ["step1", "step2", "step3"],
      "expected_results": ["result1", "result2"],
      "negative_cases": ["negative1"],
      "boundary_cases": ["boundary1"]
    }}
  ],
  "assumptions": ["assumption1"],
  "missing_information": ["missing1"],
  "confidence_score": 0.85
}}

Ensure ALL use cases are grounded in the provided context. Do not hallucinate.
"""
```

#### 3.2 Generation Engine

**UseCaseGenerator**
```python
class UseCaseGenerator:
    """Generate structured use cases using LLM."""

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.3
    ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
        min_confidence: float = 0.6
    ) -> Dict:
        """
        Generate use cases from query and context.

        Returns:
            Structured JSON output with use cases
        """
        # Check if we have sufficient context
        if not retrieved_chunks or retrieved_chunks[0].score < min_confidence:
            return {
                "insufficient_context": True,
                "clarifying_questions": self._generate_questions(query)
            }

        # Build context from chunks
        context = self._build_context(retrieved_chunks)

        # Create prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                context=context,
                query=query
            )}
        ]

        # Call LLM with JSON mode
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )

        # Parse JSON response
        output = json.loads(response.choices[0].message.content)

        # Add citations
        output = self._add_citations(output, retrieved_chunks)

        return output

    def _build_context(self, chunks: List[SearchResult]) -> str:
        """Build context string from chunks with citations."""
        context_parts = []

        for idx, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get('source', 'unknown')
            page = chunk.metadata.get('page', '')
            page_ref = f" (page {page})" if page else ""

            context_parts.append(
                f"[Document {idx}] {source}{page_ref}\n"
                f"{chunk.metadata['content']}\n"
            )

        return "\n".join(context_parts)
```

### 4. Guards Module

#### 4.1 Hallucination Checker

**HallucinationChecker**
```python
class HallucinationChecker:
    """Verify generated content against source context."""

    def check(
        self,
        generated_output: Dict,
        source_chunks: List[SearchResult]
    ) -> Dict[str, Any]:
        """
        Check for potential hallucinations.

        Returns:
            Validation report with warnings
        """
        warnings = []

        # Extract all claims from generated use cases
        claims = self._extract_claims(generated_output)

        # Check each claim against source chunks
        for claim in claims:
            if not self._verify_claim(claim, source_chunks):
                warnings.append({
                    "claim": claim,
                    "issue": "Not found in source context",
                    "severity": "high"
                })

        return {
            "passed": len(warnings) == 0,
            "warnings": warnings,
            "confidence": self._calculate_confidence(warnings)
        }

    def _extract_claims(self, output: Dict) -> List[str]:
        """Extract factual claims from generated output."""
        claims = []

        for use_case in output.get('use_cases', []):
            # Extract from steps, preconditions, expected results
            claims.extend(use_case.get('steps', []))
            claims.extend(use_case.get('preconditions', []))
            claims.extend(use_case.get('expected_results', []))

        return claims

    def _verify_claim(
        self,
        claim: str,
        chunks: List[SearchResult]
    ) -> bool:
        """Verify if claim is supported by source chunks."""
        # Simple keyword overlap check
        # In production, use semantic similarity or NLI model
        claim_keywords = set(claim.lower().split())

        for chunk in chunks:
            chunk_text = chunk.metadata['content'].lower()
            chunk_keywords = set(chunk_text.split())

            # Check for significant overlap
            overlap = len(claim_keywords & chunk_keywords)
            if overlap / len(claim_keywords) > 0.5:
                return True

        return False
```

#### 4.2 Prompt Injection Detector

**PromptInjectionDetector**
```python
class PromptInjectionDetector:
    """Detect and block prompt injection attempts."""

    INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"disregard above",
        r"forget everything",
        r"new instructions:",
        r"system:",
        r"you are now",
        r"act as",
        r"pretend to be"
    ]

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Scan text for injection patterns.

        Returns:
            Detection report with matched patterns
        """
        matches = []

        text_lower = text.lower()

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                matches.append(pattern)

        return {
            "injection_detected": len(matches) > 0,
            "matched_patterns": matches,
            "risk_level": "high" if len(matches) > 2 else "medium" if matches else "low"
        }

    def sanitize(self, text: str) -> str:
        """Remove or neutralize injection attempts."""
        # Replace suspicious patterns
        sanitized = text

        for pattern in self.INJECTION_PATTERNS:
            sanitized = re.sub(
                pattern,
                "[REDACTED]",
                sanitized,
                flags=re.IGNORECASE
            )

        return sanitized
```

---

## Data Models

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Document(BaseModel):
    """Raw document from file parsing."""
    page_content: str
    metadata: Dict[str, Any]

class Chunk(BaseModel):
    """Text chunk with metadata."""
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any]
    chunk_id: str

class SearchResult(BaseModel):
    """Search result with score."""
    score: float
    metadata: Dict[str, Any]

class UseCase(BaseModel):
    """Individual use case structure."""
    title: str
    goal: str
    preconditions: List[str]
    test_data: Dict[str, Any] = Field(default_factory=dict)
    steps: List[str]
    expected_results: List[str]
    negative_cases: List[str] = Field(default_factory=list)
    boundary_cases: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    sources: List[str] = Field(default_factory=list)

class GenerationOutput(BaseModel):
    """Complete generation output."""
    use_cases: List[UseCase]
    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    insufficient_context: bool = False
    clarifying_questions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
```

---

## REST API Endpoints (Phase 4 - NEW)

### Web Server Architecture

**Framework**: FastAPI with Uvicorn ASGI server
**Base URL**: `http://localhost:8000`
**Documentation**: Auto-generated Swagger UI at `/docs`

### API Endpoints

#### 1. Health Check

```python
GET /health
```

**Description**: Check if server is running

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-02T10:30:00Z"
}
```

**Status Codes**:
- `200 OK`: Server is healthy

---

#### 2. Upload Document

```python
POST /upload
```

**Description**: Upload and process a document or image

**Request**:
- Content-Type: `multipart/form-data`
- Body: `file` (form field)

**Supported Formats**:
- Documents: PDF, TXT, MD, DOCX, YAML, JSON
- Images: PNG, JPG, JPEG, BMP, TIFF, GIF

**Response**:
```json
{
  "success": true,
  "filename": "example.pdf",
  "chunks_added": 15,
  "total_vectors": 238,
  "processing_time": 3.2
}
```

**Status Codes**:
- `200 OK`: File processed successfully
- `400 Bad Request`: Invalid file type or format
- `500 Internal Server Error`: Processing failed

**Image Processing**:
- Images are processed using Vision API (GPT-4o-mini)
- Text extracted and indexed for retrieval
- Thumbnail preview available in web UI

---

#### 3. Query System

```python
POST /query
```

**Description**: Generate test cases based on query

**Request**:
```json
{
  "query": "Create test cases for user signup",
  "top_k": 5,
  "use_hybrid": true,
  "enable_guards": true
}
```

**Response**:
```json
{
  "use_cases": [
    {
      "title": "Successful signup with valid credentials",
      "goal": "Verify user can create account",
      "preconditions": ["User logged out", "Email not registered"],
      "test_data": {
        "email": "user@example.com",
        "password": "SecurePass123!"
      },
      "steps": [
        "Navigate to signup page",
        "Enter email and password",
        "Click submit"
      ],
      "expected_results": [
        "Account created",
        "Verification email sent"
      ],
      "negative_cases": ["Duplicate email rejected"],
      "boundary_cases": ["Min password length"]
    }
  ],
  "assumptions": [],
  "missing_information": [],
  "confidence_score": 0.85,
  "processing_time": 4.5
}
```

**Status Codes**:
- `200 OK`: Query processed successfully
- `400 Bad Request`: Invalid query format
- `404 Not Found`: No documents in vector store
- `500 Internal Server Error`: Generation failed

---

#### 4. Get Statistics

```python
GET /stats
```

**Description**: Get system statistics

**Response**:
```json
{
  "total_vectors": 238,
  "dimension": 1536,
  "unique_sources": 12,
  "bm25_indexed": true,
  "last_ingestion": "2026-01-02T09:15:00Z"
}
```

**Status Codes**:
- `200 OK`: Statistics retrieved

---

#### 5. Web Application

```python
GET /app
```

**Description**: Serve web interface HTML

**Response**: HTML page with Tailwind CSS styling

**Features**:
- Drag-and-drop file upload
- Image upload with preview
- Query interface
- Results display
- Statistics dashboard

---

### Frontend Architecture

**Technology**: Vanilla JavaScript + Tailwind CSS
**File**: `web/index.html`, `web/static/app.js`

**Key Features**:
1. **File Upload Component**
   - Drag-and-drop zone
   - File type validation
   - Image thumbnail preview
   - Upload progress feedback

2. **Query Interface**
   - Text input for queries
   - Generate button
   - JSON result display
   - Copy to clipboard

3. **Statistics Dashboard**
   - Real-time stats from `/stats` endpoint
   - Document count
   - Vector dimensions
   - Source tracking

**API Communication**:
```javascript
// Upload file
const formData = new FormData();
formData.append('file', file);
const response = await fetch('/upload', {
    method: 'POST',
    body: formData
});

// Query system
const response = await fetch('/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: userQuery})
});
```

---

## Configuration

```yaml
# config.yaml

# LLM Settings
llm:
  provider: "openai"  # openai, anthropic, ollama
  model: "gpt-4-turbo-preview"
  temperature: 0.3
  max_tokens: 4000

# Embedding Settings
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
  dimension: 1536

# Chunking Settings
chunking:
  chunk_size: 1000
  chunk_overlap: 150
  separators: ["\n\n", "\n", ". ", " "]

# Retrieval Settings
retrieval:
  top_k: 5
  vector_weight: 0.6
  keyword_weight: 0.4
  min_confidence: 0.6
  enable_reranking: false

# Guards Settings
guards:
  enable_hallucination_check: true
  enable_injection_detection: true
  min_evidence_threshold: 0.6
  enable_deduplication: true

# Observability Settings
observability:
  enable_logging: true
  log_level: "INFO"
  enable_metrics: true
  enable_debug_mode: false

# Storage Settings
storage:
  vector_db_path: "./data/storage/vector_db"
  metadata_db_path: "./data/storage/metadata.db"
  cache_enabled: true

# Web Server Settings (Phase 4)
web:
  host: "0.0.0.0"
  port: 8000
  enable_cors: true
  reload: false  # Set to true for development

# Vision API Settings (Phase 4)
vision:
  enable_vision_api: true
  model: "gpt-4o-mini"
  enable_ocr_fallback: false
```

---

## Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| PDF Ingestion | < 2s per page | < 5s |
| Image OCR | < 3s per image | < 8s |
| Chunk Generation | < 1s per document | < 3s |
| Embedding Generation | < 0.5s per chunk | < 2s |
| Retrieval (vector) | < 0.5s | < 2s |
| Retrieval (hybrid) | < 1s | < 3s |
| LLM Generation | < 8s | < 20s |
| End-to-End Query | < 15s | < 30s |
| Memory Usage | < 2GB | < 4GB |

---

**Document Version**: 2.0 (Added Phase 4: Web Interface & API)
**Last Updated**: January 2, 2026
**Status**: Production Ready - All Phases Complete
