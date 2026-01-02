# Production Setup Instructions - Multimodal RAG System

## Overview

This guide covers the complete setup process for the Multimodal RAG System, from initial installation to running queries with safety guardrails.

**Current Status:** Production-Ready with All Phases Complete (v4.0 Final)
- **Phase 4**: Web interface with FastAPI + Tailwind CSS
- **Phase 4**: Image upload with Vision API (GPT-4o-mini)
- **Phase 4**: REST API with automatic documentation
- Multimodal document processing (PDF, DOCX, Images, Text)
- Hybrid retrieval (Vector + Keyword search)
- Safety guardrails (Hallucination detection, Prompt injection protection)
- Comprehensive test suite (163 tests, 88.3% pass rate)
- Evaluation framework

---

## Prerequisites

- Python 3.8 or higher (3.10+ recommended)
- Git
- Text editor (VS Code, PyCharm, etc.)
- OpenAI API key
- System dependencies (tesseract, poppler - see below)

---

## Quick Start (Existing Project)

If you've already cloned the repository, follow these steps:

```bash
# 1. Navigate to project
cd D:\multimodal-rag-system\multimodal-rag-system

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install package in development mode
pip install -e .[dev]

# 4. Configure environment
copy .env.example .env         # Windows
# cp .env.example .env         # Linux/Mac
# Then edit .env and add your OPENAI_API_KEY

# 5. Verify installation
python -c "import src; print('✓ Installation successful')"
pytest tests/unit/ -v          # Run tests to verify
```

---

## Web Interface Setup (Recommended)

### Quick Web Interface Start

The fastest way to get started is using the web interface:

```bash
# 1. Navigate to project
cd D:\multimodal-rag-system\multimodal-rag-system

# 2. Activate virtual environment (if not already activated)
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Start web server
python run_web.py
```

The server will start on `http://localhost:8000`

### Access Points

Once the server is running, open your browser to:

1. **Web Application**: http://localhost:8000/app
   - Modern UI with drag-and-drop file upload
   - Interactive query interface
   - Real-time processing feedback
   - Document statistics dashboard

2. **API Documentation**: http://localhost:8000/docs
   - Interactive Swagger/OpenAPI documentation
   - Test API endpoints directly from browser
   - View request/response schemas

### Web Interface Features

**Document Upload:**
- Drag and drop files or click to browse
- Supported formats: PDF, TXT, DOCX, MD, YAML, JSON, PNG, JPG, JPEG, BMP, TIFF, GIF
- **Image Upload**: Automatic processing with Vision API (GPT-4o-mini)
- File preview before upload
- Real-time upload progress
- File type icons for each document

**Query Interface:**
- Type queries in natural language
- Real-time test case generation
- JSON formatted output
- Copy results with one click

**Statistics Dashboard:**
- Total documents indexed
- Total vector embeddings
- Embedding dimensions
- Unique sources count

### Image Upload with Vision API

The web interface supports advanced image processing:

1. **Upload Image**: Drag image file or click to select
2. **Preview**: View image thumbnail before processing
3. **Process**: Click "Upload" - image is processed with GPT-4o-mini Vision
4. **Index**: Image content is extracted and indexed for retrieval
5. **Query**: Use specific terms from the image in your queries for best results

**Supported Image Formats:**
- PNG, JPG, JPEG
- BMP, TIFF, GIF

**Example Workflow:**
```
1. Upload flight booking screenshot (booking.png)
2. System extracts text using Vision API
3. Content is indexed in vector database
4. Query: "Create test cases for flight filter functionality"
5. System retrieves relevant content from the screenshot
```

---

## Detailed Setup Instructions

## Step 1: Clone or Navigate to Project

### On Windows (PowerShell or CMD):

```batch
REM Navigate to where you want the project
cd D:\

REM Create main project folder
mkdir multimodal-rag-system
cd multimodal-rag-system

REM Create all subfolders
mkdir src\ingestion\parsers
mkdir src\ingestion\chunkers
mkdir src\ingestion\embedders
mkdir src\retrieval
mkdir src\generation
mkdir src\guards
mkdir src\storage
mkdir src\observability
mkdir tests
mkdir data\sample_dataset\user-signup
mkdir data\storage\vector_db
mkdir data\storage\cache
mkdir docs
mkdir scripts
mkdir logs
mkdir ui

REM Create __init__.py files
type nul > src\__init__.py
type nul > src\ingestion\__init__.py
type nul > src\ingestion\parsers\__init__.py
type nul > src\ingestion\chunkers\__init__.py
type nul > src\ingestion\embedders\__init__.py
type nul > src\retrieval\__init__.py
type nul > src\generation\__init__.py
type nul > src\guards\__init__.py
type nul > src\storage\__init__.py
type nul > src\observability\__init__.py
type nul > tests\__init__.py
```

### On Linux/Mac (Terminal):

```bash
# Navigate to where you want the project
cd ~

# Create main project folder
mkdir multimodal-rag-system
cd multimodal-rag-system

# Create all subfolders
mkdir -p src/ingestion/{parsers,chunkers,embedders}
mkdir -p src/{retrieval,generation,guards,storage,observability}
mkdir -p tests
mkdir -p data/sample_dataset/user-signup
mkdir -p data/storage/{vector_db,cache}
mkdir -p docs scripts logs ui

# Create __init__.py files
touch src/__init__.py
touch src/ingestion/__init__.py
touch src/ingestion/parsers/__init__.py
touch src/ingestion/chunkers/__init__.py
touch src/ingestion/embedders/__init__.py
touch src/retrieval/__init__.py
touch src/generation/__init__.py
touch src/guards/__init__.py
touch src/storage/__init__.py
touch src/observability/__init__.py
touch tests/__init__.py
```

---

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# You should see (venv) in your terminal

# Upgrade pip
pip install --upgrade pip
```

---

## Step 3: Install Dependencies

### Option A: Using setup.py (Recommended)

```bash
# Install package in editable mode with all dependencies
pip install -e .

# Or install with development dependencies (for testing)
pip install -e .[dev]

# Or install everything (production + development)
pip install -e .[all]
```

After installation, you can use the `rag-system` command directly:
```bash
rag-system --help
```

### Option B: Using requirements.txt

```bash
# Install all dependencies
pip install -r requirements.txt

# This will take 2-5 minutes
```

---

## Step 4: Install System Dependencies

These are required for OCR and PDF processing.

**For Windows:**
1. Download Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it
3. Add to PATH: `C:\Program Files\Tesseract-OCR`

4. Download Poppler: https://github.com/oschwartz10612/poppler-windows/releases
5. Extract and add `bin` folder to PATH

**For Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils
```

**For Mac:**
```bash
brew install tesseract
brew install poppler
```

**Verify Installation:**
```bash
tesseract --version    # Should show Tesseract version
pdftoppm -v           # Should show Poppler version
```

---

## Step 5: Configure Environment Variables

### Create .env File

```bash
# Copy template
# Windows:
copy .env.example .env

# Linux/Mac:
cp .env.example .env
```

### Edit .env and Add Your API Key

Open `.env` in a text editor and update:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional: Enable features
ENABLE_HYBRID_RETRIEVAL=true
ENABLE_HALLUCINATION_CHECK=true
ENABLE_INJECTION_DETECTION=true

# Optional: Model configuration
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0.3

# Optional: Retrieval configuration
TOP_K=5
MIN_CONFIDENCE=0.6
VECTOR_WEIGHT=0.6
KEYWORD_WEIGHT=0.4

# Optional: Chunking configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

---

## Step 6: Verify Installation

```bash
# Test Python imports
python -c "import src; print('✓ Package installed')"
python -c "import openai, faiss, pdfplumber; print('✓ Dependencies working')"

# Check if main module works
python -m src.main --help

# Or if you installed with setup.py:
rag-system --help

# Run a quick test
pytest tests/unit/test_parsers.py::TestTextParser::test_parse_simple_text -v
```

---

## Step 7: Create Core Files (If Starting From Scratch)

### Create `.gitignore`

```bash
# Windows
notepad .gitignore

# Linux/Mac
nano .gitignore
```

**Paste this content:**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env

# Data & Storage
data/storage/
logs/
*.pkl
*.index

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints

# pytest
.pytest_cache/
```

### Create `requirements.txt`

```bash
# Windows
notepad requirements.txt

# Linux/Mac
nano requirements.txt
```

**Paste this content:**

```txt
# Core LLM & Embeddings
openai==1.12.0
anthropic==0.18.0
sentence-transformers==2.3.1

# Vector Search & Retrieval
faiss-cpu==1.7.4
rank-bm25==0.2.2

# RAG Framework
langchain==0.1.6
langchain-community==0.0.19
langchain-openai==0.0.5

# Document Processing
pdfplumber==0.10.3
pdf2image==1.16.3
python-docx==1.1.0
Pillow==10.2.0
pytesseract==0.3.10

# NLP & Text Processing
tiktoken==0.5.2
spacy==3.7.2

# Data Validation
pydantic==2.6.0

# Utilities
python-dotenv==1.0.0
loguru==0.7.2
pyyaml==6.0.1
tqdm==4.66.1

# Development
pytest==7.4.4
black==24.1.1
flake8==7.0.0
mypy==1.8.0

# Optional UI
streamlit==1.30.0
gradio==4.16.0
```

### Create `.env.example`

```bash
# Windows
notepad .env.example

# Linux/Mac
nano .env.example
```

**Paste this content:**

```bash
# OpenAI Configuration (Primary)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration (Optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0.3

# Retrieval Configuration
TOP_K=5
MIN_CONFIDENCE=0.6
VECTOR_WEIGHT=0.6
KEYWORD_WEIGHT=0.4

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=150

# Guardrails
ENABLE_HALLUCINATION_CHECK=true
ENABLE_INJECTION_DETECTION=true
MIN_EVIDENCE_THRESHOLD=0.6

# Logging
LOG_LEVEL=INFO
ENABLE_DEBUG_MODE=false
```

### Create `.env` (Your actual config)

```bash
# Copy template
# Windows:
copy .env.example .env

# Linux/Mac:
cp .env.example .env

# Now edit .env and add your actual API key
```

---

## Usage Guide

Now that everything is set up, here's how to use the system:

### 1. Ingest Documents

Ingest documents from a folder into the vector database:

```bash
# Using Python module
python -m src.main --ingest data/sample_dataset/user-signup/

# Or if you installed with setup.py:
rag-system --ingest data/sample_dataset/user-signup/

# Ingest with hybrid mode enabled (builds BM25 index)
python -m src.main --ingest data/sample_dataset/user-signup/ --hybrid
```

This will:
- Parse all supported files (PDF, DOCX, TXT, MD, YAML, JSON, images)
- Chunk documents semantically
- Generate embeddings using OpenAI
- Store in FAISS vector database
- Optionally build BM25 keyword index (with --hybrid)

### 2. Query the System

**Basic Query:**
```bash
python -m src.main --query "Create use cases for user signup"
```

**With Hybrid Retrieval (Vector + Keyword):**
```bash
python -m src.main --query "Create test cases for authentication" --hybrid
```

**With Safety Guardrails:**
```bash
python -m src.main --query "Generate negative test cases" --enable-guards
```

**With Everything (Hybrid + Guards + Debug):**
```bash
python -m src.main --query "Create boundary test cases" --hybrid --enable-guards --debug
```

**Save Output to File:**
```bash
python -m src.main --query "Create test cases" --output results.json
```

### 3. Check Statistics

```bash
python -m src.main --stats
```

This shows:
- Number of documents in vector store
- Number of chunks
- Embedding dimensions
- BM25 index status (if hybrid mode)

---

## Sample Dataset

The project includes sample documents in `data/sample_dataset/user-signup/`:

- `PRD_Signup.md` - Product requirements for user signup
- `API_Spec.yaml` - API specification
- `Error_Codes.txt` - Error code documentation

These files are already created and ready to use for testing.

---

## Running Tests

The project includes comprehensive tests (130+ unit tests):

### Run All Tests

```bash
pytest
```

### Run Specific Test Suites

```bash
# Unit tests only
pytest tests/unit/ -v

# Guards tests
pytest tests/unit/test_guards.py -v

# Parsers tests
pytest tests/unit/test_parsers.py -v

# Integration tests (if available)
pytest tests/integration/ -v
```

### Run with Coverage Report

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term

# View HTML coverage report
start htmlcov/index.html   # Windows
open htmlcov/index.html    # Mac
xdg-open htmlcov/index.html  # Linux
```

### Run Specific Test Markers

```bash
pytest -m unit          # Unit tests
pytest -m guards        # Guard tests
pytest -m slow          # Slow tests
```

---

## Development Workflow

### Code Formatting

```bash
# Format code with Black
black src/ tests/

# Check formatting without changing
black --check src/ tests/
```

### Linting

```bash
# Run flake8
flake8 src/ tests/

# Run with specific rules
flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
```

### Type Checking

```bash
# Run mypy
mypy src/ --ignore-missing-imports
```

### Pre-commit Checks

```bash
# Run all checks before committing
black src/ tests/
flake8 src/ tests/
mypy src/ --ignore-missing-imports
pytest tests/unit/ -v
```

---

## Advanced Configuration

### Environment Variables

Edit `.env` to customize:

```bash
# LLM Configuration
LLM_PROVIDER=openai  # or "anthropic"
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0.3

# Retrieval Settings
TOP_K=5                         # Number of chunks to retrieve
MIN_CONFIDENCE=0.6              # Minimum similarity threshold
VECTOR_WEIGHT=0.6               # Weight for vector search in hybrid mode
KEYWORD_WEIGHT=0.4              # Weight for keyword search in hybrid mode
RRF_CONSTANT=60                 # Reciprocal Rank Fusion constant

# Chunking Settings
CHUNK_SIZE=1000                 # Token size per chunk
CHUNK_OVERLAP=150               # Overlap between chunks

# Safety Guardrails
ENABLE_HALLUCINATION_CHECK=true # Enable hallucination detection
ENABLE_INJECTION_DETECTION=true # Enable prompt injection protection
MIN_EVIDENCE_THRESHOLD=0.6      # Minimum evidence score

# Image Processing
ENABLE_OCR=true                 # Enable OCR for images
USE_VISION_API=false            # Use OpenAI Vision API (costs more)
OCR_DPI=200                     # DPI for image processing
EXTRACT_PDF_IMAGES=false        # Extract images from PDFs

# Logging
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
ENABLE_DEBUG_MODE=false         # Enable debug output
```

### Programmatic Usage

```python
from src.main import RAGSystem

# Initialize system
rag = RAGSystem(
    enable_hybrid=True,      # Enable hybrid retrieval
    enable_guards=True       # Enable safety guardrails
)

# Ingest documents
stats = rag.ingest("data/sample_dataset/user-signup/")
print(f"Ingested {stats['total_chunks']} chunks from {stats['files_processed']} files")

# Query system
result = rag.query(
    query_text="Create test cases for user authentication",
    debug=True  # Show retrieved chunks
)

# Check if query was blocked by safety guards
if result.get("safety_blocked"):
    print("Query blocked due to safety concerns")
    print(result.get("safety_report"))
else:
    # Process results
    for use_case in result["use_cases"]:
        print(f"- {use_case['title']}")
```

---

## Current Project Structure

Your project already has the following structure with implemented features:

```
multimodal-rag-system/
├── .env                              # Your API keys (create from .env.example)
├── .env.example                      # Template for environment variables
├── .gitignore                        # Git ignore rules
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup for pip install
├── pytest.ini                        # Pytest configuration
│
├── src/                              # ✅ Source code (fully implemented)
│   ├── config.py                     # Configuration management
│   ├── main.py                       # CLI interface
│   │
│   ├── ingestion/                    # Document processing
│   │   ├── pipeline.py               # Ingestion orchestration
│   │   ├── parsers/
│   │   │   ├── text_parser.py        # Text/MD/YAML/JSON parser
│   │   │   ├── pdf_parser.py         # PDF with OCR support
│   │   │   ├── docx_parser.py        # DOCX parser
│   │   │   └── image_parser.py       # Image OCR parser
│   │   ├── chunkers/
│   │   │   └── semantic_chunker.py   # Semantic chunking
│   │   └── embedders/
│   │       └── embedding_generator.py # OpenAI embeddings
│   │
│   ├── storage/
│   │   └── vector_store.py           # FAISS vector database
│   │
│   ├── retrieval/                    # Phase 2: Hybrid retrieval
│   │   ├── engine.py                 # Main retrieval engine
│   │   ├── keyword_search.py         # BM25 keyword search
│   │   └── hybrid_fusion.py          # RRF fusion
│   │
│   ├── generation/                   # LLM generation
│   │   ├── generator.py              # Base generator
│   │   ├── safe_generator.py         # Generator with guards
│   │   ├── prompt_templates.py       # Prompts
│   │   └── output_schemas.py         # Pydantic schemas
│   │
│   ├── guards/                       # Phase 3: Safety guardrails
│   │   ├── hallucination_detector.py # Hallucination prevention
│   │   ├── injection_detector.py     # Prompt injection protection
│   │   └── output_validator.py       # Output quality validation
│   │
│   └── evaluation/                   # Phase 3: Evaluation framework
│       ├── retrieval_metrics.py      # Recall, Precision, MRR, NDCG
│       ├── generation_metrics.py     # BLEU, ROUGE, BERTScore
│       ├── rag_metrics.py            # RAGAS metrics
│       └── evaluator.py              # Main evaluator
│
├── tests/                            # ✅ Comprehensive test suite
│   ├── conftest.py                   # Shared fixtures
│   ├── unit/
│   │   ├── test_guards.py            # Guard tests (28 tests)
│   │   ├── test_parsers.py           # Parser tests (40+ tests)
│   │   ├── test_chunkers.py          # Chunker tests (25 tests)
│   │   ├── test_embedders.py         # Embedder tests (15 tests)
│   │   └── test_retrieval.py         # Retrieval tests (25+ tests)
│   └── integration/                  # Integration tests (if available)
│
├── data/
│   ├── sample_dataset/               # ✅ Sample test documents
│   │   └── user-signup/
│   │       ├── PRD_Signup.md         # Product requirements
│   │       ├── API_Spec.yaml         # API specification
│   │       └── Error_Codes.txt       # Error codes
│   └── storage/                      # Generated during ingestion
│       └── vector_db/                # FAISS index + metadata
│
├── docs/                             # ✅ Comprehensive documentation
│   ├── PROJECT_SUMMARY.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── TECHNICAL_SPECIFICATION.md
│   ├── PHASE1_PROGRESS_REPORT.md
│   ├── PHASE2_PROGRESS_REPORT.md
│   ├── PHASE3_FINAL_COMPLETE_REPORT.md
│   └── SETUP_INSTRUCTIONS.md         # This file
│
└── scripts/                          # Utility scripts
    ├── ingest.py                     # Document ingestion script
    └── query.py                      # Query script
```

---

## Feature Status

### ✅ Phase 1: Core Infrastructure (COMPLETE)
- Text, PDF, DOCX parsing
- Semantic chunking with overlap
- OpenAI embeddings (text-embedding-3-small)
- FAISS vector storage
- Vector similarity search
- LLM generation (GPT-4-turbo)
- Structured JSON output with Pydantic validation

### ✅ Phase 2: Multimodal & Hybrid Search (COMPLETE)
- Image OCR support (Tesseract)
- PDF image extraction
- BM25 keyword search
- Hybrid retrieval (Vector + Keyword)
- Reciprocal Rank Fusion (RRF)
- Configurable fusion weights

### ✅ Phase 3: Safety & Quality (COMPLETE)
- Hallucination detection (NLI-based)
- Prompt injection protection (20+ patterns)
- Output validation (Pydantic + business rules)
- Comprehensive test suite (130+ tests)
- Evaluation framework (Recall, BLEU, ROUGE, RAGAS)

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'src'"
**Solution:**
```bash
# Make sure you're in the project root
cd D:\multimodal-rag-system

# Install package in editable mode
pip install -e .
```

#### 2. "OPENAI_API_KEY not set"
**Solution:**
```bash
# Check if .env exists
dir .env     # Windows
ls .env      # Linux/Mac

# If not, create it from template
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac

# Edit .env and add your key
notepad .env              # Windows
nano .env                 # Linux/Mac
```

#### 3. "Tesseract not found"
**Solution:**
```bash
# Windows: Download and install from
# https://github.com/UB-Mannheim/tesseract/wiki
# Then add to PATH: C:\Program Files\Tesseract-OCR

# Linux:
sudo apt-get install tesseract-ocr poppler-utils

# Mac:
brew install tesseract poppler

# Verify installation
tesseract --version
```

#### 4. "No documents found in vector store"
**Solution:**
```bash
# Ingest documents first
python -m src.main --ingest data/sample_dataset/user-signup/

# Verify ingestion
python -m src.main --stats
```

#### 5. Import errors or "module not found"
**Solution:**
```bash
# Ensure venv is activated (should see (venv) in terminal)
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

# Reinstall dependencies
pip install -e .[dev]
```

#### 6. pytest not found
**Solution:**
```bash
# Install with development dependencies
pip install -e .[dev]

# Or install pytest manually
pip install pytest pytest-cov
```

#### 7. Tests failing with API errors
**Solution:**
```bash
# Tests use mocked APIs, but if you see API errors:
# 1. Check .env has OPENAI_API_KEY set
# 2. Run specific unit tests that don't need API:
pytest tests/unit/test_parsers.py -v
pytest tests/unit/test_guards.py -v
```

---

## Quick Reference

### Essential Commands

```bash
# Setup
pip install -e .[dev]                          # Install with dev dependencies
python -c "import src; print('✓ Ready')"       # Verify installation

# Ingestion
python -m src.main --ingest data/sample_dataset/user-signup/
python -m src.main --ingest <folder> --hybrid  # With BM25 index

# Querying
python -m src.main --query "Create test cases"
python -m src.main --query "..." --hybrid --enable-guards --debug

# Statistics
python -m src.main --stats

# Testing
pytest                                         # All tests
pytest tests/unit/ -v                          # Unit tests
pytest tests/unit/test_guards.py -v            # Specific test file
pytest --cov=src --cov-report=html             # With coverage

# Code Quality
black src/ tests/                              # Format code
flake8 src/ tests/                             # Lint code
mypy src/ --ignore-missing-imports             # Type check
```

### Configuration (.env)

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional Features
ENABLE_HYBRID_RETRIEVAL=true        # Enable BM25 + Vector
ENABLE_HALLUCINATION_CHECK=true     # Enable hallucination detection
ENABLE_INJECTION_DETECTION=true     # Enable injection protection
ENABLE_OCR=true                     # Enable image OCR

# Tuning
TOP_K=5                             # Results per query
CHUNK_SIZE=1000                     # Tokens per chunk
VECTOR_WEIGHT=0.6                   # Vector weight in hybrid mode
KEYWORD_WEIGHT=0.4                  # Keyword weight in hybrid mode
```

---

## Next Steps

### If You're Running This for the First Time

1. **Verify Installation:**
   ```bash
   python -c "import src; print('✓ Installation successful')"
   pytest tests/unit/test_parsers.py::TestTextParser::test_parse_simple_text -v
   ```

2. **Ingest Sample Documents:**
   ```bash
   python -m src.main --ingest data/sample_dataset/user-signup/ --hybrid
   ```

3. **Run a Test Query:**
   ```bash
   python -m src.main --query "Create use cases for user signup" --enable-guards --debug
   ```

4. **Explore the Output:**
   - Check the JSON output
   - Review retrieved chunks (in debug mode)
   - Check safety report (if guards enabled)

### For Development

1. **Run Tests:**
   ```bash
   pytest tests/unit/ -v
   pytest --cov=src --cov-report=html
   open htmlcov/index.html
   ```

2. **Check Code Quality:**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/ --ignore-missing-imports
   ```

3. **Try Different Modes:**
   ```bash
   # Vector-only
   python -m src.main --query "..."

   # Hybrid retrieval
   python -m src.main --query "..." --hybrid

   # With safety guards
   python -m src.main --query "..." --enable-guards

   # All features
   python -m src.main --query "..." --hybrid --enable-guards --debug
   ```

---

## Resources

- **Project Documentation:** `docs/` folder
- **API Documentation:** See docstrings in source code
- **Test Examples:** `tests/` folder
- **Sample Data:** `data/sample_dataset/`
- **Configuration:** `.env.example`

---

**Setup Complete!** You have a production-ready RAG system with multimodal support, hybrid retrieval, and safety guardrails. 
