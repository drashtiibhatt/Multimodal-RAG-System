# Web Interface Documentation

Modern web interface for the Multimodal RAG System with FastAPI backend and Tailwind CSS frontend.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Setup](#setup)
4. [Usage](#usage)
5. [API Reference](#api-reference)
6. [Architecture](#architecture)
7. [Screenshots](#screenshots)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The web interface provides a user-friendly way to interact with the RAG system through a modern, responsive UI.

### Technology Stack

**Backend:**
- FastAPI - Modern, fast Python web framework
- Uvicorn - ASGI server
- Pydantic - Data validation

**Frontend:**
- Tailwind CSS - Utility-first CSS framework
- Vanilla JavaScript - No framework overhead
- Responsive design - Works on all screen sizes

---

## Features

### 1. Query Interface
- Natural language query input
- Configurable search parameters (top-k, min-confidence)
- Hybrid search toggle
- Safety guards toggle (PII detection, hallucination check)
- Real-time use case generation
- Structured results display

### 2. Document Management
- Drag-and-drop file upload
- Support for PDF, TXT, DOCX
- View indexed documents
- Delete individual documents
- Clear all documents
- Real-time indexing status

### 3. System Settings
- LLM provider selection (OpenAI/Ollama)
- Model configuration
- Temperature adjustment
- Feature toggles:
  - Vision API
  - Hybrid retrieval
  - OCR
  - Hallucination detection
  - Injection detection

### 4. Statistics Dashboard
- Vector store statistics
- Cache metrics
- System health monitoring
- Auto-refresh every 30 seconds

### 5. Real-time Notifications
- Toast notifications for actions
- Success/error feedback
- Loading states

---

## Setup

### 1. Install Dependencies

```bash
# Install web dependencies
pip install fastapi uvicorn[standard] python-multipart

# Or install all requirements
pip install -r requirements.txt
```

### 2. Configure Environment

Ensure your `.env` file has the necessary configuration:

```bash
# LLM Configuration
OPENAI_API_KEY=your_api_key_here
LLM_PROVIDER=openai  # or ollama
LLM_MODEL=gpt-4-turbo-preview

# Vector Store
VECTOR_DB_PATH=data/vector_store

# Optional Features
USE_VISION_API=true
ENABLE_HYBRID_RETRIEVAL=true
ENABLE_HALLUCINATION_CHECK=true
ENABLE_INJECTION_DETECTION=true
```

### 3. Start the Server

```bash
# Using the startup script
python run_web.py

# Or directly with uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the Interface

Open your browser and navigate to:
- **Web Interface**: http://localhost:8000/app
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

---

## Usage

### Query Tab

1. **Enter Your Question**
   - Type a natural language query
   - Example: "Create test cases for user login with email validation"

2. **Configure Parameters**
   - **Top K**: Number of relevant documents to retrieve (1-20)
   - **Min Confidence**: Minimum confidence score (0-1)
   - **Hybrid Search**: Combine semantic + keyword search
   - **Safety Guards**: Enable PII detection and hallucination checks

3. **Generate Use Cases**
   - Click "Generate Use Cases" button
   - View structured results with:
     - Use case title and description
     - Acceptance criteria
     - Test scenarios
     - Metadata (confidence, assumptions)

### Documents Tab

1. **Upload Document**
   - Click the upload area or drag-and-drop a file
   - Supported formats: PDF, TXT, DOCX
   - Click "Upload & Index" to process
   - View indexing progress and results

2. **Manage Documents**
   - View list of indexed documents
   - See chunk count for each document
   - Delete individual documents
   - Clear all documents at once

### Settings Tab

1. **LLM Configuration**
   - Select provider (OpenAI/Ollama)
   - Choose model
   - Adjust temperature (0-1)

2. **Feature Toggles**
   - Enable/disable Vision API
   - Toggle hybrid retrieval
   - Configure OCR settings
   - Enable safety features

3. **Save Settings**
   - Click "Save Settings"
   - Some changes require server restart

### Statistics Tab

- **Vector Store**: Total vectors, unique sources
- **Embedding Cache**: Cached embeddings, hit rate
- **Query Cache**: Cached queries, performance metrics
- **Clear Cache**: Remove all cached data

---

## API Reference

### Health & Info

**GET /**
- Root endpoint with API information

**GET /health**
- Health check endpoint
- Response: `{ "status": "healthy", "total_vectors": 1234 }`

**GET /api/info**
- System configuration and status

### Query Endpoints

**POST /api/query**
- Process query and generate use cases

Request:
```json
{
  "query": "Create test cases for user login",
  "top_k": 5,
  "use_hybrid": false,
  "use_guards": true,
  "min_confidence": 0.6,
  "provider": "openai",
  "model": "gpt-4-turbo-preview"
}
```

Response:
```json
{
  "use_cases": [
    {
      "title": "User Login Test",
      "description": "...",
      "acceptance_criteria": [...],
      "test_scenarios": [...]
    }
  ],
  "metadata": {
    "query": "...",
    "confidence_score": 0.85,
    "assumptions": [...],
    "missing_information": [...]
  },
  "safety_report": {
    "has_pii": false,
    "hallucination_detected": false
  }
}
```

### Document Management

**POST /api/documents/upload**
- Upload and index a document
- Content-Type: `multipart/form-data`
- Field: `file`

**GET /api/documents**
- List all indexed documents

**DELETE /api/documents/{source}**
- Delete a specific document
- Source must be URL-encoded

**POST /api/documents/clear**
- Clear all documents from the index

### Statistics & Cache

**GET /api/stats**
- Get system statistics

**POST /api/cache/clear**
- Clear all caches

### Settings

**GET /api/settings**
- Get current settings

**PUT /api/settings**
- Update settings

Request:
```json
{
  "llm_provider": "ollama",
  "llm_model": "llama2",
  "temperature": 0.7,
  "use_vision_api": true,
  "enable_hybrid_retrieval": true
}
```

---

## Architecture

### Backend Architecture

```
src/api/
├── __init__.py          # Module initialization
└── main.py              # FastAPI application
    ├── Request/Response Models (Pydantic)
    ├── Startup/Shutdown Events
    ├── API Endpoints
    ├── CORS Middleware
    └── Static File Serving
```

### Frontend Architecture

```
web/
├── index.html           # Main HTML page
│   ├── Header with gradient
│   ├── Tab navigation
│   ├── Query interface
│   ├── Document management
│   ├── Settings panel
│   └── Statistics dashboard
└── static/
    └── app.js           # Frontend JavaScript
        ├── API client functions
        ├── UI update functions
        ├── Event handlers
        └── Utility functions
```

### Data Flow

```
User Input → Frontend (app.js)
    ↓
    fetch() API Request
    ↓
Backend (FastAPI) → Validate Request (Pydantic)
    ↓
Process Request:
  - Query → EmbeddingGenerator → VectorStore → Generator
  - Upload → IngestionPipeline → VectorStore
  - Settings → Config Updates
    ↓
Return Response (JSON)
    ↓
Frontend Updates UI
```

### Component Interaction

```
┌─────────────────────────────────────────────────┐
│               Browser (Frontend)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Query   │  │Documents │  │ Settings │      │
│  │   Tab    │  │   Tab    │  │   Tab    │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │             │              │             │
│       └─────────────┴──────────────┘             │
│                     │                            │
│              fetch() API Calls                   │
└─────────────────────┼───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│           FastAPI Backend (Server)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Query   │  │Document  │  │ Settings │      │
│  │Endpoints │  │Endpoints │  │Endpoints │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │             │              │             │
│       └─────────────┴──────────────┘             │
│                     │                            │
└─────────────────────┼───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              Core RAG System                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Vector  │  │Ingestion │  │Generator │      │
│  │  Store   │  │ Pipeline │  │(LLM)     │      │
│  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────┘
```

---

## Screenshots

### Query Interface
```
┌─────────────────────────────────────────────────────────┐
│  🤖 Multimodal RAG System                    ● System Ready │
├─────────────────────────────────────────────────────────┤
│  🔍 Query  │  📄 Documents  │  ⚙️ Settings  │  📊 Stats  │
├─────────────────────────────────────────────────────────┤
│ ┌──────────────────┐  ┌────────────────────────────┐   │
│ │ Ask a Question   │  │ Results                     │   │
│ │                  │  │                             │   │
│ │ [Text Area]      │  │ Use Case: User Login Test   │   │
│ │                  │  │ Description: ...            │   │
│ │ Top K:     [5]   │  │ Acceptance Criteria:        │   │
│ │ Min Conf: [0.6]  │  │ - Valid email format        │   │
│ │                  │  │ - Password requirements     │   │
│ │ ☑ Hybrid Search  │  │                             │   │
│ │ ☑ Safety Guards  │  │ Test Scenarios:             │   │
│ │                  │  │ 1. Valid credentials        │   │
│ │ [Generate]       │  │ 2. Invalid email            │   │
│ │                  │  │ 3. Wrong password           │   │
│ └──────────────────┘  └────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Document Management
```
┌─────────────────────────────────────────────────────────┐
│ ┌──────────────────┐  ┌────────────────────────────┐   │
│ │ Upload Document  │  │ Indexed Documents           │   │
│ │                  │  │                             │   │
│ │ [Drop Zone]      │  │ 📄 user_manual.pdf          │   │
│ │ PDF, TXT, DOCX   │  │    10 chunks          [Del] │   │
│ │                  │  │                             │   │
│ │ [Upload & Index] │  │ 📄 test_plan.docx           │   │
│ │ [Clear All]      │  │    5 chunks           [Del] │   │
│ └──────────────────┘  └────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Issue: "Module not found" Error

**Solution:**
```bash
# Ensure you're in the project root
cd multimodal-rag-system

# Install dependencies
pip install -r requirements.txt

# Run from project root
python run_web.py
```

### Issue: Port 8000 Already in Use

**Solution:**
```bash
# Use a different port
uvicorn src.api.main:app --port 8080

# Or kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9
```

### Issue: CORS Errors

**Solution:**

The API has CORS enabled for all origins by default. For production, update `src/api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: File Upload Fails

**Solution:**

1. Check file size limits (FastAPI default is 1MB)
2. Ensure `python-multipart` is installed:
   ```bash
   pip install python-multipart
   ```
3. Check file permissions in upload directory

### Issue: Static Files Not Loading

**Solution:**

Ensure the directory structure is correct:
```
multimodal-rag-system/
├── web/
│   ├── index.html
│   └── static/
│       └── app.js
└── src/
    └── api/
        └── main.py
```

The API mounts static files from `web/static/`.

### Issue: "No documents indexed" Error

**Solution:**

1. Upload documents via the Documents tab
2. Or use CLI to ingest documents:
   ```bash
   python -m src.main ingest docs/
   ```
3. Check vector store path in `.env`

### Issue: Slow Query Responses

**Solutions:**

1. **Enable Caching**: Already enabled by default
2. **Reduce top_k**: Use fewer results (e.g., top_k=3)
3. **Use Hybrid Search**: Can be faster for specific queries
4. **Disable Guards**: Turn off safety guards for faster responses
5. **Use Ollama**: Local models can be faster (no API latency)

### Issue: Settings Not Persisting

**Note**: Settings are stored in memory and reset on server restart.

**Solution for Persistent Settings**:

1. Update `.env` file directly
2. Restart the server
3. Or implement database-backed settings storage

---

## Performance Optimization

### Frontend Optimization

1. **Debounce Search**: Add delay to search input
2. **Pagination**: For large document lists
3. **Lazy Loading**: Load statistics on-demand
4. **Caching**: Browser caching for static assets

### Backend Optimization

1. **Connection Pooling**: Reuse database connections
2. **Background Tasks**: Use FastAPI BackgroundTasks for async operations
3. **Streaming**: Implement streaming responses for long queries
4. **Caching**: Redis for distributed caching
5. **Rate Limiting**: Add rate limiting middleware

### Example: Streaming Responses

```python
from fastapi.responses import StreamingResponse

@app.post("/api/query/stream")
async def stream_query(request: QueryRequest):
    async def generate():
        # Yield chunks as they're generated
        for chunk in generator.generate_stream(query):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## Security Considerations

### Production Deployment

1. **HTTPS**: Always use HTTPS in production
2. **API Keys**: Never expose API keys in frontend
3. **CORS**: Restrict origins to your domain
4. **Rate Limiting**: Implement rate limiting
5. **Authentication**: Add user authentication
6. **Input Validation**: Already implemented via Pydantic
7. **File Upload**: Validate file types and sizes
8. **SQL Injection**: Not applicable (no SQL database)

### Example: Adding Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

@app.post("/api/query", dependencies=[Depends(verify_token)])
async def process_query(request: QueryRequest):
    # Protected endpoint
    pass
```

---

## Deployment

### Local Development

```bash
python run_web.py
```

### Production (Uvicorn)

```bash
uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VECTOR_DB_PATH=/app/data/vector_store
    restart: unless-stopped
```

---

## Future Enhancements

1. **Streaming Responses**: Real-time token streaming
2. **WebSocket Support**: Live updates
3. **User Authentication**: Multi-user support
4. **File Preview**: Preview documents before upload
5. **Advanced Filters**: Filter by document type, date
6. **Batch Operations**: Upload multiple files
7. **Export Results**: Download as PDF, CSV, JSON
8. **Dark Mode**: Theme toggle
9. **Mobile App**: Native mobile interface
10. **Collaboration**: Share queries and results

---

## Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Tailwind CSS**: https://tailwindcss.com/
- **API Docs (Interactive)**: http://localhost:8000/docs
- **Source Code**: `src/api/main.py`, `web/`
- **Startup Script**: `run_web.py`

---

## Summary

The Web Interface provides:

✅ **Modern UI** with Tailwind CSS
✅ **RESTful API** with FastAPI
✅ **Document Management** with upload/delete
✅ **Query Interface** with configurable parameters
✅ **Settings Management** for system configuration
✅ **Statistics Dashboard** for monitoring
✅ **Real-time Updates** with auto-refresh
✅ **Toast Notifications** for user feedback
✅ **Responsive Design** for all devices
✅ **Interactive API Docs** at /docs

Quick Start:
```bash
pip install -r requirements.txt
python run_web.py
# Open http://localhost:8000/app
```
