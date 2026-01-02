"""FastAPI application for RAG system web interface."""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import logging
from pathlib import Path
import tempfile
import shutil

from ..config import get_settings
from ..storage import VectorStore
from ..generation import UseCaseGenerator
from ..generation.safe_generator import SafeGenerator
from ..ingestion import IngestionPipeline
from ..caching import CacheManager

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG System API",
    description="REST API for document ingestion and intelligent query processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vector_store: Optional[VectorStore] = None
pipeline: Optional[IngestionPipeline] = None
cache_manager = CacheManager()


# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_hybrid: bool = False
    use_guards: bool = True
    min_confidence: float = 0.6
    provider: Optional[str] = None
    model: Optional[str] = None


class QueryResponse(BaseModel):
    use_cases: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    safety_report: Optional[Dict[str, Any]] = None


class DocumentInfo(BaseModel):
    source: str
    total_chunks: int
    chunk_ids: List[str]


class SettingsUpdate(BaseModel):
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: Optional[float] = None
    use_vision_api: Optional[bool] = None
    enable_hybrid_retrieval: Optional[bool] = None


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global vector_store, pipeline

    logger.info("Starting RAG System API...")

    try:
        # Initialize vector store
        settings = get_settings()
        vector_store = VectorStore()

        # Try to load existing index
        if Path(settings.vector_db_path).exists():
            vector_store.load(settings.vector_db_path)
            logger.info(f"Loaded vector store: {vector_store.get_stats()}")
        else:
            logger.info("No existing vector store found")

        # Initialize ingestion pipeline
        pipeline = IngestionPipeline(vector_store=vector_store)

        logger.info("RAG System API ready!")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG System API...")


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multimodal RAG System API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "query": "/api/query",
            "upload": "/api/documents/upload",
            "stats": "/api/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = {}

    if vector_store:
        stats = vector_store.get_stats()

    return {
        "status": "healthy",
        "vector_store_loaded": vector_store is not None,
        "total_vectors": stats.get("total_vectors", 0),
        "unique_sources": stats.get("unique_sources", 0)
    }


@app.get("/api/info")
async def get_system_info():
    """Get system configuration and status."""
    settings = get_settings()

    info = {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "temperature": settings.temperature,
        "use_vision_api": settings.use_vision_api,
        "enable_hybrid_retrieval": settings.enable_hybrid_retrieval,
        "enable_hallucination_check": settings.enable_hallucination_check,
        "enable_injection_detection": settings.enable_injection_detection
    }

    if vector_store:
        info["vector_store"] = vector_store.get_stats()

    return info


# ============================================================================
# Query Endpoints
# ============================================================================

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query and generate use cases."""
    if not vector_store or vector_store.get_stats()["total_vectors"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please upload documents first."
        )

    try:
        logger.info(f"Processing query: {request.query}")

        # Search vector store
        from ..ingestion import EmbeddingGenerator
        embedder = EmbeddingGenerator()

        query_embedding = embedder.generate([request.query])

        # Get search results
        if request.use_hybrid:
            # TODO: Implement hybrid search
            results = vector_store.search(query_embedding[0], top_k=request.top_k)
        else:
            results = vector_store.search(query_embedding[0], top_k=request.top_k)

        # Format context
        context = "\n\n---\n\n".join([
            f"[Document {i+1}] {r.metadata.get('source', 'unknown')}\n{r.content}"
            for i, r in enumerate(results)
        ])

        # Generate use cases
        if request.use_guards:
            generator = SafeGenerator(
                model=request.model,
                temperature=get_settings().temperature
            )
            output, safety_report = generator.generate(
                query=request.query,
                context=context,
                min_confidence=request.min_confidence,
                debug=False
            )
        else:
            generator = UseCaseGenerator(
                model=request.model,
                provider=request.provider
            )
            output = generator.generate(
                query=request.query,
                context=context,
                min_confidence=request.min_confidence,
                debug=False
            )
            safety_report = None

        # Prepare response
        response = QueryResponse(
            use_cases=output.get("use_cases", []),
            metadata={
                "query": request.query,
                "top_k": request.top_k,
                "results_found": len(results),
                "confidence_score": output.get("confidence_score", 0.0),
                "assumptions": output.get("assumptions", []),
                "missing_information": output.get("missing_information", [])
            },
            safety_report=safety_report.__dict__ if safety_report else None
        )

        logger.info(f"Generated {len(response.use_cases)} use cases")
        return response

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and index a document."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        # Create temp directory for uploads
        temp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
        temp_dir.mkdir(exist_ok=True)

        # Save with original filename to temp directory
        temp_file_path = temp_dir / file.filename

        # If file exists, add timestamp to make it unique
        if temp_file_path.exists():
            import time
            stem = temp_file_path.stem
            suffix = temp_file_path.suffix
            temp_file_path = temp_dir / f"{stem}_{int(time.time())}{suffix}"

        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"Processing uploaded file: {file.filename}")

        # Process document (source will be the temp path with original filename)
        result = pipeline.ingest_file(str(temp_file_path))

        # Check if processing succeeded
        if not result.get("success", True):
            error_msg = result.get("error", "Unknown error during processing")
            logger.error(f"Document processing failed: {error_msg}")
            # Cleanup temp file
            if temp_file_path.exists():
                temp_file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Processing failed: {error_msg}")

        # Update metadata to use just the filename as source for cleaner display
        # Find all chunks with this temp path and update their source
        temp_path_str = str(temp_file_path)
        for idx, metadata in enumerate(vector_store.chunk_metadata):
            if metadata.get("source") == temp_path_str:
                vector_store.chunk_metadata[idx]["source"] = file.filename

        # Save updated index
        settings = get_settings()
        vector_store.save(settings.vector_db_path)

        # Cleanup temp file
        temp_file_path.unlink()

        stats = vector_store.get_stats()

        return {
            "status": "success",
            "filename": file.filename,
            "message": f"Document indexed successfully",
            "total_vectors": stats["total_vectors"],
            "unique_sources": stats["unique_sources"]
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        # Cleanup on error
        if 'temp_file_path' in locals() and Path(temp_file_path).exists():
            Path(temp_file_path).unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all indexed documents."""
    if not vector_store:
        return []

    try:
        sources = vector_store.list_sources()
        documents = []

        for source in sources:
            stats = vector_store.get_source_stats(source)
            documents.append(DocumentInfo(
                source=stats["source"],
                total_chunks=stats["total_chunks"],
                chunk_ids=stats["chunk_ids"]
            ))

        return documents

    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{source}")
async def delete_document(source: str):
    """Delete a document from the index."""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        # Decode source (URL encoded)
        from urllib.parse import unquote
        source = unquote(source)

        deleted_count = vector_store.delete_by_source(source)

        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Document not found: {source}")

        # Save updated index
        settings = get_settings()
        vector_store.save(settings.vector_db_path)

        return {
            "status": "success",
            "source": source,
            "chunks_deleted": deleted_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/clear")
async def clear_all_documents():
    """Clear all documents from the index."""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        stats_before = vector_store.get_stats()
        vector_store.clear()

        # Save empty index
        settings = get_settings()
        vector_store.save(settings.vector_db_path)

        return {
            "status": "success",
            "message": "All documents cleared",
            "vectors_deleted": stats_before["total_vectors"]
        }

    except Exception as e:
        logger.error(f"Clear failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Statistics & Cache Endpoints
# ============================================================================

@app.get("/api/stats")
async def get_statistics():
    """Get system statistics."""
    stats = {
        "vector_store": {},
        "cache": {}
    }

    if vector_store:
        stats["vector_store"] = vector_store.get_stats()

    # Get cache stats
    stats["cache"] = cache_manager.get_all_stats()

    return stats


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all caches."""
    try:
        result = cache_manager.clear_all()

        return {
            "status": "success",
            "embeddings_cleared": result["embeddings_cleared"],
            "queries_cleared": result["queries_cleared"]
        }

    except Exception as e:
        logger.error(f"Cache clear failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Settings Endpoints
# ============================================================================

@app.get("/api/settings")
async def get_settings_api():
    """Get current settings."""
    settings = get_settings()

    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "ollama_model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
        "temperature": settings.temperature,
        "use_vision_api": settings.use_vision_api,
        "vision_model": settings.vision_model,
        "enable_hybrid_retrieval": settings.enable_hybrid_retrieval,
        "enable_ocr": settings.enable_ocr,
        "enable_hallucination_check": settings.enable_hallucination_check,
        "enable_injection_detection": settings.enable_injection_detection,
        "top_k": settings.top_k,
        "min_confidence": settings.min_confidence
    }


@app.put("/api/settings")
async def update_settings_api(update: SettingsUpdate):
    """Update settings (note: requires restart for some changes)."""
    settings = get_settings()

    updated_fields = []

    if update.llm_provider is not None:
        settings.llm_provider = update.llm_provider
        updated_fields.append("llm_provider")

    if update.llm_model is not None:
        settings.llm_model = update.llm_model
        updated_fields.append("llm_model")

    if update.temperature is not None:
        settings.temperature = update.temperature
        updated_fields.append("temperature")

    if update.use_vision_api is not None:
        settings.use_vision_api = update.use_vision_api
        updated_fields.append("use_vision_api")

    if update.enable_hybrid_retrieval is not None:
        settings.enable_hybrid_retrieval = update.enable_hybrid_retrieval
        updated_fields.append("enable_hybrid_retrieval")

    return {
        "status": "success",
        "message": "Settings updated (some changes require restart)",
        "updated_fields": updated_fields
    }


# ============================================================================
# Mount static files (frontend)
# ============================================================================

# Mount static files for frontend
static_dir = Path(__file__).parent.parent.parent / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve frontend
frontend_dir = Path(__file__).parent.parent.parent / "web"
if (frontend_dir / "index.html").exists():
    @app.get("/app")
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
