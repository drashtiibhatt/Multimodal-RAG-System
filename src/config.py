"""Configuration management for the RAG system."""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

    # Model Configuration
    llm_provider: str = Field("openai", env="LLM_PROVIDER")  # "openai" or "ollama"
    llm_model: str = Field("gpt-4-turbo-preview", env="LLM_MODEL")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    temperature: float = Field(0.3, env="TEMPERATURE")

    # Ollama Configuration (for local models)
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama2", env="OLLAMA_MODEL")

    # Retrieval Configuration
    top_k: int = Field(5, env="TOP_K")
    min_confidence: float = Field(0.6, env="MIN_CONFIDENCE")

    # Phase 2: Hybrid Retrieval Configuration
    enable_hybrid_retrieval: bool = Field(False, env="ENABLE_HYBRID_RETRIEVAL")
    vector_weight: float = Field(0.6, env="VECTOR_WEIGHT")
    keyword_weight: float = Field(0.4, env="KEYWORD_WEIGHT")
    rrf_constant: int = Field(60, env="RRF_CONSTANT")

    # Chunking Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(150, env="CHUNK_OVERLAP")

    # Image Processing Configuration
    enable_ocr: bool = Field(True, env="ENABLE_OCR")
    use_vision_api: bool = Field(False, env="USE_VISION_API")
    vision_model: str = Field("gpt-4o-mini", env="VISION_MODEL")
    tesseract_path: Optional[str] = Field(None, env="TESSERACT_PATH")
    poppler_path: Optional[str] = Field(None, env="POPPLER_PATH")
    ocr_dpi: int = Field(200, env="OCR_DPI")
    extract_pdf_images: bool = Field(False, env="EXTRACT_PDF_IMAGES")

    # Guardrails Configuration
    enable_hallucination_check: bool = Field(True, env="ENABLE_HALLUCINATION_CHECK")
    enable_injection_detection: bool = Field(True, env="ENABLE_INJECTION_DETECTION")
    min_evidence_threshold: float = Field(0.6, env="MIN_EVIDENCE_THRESHOLD")

    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_debug_mode: bool = Field(False, env="ENABLE_DEBUG_MODE")

    # Storage Paths
    vector_db_path: str = Field("data/storage/vector_db", env="VECTOR_DB_PATH")
    metadata_db_path: str = Field("data/storage/metadata.json", env="METADATA_DB_PATH")
    cache_path: str = Field("data/storage/cache", env="CACHE_PATH")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Convenience functions
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings
