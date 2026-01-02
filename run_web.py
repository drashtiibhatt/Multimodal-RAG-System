"""Web server startup script for RAG System."""

import sys
import logging
from pathlib import Path
import os
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Clear Python cache to avoid import issues
def clear_cache():
    """Clear Python bytecode cache."""
    cache_dirs = []
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            cache_dir = os.path.join(root, "__pycache__")
            cache_dirs.append(cache_dir)

    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
        except Exception:
            pass

clear_cache()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Start the web server."""
    try:
        import uvicorn
        from src.api import app

        logger.info("=" * 60)
        logger.info("Starting Multimodal RAG System Web Interface")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Web Interface: http://localhost:8000/app")
        logger.info("API Documentation: http://localhost:8000/docs")
        logger.info("API Base URL: http://localhost:8000")
        logger.info("")
        logger.info("Press CTRL+C to stop the server")
        logger.info("=" * 60)

        # Start server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install web dependencies:")
        logger.error("  pip install fastapi uvicorn python-multipart")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
