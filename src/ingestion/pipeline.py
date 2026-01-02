"""Ingestion pipeline for processing and indexing documents."""

from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import logging

from .parsers import TextParser, PDFParser, ImageParser, DOCXParser, Document
from .chunkers import SemanticChunker, Chunk
from .embedders import EmbeddingGenerator
from ..storage import VectorStore
from ..retrieval import KeywordSearch
from ..config import get_settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Complete pipeline for ingesting documents into the vector store.

    Pipeline stages:
    1. Parse files -> Documents
    2. Chunk documents -> Chunks
    3. Generate embeddings -> Vectors
    4. Store in vector database
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        keyword_search: Optional[KeywordSearch] = None,
        build_bm25: bool = False
    ):
        """
        Initialize ingestion pipeline.

        Args:
            vector_store: Existing vector store (or create new one)
            keyword_search: Optional BM25 keyword search engine
            build_bm25: Whether to build BM25 index during ingestion
        """
        settings = get_settings()

        # Initialize parsers
        self.text_parser = TextParser()
        self.pdf_parser = PDFParser(
            extract_images=settings.extract_pdf_images
        )

        # Phase 2: Image and DOCX parsers
        try:
            self.image_parser = ImageParser(
                use_vision_api=settings.use_vision_api,
                tesseract_path=settings.tesseract_path
            )
        except ImportError:
            logger.warning("Image parser dependencies not available")
            self.image_parser = None

        try:
            self.docx_parser = DOCXParser()
        except ImportError:
            logger.warning("DOCX parser dependencies not available")
            self.docx_parser = None

        # Chunking and embedding
        self.chunker = SemanticChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.embedder = EmbeddingGenerator()

        # Vector store
        if vector_store is None:
            embedding_dim = 1536  # OpenAI text-embedding-3-small dimension
            self.vector_store = VectorStore(dimension=embedding_dim)
        else:
            self.vector_store = vector_store

        # Phase 2: Keyword search
        self.keyword_search = keyword_search
        self.build_bm25 = build_bm25 or (keyword_search is not None)

        # Statistics
        self.stats = {
            "files_processed": 0,
            "documents_created": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "bm25_indexed": False,
            "errors": []
        }

        logger.info(
            f"IngestionPipeline initialized (BM25: {self.build_bm25}, "
            f"OCR: {self.image_parser is not None}, "
            f"DOCX: {self.docx_parser is not None})"
        )

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Ingest pre-parsed documents directly.

        Args:
            documents: List of Document objects to ingest

        Returns:
            Dictionary with ingestion statistics
        """
        try:
            # Step 1: Chunk documents
            chunks = self.chunker.chunk_documents(documents)

            # Step 2: Generate embeddings
            if chunks:
                embeddings = self.embedder.generate_for_chunks(chunks)

                # Step 3: Store in vector database
                contents = [chunk.content for chunk in chunks]
                metadata = [chunk.metadata for chunk in chunks]

                self.vector_store.add_vectors(embeddings, contents, metadata)

                # Update stats
                self.stats["documents_created"] += len(documents)
                self.stats["chunks_created"] += len(chunks)
                self.stats["embeddings_generated"] += len(embeddings)

            return {
                "success": True,
                "documents": len(documents),
                "chunks": len(chunks)
            }

        except Exception as e:
            error_msg = f"Error processing documents: {str(e)}"
            self.stats["errors"].append(error_msg)
            return {
                "success": False,
                "error": str(e)
            }

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single file.

        Args:
            file_path: Path to file to ingest

        Returns:
            Dictionary with ingestion statistics
        """
        path = Path(file_path)

        try:
            # Step 1: Parse file
            documents = self._parse_file(file_path)

            # Step 2: Chunk documents
            chunks = self.chunker.chunk_documents(documents)

            # Step 3: Generate embeddings
            if chunks:
                embeddings = self.embedder.generate_for_chunks(chunks)

                # Step 4: Store in vector database
                contents = [chunk.content for chunk in chunks]
                metadata = [chunk.metadata for chunk in chunks]

                self.vector_store.add_vectors(embeddings, contents, metadata)

                # Update stats
                self.stats["files_processed"] += 1
                self.stats["documents_created"] += len(documents)
                self.stats["chunks_created"] += len(chunks)
                self.stats["embeddings_generated"] += len(embeddings)

            return {
                "file": file_path,
                "success": True,
                "documents": len(documents),
                "chunks": len(chunks)
            }

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.stats["errors"].append(error_msg)
            return {
                "file": file_path,
                "success": False,
                "error": str(e)
            }

    def ingest_folder(self, folder_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Ingest all supported files in a folder.

        Args:
            folder_path: Path to folder containing files
            recursive: Whether to search subfolders

        Returns:
            Dictionary with ingestion statistics
        """
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find all supported files
        files = self._find_files(folder, recursive)

        if not files:
            print(f"No supported files found in {folder_path}")
            return self.stats

        print(f"\n[FOLDER] Ingesting {len(files)} files from {folder_path}")
        print("=" * 60)

        # Process each file with progress bar
        for file_path in tqdm(files, desc="Processing files"):
            result = self.ingest_file(str(file_path))

            if result["success"]:
                print(f"[OK] {file_path.name}: {result['chunks']} chunks created")
            else:
                print(f"[X] {file_path.name}: {result['error']}")

        # Phase 2: Build BM25 index if enabled
        if self.build_bm25 and self.keyword_search:
            self._build_bm25_index()

        # Print summary
        self._print_summary()

        return self.stats

    def save_vector_store(self, save_dir: str = None) -> None:
        """
        Save vector store to disk.

        Args:
            save_dir: Directory to save (defaults to config setting)
        """
        settings = get_settings()
        save_dir = save_dir or settings.vector_db_path
        self.vector_store.save(save_dir)

        # Phase 2: Save BM25 index if available
        if self.keyword_search and self.keyword_search.is_indexed:
            self.keyword_search.save(save_dir)
            logger.info(f"BM25 index saved to {save_dir}")

    def load_vector_store(self, load_dir: str = None) -> None:
        """
        Load vector store from disk.

        Args:
            load_dir: Directory to load from (defaults to config setting)
        """
        settings = get_settings()
        load_dir = load_dir or settings.vector_db_path
        self.vector_store.load(load_dir)

        # Phase 2: Load BM25 index if available
        if self.keyword_search:
            try:
                self.keyword_search.load(load_dir)
                self.stats["bm25_indexed"] = True
                logger.info(f"BM25 index loaded from {load_dir}")
            except FileNotFoundError:
                logger.info("BM25 index not found, will need to rebuild")

    def _parse_file(self, file_path: str) -> List[Document]:
        """
        Parse file based on extension.

        Args:
            file_path: Path to file

        Returns:
            List of Document objects
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        # Text files
        if extension in TextParser.SUPPORTED_EXTENSIONS:
            return self.text_parser.parse(file_path)

        # PDF files
        elif extension == '.pdf':
            return self.pdf_parser.parse(file_path)

        # Phase 2: Image files
        elif extension in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}:
            if self.image_parser:
                return self.image_parser.parse(file_path)
            else:
                raise ValueError("Image parser not available. Install pytesseract and pillow.")

        # Phase 2: DOCX files
        elif extension == '.docx':
            if self.docx_parser:
                return self.docx_parser.parse(file_path)
            else:
                raise ValueError("DOCX parser not available. Install python-docx.")

        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def _find_files(self, folder: Path, recursive: bool) -> List[Path]:
        """
        Find all supported files in folder.

        Args:
            folder: Folder path
            recursive: Whether to search subfolders

        Returns:
            List of file paths
        """
        # Phase 1 extensions
        supported_extensions = {'.txt', '.md', '.markdown', '.yaml', '.yml', '.json', '.pdf'}

        # Phase 2 extensions
        if self.image_parser:
            supported_extensions.update({'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'})
        if self.docx_parser:
            supported_extensions.add('.docx')

        files = []

        if recursive:
            for ext in supported_extensions:
                files.extend(folder.rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                files.extend(folder.glob(f"*{ext}"))

        return sorted(files)

    def _build_bm25_index(self):
        """Build BM25 index from vector store contents."""
        if not self.keyword_search:
            return

        print("\n[QUERY] Building BM25 keyword index...")

        # Get all chunks from vector store
        contents = self.vector_store.chunk_contents
        chunk_ids = self.vector_store.chunk_ids
        metadata = self.vector_store.chunk_metadata

        if not contents:
            logger.warning("No content to index for BM25")
            return

        # Build BM25 index
        self.keyword_search.build_index(
            contents=contents,
            chunk_ids=chunk_ids,
            metadata=metadata
        )

        self.stats["bm25_indexed"] = True
        print(f"[OK] BM25 index built with {len(contents)} documents")

    def _print_summary(self) -> None:
        """Print ingestion summary."""
        print("\n" + "=" * 60)
        print("[RESULTS] Ingestion Summary")
        print("=" * 60)
        print(f"Files processed:      {self.stats['files_processed']}")
        print(f"Documents created:    {self.stats['documents_created']}")
        print(f"Chunks created:       {self.stats['chunks_created']}")
        print(f"Embeddings generated: {self.stats['embeddings_generated']}")

        if self.stats['errors']:
            print(f"\n[WARN]  Errors: {len(self.stats['errors'])}")
            for error in self.stats['errors']:
                print(f"  - {error}")

        print("=" * 60)

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            **self.stats,
            "vector_store_stats": self.vector_store.get_stats()
        }
