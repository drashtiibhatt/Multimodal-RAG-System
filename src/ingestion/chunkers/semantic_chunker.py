"""Semantic text chunking with overlap for context preservation."""

from typing import List, Dict, Any
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..parsers.base import Document


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    content: str
    metadata: Dict[str, Any]
    chunk_id: str

    def __repr__(self) -> str:
        """String representation."""
        return f"Chunk(id={self.chunk_id}, length={len(self.content)})"


class SemanticChunker:
    """
    Intelligent text chunking that preserves semantic boundaries.

    Uses recursive character splitting to maintain context while
    creating manageable chunks for embedding.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: List[str] = None
    ):
        """
        Initialize semantic chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to split on (in priority order)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        # Initialize LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split document into semantic chunks.

        Args:
            document: Document object to chunk

        Returns:
            List of Chunk objects with preserved metadata
        """
        # Get text content
        text = document.page_content

        # Split text into chunks
        chunk_texts = self.splitter.split_text(text)

        # Create Chunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            # Generate unique chunk ID
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page", "")
            page_suffix = f":p{page}" if page else ""
            chunk_id = f"{source}{page_suffix}:chunk_{idx}"

            # Create chunk metadata (inherit from document + add chunk info)
            chunk_metadata = {
                **document.metadata,
                "chunk_index": idx,
                "chunk_id": chunk_id,
                "total_chunks": len(chunk_texts),
                "chunk_char_count": len(chunk_text)
            }

            # Create Chunk object
            chunk = Chunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            )

            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of Document objects

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)

        return all_chunks
