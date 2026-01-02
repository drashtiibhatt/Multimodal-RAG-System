"""Base parser class for all document parsers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Represents a parsed document with content and metadata."""

    page_content: str
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        """String representation."""
        return f"Document(content_length={len(self.page_content)}, metadata={self.metadata})"


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """
        Parse a file and return list of Document objects.

        Args:
            file_path: Path to the file to parse

        Returns:
            List of Document objects with content and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        pass

    def _validate_file_exists(self, file_path: str) -> Path:
        """
        Validate that file exists.

        Args:
            file_path: Path to validate

        Returns:
            Path object

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        return path

    def _create_metadata(
        self,
        file_path: str,
        file_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create metadata dictionary for a document.

        Args:
            file_path: Path to source file
            file_type: Type of file (text, pdf, etc.)
            **kwargs: Additional metadata fields

        Returns:
            Dictionary with metadata
        """
        path = Path(file_path)
        metadata = {
            "source": str(path),
            "file_name": path.name,
            "file_type": file_type,
            **kwargs
        }
        return metadata
