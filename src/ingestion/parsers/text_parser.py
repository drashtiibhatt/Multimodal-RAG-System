"""Parser for text and markdown files."""

from typing import List
from pathlib import Path
from .base import BaseParser, Document


class TextParser(BaseParser):
    """Parser for plain text and markdown files."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".yaml", ".yml", ".json"}

    def parse(self, file_path: str) -> List[Document]:
        """
        Parse text or markdown file.

        Args:
            file_path: Path to text file

        Returns:
            List containing single Document object

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is invalid
        """
        # Validate file exists
        path = self._validate_file_exists(file_path)

        # Check if file extension is supported
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        # Read file content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(path, 'r', encoding='latin-1') as f:
                content = f.read()

        # Create metadata
        metadata = self._create_metadata(
            file_path=file_path,
            file_type="text",
            char_count=len(content),
            line_count=content.count('\n') + 1,
            extension=path.suffix
        )

        # Create Document object
        document = Document(
            page_content=content,
            metadata=metadata
        )

        return [document]
