"""
Unit tests for document parsers.

Tests text, PDF, image, and DOCX parsers with various
file formats and edge cases.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

from src.ingestion.parsers import (
    BaseParser,
    Document,
    TextParser,
    PDFParser,
    ImageParser,
    DOCXParser
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_text_content():
    """Provide sample text content for testing."""
    return """User Authentication System

This document describes the user authentication system.

Requirements:
1. Users must provide valid email
2. Password must be secure
3. Session management required

Contact: admin@example.com"""


@pytest.fixture
def sample_markdown_content():
    """Provide sample markdown content."""
    return """# User Guide

## Features

- Feature 1: User signup
- Feature 2: User login
- Feature 3: Password reset

## Installation

Run the following command:
```bash
pip install requirements
```"""


@pytest.fixture
def create_temp_text_file(temp_dir):
    """Factory fixture to create temporary text files."""
    def _create_file(filename: str, content: str, encoding: str = 'utf-8'):
        file_path = temp_dir / filename
        file_path.write_text(content, encoding=encoding)
        return str(file_path)
    return _create_file


@pytest.fixture
def create_temp_image(temp_dir):
    """Factory fixture to create temporary image files."""
    def _create_image(filename: str, width: int = 100, height: int = 100, color: str = "white"):
        file_path = temp_dir / filename
        image = Image.new('RGB', (width, height), color=color)
        image.save(file_path)
        return str(file_path)
    return _create_image


# ==================== TextParser Tests ====================

class TestTextParser:
    """Tests for TextParser."""

    @pytest.fixture
    def parser(self):
        """Create a TextParser instance."""
        return TextParser()

    def test_parser_initialization(self, parser):
        """Test parser initializes correctly."""
        assert parser is not None
        assert hasattr(parser, 'SUPPORTED_EXTENSIONS')
        assert '.txt' in parser.SUPPORTED_EXTENSIONS
        assert '.md' in parser.SUPPORTED_EXTENSIONS

    def test_parse_txt_file(self, parser, create_temp_text_file, sample_text_content):
        """Test parsing plain text file."""
        file_path = create_temp_text_file("test.txt", sample_text_content)
        documents = parser.parse(file_path)

        assert len(documents) == 1
        doc = documents[0]
        assert isinstance(doc, Document)
        assert doc.page_content == sample_text_content
        assert doc.metadata['file_type'] == 'text'
        assert doc.metadata['extension'] == '.txt'
        assert doc.metadata['char_count'] == len(sample_text_content)
        assert 'line_count' in doc.metadata

    def test_parse_markdown_file(self, parser, create_temp_text_file, sample_markdown_content):
        """Test parsing markdown file."""
        file_path = create_temp_text_file("test.md", sample_markdown_content)
        documents = parser.parse(file_path)

        assert len(documents) == 1
        doc = documents[0]
        assert doc.page_content == sample_markdown_content
        assert doc.metadata['extension'] == '.md'

    def test_parse_yaml_file(self, parser, create_temp_text_file):
        """Test parsing YAML file."""
        yaml_content = """
version: 1.0
features:
  - authentication
  - authorization
"""
        file_path = create_temp_text_file("config.yaml", yaml_content)
        documents = parser.parse(file_path)

        assert len(documents) == 1
        assert documents[0].metadata['extension'] == '.yaml'

    def test_parse_json_file(self, parser, create_temp_text_file):
        """Test parsing JSON file."""
        json_content = '{"name": "test", "version": "1.0"}'
        file_path = create_temp_text_file("data.json", json_content)
        documents = parser.parse(file_path)

        assert len(documents) == 1
        assert documents[0].page_content == json_content
        assert documents[0].metadata['extension'] == '.json'

    def test_parse_nonexistent_file(self, parser):
        """Test parsing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.txt")

    def test_parse_unsupported_extension(self, parser, create_temp_text_file):
        """Test parsing unsupported file extension."""
        file_path = create_temp_text_file("test.exe", "content")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            parser.parse(file_path)

    def test_parse_empty_file(self, parser, create_temp_text_file):
        """Test parsing empty file."""
        file_path = create_temp_text_file("empty.txt", "")
        documents = parser.parse(file_path)

        assert len(documents) == 1
        assert documents[0].page_content == ""
        assert documents[0].metadata['char_count'] == 0

    def test_parse_unicode_content(self, parser, create_temp_text_file):
        """Test parsing file with unicode characters."""
        unicode_content = "Hello 世界 🌍 Привет"
        file_path = create_temp_text_file("unicode.txt", unicode_content)
        documents = parser.parse(file_path)

        assert len(documents) == 1
        assert documents[0].page_content == unicode_content

    def test_encoding_fallback(self, parser, temp_dir):
        """Test fallback to latin-1 encoding."""
        # Create file with latin-1 encoding
        file_path = temp_dir / "latin1.txt"
        content = "Café résumé"
        with open(file_path, 'w', encoding='latin-1') as f:
            f.write(content)

        # Parser should handle it
        documents = parser.parse(str(file_path))
        assert len(documents) == 1


# ==================== PDFParser Tests ====================

class TestPDFParser:
    """Tests for PDFParser."""

    @pytest.fixture
    def parser(self):
        """Create a PDFParser instance."""
        return PDFParser(extract_images=False)

    @pytest.fixture
    def parser_with_images(self):
        """Create a PDFParser with image extraction."""
        return PDFParser(extract_images=True)

    def test_parser_initialization(self, parser):
        """Test parser initializes correctly."""
        assert parser is not None
        assert parser.extract_images == False

    def test_parser_with_images_initialization(self, parser_with_images):
        """Test parser with image extraction initializes."""
        assert parser_with_images.extract_images == True

    @patch('pdfplumber.open')
    def test_parse_pdf_single_page(self, mock_pdf_open, parser, temp_dir):
        """Test parsing single-page PDF."""
        # Mock PDF content
        mock_page = Mock()
        mock_page.extract_text.return_value = "This is page 1 content."

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        mock_pdf_open.return_value = mock_pdf

        # Create dummy PDF file
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        # Parse PDF
        documents = parser.parse(str(pdf_path))

        assert len(documents) == 1
        doc = documents[0]
        assert doc.page_content == "This is page 1 content."
        assert doc.metadata['file_type'] == 'pdf'
        assert doc.metadata['page'] == 1
        assert doc.metadata['total_pages'] == 1

    @patch('pdfplumber.open')
    def test_parse_pdf_multiple_pages(self, mock_pdf_open, parser, temp_dir):
        """Test parsing multi-page PDF."""
        # Mock PDF with 3 pages
        mock_pages = []
        for i in range(3):
            mock_page = Mock()
            mock_page.extract_text.return_value = f"Page {i+1} content."
            mock_pages.append(mock_page)

        mock_pdf = Mock()
        mock_pdf.pages = mock_pages
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        mock_pdf_open.return_value = mock_pdf

        pdf_path = temp_dir / "multi.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        documents = parser.parse(str(pdf_path))

        assert len(documents) == 3
        for i, doc in enumerate(documents, 1):
            assert f"Page {i} content." in doc.page_content
            assert doc.metadata['page'] == i
            assert doc.metadata['total_pages'] == 3

    @patch('pdfplumber.open')
    def test_parse_pdf_empty_page(self, mock_pdf_open, parser, temp_dir):
        """Test handling of empty PDF pages."""
        mock_page = Mock()
        mock_page.extract_text.return_value = ""

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        mock_pdf_open.return_value = mock_pdf

        pdf_path = temp_dir / "empty.pdf"
        pdf_path.write_bytes(b"dummy")

        documents = parser.parse(str(pdf_path))

        assert len(documents) == 1
        assert "[Empty page 1]" in documents[0].page_content

    def test_parse_non_pdf_file(self, parser, temp_dir):
        """Test parsing non-PDF file raises error."""
        txt_path = temp_dir / "notpdf.txt"
        txt_path.write_text("not a pdf")

        with pytest.raises(ValueError, match="not a PDF"):
            parser.parse(str(txt_path))

    def test_parse_nonexistent_pdf(self, parser):
        """Test parsing nonexistent PDF raises error."""
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.pdf")

    @patch('pdfplumber.open')
    def test_parse_corrupted_pdf(self, mock_pdf_open, parser, temp_dir):
        """Test handling corrupted PDF."""
        mock_pdf_open.side_effect = Exception("PDF corrupted")

        pdf_path = temp_dir / "corrupt.pdf"
        pdf_path.write_bytes(b"corrupted")

        with pytest.raises(Exception, match="Failed to parse PDF"):
            parser.parse(str(pdf_path))


# ==================== ImageParser Tests ====================

class TestImageParser:
    """Tests for ImageParser."""

    @pytest.fixture
    def parser_tesseract(self):
        """Create ImageParser with Tesseract."""
        return ImageParser(use_vision_api=False)

    @pytest.fixture
    def parser_vision(self):
        """Create ImageParser with Vision API."""
        # Mock settings
        with patch('src.ingestion.parsers.image_parser.get_settings') as mock_settings:
            mock_settings.return_value.openai_api_key = "test_key"
            with patch('src.ingestion.parsers.image_parser.OpenAI'):
                return ImageParser(use_vision_api=True)

    def test_parser_initialization_tesseract(self, parser_tesseract):
        """Test parser initializes with Tesseract."""
        assert parser_tesseract is not None
        assert parser_tesseract.use_vision_api == False

    def test_supported_extensions(self, parser_tesseract):
        """Test supported image extensions."""
        assert '.png' in parser_tesseract.SUPPORTED_EXTENSIONS
        assert '.jpg' in parser_tesseract.SUPPORTED_EXTENSIONS
        assert '.jpeg' in parser_tesseract.SUPPORTED_EXTENSIONS

    @patch('src.ingestion.parsers.image_parser.TESSERACT_AVAILABLE', True)
    @patch('pytesseract.image_to_string')
    @patch('PIL.Image.open')
    def test_parse_image_with_tesseract(
        self,
        mock_image_open,
        mock_tesseract,
        parser_tesseract,
        create_temp_image
    ):
        """Test parsing image with Tesseract OCR."""
        # Mock Tesseract output
        mock_tesseract.return_value = "Extracted text from image"

        # Mock PIL Image
        mock_img = Mock(spec=Image.Image)
        mock_img.format = "PNG"
        mock_img.mode = "RGB"
        mock_img.width = 100
        mock_img.height = 100
        mock_image_open.return_value = mock_img

        # Create test image
        img_path = create_temp_image("test.png")

        # Parse image
        documents = parser_tesseract.parse(img_path)

        assert len(documents) == 1
        doc = documents[0]
        assert doc.page_content == "Extracted text from image"
        assert doc.metadata['file_type'] == 'image'
        assert doc.metadata['ocr_method'] == 'tesseract'
        assert doc.metadata['format'] == 'PNG'

    def test_parse_unsupported_image_format(self, parser_tesseract, temp_dir):
        """Test parsing unsupported image format."""
        # Create file with unsupported extension
        file_path = temp_dir / "image.svg"
        file_path.write_text("<svg></svg>")

        with pytest.raises(ValueError, match="Unsupported image format"):
            parser_tesseract.parse(str(file_path))

    def test_parse_nonexistent_image(self, parser_tesseract):
        """Test parsing nonexistent image."""
        with pytest.raises(FileNotFoundError):
            parser_tesseract.parse("nonexistent.png")

    @patch('src.ingestion.parsers.image_parser.TESSERACT_AVAILABLE', False)
    def test_tesseract_not_available(self, parser_tesseract, create_temp_image):
        """Test error when Tesseract is not available."""
        img_path = create_temp_image("test.png")

        with pytest.raises(ValueError, match="Tesseract OCR is not available"):
            parser_tesseract.parse(img_path)

    @patch('src.ingestion.parsers.image_parser.VISION_AVAILABLE', True)
    @patch('PIL.Image.open')
    def test_parse_image_with_vision_api(
        self,
        mock_image_open,
        parser_vision,
        create_temp_image
    ):
        """Test parsing image with Vision API."""
        # Mock PIL Image for metadata
        mock_img = Mock(spec=Image.Image)
        mock_img.format = "JPEG"
        mock_img.mode = "RGB"
        mock_img.width = 200
        mock_img.height = 150
        mock_image_open.return_value = mock_img

        # Mock OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Text extracted via Vision API"

        parser_vision.client = Mock()
        parser_vision.client.chat.completions.create.return_value = mock_response

        # Create test image
        img_path = create_temp_image("test.jpg")

        # Parse image
        documents = parser_vision.parse(img_path)

        assert len(documents) == 1
        doc = documents[0]
        assert doc.page_content == "Text extracted via Vision API"
        assert doc.metadata['ocr_method'] == 'vision_api'

    def test_supports_file(self, parser_tesseract):
        """Test supports_file class method."""
        assert ImageParser.supports_file("image.png") == True
        assert ImageParser.supports_file("image.jpg") == True
        assert ImageParser.supports_file("document.pdf") == False
        assert ImageParser.supports_file("text.txt") == False


# ==================== DOCXParser Tests ====================

class TestDOCXParser:
    """Tests for DOCXParser."""

    @pytest.fixture
    def parser(self):
        """Create DOCXParser instance."""
        with patch('src.ingestion.parsers.docx_parser.DOCX_AVAILABLE', True):
            return DOCXParser(extract_tables=True)

    @pytest.fixture
    def parser_no_tables(self):
        """Create DOCXParser without table extraction."""
        with patch('src.ingestion.parsers.docx_parser.DOCX_AVAILABLE', True):
            return DOCXParser(extract_tables=False)

    def test_parser_initialization(self, parser):
        """Test parser initializes correctly."""
        assert parser is not None
        assert parser.extract_tables == True

    def test_parser_no_tables_initialization(self, parser_no_tables):
        """Test parser without tables initializes."""
        assert parser_no_tables.extract_tables == False

    @patch('src.ingestion.parsers.docx_parser.DOCX_AVAILABLE', False)
    def test_initialization_without_docx_library(self):
        """Test initialization fails without python-docx."""
        with pytest.raises(ImportError, match="python-docx not available"):
            DOCXParser()

    @patch('src.ingestion.parsers.docx_parser.DocxDocument')
    def test_parse_docx_with_paragraphs(self, mock_docx, parser, temp_dir):
        """Test parsing DOCX with paragraphs."""
        # Mock paragraphs
        mock_para1 = Mock()
        mock_para1.text = "Introduction paragraph."
        mock_para1.style.name = "Normal"

        mock_para2 = Mock()
        mock_para2.text = "Body paragraph."
        mock_para2.style.name = "Normal"

        # Mock document
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_doc.tables = []
        mock_doc.core_properties = Mock()
        mock_doc.core_properties.title = "Test Document"
        mock_doc.core_properties.author = "Test Author"

        mock_docx.return_value = mock_doc

        # Create dummy DOCX file
        docx_path = temp_dir / "test.docx"
        docx_path.write_bytes(b"dummy docx")

        # Parse
        documents = parser.parse(str(docx_path))

        assert len(documents) == 1
        doc = documents[0]
        assert "Introduction paragraph." in doc.page_content
        assert "Body paragraph." in doc.page_content
        assert doc.metadata['file_type'] == 'docx'
        assert doc.metadata['paragraph_count'] == 2

    @patch('src.ingestion.parsers.docx_parser.DocxDocument')
    def test_parse_docx_with_headings(self, mock_docx, parser, temp_dir):
        """Test parsing DOCX with headings."""
        # Mock heading paragraph
        mock_heading = Mock()
        mock_heading.text = "Main Heading"
        mock_heading.style.name = "Heading 1"

        mock_para = Mock()
        mock_para.text = "Content under heading."
        mock_para.style.name = "Normal"

        mock_doc = Mock()
        mock_doc.paragraphs = [mock_heading, mock_para]
        mock_doc.tables = []
        mock_doc.core_properties = Mock()

        mock_docx.return_value = mock_doc

        docx_path = temp_dir / "headings.docx"
        docx_path.write_bytes(b"dummy")

        documents = parser.parse(str(docx_path))

        assert len(documents) == 1
        assert "## Main Heading" in documents[0].page_content

    @patch('src.ingestion.parsers.docx_parser.DocxDocument')
    def test_parse_docx_with_tables(self, mock_docx, parser, temp_dir):
        """Test parsing DOCX with tables."""
        # Mock table
        mock_cell1 = Mock()
        mock_cell1.text = "Name"
        mock_cell2 = Mock()
        mock_cell2.text = "Age"

        mock_row = Mock()
        mock_row.cells = [mock_cell1, mock_cell2]

        mock_table = Mock()
        mock_table.rows = [mock_row]

        # Mock document
        mock_doc = Mock()
        mock_doc.paragraphs = []
        mock_doc.tables = [mock_table]
        mock_doc.core_properties = Mock()

        mock_docx.return_value = mock_doc

        docx_path = temp_dir / "tables.docx"
        docx_path.write_bytes(b"dummy")

        documents = parser.parse(str(docx_path))

        assert len(documents) == 1
        doc = documents[0]
        assert "TABLES" in doc.page_content
        assert "Name" in doc.page_content
        assert doc.metadata['table_count'] == 1

    def test_parse_non_docx_file(self, parser, temp_dir):
        """Test parsing non-DOCX file raises error."""
        txt_path = temp_dir / "notdocx.txt"
        txt_path.write_text("not a docx")

        with pytest.raises(ValueError, match="not a DOCX"):
            parser.parse(str(txt_path))

    def test_parse_nonexistent_docx(self, parser):
        """Test parsing nonexistent DOCX."""
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.docx")

    def test_supports_file(self, parser):
        """Test supports_file class method."""
        assert DOCXParser.supports_file("document.docx") == True
        assert DOCXParser.supports_file("document.doc") == False
        assert DOCXParser.supports_file("text.txt") == False


# ==================== BaseParser Tests ====================

class TestBaseParser:
    """Tests for BaseParser abstract class."""

    def test_cannot_instantiate_base_parser(self):
        """Test that BaseParser cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseParser()

    def test_document_dataclass(self):
        """Test Document dataclass creation."""
        doc = Document(
            page_content="Test content",
            metadata={"source": "test.txt", "page": 1}
        )

        assert doc.page_content == "Test content"
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["page"] == 1

    def test_document_repr(self):
        """Test Document string representation."""
        doc = Document(
            page_content="Hello World",
            metadata={"file": "test.txt"}
        )

        repr_str = repr(doc)
        assert "content_length=11" in repr_str
        assert "metadata=" in repr_str


# ==================== Integration Tests ====================

@pytest.mark.integration
class TestParsersIntegration:
    """Integration tests for parsers working together."""

    def test_parse_multiple_file_types(
        self,
        temp_dir,
        create_temp_text_file,
        create_temp_image,
        sample_text_content
    ):
        """Test parsing multiple file types."""
        # Create test files
        txt_path = create_temp_text_file("doc.txt", sample_text_content)
        md_path = create_temp_text_file("readme.md", "# Readme")

        # Parse with appropriate parsers
        text_parser = TextParser()

        txt_docs = text_parser.parse(txt_path)
        md_docs = text_parser.parse(md_path)

        assert len(txt_docs) == 1
        assert len(md_docs) == 1
        assert txt_docs[0].metadata['extension'] == '.txt'
        assert md_docs[0].metadata['extension'] == '.md'

    def test_metadata_consistency(self, create_temp_text_file):
        """Test that all parsers create consistent metadata."""
        parser = TextParser()
        file_path = create_temp_text_file("test.txt", "content")

        documents = parser.parse(file_path)
        metadata = documents[0].metadata

        # Check required metadata fields
        assert 'source' in metadata
        assert 'file_name' in metadata
        assert 'file_type' in metadata
        assert Path(metadata['source']).name == metadata['file_name']
