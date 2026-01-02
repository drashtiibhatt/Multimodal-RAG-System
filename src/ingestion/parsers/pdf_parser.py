"""Parser for PDF files with optional image extraction."""

from typing import List, Optional
from pathlib import Path
import pdfplumber
import logging
import tempfile

try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    IMAGE_EXTRACTION_AVAILABLE = True
except ImportError:
    IMAGE_EXTRACTION_AVAILABLE = False

from .base import BaseParser, Document
from ..vision_ocr import VisionOCR
from ...config import get_settings

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """Parser for PDF files with text extraction."""

    def __init__(
        self,
        extract_images: bool = False,
        use_vision_api: bool = False,
        vision_model: str = "gpt-4-vision-preview"
    ):
        """
        Initialize PDF parser.

        Args:
            extract_images: Whether to extract images from PDF
            use_vision_api: Use OpenAI Vision API instead of Tesseract OCR
            vision_model: Vision model to use if use_vision_api=True
        """
        self.extract_images = extract_images
        self.use_vision_api = use_vision_api
        self.vision_ocr = None

        if use_vision_api:
            try:
                self.vision_ocr = VisionOCR(model=vision_model)
                logger.info(f"Vision API enabled: {vision_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Vision API: {str(e)}")
                logger.warning("Falling back to Tesseract OCR")
                self.use_vision_api = False

    def parse(self, file_path: str) -> List[Document]:
        """
        Parse PDF file and extract text from each page.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects (one per page)

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If PDF is corrupted or unreadable
        """
        # Validate file exists
        path = self._validate_file_exists(file_path)

        # Validate PDF extension
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")

        documents = []

        # Extract text from PDF
        try:
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text from page
                    text = page.extract_text()

                    # Skip empty pages
                    if not text or text.strip() == "":
                        text = f"[Empty page {page_num}]"

                    # Create metadata for this page
                    metadata = self._create_metadata(
                        file_path=file_path,
                        file_type="pdf",
                        page=page_num,
                        total_pages=total_pages,
                        char_count=len(text)
                    )

                    # Create Document object
                    document = Document(
                        page_content=text,
                        metadata=metadata
                    )

                    documents.append(document)

        except Exception as e:
            raise Exception(f"Failed to parse PDF {file_path}: {str(e)}")

        # Extract images if enabled
        if self.extract_images:
            try:
                if self.use_vision_api and self.vision_ocr:
                    image_documents = self._extract_with_vision_api(file_path)
                    logger.info(f"Extracted {len(image_documents)} pages using Vision API from {path.name}")
                else:
                    image_documents = self._extract_images_from_pdf(file_path)
                    logger.info(f"Extracted {len(image_documents)} pages using Tesseract from {path.name}")

                documents.extend(image_documents)
            except Exception as e:
                logger.warning(f"Image extraction failed for {path.name}: {str(e)}")

        return documents

    def _extract_images_from_pdf(
        self,
        file_path: str,
        dpi: int = 200,
        poppler_path: Optional[str] = None
    ) -> List[Document]:
        """
        Extract images from PDF pages and perform OCR.

        Args:
            file_path: Path to PDF file
            dpi: DPI for image conversion (higher = better quality, slower)
            poppler_path: Custom path to Poppler binaries

        Returns:
            List of Document objects from OCR on each page image

        Raises:
            ValueError: If pdf2image or pytesseract not available
        """
        if not IMAGE_EXTRACTION_AVAILABLE:
            raise ValueError(
                "Image extraction dependencies not available. "
                "Install with: pip install pdf2image pytesseract pillow"
            )

        path = Path(file_path)
        documents = []

        try:
            # Convert PDF pages to images
            logger.info(f"Converting PDF to images: {path.name}")

            images = convert_from_path(
                file_path,
                dpi=dpi,
                poppler_path=poppler_path
            )

            # Perform OCR on each image
            for page_num, image in enumerate(images, start=1):
                try:
                    # Extract text with Tesseract
                    text = pytesseract.image_to_string(image)

                    # Skip if no text found
                    if not text or text.strip() == "":
                        logger.debug(f"No text extracted from image page {page_num}")
                        continue

                    # Create metadata
                    metadata = self._create_metadata(
                        file_path=file_path,
                        file_type="pdf_image",
                        page=page_num,
                        total_pages=len(images),
                        char_count=len(text),
                        extraction_method="ocr",
                        ocr_dpi=dpi
                    )

                    # Create document
                    document = Document(
                        page_content=text,
                        metadata=metadata
                    )

                    documents.append(document)

                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {str(e)}")
                    continue

            logger.info(
                f"OCR completed: {len(documents)}/{len(images)} pages "
                f"extracted text from {path.name}"
            )

        except Exception as e:
            logger.error(f"Failed to extract images from PDF: {str(e)}")
            raise ValueError(
                f"Image extraction failed: {str(e)}. "
                "Ensure Poppler is installed and in PATH."
            )

        return documents

    def _extract_with_vision_api(
        self,
        file_path: str,
        dpi: int = 200,
        poppler_path: Optional[str] = None
    ) -> List[Document]:
        """
        Extract text from PDF using OpenAI Vision API.

        More powerful than Tesseract for:
        - Handwritten text
        - Complex layouts
        - Tables and diagrams
        - Poor quality scans

        Args:
            file_path: Path to PDF file
            dpi: DPI for image conversion
            poppler_path: Custom path to Poppler binaries

        Returns:
            List of Document objects from Vision API

        Raises:
            ValueError: If pdf2image not available or Vision API not initialized
        """
        if not IMAGE_EXTRACTION_AVAILABLE:
            raise ValueError(
                "Image extraction dependencies not available. "
                "Install with: pip install pdf2image pillow"
            )

        if not self.vision_ocr:
            raise ValueError("Vision OCR not initialized")

        path = Path(file_path)
        documents = []

        try:
            # Convert PDF pages to images
            logger.info(f"Converting PDF to images for Vision API: {path.name}")

            settings = get_settings()
            poppler_path = poppler_path or settings.poppler_path
            dpi = dpi or settings.ocr_dpi

            images = convert_from_path(
                file_path,
                dpi=dpi,
                poppler_path=poppler_path
            )

            logger.info(f"Processing {len(images)} pages with Vision API...")

            # Process each image with Vision API
            for page_num, image in enumerate(images, start=1):
                try:
                    logger.info(f"Processing page {page_num}/{len(images)}...")

                    # Extract text using Vision API
                    text = self.vision_ocr.extract_text_from_image(image)

                    # Skip if no text found
                    if not text or text.strip() == "":
                        logger.debug(f"No text extracted from page {page_num}")
                        continue

                    # Create metadata
                    metadata = self._create_metadata(
                        file_path=file_path,
                        file_type="pdf_vision",
                        page=page_num,
                        total_pages=len(images),
                        char_count=len(text),
                        extraction_method="vision_api",
                        vision_model=self.vision_ocr.model,
                        ocr_dpi=dpi
                    )

                    # Create document
                    document = Document(
                        page_content=text,
                        metadata=metadata
                    )

                    documents.append(document)

                    logger.info(
                        f"Page {page_num}: extracted {len(text)} characters"
                    )

                except Exception as e:
                    logger.warning(f"Vision API failed for page {page_num}: {str(e)}")
                    continue

            logger.info(
                f"Vision API completed: {len(documents)}/{len(images)} pages "
                f"extracted from {path.name}"
            )

        except Exception as e:
            logger.error(f"Failed to extract with Vision API: {str(e)}")
            raise ValueError(
                f"Vision API extraction failed: {str(e)}. "
                "Ensure Poppler is installed and OpenAI API key is configured."
            )

        return documents

    def extract_images_only(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        dpi: int = 200
    ) -> List[Path]:
        """
        Extract images from PDF and save to disk.

        Args:
            file_path: Path to PDF file
            output_dir: Directory to save images (defaults to temp)
            dpi: DPI for conversion

        Returns:
            List of paths to saved images
        """
        if not IMAGE_EXTRACTION_AVAILABLE:
            raise ValueError(
                "Image extraction not available. "
                "Install with: pip install pdf2image pillow"
            )

        path = Path(file_path)

        # Use temp directory if not specified
        if output_dir is None:
            output_dir = tempfile.mkdtemp()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to images
        settings = get_settings()
        images = convert_from_path(
            file_path,
            dpi=dpi,
            poppler_path=settings.poppler_path
        )

        # Save images
        saved_paths = []
        for page_num, image in enumerate(images, start=1):
            image_path = output_path / f"{path.stem}_page_{page_num}.jpg"
            image.save(image_path, "JPEG", quality=95)
            saved_paths.append(image_path)
            logger.info(f"Saved page {page_num} to {image_path}")

        return saved_paths
