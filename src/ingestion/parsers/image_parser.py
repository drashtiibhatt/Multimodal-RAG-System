"""
Image Parser - Extracts text from image files using OCR.

Supports:
- Standalone images (.png, .jpg, .jpeg)
- OCR text extraction using Tesseract
- Optional Vision API for better quality
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from openai import OpenAI
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

from .base import BaseParser, Document
from ...config import get_settings

logger = logging.getLogger(__name__)


class ImageParser(BaseParser):
    """Parser for image files with OCR capabilities."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}

    def __init__(
        self,
        use_vision_api: bool = False,
        tesseract_path: Optional[str] = None
    ):
        """
        Initialize ImageParser.

        Args:
            use_vision_api: If True, use OpenAI Vision API instead of Tesseract
            tesseract_path: Custom path to Tesseract executable
        """
        self.use_vision_api = use_vision_api
        self.settings = get_settings()

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Initialize OpenAI client if using Vision API
        if self.use_vision_api and VISION_AVAILABLE:
            self.client = OpenAI(api_key=self.settings.openai_api_key)

        logger.info(
            f"ImageParser initialized (Vision API: {use_vision_api}, "
            f"Tesseract: {TESSERACT_AVAILABLE})"
        )

    def parse(self, file_path: str) -> List[Document]:
        """
        Parse an image file and extract text.

        Args:
            file_path: Path to image file

        Returns:
            List containing a single Document with extracted text

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If OCR is not available
        """
        path = self._validate_file_exists(file_path)

        # Validate extension
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        try:
            # Extract text using selected method
            if self.use_vision_api:
                text = self._extract_with_vision_api(path)
            else:
                text = self._extract_with_tesseract(path)

            # Get image metadata
            metadata = self._extract_metadata(path)

            # Create document
            document = Document(
                page_content=text,
                metadata=metadata
            )

            logger.info(
                f"Extracted {len(text)} characters from {path.name} "
                f"using {'Vision API' if self.use_vision_api else 'Tesseract'}"
            )

            return [document]

        except Exception as e:
            logger.error(f"Failed to parse image {file_path}: {str(e)}")
            raise

    def _extract_with_tesseract(self, image_path: Path) -> str:
        """
        Extract text using Tesseract OCR.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text

        Raises:
            ValueError: If Tesseract is not available
        """
        if not TESSERACT_AVAILABLE:
            raise ValueError(
                "Tesseract OCR is not available. "
                "Install with: pip install pytesseract pillow"
            )

        try:
            # Open image
            image = Image.open(image_path)

            # Perform OCR
            text = pytesseract.image_to_string(image)

            # Clean up text
            text = text.strip()

            if not text:
                logger.warning(f"No text extracted from {image_path.name}")

            return text

        except pytesseract.TesseractNotFoundError:
            raise ValueError(
                "Tesseract executable not found. "
                "Install Tesseract OCR and add to PATH, or set TESSERACT_PATH in .env"
            )
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {str(e)}")
            raise

    def _extract_with_vision_api(self, image_path: Path) -> str:
        """
        Extract text using OpenAI Vision API.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text

        Raises:
            ValueError: If Vision API is not available
        """
        if not VISION_AVAILABLE:
            raise ValueError(
                "OpenAI library not available. "
                "Install with: pip install openai"
            )

        try:
            import base64

            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Call Vision API
            response = self.client.chat.completions.create(
                model=self.settings.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Return only the extracted text, nothing else."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            text = response.choices[0].message.content.strip()

            if not text:
                logger.warning(f"No text extracted from {image_path.name} via Vision API")

            return text

        except Exception as e:
            logger.error(f"Vision API extraction failed: {str(e)}")
            raise

    def _extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from image file.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with image metadata
        """
        try:
            image = Image.open(image_path)

            metadata = {
                "source": str(image_path),
                "file_name": image_path.name,
                "file_type": "image",
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "size_bytes": image_path.stat().st_size,
                "ocr_method": "vision_api" if self.use_vision_api else "tesseract"
            }

            # Add EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                metadata["has_exif"] = True

            return metadata

        except Exception as e:
            logger.warning(f"Failed to extract full metadata: {str(e)}")
            return {
                "source": str(image_path),
                "file_name": image_path.name,
                "file_type": "image",
                "ocr_method": "vision_api" if self.use_vision_api else "tesseract"
            }

    @classmethod
    def supports_file(cls, file_path: str) -> bool:
        """
        Check if this parser supports the given file.

        Args:
            file_path: Path to file

        Returns:
            True if file extension is supported
        """
        path = Path(file_path)
        return path.suffix.lower() in cls.SUPPORTED_EXTENSIONS
