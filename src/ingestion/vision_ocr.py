"""Advanced OCR using OpenAI Vision API for image understanding."""

from typing import List, Dict, Any, Optional, Union
import base64
import logging
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
from openai import OpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)


class VisionOCR:
    """
    Advanced OCR using OpenAI Vision API.

    Extracts text and understanding from images using GPT-4V.
    Much more powerful than traditional OCR for:
    - Handwritten text
    - Complex layouts
    - Tables and diagrams
    - Scanned documents
    """

    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        max_tokens: int = 4096,
        detail: str = "high"
    ):
        """
        Initialize Vision OCR.

        Args:
            model: Vision model to use (gpt-4-vision-preview, gpt-4-turbo, etc.)
            max_tokens: Maximum tokens in response
            detail: Image detail level ("low" or "high")
        """
        settings = get_settings()
        self.model = model
        self.max_tokens = max_tokens
        self.detail = detail
        self.client = OpenAI(api_key=settings.openai_api_key)

        logger.info(f"VisionOCR initialized: {model} (detail={detail})")

    def extract_text_from_image(
        self,
        image: Union[str, Path, bytes, Image.Image],
        prompt: Optional[str] = None
    ) -> str:
        """
        Extract text from an image using Vision API.

        Args:
            image: Image file path, bytes, or PIL Image
            prompt: Custom prompt (defaults to OCR extraction)

        Returns:
            Extracted text

        Raises:
            Exception: If API call fails
        """
        # Encode image to base64
        image_base64 = self._encode_image(image)

        # Default OCR prompt
        if prompt is None:
            prompt = (
                "Extract all text from this image. "
                "Preserve the original formatting, layout, and structure as much as possible. "
                "Include all visible text, even if handwritten or in tables. "
                "If there are multiple columns, process left to right, top to bottom."
            )

        # Build API request
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": self.detail
                        }
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )

            extracted_text = response.choices[0].message.content
            logger.info(f"Extracted {len(extracted_text)} characters from image")

            return extracted_text

        except Exception as e:
            logger.error(f"Vision API call failed: {str(e)}")
            raise Exception(f"Failed to extract text from image: {str(e)}")

    def describe_image(
        self,
        image: Union[str, Path, bytes, Image.Image],
        focus: Optional[str] = None
    ) -> str:
        """
        Get detailed description of image content.

        Args:
            image: Image file path, bytes, or PIL Image
            focus: Specific aspect to focus on (e.g., "tables", "diagrams")

        Returns:
            Image description
        """
        image_base64 = self._encode_image(image)

        prompt = "Describe this image in detail."
        if focus:
            prompt += f" Pay special attention to {focus}."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": self.detail
                        }
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Vision API call failed: {str(e)}")
            raise Exception(f"Failed to describe image: {str(e)}")

    def extract_table_from_image(
        self,
        image: Union[str, Path, bytes, Image.Image]
    ) -> str:
        """
        Extract table data from image.

        Args:
            image: Image containing a table

        Returns:
            Table data in markdown format
        """
        image_base64 = self._encode_image(image)

        prompt = (
            "Extract the table from this image and format it as a markdown table. "
            "Preserve all rows, columns, and cell values exactly as shown. "
            "Include headers if present."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"  # Always use high detail for tables
                        }
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Vision API call failed: {str(e)}")
            raise Exception(f"Failed to extract table: {str(e)}")

    def analyze_document_structure(
        self,
        image: Union[str, Path, bytes, Image.Image]
    ) -> Dict[str, Any]:
        """
        Analyze document structure and layout.

        Args:
            image: Document image

        Returns:
            Dictionary with structure analysis
        """
        image_base64 = self._encode_image(image)

        prompt = (
            "Analyze this document's structure and provide a JSON response with:\n"
            "1. document_type: (form, table, text, diagram, mixed)\n"
            "2. has_handwriting: (true/false)\n"
            "3. has_tables: (true/false)\n"
            "4. has_images: (true/false)\n"
            "5. layout: (single_column, multi_column, mixed)\n"
            "6. quality: (excellent, good, fair, poor)\n"
            "7. notes: (any additional observations)"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": self.detail
                        }
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000
            )

            # Try to parse as JSON
            import json
            content = response.choices[0].message.content

            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)

        except Exception as e:
            logger.error(f"Structure analysis failed: {str(e)}")
            return {
                "document_type": "unknown",
                "error": str(e)
            }

    def process_scanned_page(
        self,
        image: Union[str, Path, bytes, Image.Image],
        include_structure: bool = False
    ) -> Dict[str, Any]:
        """
        Process a scanned document page comprehensively.

        Args:
            image: Scanned page image
            include_structure: Whether to include structure analysis

        Returns:
            Dictionary with extracted content
        """
        result = {
            "text": "",
            "has_tables": False,
            "has_images": False,
            "structure": None
        }

        try:
            # Analyze structure first if requested
            if include_structure:
                result["structure"] = self.analyze_document_structure(image)
                result["has_tables"] = result["structure"].get("has_tables", False)
                result["has_images"] = result["structure"].get("has_images", False)

            # Extract text
            result["text"] = self.extract_text_from_image(image)

            logger.info("Successfully processed scanned page")
            return result

        except Exception as e:
            logger.error(f"Failed to process scanned page: {str(e)}")
            result["error"] = str(e)
            return result

    def _encode_image(
        self,
        image: Union[str, Path, bytes, Image.Image]
    ) -> str:
        """
        Encode image to base64 string.

        Args:
            image: Image in various formats

        Returns:
            Base64 encoded string
        """
        # If it's a file path
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image_bytes = f.read()

        # If it's already bytes
        elif isinstance(image, bytes):
            image_bytes = image

        # If it's a PIL Image
        elif isinstance(image, Image.Image):
            buffer = BytesIO()
            # Convert to RGB if necessary (for PNG with alpha)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            image.save(buffer, format="JPEG", quality=95)
            image_bytes = buffer.getvalue()

        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Encode to base64
        return base64.b64encode(image_bytes).decode('utf-8')

    def is_available(self) -> bool:
        """
        Check if Vision API is available.

        Returns:
            True if API key is configured
        """
        return self.client is not None

    def get_info(self) -> Dict[str, Any]:
        """
        Get OCR engine information.

        Returns:
            Dictionary with OCR info
        """
        return {
            "engine": "openai_vision",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "detail_level": self.detail,
            "available": self.is_available()
        }


class ImagePreprocessor:
    """Helper class for image preprocessing before OCR."""

    @staticmethod
    def enhance_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
        """
        Enhance image contrast.

        Args:
            image: PIL Image
            factor: Contrast enhancement factor (1.0 = no change)

        Returns:
            Enhanced image
        """
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def convert_to_grayscale(image: Image.Image) -> Image.Image:
        """
        Convert image to grayscale.

        Args:
            image: PIL Image

        Returns:
            Grayscale image
        """
        return image.convert('L')

    @staticmethod
    def resize_if_large(
        image: Image.Image,
        max_size: int = 2048
    ) -> Image.Image:
        """
        Resize image if it's too large.

        Args:
            image: PIL Image
            max_size: Maximum dimension

        Returns:
            Resized image if needed
        """
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def preprocess_for_ocr(
        image: Image.Image,
        enhance_contrast: bool = True,
        convert_grayscale: bool = False,
        resize: bool = True
    ) -> Image.Image:
        """
        Apply preprocessing pipeline.

        Args:
            image: PIL Image
            enhance_contrast: Whether to enhance contrast
            convert_grayscale: Whether to convert to grayscale
            resize: Whether to resize if too large

        Returns:
            Preprocessed image
        """
        if resize:
            image = ImagePreprocessor.resize_if_large(image)

        if convert_grayscale:
            image = ImagePreprocessor.convert_to_grayscale(image)

        if enhance_contrast:
            image = ImagePreprocessor.enhance_contrast(image)

        return image
