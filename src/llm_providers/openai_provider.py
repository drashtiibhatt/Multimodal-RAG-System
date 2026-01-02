"""OpenAI LLM provider implementation."""

from typing import List, Dict, Any, Optional
import logging
from openai import OpenAI

from .base_provider import BaseLLMProvider, LLMResponse
from ..config import get_settings

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for GPT models."""

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            temperature: Sampling temperature
            api_key: OpenAI API key (defaults to config)
        """
        settings = get_settings()
        model = model or settings.llm_model
        super().__init__(model, temperature)

        self.api_key = api_key or settings.openai_api_key
        self.client = None

        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI provider initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                self.client = None

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion using OpenAI API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            json_mode: Request JSON-formatted output
            **kwargs: Additional OpenAI parameters

        Returns:
            LLMResponse with generated content

        Raises:
            Exception: If API call fails
        """
        if not self.client:
            raise Exception("OpenAI client not initialized. Check API key.")

        temp = temperature if temperature is not None else self.temperature

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }

        if max_tokens:
            api_params["max_tokens"] = max_tokens

        if json_mode:
            api_params["response_format"] = {"type": "json_object"}

        # Add any extra parameters
        api_params.update(kwargs)

        try:
            response = self.client.chat.completions.create(**api_params)

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                finish_reason=response.choices[0].finish_reason
            )

        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise Exception(f"OpenAI generation failed: {str(e)}")

    def is_available(self) -> bool:
        """
        Check if OpenAI provider is available.

        Returns:
            True if API key is configured and client is initialized
        """
        return self.client is not None

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openai"

    def supports_json_mode(self) -> bool:
        """
        Check if current model supports JSON mode.

        Returns:
            True for GPT-4 and GPT-3.5-turbo models
        """
        # JSON mode supported in gpt-4-* and gpt-3.5-turbo-* models
        return "gpt-4" in self.model.lower() or "gpt-3.5-turbo" in self.model.lower()
