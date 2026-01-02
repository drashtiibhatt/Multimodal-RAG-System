"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (OpenAI, Ollama, Anthropic, etc.) must implement this interface.
    """

    def __init__(self, model: str, temperature: float = 0.7):
        """
        Initialize provider.

        Args:
            model: Model name/identifier
            temperature: Sampling temperature (0-1)
        """
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            json_mode: Request JSON-formatted output
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with generated content

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is available and configured.

        Returns:
            True if provider can be used
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            Provider identifier (e.g., 'openai', 'ollama')
        """
        pass

    def get_model_name(self) -> str:
        """
        Get current model name.

        Returns:
            Model identifier
        """
        return self.model

    def supports_json_mode(self) -> bool:
        """
        Check if provider supports JSON mode.

        Returns:
            True if JSON mode is supported
        """
        return True

    def get_info(self) -> Dict[str, Any]:
        """
        Get provider information.

        Returns:
            Dictionary with provider details
        """
        return {
            "provider": self.get_provider_name(),
            "model": self.model,
            "temperature": self.temperature,
            "available": self.is_available(),
            "supports_json": self.supports_json_mode()
        }
