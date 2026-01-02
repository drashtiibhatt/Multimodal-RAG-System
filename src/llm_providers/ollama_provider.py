"""Ollama LLM provider for local models."""

from typing import List, Dict, Any, Optional
import logging
import json
import requests

from .base_provider import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider for local LLM models.

    Supports models like Llama, Mistral, CodeLlama, etc. running via Ollama.
    """

    def __init__(
        self,
        model: str = "llama2",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name (e.g., 'llama2', 'mistral', 'codellama')
            temperature: Sampling temperature
            base_url: Ollama server URL
        """
        super().__init__(model, temperature)
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/chat"
        self.models_url = f"{self.base_url}/api/tags"

        logger.info(f"Ollama provider initialized: {self.model} @ {self.base_url}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion using Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate (not all models support this)
            json_mode: Request JSON-formatted output
            **kwargs: Additional Ollama parameters

        Returns:
            LLMResponse with generated content

        Raises:
            Exception: If API call fails
        """
        temp = temperature if temperature is not None else self.temperature

        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
            }
        }

        # Add max tokens if specified (Ollama uses num_predict)
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # JSON mode: add format instruction to system message
        if json_mode:
            # Add JSON format instruction
            json_instruction = "\n\nYou must respond with valid JSON only. Do not include any text outside the JSON object."

            # Find system message or create one
            has_system = False
            for msg in messages:
                if msg["role"] == "system":
                    msg["content"] += json_instruction
                    has_system = True
                    break

            if not has_system:
                messages.insert(0, {
                    "role": "system",
                    "content": f"You are a helpful assistant.{json_instruction}"
                })

            # Set format to json
            payload["format"] = "json"

        # Add any extra options
        if kwargs:
            payload["options"].update(kwargs)

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # 2 minute timeout for local models
            )

            if response.status_code != 200:
                raise Exception(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )

            result = response.json()
            content = result.get("message", {}).get("content", "")

            # Extract usage stats if available
            usage = None
            if "eval_count" in result:
                usage = {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }

            return LLMResponse(
                content=content,
                model=self.model,
                provider="ollama",
                usage=usage,
                finish_reason=result.get("done_reason")
            )

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise Exception(
                f"Cannot connect to Ollama server at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            )
        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout")
            raise Exception("Ollama request timeout. Try a smaller prompt or increase timeout.")
        except Exception as e:
            logger.error(f"Ollama API call failed: {str(e)}")
            raise Exception(f"Ollama generation failed: {str(e)}")

    def is_available(self) -> bool:
        """
        Check if Ollama server is available.

        Returns:
            True if Ollama server is reachable
        """
        try:
            response = requests.get(
                self.models_url,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "ollama"

    def list_models(self) -> List[str]:
        """
        List available Ollama models.

        Returns:
            List of model names
        """
        try:
            response = requests.get(self.models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {str(e)}")
            return []

    def supports_json_mode(self) -> bool:
        """
        Check if model supports JSON mode.

        Returns:
            True (Ollama supports format parameter)
        """
        return True

    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama registry.

        Args:
            model_name: Name of model to pull (e.g., 'llama2', 'mistral')

        Returns:
            True if successful
        """
        pull_url = f"{self.base_url}/api/pull"

        try:
            logger.info(f"Pulling Ollama model: {model_name}")
            response = requests.post(
                pull_url,
                json={"name": model_name, "stream": False},
                timeout=600  # 10 minutes for model download
            )

            if response.status_code == 200:
                logger.info(f"Successfully pulled model: {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get provider information including available models.

        Returns:
            Dictionary with provider details
        """
        info = super().get_info()
        info["base_url"] = self.base_url
        info["available_models"] = self.list_models() if self.is_available() else []
        return info
