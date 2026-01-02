"""LLM-based use case generator."""

from typing import Dict, Any, List, Optional
import json

from .prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .output_schemas import GenerationOutput
from ..storage import SearchResult
from ..config import get_settings
from ..caching import QueryCache
from ..llm_providers import OpenAIProvider, OllamaProvider, BaseLLMProvider


class UseCaseGenerator:
    """Generate structured use cases using LLM."""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        use_cache: bool = True,
        provider: str = None
    ):
        """
        Initialize generator.

        Args:
            model: LLM model name (defaults to config)
            temperature: Sampling temperature (defaults to config)
            use_cache: Whether to use query caching
            provider: LLM provider ("openai" or "ollama", defaults to config)
        """
        settings = get_settings()
        self.temperature = temperature or settings.temperature
        self.use_cache = use_cache
        self.cache = QueryCache() if use_cache else None

        # Determine provider
        provider = provider or settings.llm_provider

        # Initialize provider
        if provider.lower() == "ollama":
            self.model = model or settings.ollama_model
            self.provider = OllamaProvider(
                model=self.model,
                temperature=self.temperature,
                base_url=settings.ollama_base_url
            )
        else:  # Default to OpenAI
            self.model = model or settings.llm_model
            self.provider = OpenAIProvider(
                model=self.model,
                temperature=self.temperature
            )

        # Check provider availability
        if not self.provider.is_available():
            print(f"[WARN] Provider '{provider}' is not available!")
            if provider.lower() == "ollama":
                print("[INFO] Make sure Ollama is running: ollama serve")
                print(f"[INFO] And the model is pulled: ollama pull {self.model}")

    def generate(
        self,
        query: str,
        context: str,
        min_confidence: float = 0.6,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Generate use cases from query and context.

        Args:
            query: User query
            context: Retrieved context string
            min_confidence: Minimum confidence threshold
            debug: Whether to print debug info

        Returns:
            Dictionary with use cases and metadata
        """
        if debug:
            print(f"\n[AI] Generating use cases with {self.model}")
            print(f"Temperature: {self.temperature}")
            print(f"Context length: {len(context)} characters")

        # Check if context is empty
        if not context or context.strip() == "":
            return {
                "insufficient_context": True,
                "clarifying_questions": [
                    "What feature or functionality should the test cases cover?",
                    "Do you have any documentation or requirements to provide?"
                ],
                "missing_information": ["No context documents found"],
                "use_cases": [],
                "assumptions": [],
                "confidence_score": 0.0
            }

        # Check cache first
        if self.use_cache and self.cache:
            cached_result = self.cache.get(query, context)
            if cached_result:
                if debug:
                    print(f"[CACHE] Using cached result for query")
                return cached_result

        # Build prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )

        if debug:
            print(f"Prompt length: {len(user_prompt)} characters")

        # Call LLM via provider
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            # Use provider to generate
            response = self.provider.generate(
                messages=messages,
                temperature=self.temperature,
                json_mode=True
            )

            # Parse JSON response
            output_text = response.content
            output_dict = json.loads(output_text)

            if debug:
                print(f"[OK] Generated response ({len(output_text)} characters)")
                print(f"[OK] Use cases created: {len(output_dict.get('use_cases', []))}")

            # Validate with Pydantic (optional, adds type safety)
            try:
                validated_output = GenerationOutput(**output_dict)
                output_dict = validated_output.model_dump()
            except Exception as e:
                if debug:
                    print(f"[WARN]  Validation warning: {str(e)}")
                # Continue with unvalidated output

            # Cache the result
            if self.use_cache and self.cache:
                self.cache.set(query, context, output_dict)

            return output_dict

        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse JSON response: {str(e)}",
                "use_cases": [],
                "insufficient_context": True
            }

        except Exception as e:
            return {
                "error": f"Generation failed: {str(e)}",
                "use_cases": [],
                "insufficient_context": True
            }

    def generate_from_results(
        self,
        query: str,
        results: List[SearchResult],
        min_confidence: float = 0.6,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Generate use cases from search results.

        Args:
            query: User query
            results: List of search results
            min_confidence: Minimum confidence threshold
            debug: Whether to print debug info

        Returns:
            Dictionary with use cases
        """
        # Format context from results
        context = self._format_context(results)

        # Generate use cases
        return self.generate(query, context, min_confidence, debug)

    def _format_context(self, results: List[SearchResult]) -> str:
        """
        Format search results as context.

        Args:
            results: List of search results

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = []

        for idx, result in enumerate(results, 1):
            source = result.metadata.get("source", "unknown")
            page = result.metadata.get("page", "")
            page_ref = f" (page {page})" if page else ""

            context_parts.append(
                f"[Document {idx}] {source}{page_ref}\n"
                f"{result.content}\n"
            )

        return "\n---\n\n".join(context_parts)

    def get_cache_stats(self) -> Optional[dict]:
        """
        Get query cache statistics.

        Returns:
            Cache statistics dict or None if cache disabled
        """
        if self.cache:
            return self.cache.get_stats()
        return None

    def clear_cache(self) -> int:
        """
        Clear query cache.

        Returns:
            Number of cache entries cleared
        """
        if self.cache:
            return self.cache.clear()
        return 0
