"""
Prompt Injection Detector - Detects and blocks prompt injection attacks.

Identifies malicious attempts to override system instructions or
manipulate the LLM's behavior.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class InjectionResult:
    """Result from prompt injection detection."""
    is_injection: bool
    confidence: float
    patterns_found: List[str]
    safe_to_process: bool
    sanitized_query: str
    risk_level: str  # "low", "medium", "high"


class PromptInjectionDetector:
    """
    Detect prompt injection attempts in user queries.

    Uses pattern matching and heuristics to identify
    malicious input that attempts to manipulate the system.
    """

    # Common injection patterns
    INJECTION_PATTERNS = [
        # System prompt override attempts
        (r"ignore\s+(previous|above|all|prior)\s+(instructions?|commands?|prompts?)", "high"),
        (r"disregard\s+(previous|above|all|prior)\s+(instructions?|commands?|prompts?)", "high"),
        (r"forget\s+(previous|above|all|prior)\s+(instructions?|commands?|prompts?)", "high"),
        (r"override\s+(previous|above|all|prior)\s+(instructions?|commands?|prompts?)", "high"),

        # New instructions injection
        (r"new\s+(instructions?|commands?|rules?)[\s:]+", "high"),
        (r"updated\s+(instructions?|commands?|rules?)[\s:]+", "high"),
        (r"system\s+(prompt|instructions?)[\s:]+", "high"),

        # Role-playing attacks
        (r"you\s+are\s+now\s+", "medium"),
        (r"act\s+as\s+(if\s+)?you\s+(are|were)\s+", "medium"),
        (r"pretend\s+(to\s+be|you\s+are)\s+", "medium"),
        (r"roleplay\s+as\s+", "medium"),
        (r"simulate\s+(being|a)\s+", "medium"),

        # Instruction injection markers
        (r"<!--.*?instruction.*?-->", "medium"),
        (r"\[SYSTEM\]", "high"),
        (r"\[INSTRUCTION\]", "high"),
        (r"\[ADMIN\]", "high"),
        (r"<\|system\|>", "high"),

        # Output manipulation
        (r"output\s+only\s+", "medium"),
        (r"respond\s+with\s+only\s+", "medium"),
        (r"just\s+say\s+", "low"),
        (r"simply\s+answer\s+", "low"),
        (r"answer\s+with\s+one\s+word", "low"),

        # Escaping attempts
        (r"```\s*system", "high"),
        (r"</instructions?>", "medium"),
        (r"<\|endoftext\|>", "high"),
        (r"<\|im_start\|>", "high"),

        # Character-level tricks
        (r"\x00", "high"),  # Null bytes
        (r"\\x[0-9a-fA-F]{2}", "medium"),  # Hex escapes

        # Attempting to expose system prompt
        (r"repeat\s+(the\s+)?instructions", "medium"),
        (r"what\s+(are\s+)?(your\s+)?instructions", "medium"),
        (r"show\s+(me\s+)?(your\s+)?prompt", "medium"),
    ]

    def __init__(
        self,
        max_length: int = 2000,
        suspicious_threshold: int = 1,  # Changed from 2 to 1 for stricter detection
        strict_mode: bool = False
    ):
        """
        Initialize injection detector.

        Args:
            max_length: Maximum allowed query length
            suspicious_threshold: Number of patterns to trigger alert
            strict_mode: If True, use stricter detection (may have more false positives)
        """
        self.max_length = max_length
        self.suspicious_threshold = suspicious_threshold
        self.strict_mode = strict_mode

        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.DOTALL), risk)
            for pattern, risk in self.INJECTION_PATTERNS
        ]

        logger.info(
            f"InjectionDetector initialized: max_length={max_length}, "
            f"threshold={suspicious_threshold}, strict={strict_mode}"
        )

    def detect(self, query: str) -> InjectionResult:
        """
        Detect prompt injection in query.

        Args:
            query: User query to check

        Returns:
            InjectionResult with detection details
        """
        # Check length first
        if len(query) > self.max_length:
            return InjectionResult(
                is_injection=True,
                confidence=1.0,
                patterns_found=[f"Query exceeds maximum length ({len(query)} > {self.max_length})"],
                safe_to_process=False,
                sanitized_query="",
                risk_level="high"
            )

        # Check for injection patterns
        patterns_found = []
        risk_scores = {"low": 0, "medium": 0, "high": 0}

        for pattern, risk_level in self.compiled_patterns:
            matches = pattern.findall(query)
            if matches:
                patterns_found.append(f"{pattern.pattern} (risk: {risk_level})")
                risk_scores[risk_level] += len(matches)

        # Additional heuristics
        if self._check_special_chars(query):
            patterns_found.append("Excessive special characters")
            risk_scores["medium"] += 1

        if self._check_encoding_tricks(query):
            patterns_found.append("Encoding manipulation detected")
            risk_scores["high"] += 1

        if self._check_repeated_phrases(query):
            patterns_found.append("Suspicious repeated phrases")
            risk_scores["low"] += 1

        # Calculate overall risk
        total_high = risk_scores["high"]
        total_medium = risk_scores["medium"]
        total_low = risk_scores["low"]
        total_patterns = total_high + total_medium + total_low

        # Determine risk level
        if total_high > 0:
            risk_level = "high"
        elif total_medium > 0:
            risk_level = "medium"
        elif total_low > 0:
            risk_level = "low"
        else:
            risk_level = "none"

        # Calculate confidence
        if total_high > 0:
            confidence = min(1.0, 0.7 + (total_high * 0.1))
        elif total_medium > 0:
            confidence = min(0.7, 0.4 + (total_medium * 0.1))
        elif total_low > 0:
            confidence = min(0.4, 0.2 + (total_low * 0.1))
        else:
            confidence = 0.0

        # Determine if injection
        is_injection = (
            total_patterns >= self.suspicious_threshold or
            (self.strict_mode and total_patterns > 0)
        )

        # Sanitize query if injection detected
        sanitized_query = self.sanitize(query) if is_injection else query

        return InjectionResult(
            is_injection=is_injection,
            confidence=confidence,
            patterns_found=patterns_found,
            safe_to_process=not is_injection,
            sanitized_query=sanitized_query,
            risk_level=risk_level
        )

    def _check_special_chars(self, query: str) -> bool:
        """
        Check for excessive special characters.

        Args:
            query: Query to check

        Returns:
            True if suspicious amount of special chars
        """
        special_count = sum(
            1 for c in query
            if c in '!@#$%^&*(){}[]<>|\\`~=+'
        )

        # More than 15% special characters is suspicious
        return special_count > len(query) * 0.15

    def _check_encoding_tricks(self, query: str) -> bool:
        """
        Check for encoding manipulation attempts.

        Args:
            query: Query to check

        Returns:
            True if encoding tricks detected
        """
        # Check for base64-like patterns (long alphanumeric strings)
        if re.search(r'[A-Za-z0-9+/]{40,}=*', query):
            return True

        # Check for unicode escape sequences
        if '\\u' in query or '\\U' in query:
            return True

        # Check for hex escape sequences
        if re.search(r'\\x[0-9a-fA-F]{2}', query):
            return True

        # Check for URL encoding abuse
        if query.count('%') > 3:
            return True

        return False

    def _check_repeated_phrases(self, query: str) -> bool:
        """
        Check for suspiciously repeated phrases.

        Args:
            query: Query to check

        Returns:
            True if suspicious repetition detected
        """
        # Split into words
        words = query.lower().split()

        if len(words) < 10:
            return False

        # Check for repeated n-grams
        for n in [3, 4, 5]:
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

            # If same n-gram appears 3+ times, suspicious
            from collections import Counter
            counts = Counter(ngrams)
            if any(count >= 3 for count in counts.values()):
                return True

        return False

    def sanitize(self, query: str) -> str:
        """
        Sanitize query by removing suspicious patterns.

        Args:
            query: Query to sanitize

        Returns:
            Sanitized query
        """
        sanitized = query

        # Remove pattern matches
        for pattern, _ in self.compiled_patterns:
            sanitized = pattern.sub('', sanitized)

        # Remove excessive special characters
        # Keep only common punctuation
        allowed_special = set('.,;:!?()"\'-')
        sanitized = ''.join(
            c if c.isalnum() or c.isspace() or c in allowed_special else ''
            for c in sanitized
        )

        # Limit length
        sanitized = sanitized[:self.max_length]

        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())

        # Remove HTML/XML tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)

        return sanitized.strip()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.

        Returns:
            Dictionary with detector info
        """
        return {
            "max_length": self.max_length,
            "suspicious_threshold": self.suspicious_threshold,
            "strict_mode": self.strict_mode,
            "num_patterns": len(self.compiled_patterns)
        }
