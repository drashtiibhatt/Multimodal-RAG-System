"""
PII Detector - Detects and redacts personally identifiable information.

Identifies and optionally redacts PII like emails, phone numbers,
SSNs, credit cards, and other sensitive information.
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PIIMatch:
    """Single PII match found in text."""
    pii_type: str
    value: str
    start: int
    end: int
    confidence: float


@dataclass
class PIIResult:
    """Result from PII detection."""
    has_pii: bool
    pii_found: List[PIIMatch]
    redacted_text: str
    risk_level: str  # "low", "medium", "high", "critical"
    pii_types_detected: List[str]


class PIIDetector:
    """
    Detect personally identifiable information (PII) in text.

    Identifies common PII patterns including:
    - Email addresses
    - Phone numbers
    - Social Security Numbers (SSN)
    - Credit card numbers
    - IP addresses
    - Dates of birth
    - Names (basic pattern matching)
    - Addresses
    """

    # PII detection patterns (pattern, pii_type, risk_level)
    PII_PATTERNS = [
        # Email addresses
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email', 'medium'),

        # Phone numbers (US and international formats)
        (r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b', 'phone', 'medium'),
        (r'\b\+?[0-9]{1,3}[-.\s]?(\([0-9]{1,4}\)|[0-9]{1,4})[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}\b', 'phone', 'medium'),

        # Social Security Numbers (US)
        (r'\b\d{3}-\d{2}-\d{4}\b', 'ssn', 'critical'),
        (r'\b\d{3}\s\d{2}\s\d{4}\b', 'ssn', 'critical'),

        # Credit card numbers (basic pattern)
        (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', 'credit_card', 'critical'),

        # IP addresses (IPv4)
        (r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', 'ip_address', 'low'),

        # Dates of birth (various formats)
        (r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b', 'date_of_birth', 'high'),
        (r'\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])\b', 'date_of_birth', 'high'),

        # Passport numbers (simple pattern)
        (r'\b[A-Z]{1,2}[0-9]{6,9}\b', 'passport', 'high'),

        # Driver's license (varies by state, simple pattern)
        (r'\b[A-Z]{1,2}\d{6,8}\b', 'drivers_license', 'high'),

        # Bank account numbers (8-17 digits)
        (r'\b\d{8,17}\b', 'account_number', 'medium'),

        # US ZIP codes
        (r'\b\d{5}(?:-\d{4})?\b', 'zipcode', 'low'),
    ]

    def __init__(
        self,
        redact_pii: bool = False,
        redact_placeholder: str = "[REDACTED]",
        strict_mode: bool = False
    ):
        """
        Initialize PII detector.

        Args:
            redact_pii: Whether to redact PII in output
            redact_placeholder: Placeholder for redacted PII
            strict_mode: If True, detect more aggressively (may have false positives)
        """
        self.redact_pii = redact_pii
        self.redact_placeholder = redact_placeholder
        self.strict_mode = strict_mode

        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), pii_type, risk)
            for pattern, pii_type, risk in self.PII_PATTERNS
        ]

        logger.info(
            f"PIIDetector initialized: redact={redact_pii}, strict={strict_mode}"
        )

    def detect(self, text: str) -> PIIResult:
        """
        Detect PII in text.

        Args:
            text: Text to scan for PII

        Returns:
            PIIResult with detection details
        """
        pii_matches = []
        pii_types_found = set()

        # Scan for each PII pattern
        for pattern, pii_type, risk_level in self.compiled_patterns:
            for match in pattern.finditer(text):
                # Calculate confidence based on pattern and context
                confidence = self._calculate_confidence(
                    match.group(),
                    pii_type,
                    text
                )

                # Skip low-confidence matches in non-strict mode
                if not self.strict_mode and confidence < 0.5:
                    continue

                pii_matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence
                ))

                pii_types_found.add(pii_type)

        # Additional context-based detection
        if self.strict_mode:
            context_matches = self._detect_contextual_pii(text)
            pii_matches.extend(context_matches)
            pii_types_found.update([m.pii_type for m in context_matches])

        # Determine overall risk level
        risk_level = self._calculate_risk_level(pii_matches)

        # Redact PII if enabled
        redacted_text = self._redact_pii(text, pii_matches) if self.redact_pii else text

        return PIIResult(
            has_pii=len(pii_matches) > 0,
            pii_found=pii_matches,
            redacted_text=redacted_text,
            risk_level=risk_level,
            pii_types_detected=list(pii_types_found)
        )

    def _calculate_confidence(
        self,
        match_value: str,
        pii_type: str,
        context: str
    ) -> float:
        """
        Calculate confidence for a PII match.

        Args:
            match_value: The matched value
            pii_type: Type of PII
            context: Full text context

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.7  # Base confidence

        # Adjust based on PII type
        if pii_type == 'email':
            # Check if it's a real email pattern
            if '@' in match_value and '.' in match_value.split('@')[1]:
                confidence = 0.9

        elif pii_type == 'ssn':
            # SSN pattern is quite specific
            confidence = 0.95

        elif pii_type == 'credit_card':
            # Validate with Luhn algorithm
            if self._validate_luhn(match_value):
                confidence = 0.95
            else:
                confidence = 0.4  # Likely false positive

        elif pii_type == 'phone':
            # Check format validity
            digits = re.sub(r'\D', '', match_value)
            if 10 <= len(digits) <= 15:
                confidence = 0.8
            else:
                confidence = 0.5

        elif pii_type == 'ip_address':
            # Validate IP octets
            octets = match_value.split('.')
            if all(0 <= int(o) <= 255 for o in octets):
                confidence = 0.85
            else:
                confidence = 0.3

        elif pii_type in ['account_number', 'zipcode']:
            # These patterns can have many false positives
            confidence = 0.5

        return confidence

    def _validate_luhn(self, card_number: str) -> bool:
        """
        Validate credit card using Luhn algorithm.

        Args:
            card_number: Card number to validate

        Returns:
            True if passes Luhn check
        """
        # Remove non-digits
        digits = re.sub(r'\D', '', card_number)

        if len(digits) < 13 or len(digits) > 19:
            return False

        # Luhn algorithm
        def luhn_checksum(card):
            def digits_of(n):
                return [int(d) for d in str(n)]

            digits = digits_of(card)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10

        try:
            return luhn_checksum(digits) == 0
        except:
            return False

    def _detect_contextual_pii(self, text: str) -> List[PIIMatch]:
        """
        Detect PII based on context clues.

        Args:
            text: Text to scan

        Returns:
            List of contextual PII matches
        """
        matches = []

        # Look for "Name: John Doe" patterns
        name_patterns = [
            r'(?:name|patient|customer)[\s:]+([A-Z][a-z]+ [A-Z][a-z]+)',
            r'\b([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)\b',  # John A. Smith
        ]

        for pattern in name_patterns:
            for match in re.finditer(pattern, text):
                matches.append(PIIMatch(
                    pii_type='name',
                    value=match.group(1),
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.6
                ))

        # Look for address patterns
        address_pattern = r'\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)'
        for match in re.finditer(address_pattern, text):
            matches.append(PIIMatch(
                pii_type='address',
                value=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=0.65
            ))

        return matches

    def _calculate_risk_level(self, pii_matches: List[PIIMatch]) -> str:
        """
        Calculate overall risk level based on PII found.

        Args:
            pii_matches: List of PII matches

        Returns:
            Risk level string
        """
        if not pii_matches:
            return "none"

        # Map PII types to risk scores
        risk_scores = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }

        # Get maximum risk from patterns
        pii_risks = []
        for pattern, pii_type, risk in self.PII_PATTERNS:
            for match in pii_matches:
                if match.pii_type == pii_type:
                    pii_risks.append(risk_scores.get(risk, 1))

        if not pii_risks:
            return "low"

        max_risk = max(pii_risks)

        # Determine level
        if max_risk >= 4:
            return "critical"
        elif max_risk >= 3:
            return "high"
        elif max_risk >= 2:
            return "medium"
        else:
            return "low"

    def _redact_pii(self, text: str, pii_matches: List[PIIMatch]) -> str:
        """
        Redact PII from text.

        Args:
            text: Original text
            pii_matches: PII matches to redact

        Returns:
            Redacted text
        """
        # Sort matches by position (reverse order to maintain indices)
        sorted_matches = sorted(pii_matches, key=lambda m: m.start, reverse=True)

        redacted = text
        for match in sorted_matches:
            # Create type-specific placeholder
            placeholder = f"[{match.pii_type.upper()}_REDACTED]"
            redacted = (
                redacted[:match.start] +
                placeholder +
                redacted[match.end:]
            )

        return redacted

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.

        Returns:
            Dictionary with detector info
        """
        return {
            "redact_pii": self.redact_pii,
            "strict_mode": self.strict_mode,
            "num_patterns": len(self.compiled_patterns),
            "pii_types": list(set(pii_type for _, pii_type, _ in self.PII_PATTERNS))
        }
