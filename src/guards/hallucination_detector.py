"""
Hallucination Detector - Detects when LLM generates ungrounded content.

Uses Natural Language Inference (NLI) models to check if outputs
are grounded in the retrieved context.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """Result from hallucination detection."""
    has_hallucination: bool
    confidence: float
    grounded_score: float
    fabrication_score: float
    issues: List[str]
    safe_to_use: bool
    details: Dict[str, Any]


class HallucinationDetector:
    """
    Detect hallucinations in LLM-generated output.

    Uses NLI (Natural Language Inference) models to check if
    the generated output is grounded in the retrieved context.
    """

    def __init__(
        self,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        similarity_threshold: float = 0.5,
        min_confidence: float = 0.6,
        use_nli: bool = True
    ):
        """
        Initialize hallucination detector.

        Args:
            nli_model: NLI model for entailment checking
            similarity_threshold: Minimum similarity for grounding (0-1)
            min_confidence: Minimum confidence for acceptance (0-1)
            use_nli: Whether to use NLI model (False for faster, simpler checks)
        """
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.use_nli = use_nli and NLI_AVAILABLE

        # Load NLI model if available and requested
        if self.use_nli:
            try:
                logger.info(f"Loading NLI model: {nli_model}")
                self.cross_encoder = CrossEncoder(nli_model)
                logger.info("NLI model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load NLI model: {str(e)}")
                logger.warning("Falling back to simple similarity checks")
                self.cross_encoder = None
                self.use_nli = False
        else:
            self.cross_encoder = None
            if not NLI_AVAILABLE:
                logger.warning(
                    "sentence-transformers not available. "
                    "Install with: pip install sentence-transformers"
                )

    def detect(
        self,
        output: str,
        context: str,
        query: str = ""
    ) -> HallucinationResult:
        """
        Detect hallucinations in output.

        Args:
            output: LLM-generated output to check
            context: Retrieved context that should ground the output
            query: User query (optional, for additional validation)

        Returns:
            HallucinationResult with detection details
        """
        # Step 1: Check if output is grounded in context
        grounded_score = self._check_grounding(output, context)

        # Step 2: Check for fabricated facts
        fabrication_score = self._check_fabrication(output, context)

        # Step 3: Calculate overall confidence
        confidence = self._calculate_confidence(
            grounded_score,
            fabrication_score
        )

        # Step 4: Identify specific issues
        issues = self._identify_issues(
            output,
            context,
            grounded_score,
            fabrication_score,
            confidence
        )

        # Determine if hallucination detected
        has_hallucination = (
            grounded_score < self.similarity_threshold or
            confidence < self.min_confidence or
            fabrication_score < 0.5
        )

        # Additional details
        details = {
            "method": "NLI" if self.use_nli else "simple",
            "output_length": len(output),
            "context_length": len(context),
            "num_issues": len(issues)
        }

        return HallucinationResult(
            has_hallucination=has_hallucination,
            confidence=confidence,
            grounded_score=grounded_score,
            fabrication_score=fabrication_score,
            issues=issues,
            safe_to_use=not has_hallucination,
            details=details
        )

    def _check_grounding(self, output: str, context: str) -> float:
        """
        Check if output is grounded in context using NLI.

        Args:
            output: Output to check
            context: Context to check against

        Returns:
            Grounding score (0-1), higher = better grounded
        """
        if not output or not context:
            return 0.0

        if self.use_nli and self.cross_encoder:
            try:
                # Use cross-encoder for entailment
                # Format: [premise, hypothesis]
                # We check if context entails output
                pairs = [[context, output]]
                scores = self.cross_encoder.predict(pairs)

                # Convert to probability
                entailment_score = float(scores[0])

                # Normalize to 0-1 range if needed
                if entailment_score < 0:
                    entailment_score = 1 / (1 + np.exp(-entailment_score))

                return max(0.0, min(1.0, entailment_score))

            except Exception as e:
                logger.warning(f"NLI check failed: {str(e)}")
                # Fall back to simple check
                return self._simple_grounding_check(output, context)
        else:
            return self._simple_grounding_check(output, context)

    def _simple_grounding_check(self, output: str, context: str) -> float:
        """
        Simple grounding check based on word overlap.

        Args:
            output: Output to check
            context: Context to check against

        Returns:
            Grounding score (0-1)
        """
        # Tokenize (simple whitespace split)
        output_words = set(output.lower().split())
        context_words = set(context.lower().split())

        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        output_words -= stop_words
        context_words -= stop_words

        if not output_words:
            return 0.0

        # Calculate overlap
        overlap = len(output_words & context_words)
        score = overlap / len(output_words)

        return score

    def _check_fabrication(self, output: str, context: str) -> float:
        """
        Check for fabricated facts in output.

        Args:
            output: Output to check
            context: Context to check against

        Returns:
            Fabrication score (0-1), higher = less fabricated
        """
        # Extract claims from output
        claims = self._extract_claims(output)

        if not claims:
            return 1.0  # No claims = no fabrication

        # Check each claim against context
        verified_count = 0
        for claim in claims:
            if self._verify_claim(claim, context):
                verified_count += 1

        return verified_count / len(claims)

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.

        Simple implementation: split by sentences.
        Could be improved with NER or claim extraction models.

        Args:
            text: Text to extract claims from

        Returns:
            List of claim strings
        """
        # Split by sentence-ending punctuation
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in '.!?':
                sentence = current.strip()
                if len(sentence) > 10:  # Minimum claim length
                    sentences.append(sentence)
                current = ""

        # Add remaining text
        if current.strip() and len(current.strip()) > 10:
            sentences.append(current.strip())

        return sentences

    def _verify_claim(self, claim: str, context: str) -> bool:
        """
        Verify if a claim is supported by context.

        Args:
            claim: Claim to verify
            context: Context to check against

        Returns:
            True if claim is verified, False otherwise
        """
        if self.use_nli and self.cross_encoder:
            try:
                pairs = [[context, claim]]
                scores = self.cross_encoder.predict(pairs)
                score = float(scores[0])

                # Normalize if needed
                if score < 0:
                    score = 1 / (1 + np.exp(-score))

                return score > self.similarity_threshold
            except Exception as e:
                logger.debug(f"Claim verification failed: {str(e)}")
                # Fall back to simple check
                return self._simple_claim_check(claim, context)
        else:
            return self._simple_claim_check(claim, context)

    def _simple_claim_check(self, claim: str, context: str) -> bool:
        """Simple claim verification using substring matching."""
        # Remove punctuation
        claim_clean = claim.lower().replace('.', '').replace('!', '').replace('?', '')

        # Check if key parts of claim appear in context
        claim_words = set(claim_clean.split())
        context_lower = context.lower()

        # At least 50% of claim words should be in context
        found_count = sum(1 for word in claim_words if word in context_lower)

        return found_count / len(claim_words) > 0.5 if claim_words else False

    def _calculate_confidence(
        self,
        grounded_score: float,
        fabrication_score: float
    ) -> float:
        """
        Calculate overall confidence score.

        Args:
            grounded_score: Grounding score (0-1)
            fabrication_score: Fabrication score (0-1)

        Returns:
            Confidence score (0-1)
        """
        # Weighted average: grounding is more important
        return 0.6 * grounded_score + 0.4 * fabrication_score

    def _identify_issues(
        self,
        output: str,
        context: str,
        grounded_score: float,
        fabrication_score: float,
        confidence: float
    ) -> List[str]:
        """
        Identify specific issues in output.

        Args:
            output: Output to check
            context: Context
            grounded_score: Grounding score
            fabrication_score: Fabrication score
            confidence: Confidence score

        Returns:
            List of issue descriptions
        """
        issues = []

        # Check grounding
        if grounded_score < self.similarity_threshold:
            issues.append(
                f"Output not well-grounded in context "
                f"(score: {grounded_score:.2f}, threshold: {self.similarity_threshold})"
            )

        # Check fabrication
        if fabrication_score < 0.5:
            issues.append(
                f"Potential fabricated content detected "
                f"(verification: {fabrication_score:.2f})"
            )

        # Check confidence
        if confidence < self.min_confidence:
            issues.append(
                f"Low confidence in output quality "
                f"(score: {confidence:.2f}, minimum: {self.min_confidence})"
            )

        # Check output length
        if len(output) < 50:
            issues.append("Output too short (may be incomplete)")

        # Check for uncertainty markers
        uncertainty_phrases = [
            "i don't know",
            "i'm not sure",
            "unclear",
            "cannot determine",
            "insufficient information"
        ]

        output_lower = output.lower()
        for phrase in uncertainty_phrases:
            if phrase in output_lower:
                issues.append(f"Model expressed uncertainty: '{phrase}'")
                break

        # Check for empty context
        if not context or len(context.strip()) < 10:
            issues.append("Context is empty or too short")

        return issues

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.

        Returns:
            Dictionary with detector info
        """
        return {
            "nli_available": NLI_AVAILABLE,
            "nli_enabled": self.use_nli,
            "model_loaded": self.cross_encoder is not None,
            "similarity_threshold": self.similarity_threshold,
            "min_confidence": self.min_confidence
        }
