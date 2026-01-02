"""
Safe Generator - Generator with integrated safety guardrails.

Wraps UseCaseGenerator with:
- Prompt injection detection (input validation)
- Hallucination detection (output validation)
- Output quality validation
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

from .generator import UseCaseGenerator
from ..guards import (
    PromptInjectionDetector,
    HallucinationDetector,
    OutputValidator,
    PIIDetector
)
from ..storage import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class SafetyReport:
    """Detailed safety check results."""
    # Overall status
    passed: bool
    warnings: List[str]
    blocked_reasons: List[str]

    # Individual guard results
    injection_detected: bool
    hallucination_detected: bool
    validation_failed: bool
    pii_detected: bool

    # Detailed information
    injection_risk_level: Optional[str] = None
    hallucination_confidence: Optional[float] = None
    validation_quality_score: Optional[float] = None
    pii_risk_level: Optional[str] = None
    pii_types_found: Optional[List[str]] = None

    # Summary metrics
    total_checks_run: int = 0
    checks_passed: int = 0
    checks_warned: int = 0
    checks_failed: int = 0


class SafeGenerator:
    """
    Safe use case generator with integrated guardrails.

    Safety pipeline:
    1. Check query for prompt injection
    2. Generate response
    3. Check response for hallucinations
    4. Validate output structure and quality
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        enable_injection_check: bool = True,
        enable_hallucination_check: bool = True,
        enable_validation: bool = True,
        enable_pii_check: bool = True,
        use_nli: bool = False
    ):
        """
        Initialize safe generator.

        Args:
            model: LLM model name
            temperature: Sampling temperature
            enable_injection_check: Enable prompt injection detection
            enable_hallucination_check: Enable hallucination detection
            enable_validation: Enable output validation
            enable_pii_check: Enable PII detection
            use_nli: Use NLI model for hallucination detection (slower but more accurate)
        """
        # Initialize base generator
        self.generator = UseCaseGenerator(model=model, temperature=temperature)

        # Initialize guards
        self.enable_injection_check = enable_injection_check
        self.enable_hallucination_check = enable_hallucination_check
        self.enable_validation = enable_validation
        self.enable_pii_check = enable_pii_check

        if self.enable_injection_check:
            self.injection_detector = PromptInjectionDetector(
                max_length=2000,
                suspicious_threshold=1
            )
            logger.info("Prompt injection detection enabled")

        if self.enable_hallucination_check:
            self.hallucination_detector = HallucinationDetector(
                similarity_threshold=0.5,
                min_confidence=0.6,
                use_nli=use_nli
            )
            logger.info(f"Hallucination detection enabled (NLI={use_nli})")

        if self.enable_validation:
            self.output_validator = OutputValidator(
                schema=None,  # Use default validation
                min_use_cases=1,
                max_use_cases=10,
                min_steps_per_case=2,
                max_steps_per_case=20
            )
            logger.info("Output validation enabled")

        if self.enable_pii_check:
            self.pii_detector = PIIDetector(
                redact_pii=False,  # Don't redact by default, just detect
                strict_mode=False
            )
            logger.info("PII detection enabled")

        logger.info("SafeGenerator initialized with guardrails")

    def generate(
        self,
        query: str,
        context: str,
        min_confidence: float = 0.6,
        debug: bool = False,
        skip_guards: bool = False
    ) -> Tuple[Dict[str, Any], SafetyReport]:
        """
        Safely generate use cases with comprehensive guard checks.

        Args:
            query: User query
            context: Retrieved context
            min_confidence: Minimum confidence threshold
            debug: Whether to print debug info
            skip_guards: Skip safety checks (for testing/debugging)

        Returns:
            Tuple of (output_dict, safety_report)
        """
        warnings = []
        blocked_reasons = []

        # Track detailed metrics
        total_checks = 0
        checks_passed = 0
        checks_warned = 0
        checks_failed = 0

        # Guard results
        injection_detected = False
        injection_risk_level = None
        hallucination_detected = False
        hallucination_confidence = None
        validation_failed = False
        validation_quality_score = None
        pii_detected = False
        pii_risk_level = None
        pii_types_found = []

        # Step 1: Check query for prompt injection
        if self.enable_injection_check and not skip_guards:
            total_checks += 1
            injection_result = self.injection_detector.detect(query)
            injection_risk_level = injection_result.risk_level

            if injection_result.is_injection:
                logger.warning(
                    f"Prompt injection detected! Risk: {injection_result.risk_level}, "
                    f"Patterns: {injection_result.patterns_found}"
                )

                injection_detected = True
                checks_failed += 1
                blocked_reasons.append(
                    f"Prompt injection detected ({injection_result.risk_level} risk)"
                )

                # Return early with error
                safety_report = SafetyReport(
                    passed=False,
                    warnings=warnings,
                    blocked_reasons=blocked_reasons,
                    injection_detected=True,
                    hallucination_detected=False,
                    validation_failed=False,
                    pii_detected=False,
                    injection_risk_level=injection_risk_level,
                    total_checks_run=total_checks,
                    checks_passed=checks_passed,
                    checks_warned=checks_warned,
                    checks_failed=checks_failed
                )

                return {
                    "error": "Query blocked by safety filter",
                    "safety_blocked": True,
                    "blocked_reason": blocked_reasons[0],
                    "use_cases": []
                }, safety_report

            if not injection_result.safe_to_process:
                warnings.append("Query flagged as potentially suspicious")
                checks_warned += 1
            else:
                checks_passed += 1

            if debug:
                print(f"[OK] Injection check passed (confidence: {injection_result.confidence:.2f})")

        # Step 2: Check query and context for PII
        if self.enable_pii_check and not skip_guards:
            total_checks += 1

            # Check both query and context
            query_pii_result = self.pii_detector.detect(query)
            context_pii_result = self.pii_detector.detect(context)

            if query_pii_result.has_pii or context_pii_result.has_pii:
                pii_detected = True

                # Combine PII types from both
                all_pii_types = set(query_pii_result.pii_types_detected + context_pii_result.pii_types_detected)
                pii_types_found = list(all_pii_types)

                # Use highest risk level
                risk_levels = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                query_risk = risk_levels.get(query_pii_result.risk_level, 0)
                context_risk = risk_levels.get(context_pii_result.risk_level, 0)
                max_risk = max(query_risk, context_risk)
                pii_risk_level = [k for k, v in risk_levels.items() if v == max_risk][0]

                logger.warning(
                    f"PII detected! Risk: {pii_risk_level}, "
                    f"Types: {pii_types_found}"
                )

                warnings.append(f"PII detected ({pii_risk_level} risk): {', '.join(pii_types_found)}")

                # Block if critical PII found
                if pii_risk_level == 'critical':
                    checks_failed += 1
                    blocked_reasons.append(
                        f"Critical PII detected: {', '.join(pii_types_found)}"
                    )
                else:
                    checks_warned += 1
            else:
                checks_passed += 1

            if debug:
                if pii_detected:
                    print(f"[WARN]  PII detected (risk: {pii_risk_level}): {', '.join(pii_types_found)}")
                else:
                    print("[OK] No PII detected")

        # Step 3: Generate response
        if debug:
            print("\n[AI] Generating with safety guardrails...")

        output = self.generator.generate(
            query=query,
            context=context,
            min_confidence=min_confidence,
            debug=debug
        )

        # Check for generation errors
        if "error" in output:
            safety_report = SafetyReport(
                passed=False,
                warnings=warnings + ["Generation failed"],
                blocked_reasons=blocked_reasons,
                injection_detected=injection_detected,
                hallucination_detected=False,
                validation_failed=False,
                pii_detected=pii_detected,
                injection_risk_level=injection_risk_level,
                pii_risk_level=pii_risk_level,
                pii_types_found=pii_types_found if pii_types_found else None,
                total_checks_run=total_checks,
                checks_passed=checks_passed,
                checks_warned=checks_warned,
                checks_failed=checks_failed
            )
            return output, safety_report

        # Step 4: Check generated output for PII
        if self.enable_pii_check and not skip_guards:
            total_checks += 1
            output_text = self._extract_text_for_hallucination_check(output)
            output_pii_result = self.pii_detector.detect(output_text)

            if output_pii_result.has_pii:
                if not pii_detected:
                    pii_detected = True
                    pii_types_found = output_pii_result.pii_types_detected
                    pii_risk_level = output_pii_result.risk_level
                else:
                    # Merge with existing PII findings
                    all_types = set(pii_types_found + output_pii_result.pii_types_detected)
                    pii_types_found = list(all_types)

                logger.warning(f"PII detected in generated output: {output_pii_result.pii_types_detected}")
                warnings.append(f"Generated output contains PII: {', '.join(output_pii_result.pii_types_detected)}")
                checks_warned += 1

                # Add PII info to output
                output["pii_detected"] = True
                output["pii_types"] = output_pii_result.pii_types_detected
                output["pii_risk"] = output_pii_result.risk_level
            else:
                checks_passed += 1

        # Step 5: Check for hallucinations
        if self.enable_hallucination_check and not skip_guards:
            total_checks += 1
            # Extract generated text for hallucination check
            output_text = self._extract_text_for_hallucination_check(output)

            hallucination_result = self.hallucination_detector.detect(
                output=output_text,
                context=context,
                query=query
            )

            hallucination_confidence = hallucination_result.confidence

            if hallucination_result.has_hallucination:
                logger.warning(
                    f"Hallucination detected! Confidence: {hallucination_result.confidence:.2f}, "
                    f"Issues: {hallucination_result.issues}"
                )

                hallucination_detected = True
                checks_warned += 1
                warnings.extend([f"Hallucination: {issue}" for issue in hallucination_result.issues])

                # Add warning to output
                output["hallucination_warning"] = True
                output["hallucination_issues"] = hallucination_result.issues

                # Optionally block if hallucination is severe
                if hallucination_result.confidence < 0.3:
                    checks_failed += 1
                    blocked_reasons.append(
                        f"Severe hallucination detected (confidence: {hallucination_result.confidence:.2f})"
                    )
            else:
                checks_passed += 1

            if debug:
                if hallucination_result.has_hallucination:
                    print(f"[WARN]  Hallucination warning (confidence: {hallucination_result.confidence:.2f})")
                else:
                    print(f"[OK] Hallucination check passed (confidence: {hallucination_result.confidence:.2f})")

        # Step 6: Validate output
        if self.enable_validation and not skip_guards:
            total_checks += 1
            validation_result = self.output_validator.validate(output)
            validation_quality_score = validation_result.quality_score

            if not validation_result.is_valid:
                logger.warning(
                    f"Validation failed! Errors: {validation_result.errors}"
                )

                validation_failed = True
                checks_failed += 1
                warnings.extend([f"Validation: {error}" for error in validation_result.errors])
                warnings.extend([f"Warning: {warning}" for warning in validation_result.warnings])

                # Add validation info to output
                output["validation_failed"] = True
                output["validation_errors"] = validation_result.errors
                output["validation_warnings"] = validation_result.warnings

                # Optionally block if validation fails critically
                if validation_result.quality_score < 0.3:
                    blocked_reasons.append(
                        f"Output quality too low (score: {validation_result.quality_score:.2f})"
                    )
            elif len(validation_result.warnings) > 0:
                checks_warned += 1
            else:
                checks_passed += 1

            if debug:
                if validation_failed:
                    print(f"[WARN]  Validation warnings (quality: {validation_result.quality_score:.2f})")
                else:
                    print(f"[OK] Validation passed (quality: {validation_result.quality_score:.2f})")

        # Create comprehensive safety report
        safety_report = SafetyReport(
            passed=len(blocked_reasons) == 0,
            warnings=warnings,
            blocked_reasons=blocked_reasons,
            injection_detected=injection_detected,
            hallucination_detected=hallucination_detected,
            validation_failed=validation_failed,
            pii_detected=pii_detected,
            injection_risk_level=injection_risk_level,
            hallucination_confidence=hallucination_confidence,
            validation_quality_score=validation_quality_score,
            pii_risk_level=pii_risk_level,
            pii_types_found=pii_types_found if pii_types_found else None,
            total_checks_run=total_checks,
            checks_passed=checks_passed,
            checks_warned=checks_warned,
            checks_failed=checks_failed
        )

        # Add safety metadata to output
        output["safety_passed"] = safety_report.passed
        output["safety_warnings"] = warnings
        output["safety_checks_summary"] = {
            "total": total_checks,
            "passed": checks_passed,
            "warned": checks_warned,
            "failed": checks_failed
        }

        if debug:
            if safety_report.passed:
                print(f"\n[OK] All safety checks passed ({checks_passed}/{total_checks})")
            else:
                print(f"\n[WARN]  Safety checks failed: {blocked_reasons}")
                print(f"Summary: {checks_passed} passed, {checks_warned} warned, {checks_failed} failed")

        return output, safety_report

    def generate_from_results(
        self,
        query: str,
        results: List[SearchResult],
        min_confidence: float = 0.6,
        debug: bool = False,
        skip_guards: bool = False
    ) -> Tuple[Dict[str, Any], SafetyReport]:
        """
        Safely generate from search results.

        Args:
            query: User query
            results: Search results
            min_confidence: Minimum confidence
            debug: Debug mode
            skip_guards: Skip safety checks

        Returns:
            Tuple of (output_dict, safety_report)
        """
        # Format context from results
        context = self.generator._format_context(results)

        # Generate safely
        return self.generate(
            query=query,
            context=context,
            min_confidence=min_confidence,
            debug=debug,
            skip_guards=skip_guards
        )

    def _extract_text_for_hallucination_check(self, output: Dict[str, Any]) -> str:
        """
        Extract text from output for hallucination checking.

        Args:
            output: Generation output dictionary

        Returns:
            Concatenated text from output
        """
        text_parts = []

        # Extract use case content
        use_cases = output.get("use_cases", [])

        for uc in use_cases:
            # Add title and goal
            if "title" in uc:
                text_parts.append(uc["title"])
            if "goal" in uc:
                text_parts.append(uc["goal"])

            # Add steps
            if "steps" in uc:
                text_parts.extend(uc["steps"])

            # Add expected results
            if "expected_results" in uc:
                text_parts.extend(uc["expected_results"])

        # Extract assumptions
        assumptions = output.get("assumptions", [])
        text_parts.extend(assumptions)

        return " ".join(text_parts)

    def get_guard_stats(self) -> Dict[str, Any]:
        """
        Get statistics about guard configurations.

        Returns:
            Dictionary with guard stats
        """
        stats = {
            "injection_check_enabled": self.enable_injection_check,
            "hallucination_check_enabled": self.enable_hallucination_check,
            "validation_enabled": self.enable_validation,
            "pii_check_enabled": self.enable_pii_check
        }

        if self.enable_injection_check:
            stats["injection_detector"] = self.injection_detector.get_stats()

        if self.enable_hallucination_check:
            stats["hallucination_detector"] = self.hallucination_detector.get_stats()

        if self.enable_validation:
            stats["output_validator"] = self.output_validator.get_stats()

        if self.enable_pii_check:
            stats["pii_detector"] = self.pii_detector.get_stats()

        return stats
