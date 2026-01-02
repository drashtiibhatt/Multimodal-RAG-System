"""
Output Validator - Validates LLM output meets quality standards.

Ensures that generated outputs conform to expected schema
and satisfy business rules.
"""

from typing import Dict, Any, List, Optional, Type, Tuple, Callable
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from output validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validated_output: Optional[Dict[str, Any]]
    quality_score: float  # 0-1


class OutputValidator:
    """
    Validate LLM-generated output.

    Performs schema validation and business rule checks
    to ensure output quality and completeness.
    """

    def __init__(
        self,
        schema: Optional[Type[BaseModel]] = None,
        min_use_cases: int = 1,
        max_use_cases: int = 10,
        min_steps_per_case: int = 2,
        max_steps_per_case: int = 20,
        custom_validators: Optional[List[Callable[[Dict[str, Any]], Tuple[List[str], List[str]]]]] = None
    ):
        """
        Initialize output validator.

        Args:
            schema: Pydantic schema for output validation
            min_use_cases: Minimum number of use cases required
            max_use_cases: Maximum number of use cases allowed
            min_steps_per_case: Minimum steps per use case
            max_steps_per_case: Maximum steps per use case
            custom_validators: List of custom validation functions that take output dict
                             and return (errors, warnings) tuple
        """
        self.schema = schema
        self.min_use_cases = min_use_cases
        self.max_use_cases = max_use_cases
        self.min_steps_per_case = min_steps_per_case
        self.max_steps_per_case = max_steps_per_case
        self.custom_validators = custom_validators or []

        logger.info(
            f"OutputValidator initialized: "
            f"use_cases={min_use_cases}-{max_use_cases}, "
            f"steps={min_steps_per_case}-{max_steps_per_case}, "
            f"custom_validators={len(self.custom_validators)}"
        )

    def validate(self, output: Dict[str, Any]) -> ValidationResult:
        """
        Validate output against schema and business rules.

        Args:
            output: LLM output dictionary to validate

        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        validated_output = None

        # Step 1: Schema validation (if schema provided)
        if self.schema:
            schema_result = self._validate_schema(output)
            if not schema_result["valid"]:
                errors.extend(schema_result["errors"])
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    validated_output=None,
                    quality_score=0.0
                )
            validated_output = schema_result["validated"]
        else:
            validated_output = output

        # Step 2: Business rule validation
        business_errors, business_warnings = self._validate_business_rules(output)
        errors.extend(business_errors)
        warnings.extend(business_warnings)

        # Step 3: Content quality validation
        quality_errors, quality_warnings = self._validate_content_quality(output)
        errors.extend(quality_errors)
        warnings.extend(quality_warnings)

        # Step 4: Completeness check
        completeness_warnings = self._check_completeness(output)
        warnings.extend(completeness_warnings)

        # Step 5: Run custom validators
        custom_errors, custom_warnings = self._run_custom_validators(output)
        errors.extend(custom_errors)
        warnings.extend(custom_warnings)

        # Step 6: Calculate quality score
        quality_score = self._calculate_quality_score(
            output,
            len(errors),
            len(warnings)
        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_output=validated_output if len(errors) == 0 else None,
            quality_score=quality_score
        )

    def _validate_schema(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate output against Pydantic schema.

        Args:
            output: Output to validate

        Returns:
            Dict with validation results
        """
        try:
            validated = self.schema(**output)
            return {
                "valid": True,
                "validated": validated.model_dump(),
                "errors": []
            }
        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                field = '.'.join(str(x) for x in error['loc'])
                message = error['msg']
                error_messages.append(f"Schema error in {field}: {message}")

            return {
                "valid": False,
                "validated": None,
                "errors": error_messages
            }
        except Exception as e:
            return {
                "valid": False,
                "validated": None,
                "errors": [f"Schema validation failed: {str(e)}"]
            }

    def _validate_business_rules(
        self,
        output: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate business rules.

        Args:
            output: Output to validate

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Check use cases count
        use_cases = output.get("use_cases", [])
        num_use_cases = len(use_cases)

        if num_use_cases < self.min_use_cases:
            errors.append(
                f"Too few use cases: {num_use_cases} "
                f"(minimum: {self.min_use_cases})"
            )

        if num_use_cases > self.max_use_cases:
            warnings.append(
                f"Many use cases: {num_use_cases} "
                f"(recommended maximum: {self.max_use_cases})"
            )

        # Validate each use case
        for idx, uc in enumerate(use_cases, 1):
            uc_errors, uc_warnings = self._validate_use_case(uc, idx)
            errors.extend(uc_errors)
            warnings.extend(uc_warnings)

        return errors, warnings

    def _validate_use_case(
        self,
        use_case: Dict,
        idx: int
    ) -> Tuple[List[str], List[str]]:
        """
        Validate individual use case.

        Args:
            use_case: Use case to validate
            idx: Use case index (1-based)

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Check required fields
        if not use_case.get("title"):
            errors.append(f"Use case {idx}: Missing title")
        elif len(use_case["title"]) < 5:
            warnings.append(f"Use case {idx}: Title too short")

        if not use_case.get("goal"):
            errors.append(f"Use case {idx}: Missing goal")

        if not use_case.get("steps"):
            errors.append(f"Use case {idx}: Missing steps")
        else:
            # Check steps count
            steps = use_case["steps"]
            num_steps = len(steps)

            if num_steps < self.min_steps_per_case:
                errors.append(
                    f"Use case {idx}: Too few steps ({num_steps}, "
                    f"minimum: {self.min_steps_per_case})"
                )

            if num_steps > self.max_steps_per_case:
                warnings.append(
                    f"Use case {idx}: Many steps ({num_steps}, "
                    f"recommended maximum: {self.max_steps_per_case})"
                )

            # Check for empty steps
            for step_idx, step in enumerate(steps, 1):
                if not step or len(str(step).strip()) < 3:
                    warnings.append(
                        f"Use case {idx}, step {step_idx}: "
                        f"Step appears empty or too short"
                    )

        # Check for expected results
        if not use_case.get("expected_results"):
            warnings.append(
                f"Use case {idx}: Missing expected results"
            )

        return errors, warnings

    def _validate_content_quality(
        self,
        output: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate content quality.

        Args:
            output: Output to validate

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        use_cases = output.get("use_cases", [])

        # Check for duplicate titles
        titles = [uc.get("title", "") for uc in use_cases]
        if len(titles) != len(set(titles)):
            warnings.append("Duplicate use case titles detected")

        # Check for overly generic content
        generic_phrases = ["test the system", "verify functionality", "check if works"]
        for idx, uc in enumerate(use_cases, 1):
            title = uc.get("title", "").lower()
            if any(phrase in title for phrase in generic_phrases):
                warnings.append(
                    f"Use case {idx}: Title appears generic or vague"
                )

        return errors, warnings

    def _check_completeness(self, output: Dict[str, Any]) -> List[str]:
        """
        Check output completeness.

        Args:
            output: Output to check

        Returns:
            List of completeness warnings
        """
        warnings = []

        # Check insufficient context flag
        if output.get("insufficient_context"):
            if not output.get("missing_information"):
                warnings.append(
                    "Insufficient context flag set but "
                    "no missing information listed"
                )

            if not output.get("clarifying_questions"):
                warnings.append(
                    "Insufficient context flag set but "
                    "no clarifying questions provided"
                )

        # Check confidence score
        confidence = output.get("confidence_score", 1.0)
        if confidence < 0.5:
            warnings.append(
                f"Low confidence score: {confidence:.2f}"
            )

        # Check assumptions
        use_cases = output.get("use_cases", [])
        assumptions = output.get("assumptions", [])

        if len(use_cases) > 0 and len(assumptions) == 0:
            warnings.append(
                "Use cases generated but no assumptions listed "
                "(consider documenting assumptions)"
            )

        return warnings

    def _calculate_quality_score(
        self,
        output: Dict[str, Any],
        num_errors: int,
        num_warnings: int
    ) -> float:
        """
        Calculate overall quality score.

        Args:
            output: Output being validated
            num_errors: Number of errors found
            num_warnings: Number of warnings found

        Returns:
            Quality score (0-1)
        """
        # Start with perfect score
        score = 1.0

        # Deduct for errors (major)
        score -= num_errors * 0.2

        # Deduct for warnings (minor)
        score -= num_warnings * 0.05

        # Bonus for good practices
        use_cases = output.get("use_cases", [])

        # Bonus for including negative cases
        has_negative = any(
            uc.get("negative_cases") and len(uc["negative_cases"]) > 0
            for uc in use_cases
        )
        if has_negative:
            score += 0.05

        # Bonus for including boundary cases
        has_boundary = any(
            uc.get("boundary_cases") and len(uc["boundary_cases"]) > 0
            for uc in use_cases
        )
        if has_boundary:
            score += 0.05

        # Bonus for including test data
        has_test_data = any(
            uc.get("test_data") and len(uc["test_data"]) > 0
            for uc in use_cases
        )
        if has_test_data:
            score += 0.05

        # Clamp to 0-1 range
        return max(0.0, min(1.0, score))

    def _run_custom_validators(
        self,
        output: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Run custom validation functions.

        Args:
            output: Output to validate

        Returns:
            Tuple of (errors, warnings) from all custom validators
        """
        all_errors = []
        all_warnings = []

        for validator_func in self.custom_validators:
            try:
                errors, warnings = validator_func(output)
                all_errors.extend(errors)
                all_warnings.extend(warnings)
            except Exception as e:
                logger.error(f"Custom validator failed: {str(e)}")
                all_warnings.append(f"Custom validator error: {str(e)}")

        return all_errors, all_warnings

    def add_custom_validator(
        self,
        validator_func: Callable[[Dict[str, Any]], Tuple[List[str], List[str]]]
    ) -> None:
        """
        Add a custom validation function.

        Args:
            validator_func: Function that takes output dict and returns (errors, warnings)

        Example:
            def my_validator(output):
                errors = []
                warnings = []
                if len(output.get("use_cases", [])) == 0:
                    errors.append("No use cases generated")
                return errors, warnings

            validator.add_custom_validator(my_validator)
        """
        self.custom_validators.append(validator_func)
        logger.info("Custom validator added")

    def remove_all_custom_validators(self) -> int:
        """
        Remove all custom validators.

        Returns:
            Number of validators removed
        """
        count = len(self.custom_validators)
        self.custom_validators = []
        logger.info(f"Removed {count} custom validators")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get validator statistics.

        Returns:
            Dictionary with validator info
        """
        return {
            "has_schema": self.schema is not None,
            "min_use_cases": self.min_use_cases,
            "max_use_cases": self.max_use_cases,
            "min_steps_per_case": self.min_steps_per_case,
            "max_steps_per_case": self.max_steps_per_case,
            "custom_validators_count": len(self.custom_validators)
        }
