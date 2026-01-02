"""Pydantic schemas for structured output validation."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class UseCase(BaseModel):
    """Individual use case structure."""

    title: str = Field(..., description="Clear, descriptive title")
    goal: str = Field(..., description="What this test achieves")
    preconditions: List[str] = Field(default_factory=list, description="Required preconditions")
    test_data: Dict[str, Any] = Field(default_factory=dict, description="Test data")
    steps: List[str] = Field(..., description="Detailed test steps")
    expected_results: List[str] = Field(..., description="Expected outcomes")
    negative_cases: List[str] = Field(default_factory=list, description="Negative test scenarios")
    boundary_cases: List[str] = Field(default_factory=list, description="Boundary test scenarios")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "title": "Successful user signup with valid credentials",
                "goal": "Verify user can create account with valid email and password",
                "preconditions": ["User is logged out", "Email is not registered"],
                "test_data": {
                    "email": "newuser@example.com",
                    "password": "SecurePass123!"
                },
                "steps": [
                    "Navigate to signup page",
                    "Enter valid email address",
                    "Enter valid password",
                    "Click 'Create Account' button"
                ],
                "expected_results": [
                    "Account is created successfully",
                    "Verification email is sent",
                    "User sees 'Check your email' message"
                ],
                "negative_cases": ["Duplicate email rejected with ERR_001"],
                "boundary_cases": ["Password at minimum length (8 characters)"]
            }
        }


class GenerationOutput(BaseModel):
    """Complete generation output structure."""

    use_cases: List[UseCase] = Field(..., description="List of generated use cases")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made")
    missing_information: List[str] = Field(default_factory=list, description="Missing information")
    insufficient_context: bool = Field(False, description="Whether context was insufficient")
    clarifying_questions: List[str] = Field(default_factory=list, description="Questions to ask user")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence")
    timestamp: datetime = Field(default_factory=datetime.now, description="Generation timestamp")

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
