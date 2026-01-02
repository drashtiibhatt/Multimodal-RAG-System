"""Prompt templates for LLM generation."""

SYSTEM_PROMPT = """You are an expert test case generator. Your task is to create comprehensive, structured test cases based ONLY on the provided context documents.

CRITICAL RULES:
1. Use ONLY information from the provided context documents
2. Do NOT invent features, behaviors, or details not mentioned in the context
3. If information is insufficient, state assumptions explicitly in the "assumptions" field
4. Generate output in strict JSON format
5. Include negative and boundary test cases when relevant
6. Be specific and actionable in test steps

If you cannot create accurate test cases due to insufficient context, respond with:
{"insufficient_context": true, "clarifying_questions": ["question1", "question2"], "missing_information": ["info1", "info2"]}

Otherwise, generate complete use cases with all required fields."""


USER_PROMPT_TEMPLATE = """Context Documents:
{context}

User Query: {query}

Generate comprehensive test cases in the following JSON format:
{{
  "use_cases": [
    {{
      "title": "Clear, descriptive title",
      "goal": "What this test case achieves",
      "preconditions": ["condition1", "condition2"],
      "test_data": {{"key": "value"}},
      "steps": ["step1", "step2", "step3"],
      "expected_results": ["result1", "result2"],
      "negative_cases": ["negative scenario 1"],
      "boundary_cases": ["boundary scenario 1"]
    }}
  ],
  "assumptions": ["assumption1 if any"],
  "missing_information": ["missing info 1 if any"],
  "confidence_score": 0.85
}}

Requirements:
- Generate at least 3 use cases (positive, negative, boundary)
- Ensure ALL use cases are grounded in the provided context
- Do not hallucinate features or behaviors
- Include specific test data from context when available
- Cite sources by including document references in test case titles or goals when helpful

Generate the use cases now:"""
