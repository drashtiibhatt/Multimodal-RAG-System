"""
RAG Metrics - End-to-end RAG evaluation using RAGAS framework.

Implements RAG-specific metrics:
- Context Precision: How relevant are retrieved contexts
- Context Recall: Coverage of ground truth in retrieved contexts
- Faithfulness: Factual consistency with retrieved contexts
- Answer Relevancy: Relevance of answer to the question
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from RAG evaluation."""
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    ragas_score: float  # Overall RAGAS score
    num_samples: int


class RAGMetrics:
    """
    Evaluate end-to-end RAG performance using RAGAS framework.

    RAGAS Metrics:
    - Context Precision: Relevance of retrieved contexts to question
    - Context Recall: Coverage of ground truth answer in contexts
    - Faithfulness: Factual consistency of answer with contexts
    - Answer Relevancy: Relevance of generated answer to question
    """

    def __init__(self, use_ragas: bool = True):
        """
        Initialize RAG metrics.

        Args:
            use_ragas: Whether to use RAGAS library (requires transformers)
        """
        self.use_ragas = use_ragas
        self.ragas_available = False

        # Try to import RAGAS if requested
        if self.use_ragas:
            try:
                from ragas import evaluate
                from ragas.metrics import (
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy
                )
                from datasets import Dataset

                self.ragas_evaluate = evaluate
                self.context_precision_metric = context_precision
                self.context_recall_metric = context_recall
                self.faithfulness_metric = faithfulness
                self.answer_relevancy_metric = answer_relevancy
                self.Dataset = Dataset

                self.ragas_available = True
                logger.info("RAGAS framework available")
            except ImportError as e:
                logger.warning(
                    f"RAGAS requested but not available: {e}. "
                    "Install with: pip install ragas"
                )
                self.use_ragas = False

        logger.info(f"RAGMetrics initialized (RAGAS={self.ragas_available})")

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> RAGResult:
        """
        Evaluate RAG performance using RAGAS.

        Args:
            questions: List of user questions
            answers: List of generated answers
            contexts: List of retrieved context lists for each question
            ground_truths: Optional list of ground truth answers

        Returns:
            RAGResult with all computed metrics

        Raises:
            ValueError: If input lists have mismatched lengths
            RuntimeError: If RAGAS is not available
        """
        if not self.ragas_available:
            logger.warning("RAGAS not available, using fallback metrics")
            return self._evaluate_fallback(
                questions, answers, contexts, ground_truths
            )

        # Validate inputs
        if not (len(questions) == len(answers) == len(contexts)):
            raise ValueError(
                f"Mismatch: {len(questions)} questions, "
                f"{len(answers)} answers, {len(contexts)} contexts"
            )

        if ground_truths and len(ground_truths) != len(questions):
            raise ValueError(
                f"Mismatch: {len(ground_truths)} ground truths, "
                f"{len(questions)} questions"
            )

        num_samples = len(questions)

        try:
            # Create dataset for RAGAS
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts
            }

            if ground_truths:
                data["ground_truth"] = ground_truths

            dataset = self.Dataset.from_dict(data)

            # Define metrics to compute
            metrics = [
                self.context_precision_metric,
                self.faithfulness_metric,
                self.answer_relevancy_metric
            ]

            # Add context recall if ground truths provided
            if ground_truths:
                metrics.append(self.context_recall_metric)

            # Evaluate
            logger.info(f"Evaluating {num_samples} samples with RAGAS...")
            result = self.ragas_evaluate(dataset, metrics=metrics)

            # Extract scores
            context_precision = result.get("context_precision", 0.0)
            context_recall = result.get("context_recall", 0.0) if ground_truths else 0.0
            faithfulness = result.get("faithfulness", 0.0)
            answer_relevancy = result.get("answer_relevancy", 0.0)

            # Calculate overall RAGAS score (average of all metrics)
            scores = [context_precision, faithfulness, answer_relevancy]
            if ground_truths:
                scores.append(context_recall)

            ragas_score = sum(scores) / len(scores) if scores else 0.0

            logger.info(
                f"RAGAS evaluation complete: "
                f"Faithfulness={faithfulness:.3f}, "
                f"Answer Relevancy={answer_relevancy:.3f}"
            )

            return RAGResult(
                context_precision=context_precision,
                context_recall=context_recall,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                ragas_score=ragas_score,
                num_samples=num_samples
            )

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            logger.warning("Falling back to simple metrics")
            return self._evaluate_fallback(
                questions, answers, contexts, ground_truths
            )

    def _evaluate_fallback(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> RAGResult:
        """
        Fallback evaluation without RAGAS library.

        Uses simple heuristics:
        - Context Precision: Average context length (proxy for relevance)
        - Context Recall: Overlap between contexts and ground truth
        - Faithfulness: Overlap between answer and contexts
        - Answer Relevancy: Overlap between answer and question

        Args:
            questions: List of questions
            answers: List of answers
            contexts: List of context lists
            ground_truths: Optional ground truths

        Returns:
            RAGResult with simple metric estimates
        """
        num_samples = len(questions)

        context_precision_scores = []
        context_recall_scores = []
        faithfulness_scores = []
        answer_relevancy_scores = []

        for idx in range(num_samples):
            question = questions[idx]
            answer = answers[idx]
            context_list = contexts[idx]

            # Context Precision: Normalized context length
            context_precision_scores.append(
                self._simple_context_precision(context_list)
            )

            # Context Recall: Overlap with ground truth if available
            if ground_truths and idx < len(ground_truths):
                context_recall_scores.append(
                    self._simple_context_recall(context_list, ground_truths[idx])
                )
            else:
                context_recall_scores.append(0.0)

            # Faithfulness: Answer overlap with contexts
            faithfulness_scores.append(
                self._simple_faithfulness(answer, context_list)
            )

            # Answer Relevancy: Answer overlap with question
            answer_relevancy_scores.append(
                self._simple_answer_relevancy(answer, question)
            )

        # Aggregate scores
        context_precision = sum(context_precision_scores) / num_samples
        context_recall = sum(context_recall_scores) / num_samples if ground_truths else 0.0
        faithfulness = sum(faithfulness_scores) / num_samples
        answer_relevancy = sum(answer_relevancy_scores) / num_samples

        scores = [context_precision, faithfulness, answer_relevancy]
        if ground_truths:
            scores.append(context_recall)

        ragas_score = sum(scores) / len(scores)

        logger.info(
            f"Fallback evaluation complete: "
            f"Faithfulness={faithfulness:.3f}, "
            f"Answer Relevancy={answer_relevancy:.3f}"
        )

        return RAGResult(
            context_precision=context_precision,
            context_recall=context_recall,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            ragas_score=ragas_score,
            num_samples=num_samples
        )

    def _simple_context_precision(self, contexts: List[str]) -> float:
        """Simple context precision proxy."""
        if not contexts:
            return 0.0

        # Assume longer contexts are more informative (up to a point)
        avg_length = sum(len(c.split()) for c in contexts) / len(contexts)
        # Normalize to 0-1 (assume 100 words is optimal)
        return min(avg_length / 100.0, 1.0)

    def _simple_context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """Simple context recall based on word overlap."""
        if not contexts or not ground_truth:
            return 0.0

        context_text = " ".join(contexts).lower()
        ground_truth_words = set(ground_truth.lower().split())

        if not ground_truth_words:
            return 0.0

        # Count overlap
        overlap = sum(1 for word in ground_truth_words if word in context_text)
        return overlap / len(ground_truth_words)

    def _simple_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Simple faithfulness based on word overlap."""
        if not answer or not contexts:
            return 0.0

        context_text = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())

        if not answer_words:
            return 0.0

        # Count overlap
        overlap = sum(1 for word in answer_words if word in context_text)
        return overlap / len(answer_words)

    def _simple_answer_relevancy(self, answer: str, question: str) -> float:
        """Simple answer relevancy based on word overlap."""
        if not answer or not question:
            return 0.0

        answer_words = set(answer.lower().split())
        question_words = set(question.lower().split())

        if not question_words:
            return 0.0

        # Count overlap
        overlap = len(answer_words.intersection(question_words))
        return overlap / len(question_words)

    def get_summary(self, result: RAGResult) -> str:
        """
        Get human-readable summary of results.

        Args:
            result: RAGResult to summarize

        Returns:
            Formatted summary string
        """
        lines = [
            f"RAG Evaluation Summary ({result.num_samples} samples)",
            "=" * 60,
            "",
            "RAGAS Metrics:",
            f"  Context Precision:  {result.context_precision:.4f}",
            f"  Context Recall:     {result.context_recall:.4f}",
            f"  Faithfulness:       {result.faithfulness:.4f}",
            f"  Answer Relevancy:   {result.answer_relevancy:.4f}",
            "",
            f"Overall RAGAS Score:  {result.ragas_score:.4f}",
        ]

        if not self.ragas_available:
            lines.append("")
            lines.append("Note: Using fallback metrics (RAGAS not available)")

        return "\n".join(lines)
