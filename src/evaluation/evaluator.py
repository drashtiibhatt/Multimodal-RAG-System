"""
RAG Evaluator - Main class for comprehensive RAG system evaluation.

Orchestrates retrieval, generation, and end-to-end RAG metrics.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
import json
from pathlib import Path

from .retrieval_metrics import RetrievalMetrics, RetrievalResult
from .generation_metrics import GenerationMetrics, GenerationResult
from .rag_metrics import RAGMetrics, RAGResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result containing all metrics."""
    retrieval: Optional[RetrievalResult] = None
    generation: Optional[GenerationResult] = None
    rag: Optional[RAGResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}

        if self.retrieval:
            result["retrieval"] = asdict(self.retrieval)

        if self.generation:
            result["generation"] = asdict(self.generation)

        if self.rag:
            result["rag"] = asdict(self.rag)

        return result


class RAGEvaluator:
    """
    Comprehensive RAG system evaluator.

    Provides evaluation for:
    - Retrieval quality (Recall@k, Precision@k, MRR, NDCG)
    - Generation quality (BLEU, ROUGE, BERTScore)
    - End-to-end RAG (RAGAS metrics)
    """

    def __init__(
        self,
        k_values: List[int] = None,
        use_bertscore: bool = False,
        use_ragas: bool = True
    ):
        """
        Initialize RAG evaluator.

        Args:
            k_values: K values for retrieval metrics (default: [1, 3, 5, 10])
            use_bertscore: Whether to compute BERTScore for generation
            use_ragas: Whether to use RAGAS framework for RAG metrics
        """
        self.retrieval_metrics = RetrievalMetrics(k_values=k_values)
        self.generation_metrics = GenerationMetrics(use_bertscore=use_bertscore)
        self.rag_metrics = RAGMetrics(use_ragas=use_ragas)

        logger.info(
            f"RAGEvaluator initialized "
            f"(k_values={k_values}, "
            f"BERTScore={use_bertscore}, "
            f"RAGAS={use_ragas})"
        )

    def evaluate_retrieval(
        self,
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        relevance_scores: List[List[float]] = None
    ) -> RetrievalResult:
        """
        Evaluate retrieval performance.

        Args:
            retrieved_docs: Retrieved document IDs for each query
            relevant_docs: Relevant document IDs for each query
            relevance_scores: Optional graded relevance scores

        Returns:
            RetrievalResult with all metrics
        """
        logger.info(f"Evaluating retrieval for {len(retrieved_docs)} queries")

        return self.retrieval_metrics.evaluate(
            retrieved_docs=retrieved_docs,
            relevant_docs=relevant_docs,
            relevance_scores=relevance_scores
        )

    def evaluate_generation(
        self,
        predictions: List[str],
        references: List[str]
    ) -> GenerationResult:
        """
        Evaluate generation quality.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            GenerationResult with all metrics
        """
        logger.info(f"Evaluating generation for {len(predictions)} samples")

        return self.generation_metrics.evaluate(
            predictions=predictions,
            references=references
        )

    def evaluate_rag(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> RAGResult:
        """
        Evaluate end-to-end RAG performance.

        Args:
            questions: User questions
            answers: Generated answers
            contexts: Retrieved contexts for each question
            ground_truths: Optional ground truth answers

        Returns:
            RAGResult with RAGAS metrics
        """
        logger.info(f"Evaluating RAG for {len(questions)} samples")

        return self.rag_metrics.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths
        )

    def evaluate_complete(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        retrieved_doc_ids: List[List[str]] = None,
        relevant_doc_ids: List[List[str]] = None,
        ground_truth_answers: List[str] = None
    ) -> EvaluationResult:
        """
        Perform complete evaluation (retrieval + generation + RAG).

        Args:
            questions: User questions
            answers: Generated answers
            contexts: Retrieved contexts
            retrieved_doc_ids: Retrieved document IDs (for retrieval metrics)
            relevant_doc_ids: Relevant document IDs (for retrieval metrics)
            ground_truth_answers: Ground truth answers (for generation/RAG metrics)

        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Starting complete evaluation for {len(questions)} samples")

        result = EvaluationResult()

        # Evaluate retrieval if document IDs provided
        if retrieved_doc_ids and relevant_doc_ids:
            try:
                result.retrieval = self.evaluate_retrieval(
                    retrieved_docs=retrieved_doc_ids,
                    relevant_docs=relevant_doc_ids
                )
                logger.info("Retrieval evaluation complete")
            except Exception as e:
                logger.error(f"Retrieval evaluation failed: {e}")

        # Evaluate generation if ground truths provided
        if ground_truth_answers:
            try:
                result.generation = self.evaluate_generation(
                    predictions=answers,
                    references=ground_truth_answers
                )
                logger.info("Generation evaluation complete")
            except Exception as e:
                logger.error(f"Generation evaluation failed: {e}")

        # Evaluate end-to-end RAG
        try:
            result.rag = self.evaluate_rag(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truth_answers
            )
            logger.info("RAG evaluation complete")
        except Exception as e:
            logger.error(f"RAG evaluation failed: {e}")

        logger.info("Complete evaluation finished")

        return result

    def get_summary(self, result: EvaluationResult) -> str:
        """
        Get human-readable summary of all results.

        Args:
            result: EvaluationResult to summarize

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 70,
            "RAG SYSTEM EVALUATION REPORT",
            "=" * 70,
            ""
        ]

        # Retrieval metrics
        if result.retrieval:
            lines.append(self.retrieval_metrics.get_summary(result.retrieval))
            lines.append("")
            lines.append("=" * 70)
            lines.append("")

        # Generation metrics
        if result.generation:
            lines.append(self.generation_metrics.get_summary(result.generation))
            lines.append("")
            lines.append("=" * 70)
            lines.append("")

        # RAG metrics
        if result.rag:
            lines.append(self.rag_metrics.get_summary(result.rag))
            lines.append("")
            lines.append("=" * 70)

        return "\n".join(lines)

    def save_results(
        self,
        result: EvaluationResult,
        output_path: str
    ):
        """
        Save evaluation results to JSON file.

        Args:
            result: EvaluationResult to save
            output_path: Path to output JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Evaluation results saved to {output_path}")

    def load_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load evaluation results from JSON file.

        Args:
            input_path: Path to input JSON file

        Returns:
            Dictionary with evaluation results
        """
        with open(input_path, 'r') as f:
            results = json.load(f)

        logger.info(f"Evaluation results loaded from {input_path}")

        return results

    def compare_results(
        self,
        result1: EvaluationResult,
        result2: EvaluationResult,
        names: tuple = ("Model A", "Model B")
    ) -> str:
        """
        Compare two evaluation results side-by-side.

        Args:
            result1: First evaluation result
            result2: Second evaluation result
            names: Names for the two models/systems

        Returns:
            Formatted comparison string
        """
        lines = [
            "=" * 80,
            f"COMPARISON: {names[0]} vs {names[1]}",
            "=" * 80,
            ""
        ]

        # Compare retrieval metrics
        if result1.retrieval and result2.retrieval:
            lines.append("RETRIEVAL METRICS")
            lines.append("-" * 80)

            for k in sorted(result1.retrieval.recall_at_k.keys()):
                lines.append(
                    f"Recall@{k:2d}:  "
                    f"{names[0]:>15s}: {result1.retrieval.recall_at_k[k]:.4f}  "
                    f"{names[1]:>15s}: {result2.retrieval.recall_at_k[k]:.4f}"
                )

            lines.append(
                f"MRR:       "
                f"{names[0]:>15s}: {result1.retrieval.mrr:.4f}  "
                f"{names[1]:>15s}: {result2.retrieval.mrr:.4f}"
            )

            lines.append("")

        # Compare generation metrics
        if result1.generation and result2.generation:
            lines.append("GENERATION METRICS")
            lines.append("-" * 80)

            lines.append(
                f"BLEU-4:    "
                f"{names[0]:>15s}: {result1.generation.bleu_4:.4f}  "
                f"{names[1]:>15s}: {result2.generation.bleu_4:.4f}"
            )

            lines.append(
                f"ROUGE-L:   "
                f"{names[0]:>15s}: {result1.generation.rouge_l:.4f}  "
                f"{names[1]:>15s}: {result2.generation.rouge_l:.4f}"
            )

            lines.append("")

        # Compare RAG metrics
        if result1.rag and result2.rag:
            lines.append("RAG METRICS")
            lines.append("-" * 80)

            lines.append(
                f"Faithfulness:      "
                f"{names[0]:>15s}: {result1.rag.faithfulness:.4f}  "
                f"{names[1]:>15s}: {result2.rag.faithfulness:.4f}"
            )

            lines.append(
                f"Answer Relevancy:  "
                f"{names[0]:>15s}: {result1.rag.answer_relevancy:.4f}  "
                f"{names[1]:>15s}: {result2.rag.answer_relevancy:.4f}"
            )

            lines.append(
                f"RAGAS Score:       "
                f"{names[0]:>15s}: {result1.rag.ragas_score:.4f}  "
                f"{names[1]:>15s}: {result2.rag.ragas_score:.4f}"
            )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)
