"""
Generation Metrics - Evaluate quality of generated text.

Implements standard NLG metrics:
- BLEU: N-gram overlap with reference
- ROUGE: Recall-oriented text overlap
- BERTScore: Semantic similarity using embeddings
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import re
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from generation evaluation."""
    bleu_1: float
    bleu_2: float
    bleu_4: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bertscore_f1: float
    num_samples: int


class GenerationMetrics:
    """
    Evaluate generation quality using NLG metrics.

    Metrics implemented:
    - BLEU: Precision-based n-gram overlap
    - ROUGE: Recall-based text overlap
    - BERTScore: Contextual embedding similarity (if available)
    """

    def __init__(self, use_bertscore: bool = False):
        """
        Initialize generation metrics.

        Args:
            use_bertscore: Whether to compute BERTScore (requires transformers)
        """
        self.use_bertscore = use_bertscore

        # Try to import BERTScore if requested
        if self.use_bertscore:
            try:
                from bert_score import score as bert_score
                self.bert_score = bert_score
                logger.info("BERTScore available")
            except ImportError:
                logger.warning(
                    "BERTScore requested but not available. "
                    "Install with: pip install bert-score"
                )
                self.use_bertscore = False

        logger.info(f"GenerationMetrics initialized (BERTScore={self.use_bertscore})")

    def evaluate(
        self,
        predictions: List[str],
        references: List[str]
    ) -> GenerationResult:
        """
        Evaluate generation quality across multiple examples.

        Args:
            predictions: List of generated texts
            references: List of reference texts

        Returns:
            GenerationResult with all computed metrics

        Raises:
            ValueError: If input lists have different lengths
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Mismatch: {len(predictions)} predictions, "
                f"{len(references)} references"
            )

        num_samples = len(predictions)

        # Calculate BLEU scores
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_4_scores = []

        for pred, ref in zip(predictions, references):
            bleu_1_scores.append(self._bleu_score(pred, ref, n=1))
            bleu_2_scores.append(self._bleu_score(pred, ref, n=2))
            bleu_4_scores.append(self._bleu_score(pred, ref, n=4))

        # Calculate ROUGE scores
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []

        for pred, ref in zip(predictions, references):
            rouge_1_scores.append(self._rouge_n(pred, ref, n=1))
            rouge_2_scores.append(self._rouge_n(pred, ref, n=2))
            rouge_l_scores.append(self._rouge_l(pred, ref))

        # Calculate BERTScore if available
        bertscore_f1 = 0.0
        if self.use_bertscore:
            try:
                P, R, F1 = self.bert_score(
                    predictions,
                    references,
                    lang="en",
                    verbose=False
                )
                bertscore_f1 = float(F1.mean())
            except Exception as e:
                logger.warning(f"BERTScore calculation failed: {e}")

        result = GenerationResult(
            bleu_1=np.mean(bleu_1_scores),
            bleu_2=np.mean(bleu_2_scores),
            bleu_4=np.mean(bleu_4_scores),
            rouge_1=np.mean(rouge_1_scores),
            rouge_2=np.mean(rouge_2_scores),
            rouge_l=np.mean(rouge_l_scores),
            bertscore_f1=bertscore_f1,
            num_samples=num_samples
        )

        logger.info(
            f"Generation evaluation complete: "
            f"{num_samples} samples, "
            f"BLEU-4={result.bleu_4:.3f}, "
            f"ROUGE-L={result.rouge_l:.3f}"
        )

        return result

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and split on whitespace/punctuation
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """
        Extract n-grams from tokens.

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            List of n-gram tuples
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams

    def _bleu_score(
        self,
        prediction: str,
        reference: str,
        n: int = 4
    ) -> float:
        """
        Calculate BLEU score.

        Simple implementation of BLEU without brevity penalty.

        Args:
            prediction: Generated text
            reference: Reference text
            n: Maximum n-gram size

        Returns:
            BLEU score (0-1)
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Calculate precision for each n-gram level
        precisions = []

        for i in range(1, n + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, i)
            ref_ngrams = self._get_ngrams(ref_tokens, i)

            if not pred_ngrams:
                precisions.append(0.0)
                continue

            # Count matches
            pred_counter = Counter(pred_ngrams)
            ref_counter = Counter(ref_ngrams)

            matches = sum(
                min(pred_counter[ng], ref_counter[ng])
                for ng in pred_counter
            )

            precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
            precisions.append(precision)

        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            return 0.0

        # Simple average for smoothing
        bleu = np.exp(np.mean([np.log(p) if p > 0 else -np.inf for p in precisions]))

        return min(bleu, 1.0)

    def _rouge_n(
        self,
        prediction: str,
        reference: str,
        n: int = 1
    ) -> float:
        """
        Calculate ROUGE-N score (recall-based).

        Args:
            prediction: Generated text
            reference: Reference text
            n: N-gram size

        Returns:
            ROUGE-N F1 score (0-1)
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)

        if not ref_ngrams:
            return 0.0

        # Count matches
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)

        matches = sum(
            min(pred_counter[ng], ref_counter[ng])
            for ng in ref_counter
        )

        # Calculate recall and precision
        recall = matches / len(ref_ngrams) if ref_ngrams else 0.0
        precision = matches / len(pred_ngrams) if pred_ngrams else 0.0

        # F1 score
        if recall + precision == 0:
            return 0.0

        f1 = 2 * (recall * precision) / (recall + precision)

        return f1

    def _rouge_l(
        self,
        prediction: str,
        reference: str
    ) -> float:
        """
        Calculate ROUGE-L score (longest common subsequence).

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            ROUGE-L F1 score (0-1)
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Find LCS length
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)

        # Calculate recall and precision
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0

        # F1 score
        if recall + precision == 0:
            return 0.0

        f1 = 2 * (recall * precision) / (recall + precision)

        return f1

    def _lcs_length(
        self,
        seq1: List[str],
        seq2: List[str]
    ) -> int:
        """
        Calculate longest common subsequence length.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Length of LCS
        """
        m, n = len(seq1), len(seq2)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def evaluate_single(
        self,
        prediction: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction.

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "bleu_1": self._bleu_score(prediction, reference, n=1),
            "bleu_2": self._bleu_score(prediction, reference, n=2),
            "bleu_4": self._bleu_score(prediction, reference, n=4),
            "rouge_1": self._rouge_n(prediction, reference, n=1),
            "rouge_2": self._rouge_n(prediction, reference, n=2),
            "rouge_l": self._rouge_l(prediction, reference)
        }

        if self.use_bertscore:
            try:
                P, R, F1 = self.bert_score(
                    [prediction],
                    [reference],
                    lang="en",
                    verbose=False
                )
                metrics["bertscore_f1"] = float(F1[0])
            except Exception as e:
                logger.warning(f"BERTScore failed: {e}")
                metrics["bertscore_f1"] = 0.0

        return metrics

    def get_summary(self, result: GenerationResult) -> str:
        """
        Get human-readable summary of results.

        Args:
            result: GenerationResult to summarize

        Returns:
            Formatted summary string
        """
        lines = [
            f"Generation Evaluation Summary ({result.num_samples} samples)",
            "=" * 60,
            "",
            "BLEU Scores:",
            f"  BLEU-1: {result.bleu_1:.4f}",
            f"  BLEU-2: {result.bleu_2:.4f}",
            f"  BLEU-4: {result.bleu_4:.4f}",
            "",
            "ROUGE Scores:",
            f"  ROUGE-1: {result.rouge_1:.4f}",
            f"  ROUGE-2: {result.rouge_2:.4f}",
            f"  ROUGE-L: {result.rouge_l:.4f}",
        ]

        if self.use_bertscore and result.bertscore_f1 > 0:
            lines.append("")
            lines.append("BERTScore:")
            lines.append(f"  F1: {result.bertscore_f1:.4f}")

        return "\n".join(lines)
