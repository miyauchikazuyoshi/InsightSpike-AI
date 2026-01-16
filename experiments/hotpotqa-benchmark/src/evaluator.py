"""HotpotQA evaluation metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Compute F1 score between prediction and ground truth.

    Returns:
        (f1, precision, recall)
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return (
            float(prediction_tokens == ground_truth_tokens),
            float(prediction_tokens == ground_truth_tokens),
            float(prediction_tokens == ground_truth_tokens),
        )

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def supporting_facts_em(
    predicted_facts: list[tuple[str, int]], gold_facts: list[tuple[str, int]]
) -> float:
    """Compute exact match for supporting facts."""
    return float(set(predicted_facts) == set(gold_facts))


def supporting_facts_f1(
    predicted_facts: list[tuple[str, int]], gold_facts: list[tuple[str, int]]
) -> tuple[float, float, float]:
    """Compute F1 score for supporting facts.

    Returns:
        (f1, precision, recall)
    """
    predicted_set = set(predicted_facts)
    gold_set = set(gold_facts)

    if not predicted_set or not gold_set:
        return (
            float(predicted_set == gold_set),
            float(predicted_set == gold_set),
            float(predicted_set == gold_set),
        )

    common = predicted_set & gold_set

    if not common:
        return 0.0, 0.0, 0.0

    precision = len(common) / len(predicted_set)
    recall = len(common) / len(gold_set)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


@dataclass
class EvaluationResult:
    """Result of evaluating a single prediction."""

    example_id: str
    em: float
    f1: float
    precision: float
    recall: float
    sf_em: float = 0.0
    sf_f1: float = 0.0
    sf_precision: float = 0.0
    sf_recall: float = 0.0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AggregatedResults:
    """Aggregated evaluation results over multiple examples."""

    count: int
    em: float
    f1: float
    precision: float
    recall: float
    sf_em: float
    sf_f1: float
    sf_precision: float
    sf_recall: float
    latency_p50_ms: float
    latency_p95_ms: float

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "em": round(self.em, 4),
            "f1": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "sf_em": round(self.sf_em, 4),
            "sf_f1": round(self.sf_f1, 4),
            "sf_precision": round(self.sf_precision, 4),
            "sf_recall": round(self.sf_recall, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
        }


class HotpotQAEvaluator:
    """Evaluate predictions on HotpotQA."""

    def __init__(self):
        self.results: list[EvaluationResult] = []

    def evaluate_single(
        self,
        example_id: str,
        prediction: str,
        ground_truth: str,
        predicted_facts: list[tuple[str, int]] | None = None,
        gold_facts: list[tuple[str, int]] | None = None,
        latency_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> EvaluationResult:
        """Evaluate a single prediction."""
        em = exact_match(prediction, ground_truth)
        f1, prec, rec = f1_score(prediction, ground_truth)

        sf_em, sf_f1, sf_prec, sf_rec = 0.0, 0.0, 0.0, 0.0
        if predicted_facts is not None and gold_facts is not None:
            sf_em = supporting_facts_em(predicted_facts, gold_facts)
            sf_f1, sf_prec, sf_rec = supporting_facts_f1(predicted_facts, gold_facts)

        result = EvaluationResult(
            example_id=example_id,
            em=em,
            f1=f1,
            precision=prec,
            recall=rec,
            sf_em=sf_em,
            sf_f1=sf_f1,
            sf_precision=sf_prec,
            sf_recall=sf_rec,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        self.results.append(result)
        return result

    def aggregate(self) -> AggregatedResults:
        """Aggregate all results."""
        if not self.results:
            return AggregatedResults(
                count=0,
                em=0.0,
                f1=0.0,
                precision=0.0,
                recall=0.0,
                sf_em=0.0,
                sf_f1=0.0,
                sf_precision=0.0,
                sf_recall=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
            )

        n = len(self.results)
        latencies = sorted([r.latency_ms for r in self.results])
        p50_idx = int(n * 0.5)
        p95_idx = min(int(n * 0.95), n - 1)

        return AggregatedResults(
            count=n,
            em=sum(r.em for r in self.results) / n,
            f1=sum(r.f1 for r in self.results) / n,
            precision=sum(r.precision for r in self.results) / n,
            recall=sum(r.recall for r in self.results) / n,
            sf_em=sum(r.sf_em for r in self.results) / n,
            sf_f1=sum(r.sf_f1 for r in self.results) / n,
            sf_precision=sum(r.sf_precision for r in self.results) / n,
            sf_recall=sum(r.sf_recall for r in self.results) / n,
            latency_p50_ms=latencies[p50_idx],
            latency_p95_ms=latencies[p95_idx],
        )

    def reset(self) -> None:
        """Reset all results."""
        self.results = []
