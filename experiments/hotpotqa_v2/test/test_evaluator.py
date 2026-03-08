"""Unit tests for HotpotQA v2 evaluator — regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure experiment root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.hotpotqa_v2.src.evaluator import (
    normalize_answer,
    exact_match,
    f1_score,
    supporting_facts_f1,
    HotpotQAEvaluator,
)


class TestNormalizeAnswer:
    def test_lower_case(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the quick brown fox") == "quick brown fox"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Cat's Meow!") == "cats meow"


class TestExactMatch:
    def test_exact(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert exact_match("paris", "Paris") == 1.0

    def test_mismatch(self):
        assert exact_match("London", "Paris") == 0.0

    def test_article_ignored(self):
        assert exact_match("The Beatles", "Beatles") == 1.0


class TestF1Score:
    def test_perfect(self):
        f1, p, r = f1_score("the quick brown fox", "quick brown fox")
        assert f1 == 1.0
        assert p == 1.0
        assert r == 1.0

    def test_partial_overlap(self):
        f1, p, r = f1_score("quick brown fox jumps", "quick brown fox sits")
        assert 0.0 < f1 < 1.0
        assert p > 0.0
        assert r > 0.0

    def test_no_overlap(self):
        f1, p, r = f1_score("hello world", "foo bar")
        assert f1 == 0.0

    def test_empty_prediction(self):
        f1, _, _ = f1_score("", "answer")
        assert f1 == 0.0


class TestSupportingFactsF1:
    def test_perfect(self):
        pred = [("Doc1", 0), ("Doc2", 1)]
        gold = [("Doc1", 0), ("Doc2", 1)]
        f1, p, r = supporting_facts_f1(pred, gold)
        assert f1 == 1.0

    def test_partial(self):
        pred = [("Doc1", 0), ("Doc3", 2)]
        gold = [("Doc1", 0), ("Doc2", 1)]
        f1, p, r = supporting_facts_f1(pred, gold)
        assert 0.0 < f1 < 1.0

    def test_no_overlap(self):
        pred = [("Doc3", 2)]
        gold = [("Doc1", 0)]
        f1, _, _ = supporting_facts_f1(pred, gold)
        assert f1 == 0.0


class TestEvaluatorAggregation:
    def test_aggregate_by_type(self):
        evaluator = HotpotQAEvaluator()
        evaluator.evaluate_single("q1", "Paris", "Paris", question_type="bridge")
        evaluator.evaluate_single("q2", "London", "Paris", question_type="bridge")
        evaluator.evaluate_single("q3", "yes", "yes", question_type="comparison")

        agg = evaluator.aggregate_by_type()
        assert "all" in agg
        assert "bridge" in agg
        assert "comparison" in agg
        assert agg["all"].count == 3
        assert agg["bridge"].count == 2
        assert agg["comparison"].count == 1
        assert agg["comparison"].em == 1.0  # "yes" == "yes"

    def test_aggregate_empty(self):
        evaluator = HotpotQAEvaluator()
        agg = evaluator.aggregate()
        assert agg.count == 0
        assert agg.em == 0.0

    def test_reset(self):
        evaluator = HotpotQAEvaluator()
        evaluator.evaluate_single("q1", "a", "a")
        assert len(evaluator.results) == 1
        evaluator.reset()
        assert len(evaluator.results) == 0
