"""Tests for evaluation metrics and harness."""

from __future__ import annotations

import asyncio
from collections import Counter

import numpy as np
import pytest

from ragnaros.evaluation.metrics import (
    accuracy_per_dollar,
    estimate_tokens,
    exact_match_accuracy,
    k_distribution,
    k_statistics,
    total_cost_usd,
)
from ragnaros.evaluation.harness import EvaluationHarness, RunConfig


# ---------------------------------------------------------------------------
# Metrics: estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") >= 1

    def test_scales_with_length(self):
        short = estimate_tokens("hello")
        long = estimate_tokens("hello world foo bar baz qux quux")
        assert long > short

    def test_consistent(self):
        assert estimate_tokens("same text") == estimate_tokens("same text")


# ---------------------------------------------------------------------------
# Metrics: total_cost_usd
# ---------------------------------------------------------------------------


class TestTotalCostUsd:
    def test_zero_inputs(self):
        cost = total_cost_usd([], [])
        assert cost == 0.0

    def test_positive_cost(self):
        cost = total_cost_usd(
            ["This is a context document with some words."],
            ["Paris"],
        )
        assert cost > 0

    def test_more_text_higher_cost(self):
        short = total_cost_usd(["short"], ["a"])
        long = total_cost_usd(["a " * 500], ["long answer with more words"])
        assert long > short


# ---------------------------------------------------------------------------
# Metrics: exact_match_accuracy
# ---------------------------------------------------------------------------


class TestExactMatchAccuracy:
    def test_all_correct(self):
        preds = ["Paris", "France", "River Seine"]
        gts = ["Paris", "France", "River Seine"]
        assert exact_match_accuracy(preds, gts) == 1.0

    def test_all_wrong(self):
        assert exact_match_accuracy(["Berlin"], ["Paris"]) == 0.0

    def test_partial(self):
        preds = ["Paris", "Berlin"]
        gts = ["Paris", "Paris"]
        assert exact_match_accuracy(preds, gts) == 0.5

    def test_case_insensitive(self):
        assert exact_match_accuracy(["paris"], ["Paris"]) == 1.0

    def test_empty(self):
        assert exact_match_accuracy([], []) == 0.0


# ---------------------------------------------------------------------------
# Metrics: k_statistics
# ---------------------------------------------------------------------------


class TestKStatistics:
    def test_empty(self):
        stats = k_statistics([])
        assert stats["mean"] == 0.0

    def test_single_value(self):
        stats = k_statistics([5])
        assert stats["mean"] == 5.0
        assert stats["min"] == 5
        assert stats["max"] == 5

    def test_distribution(self):
        ks = [1, 2, 3, 4, 5]
        stats = k_statistics(ks)
        assert stats["mean"] == 3.0
        assert stats["min"] == 1
        assert stats["max"] == 5


# ---------------------------------------------------------------------------
# Metrics: k_distribution
# ---------------------------------------------------------------------------


class TestKDistribution:
    def test_counter_returned(self):
        result = k_distribution([1, 2, 2, 3, 3, 3])
        assert isinstance(result, Counter)
        assert result[1] == 1
        assert result[2] == 2
        assert result[3] == 3

    def test_empty(self):
        assert k_distribution([]) == Counter()


# ---------------------------------------------------------------------------
# Metrics: accuracy_per_dollar
# ---------------------------------------------------------------------------


class TestAccuracyPerDollar:
    def test_basic(self):
        assert accuracy_per_dollar(0.5, 0.1) == pytest.approx(5.0)

    def test_zero_cost(self):
        assert accuracy_per_dollar(0.9, 0.0) == 0.0

    def test_zero_accuracy(self):
        assert accuracy_per_dollar(0.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# EvaluationHarness (unit test with mocks — no real LLM)
# ---------------------------------------------------------------------------


class TestEvaluationHarness:
    """Test the harness with a deterministic fake answer function."""

    @pytest.fixture
    def harness(self, fake_vectorstore, fake_embeddings_model, null_distribution):
        questions = [
            "What is the capital of France?",
            "Where is the Eiffel Tower?",
            "Who was Napoleon Bonaparte?",
        ]
        ground_truths = ["Paris", "Paris", "French military leader"]

        async def answer_fn(question, docs):
            # Simple deterministic answer: always return "Paris"
            return "Paris"

        config = RunConfig(
            n_questions=3,
            seed=42,
            max_concurrent=2,
        )
        return EvaluationHarness(
            questions=questions,
            ground_truths=ground_truths,
            answer_fn=answer_fn,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
            vectorstore=fake_vectorstore,
            config=config,
        )

    def test_run_fixed_k_returns_results(self, harness):
        results = harness.run(fixed_k_values=[3])
        assert len(results) == 1
        r = results[0]
        assert r.method_name == "fixed_k=3"
        assert 0.0 <= r.accuracy <= 1.0
        assert r.total_cost_usd > 0

    def test_run_dynamic_returns_results(self, harness):
        results = harness.run(estimator_names=["higher_criticism"])
        assert len(results) == 1
        r = results[0]
        assert r.method_name == "higher_criticism"
        assert r.mean_k >= 1

    def test_run_mixed(self, harness):
        results = harness.run(
            fixed_k_values=[1, 5],
            estimator_names=["higher_criticism"],
        )
        assert len(results) == 3
        names = {r.method_name for r in results}
        assert "fixed_k=1" in names
        assert "fixed_k=5" in names
        assert "higher_criticism" in names

    def test_k_dist_populated(self, harness):
        results = harness.run(estimator_names=["bonferroni"])
        r = results[0]
        assert sum(r.k_dist.values()) == len(r.question_results)

    def test_mismatched_lengths_raises(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        async def noop(q, d):
            return ""

        with pytest.raises(ValueError, match="same length"):
            EvaluationHarness(
                questions=["q1", "q2"],
                ground_truths=["a1"],
                answer_fn=noop,
                embeddings=fake_embeddings_model,
                null_distribution=null_distribution,
                vectorstore=fake_vectorstore,
            )

    def test_method_result_summary(self, harness):
        results = harness.run(fixed_k_values=[3])
        summary = results[0].summary()
        assert "fixed_k=3" in summary
        assert "accuracy=" in summary
