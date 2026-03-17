"""
EvaluationHarness: runs fixed-k and dynamic-k evaluations in parallel and
returns structured :class:`MethodResult` objects for comparison.

Design
------
- The harness is decoupled from any specific LLM or retriever implementation
  via an ``answer_fn`` callable:

      answer_fn(question: str, docs: list[Document]) -> str

  This lets you plug in any LangChain chain, OpenAI call, or custom function.

- Accuracy is measured via semantic cosine similarity between the predicted
  answer embedding and the ground-truth answer embedding (threshold 0.85 by
  default — matches the original research).

- Evaluation is run asynchronously with a configurable concurrency limit.
  A synchronous ``run`` method is provided for convenience (uses
  ``asyncio.run`` internally).
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sklearn.metrics.pairwise import cosine_similarity

from ragnaros._types import EstimatorName
from ragnaros.evaluation.metrics import (
    accuracy_per_dollar,
    estimate_tokens,
    k_distribution,
    k_statistics,
    total_cost_usd,
)
from ragnaros.null_distribution import NullDistribution
from ragnaros.retriever import DynamicRetriever


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Configuration shared across all evaluation runs.

    Attributes:
        n_questions: Questions to evaluate per run. Default 100.
        seed: Random seed for question sampling. Default 42.
        alpha: Statistical significance level for dynamic estimators. Default 0.05.
        max_k: Upper bound on k. Default 10.
        max_candidates: Candidates fetched in candidates mode. Default 50.
        max_concurrent: Max simultaneous LLM + retriever calls. Default 8.
        accuracy_threshold: Cosine similarity threshold for a correct answer.
            Default 0.85.
        input_price_per_million: USD per 1M input tokens. Default 1.10.
        output_price_per_million: USD per 1M output tokens. Default 4.40.
    """

    n_questions: int = 100
    seed: int = 42
    alpha: float = 0.05
    max_k: int = 10
    max_candidates: int = 50
    max_concurrent: int = 8
    accuracy_threshold: float = 0.85
    input_price_per_million: float = 1.10
    output_price_per_million: float = 4.40


@dataclass
class QuestionResult:
    """Result for a single question.

    Attributes:
        question: The input question.
        ground_truth: The reference answer.
        prediction: The model's answer.
        k_used: Number of documents retrieved for this question.
        retrieved_docs: The documents that were retrieved.
        is_correct: Whether prediction is semantically correct.
        similarity_score: Cosine similarity between prediction and ground truth.
        input_tokens: Estimated input token count.
        output_tokens: Estimated output token count.
    """

    question: str
    ground_truth: str
    prediction: str
    k_used: int
    retrieved_docs: list[Document]
    is_correct: bool
    similarity_score: float
    input_tokens: int
    output_tokens: int


@dataclass
class MethodResult:
    """Aggregated results for one evaluation method (fixed-k or a dynamic estimator).

    Attributes:
        method_name: Human-readable name, e.g. ``"fixed_k=7"`` or
            ``"higher_criticism"``.
        accuracy: Fraction of correctly answered questions.
        mean_k: Average k used across all questions.
        median_k: Median k.
        total_cost_usd: Estimated total API cost.
        accuracy_per_dollar: ``accuracy / total_cost_usd``.
        k_dist: Counter of k value frequencies.
        question_results: Per-question detail.
    """

    method_name: str
    accuracy: float
    mean_k: float
    median_k: float
    total_cost_usd: float
    accuracy_per_dollar: float
    k_dist: Counter[int]
    question_results: list[QuestionResult]

    def summary(self) -> str:
        """One-line summary string."""
        return (
            f"{self.method_name}: "
            f"accuracy={self.accuracy:.1%}, "
            f"mean_k={self.mean_k:.1f}, "
            f"cost=${self.total_cost_usd:.4f}, "
            f"acc/dollar={self.accuracy_per_dollar:.2f}"
        )


# ---------------------------------------------------------------------------
# EvaluationHarness
# ---------------------------------------------------------------------------


AnswerFn = Callable[[str, list[Document]], Awaitable[str]]


class EvaluationHarness:
    """Runs comparative evaluations of fixed-k and dynamic-k retrieval strategies.

    Args:
        questions: List of question strings.
        ground_truths: Corresponding ground-truth answers (same length as
            ``questions``).
        answer_fn: Async callable ``(question, docs) -> answer_str``.
            This is where your LLM lives — pass any async function or wrap a
            LangChain chain::

                async def answer_fn(question: str, docs: list[Document]) -> str:
                    context = "\\n\\n".join(d.page_content for d in docs)
                    result = await chain.ainvoke({"question": question, "context": context})
                    return result["answer"]

        embeddings: Embeddings model for semantic accuracy evaluation.
            Should be the same model used at retrieval time.
        null_distribution: Pre-built :class:`~ragnaros.null_distribution.NullDistribution`.
        vectorstore: The vector store to retrieve from.
        config: :class:`RunConfig` with evaluation parameters.

    Example::

        harness = EvaluationHarness(
            questions=qa_dataset["question"],
            ground_truths=qa_dataset["answer"],
            answer_fn=my_llm_fn,
            embeddings=OpenAIEmbeddings(),
            null_distribution=null_dist,
            vectorstore=chroma_store,
        )
        results = harness.run(
            fixed_k_values=[1, 5, 10],
            estimator_names=["higher_criticism", "benjamini_hochberg"],
        )
        for r in results:
            print(r.summary())
    """

    def __init__(
        self,
        questions: list[str],
        ground_truths: list[str],
        answer_fn: AnswerFn,
        embeddings: Embeddings,
        null_distribution: NullDistribution,
        vectorstore: VectorStore,
        config: RunConfig | None = None,
    ) -> None:
        if len(questions) != len(ground_truths):
            raise ValueError("questions and ground_truths must have the same length.")
        self.questions = questions
        self.ground_truths = ground_truths
        self.answer_fn = answer_fn
        self.embeddings = embeddings
        self.null_distribution = null_distribution
        self.vectorstore = vectorstore
        self.config = config or RunConfig()

        # Sample fixed question indices once for all methods.
        rng = random.Random(self.config.seed)
        indices = list(range(len(questions)))
        rng.shuffle(indices)
        self._selected = indices[: self.config.n_questions]

    # ------------------------------------------------------------------
    # Public API — synchronous wrapper
    # ------------------------------------------------------------------

    def run(
        self,
        fixed_k_values: list[int] | None = None,
        estimator_names: list[EstimatorName] | None = None,
    ) -> list[MethodResult]:
        """Run all specified methods and return a list of :class:`MethodResult`.

        Args:
            fixed_k_values: List of k constants to evaluate (e.g. ``[1, 5, 10]``).
            estimator_names: Dynamic estimators to evaluate (e.g.
                ``["higher_criticism", "bonferroni"]``).

        Returns:
            One :class:`MethodResult` per method, in the order provided.
        """
        return asyncio.run(
            self.arun(
                fixed_k_values=fixed_k_values,
                estimator_names=estimator_names,
            )
        )

    # ------------------------------------------------------------------
    # Public API — async
    # ------------------------------------------------------------------

    async def arun(
        self,
        fixed_k_values: list[int] | None = None,
        estimator_names: list[EstimatorName] | None = None,
    ) -> list[MethodResult]:
        """Async version of :meth:`run`."""
        results: list[MethodResult] = []

        for k in fixed_k_values or []:
            results.append(await self._eval_fixed_k(k))

        for name in estimator_names or []:
            results.append(await self._eval_dynamic(name))

        return results

    # ------------------------------------------------------------------
    # Internal evaluation loops
    # ------------------------------------------------------------------

    async def _eval_fixed_k(self, k: int) -> MethodResult:
        """Evaluate a single fixed-k configuration."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        question_results: list[tuple[int, QuestionResult]] = []

        async def _run_one(idx: int, question: str, ground_truth: str) -> tuple[int, QuestionResult]:
            async with semaphore:
                docs = await self.vectorstore.asimilarity_search(question, k=k)
                prediction = await self.answer_fn(question, docs)
                qr = await self._score(question, ground_truth, prediction, k, docs)
                return idx, qr

        tasks = [
            _run_one(
                idx,
                self.questions[i],
                self.ground_truths[i],
            )
            for idx, i in enumerate(self._selected)
        ]
        for coro in asyncio.as_completed(tasks):
            idx, qr = await coro
            question_results.append((idx, qr))

        question_results.sort(key=lambda x: x[0])
        qrs = [qr for _, qr in question_results]
        return self._aggregate(f"fixed_k={k}", qrs)

    async def _eval_dynamic(self, estimator_name: EstimatorName) -> MethodResult:
        """Evaluate a dynamic estimator."""
        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=self.vectorstore,
            embeddings=self.embeddings,
            null_distribution=self.null_distribution,
            estimator=estimator_name,
            alpha=self.config.alpha,
            max_k=self.config.max_k,
            max_candidates=self.config.max_candidates,
        )

        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        question_results: list[tuple[int, QuestionResult]] = []

        async def _run_one(idx: int, question: str, ground_truth: str) -> tuple[int, QuestionResult]:
            async with semaphore:
                docs = await retriever.ainvoke(question)
                k_used = len(docs)
                prediction = await self.answer_fn(question, docs)
                qr = await self._score(question, ground_truth, prediction, k_used, docs)
                return idx, qr

        tasks = [
            _run_one(
                idx,
                self.questions[i],
                self.ground_truths[i],
            )
            for idx, i in enumerate(self._selected)
        ]
        for coro in asyncio.as_completed(tasks):
            idx, qr = await coro
            question_results.append((idx, qr))

        question_results.sort(key=lambda x: x[0])
        qrs = [qr for _, qr in question_results]
        return self._aggregate(estimator_name, qrs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _score(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
        k_used: int,
        docs: list[Document],
    ) -> QuestionResult:
        """Compute semantic accuracy for a single question."""
        context_text = "\n\n".join(d.page_content for d in docs)
        input_tokens = estimate_tokens(context_text) + estimate_tokens(question)
        output_tokens = estimate_tokens(prediction)

        pred_emb, gt_emb = await asyncio.gather(
            asyncio.get_running_loop().run_in_executor(
                None, self.embeddings.embed_query, prediction
            ),
            asyncio.get_running_loop().run_in_executor(
                None, self.embeddings.embed_query, ground_truth
            ),
        )
        sim = float(cosine_similarity([pred_emb], [gt_emb])[0][0])
        is_correct = sim >= self.config.accuracy_threshold

        return QuestionResult(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
            k_used=k_used,
            retrieved_docs=docs,
            is_correct=is_correct,
            similarity_score=sim,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _aggregate(self, method_name: str, qrs: list[QuestionResult]) -> MethodResult:
        """Aggregate per-question results into a :class:`MethodResult`."""
        if not qrs:
            return MethodResult(
                method_name=method_name,
                accuracy=0.0,
                mean_k=0.0,
                median_k=0.0,
                total_cost_usd=0.0,
                accuracy_per_dollar=0.0,
                k_dist=Counter(),
                question_results=[],
            )

        ks = [qr.k_used for qr in qrs]
        stats = k_statistics(ks)
        correct = sum(1 for qr in qrs if qr.is_correct)
        acc = correct / len(qrs)

        input_texts = []
        output_texts = []
        for qr in qrs:
            input_texts.extend(d.page_content for d in qr.retrieved_docs)
            input_texts.append(qr.question)
            output_texts.append(qr.prediction)

        cost = total_cost_usd(
            input_texts,
            output_texts,
            self.config.input_price_per_million,
            self.config.output_price_per_million,
        )

        return MethodResult(
            method_name=method_name,
            accuracy=acc,
            mean_k=stats["mean"],
            median_k=stats["median"],
            total_cost_usd=cost,
            accuracy_per_dollar=accuracy_per_dollar(acc, cost),
            k_dist=k_distribution(ks),
            question_results=qrs,
        )
