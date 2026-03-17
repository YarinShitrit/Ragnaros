"""
Pure metric functions for evaluating RAG retrieval quality.

All functions are stateless — pass in result data and receive a scalar or
structured value back. No LangChain or numpy imports are required to call
these (numpy is used internally but always available as a core dependency).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Token / cost estimation
# ---------------------------------------------------------------------------

# GPT-4o-mini pricing used in the research (USD per 1M tokens).
_DEFAULT_INPUT_PRICE_PER_M = 1.10
_DEFAULT_OUTPUT_PRICE_PER_M = 4.40
_WORDS_TO_TOKENS = 1.3  # rough approximation: 1 word ≈ 1.3 tokens


def estimate_tokens(text: str) -> int:
    """Rough token count: ``len(text.split()) * 1.3``, rounded up."""
    return int(len(text.split()) * _WORDS_TO_TOKENS) + 1


def total_cost_usd(
    input_texts: list[str],
    output_texts: list[str],
    input_price_per_million: float = _DEFAULT_INPUT_PRICE_PER_M,
    output_price_per_million: float = _DEFAULT_OUTPUT_PRICE_PER_M,
) -> float:
    """Estimate total API cost in USD using a word-count proxy for tokens.

    Args:
        input_texts: All text sent as input to the LLM (documents + prompts).
        output_texts: All text received as output.
        input_price_per_million: USD per 1M input tokens. Default 1.10.
        output_price_per_million: USD per 1M output tokens. Default 4.40.

    Returns:
        Estimated cost in USD.
    """
    in_tokens = sum(estimate_tokens(t) for t in input_texts)
    out_tokens = sum(estimate_tokens(t) for t in output_texts)
    return (
        in_tokens * input_price_per_million / 1_000_000
        + out_tokens * output_price_per_million / 1_000_000
    )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def semantic_accuracy(
    predictions: list[str],
    ground_truths: list[str],
    embeddings_model: Any,
    threshold: float = 0.85,
) -> float:
    """Compute semantic accuracy using cosine similarity between embeddings.

    A prediction is considered correct if the cosine similarity between its
    embedding and the ground-truth embedding exceeds ``threshold``.

    Args:
        predictions: List of predicted answer strings.
        ground_truths: List of ground-truth answer strings.
        embeddings_model: Any LangChain ``Embeddings`` instance.
        threshold: Similarity threshold for correctness. Default 0.85.

    Returns:
        Accuracy as a float in ``[0, 1]``.

    Raises:
        ValueError: If ``predictions`` and ``ground_truths`` have different lengths.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truths "
            f"({len(ground_truths)}) must have the same length."
        )
    if not predictions:
        return 0.0

    all_texts = predictions + ground_truths
    all_embs = embeddings_model.embed_documents(all_texts)
    n = len(predictions)
    pred_embs = all_embs[:n]
    gt_embs = all_embs[n:]

    correct = sum(
        1
        for p, g in zip(pred_embs, gt_embs)
        if cosine_similarity([p], [g])[0][0] >= threshold
    )
    return correct / n


def exact_match_accuracy(
    predictions: list[str],
    ground_truths: list[str],
) -> float:
    """Fraction of predictions that exactly match (case-insensitive) the ground truth.

    Args:
        predictions: Predicted answers.
        ground_truths: Ground-truth answers.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    if not predictions:
        return 0.0
    return sum(
        p.strip().lower() == g.strip().lower()
        for p, g in zip(predictions, ground_truths)
    ) / len(predictions)


# ---------------------------------------------------------------------------
# k statistics
# ---------------------------------------------------------------------------


def k_statistics(ks: list[int]) -> dict[str, float]:
    """Summarise a list of per-query k values.

    Args:
        ks: List of k values used, one per query.

    Returns:
        Dict with keys: ``mean``, ``median``, ``std``, ``min``, ``max``.
    """
    if not ks:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0, "max": 0}
    arr = np.asarray(ks, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def k_distribution(ks: list[int]) -> Counter[int]:
    """Count how often each k value was used.

    Args:
        ks: List of per-query k values.

    Returns:
        :class:`collections.Counter` mapping ``k`` → count.
    """
    return Counter(ks)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def accuracy_per_dollar(accuracy: float, cost_usd: float) -> float:
    """Accuracy divided by cost — a compound efficiency metric.

    Args:
        accuracy: Accuracy in ``[0, 1]``.
        cost_usd: Total cost in USD.

    Returns:
        Accuracy per dollar, or 0.0 if cost is 0.
    """
    if cost_usd <= 0:
        return 0.0
    return accuracy / cost_usd

