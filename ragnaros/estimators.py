"""
Core k-estimation algorithms.

All three functions share the same signature:

    fn(question_emb, doc_embs, null_distribution, alpha, max_k) -> int

They expect ``doc_embs`` to be ordered by descending similarity to the query
(most-similar first). The returned ``k`` is always in ``[1, max_k]``.

The statistical logic:
    1. Compute cosine similarity between the query and each candidate doc.
    2. Convert each similarity to an empirical p-value:
       ``p = mean(null_distribution >= sim)``
       i.e. the fraction of null scores at least as large as the observed score.
    3. Apply a statistical test to decide how many docs are "significantly similar".
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ragnaros._types import Embedding, EstimatorFn, EstimatorName, NullDist


def higher_criticism(
    question_emb: Embedding,
    doc_embs: list[Embedding],
    null_distribution: NullDist,
    alpha: float = 0.05,
    max_k: int = 10,
) -> int:
    """Select k using the Higher Criticism statistic (Donoho & Jin, 2004).

    Designed to detect sparse, weak signals by finding the index where the
    gap between observed and expected p-values is maximised. Performs best
    when only a few documents are genuinely relevant.

    Args:
        question_emb: Query embedding, shape ``(D,)``.
        doc_embs: Candidate document embeddings ordered by descending similarity,
            each of shape ``(D,)``. Only the top ``max_k`` are examined.
        null_distribution: 1-D array of background cosine similarity scores.
        alpha: Significance level (unused in HC; kept for API consistency).
        max_k: Maximum number of documents to return.

    Returns:
        Estimated optimal k in ``[1, max_k]``.
    """
    similarities = cosine_similarity([question_emb], doc_embs)[0]
    sorted_similarities = np.sort(similarities)[::-1]
    n = min(max_k, len(sorted_similarities))

    p_values = np.array(
        [(null_distribution >= sim).mean() for sim in sorted_similarities[:n]]
    )
    p_values = np.clip(p_values, 1e-10, 1 - 1e-10)

    i = np.arange(1, n + 1)
    # Use i/(n+1) — the expected value of the i-th order statistic of n Uniform(0,1)
    # variates. Using i/n inflates the last entry to 1.0 and biases the statistic.
    expected = i / (n + 1)
    hc_stats = np.sqrt(n) * (expected - p_values) / np.sqrt(p_values * (1 - p_values))

    best_k = int(np.argmax(hc_stats)) + 1
    return min(max(best_k, 1), max_k)


def benjamini_hochberg(
    question_emb: Embedding,
    doc_embs: list[Embedding],
    null_distribution: NullDist,
    alpha: float = 0.05,
    max_k: int = 10,
) -> int:
    """Select k using the Benjamini-Hochberg step-up procedure.

    Controls the False Discovery Rate at level ``alpha``. Selects the largest
    set of documents where the BH threshold ``p_i <= (i/n) * alpha`` holds.

    Args:
        question_emb: Query embedding, shape ``(D,)``.
        doc_embs: Candidate document embeddings. Order does not matter; the
            function sorts by p-value internally.
        null_distribution: 1-D array of background cosine similarity scores.
        alpha: FDR level. Default 0.05.
        max_k: Maximum number of documents to return.

    Returns:
        Estimated optimal k in ``[1, max_k]``.

    Note:
        The BH correction factor depends on ``n = len(doc_embs)``. In
        *candidates* retrieval mode, ``n = max_candidates`` (typically 50),
        which is far less strict than *exact* mode where ``n`` equals the
        full corpus size. Results will differ between modes.
    """
    n_docs = len(doc_embs)
    similarities = cosine_similarity([question_emb], doc_embs)[0]

    p_values = np.array(
        [np.mean(null_distribution >= sim) for sim in similarities]
    )

    sorted_p_values = np.sort(p_values)

    k_selected = 0
    for i, p in enumerate(sorted_p_values, start=1):
        if p <= (i / n_docs) * alpha:
            k_selected = i

    return max(min(k_selected, max_k), 1)


def bonferroni_correction(
    question_emb: Embedding,
    doc_embs: list[Embedding],
    null_distribution: NullDist,
    alpha: float = 0.05,
    max_k: int = 10,
) -> int:
    """Select k using Bonferroni correction.

    Controls the Family-Wise Error Rate at level ``alpha``. The most
    conservative of the three estimators — corrects the threshold by dividing
    by the total number of candidate documents.

    Args:
        question_emb: Query embedding, shape ``(D,)``.
        doc_embs: Candidate document embeddings. The function sorts by
            similarity internally.
        null_distribution: 1-D array of background cosine similarity scores.
        alpha: FWER level. Default 0.05.
        max_k: Maximum number of documents to return.

    Returns:
        Estimated optimal k in ``[1, max_k]``.
    """
    similarities = cosine_similarity([question_emb], doc_embs)[0]
    sorted_similarities = np.sort(similarities)[::-1]

    alpha_prime = alpha / len(sorted_similarities)

    pseudo_pvals = np.array(
        [np.mean(null_distribution >= sim) for sim in sorted_similarities]
    )

    k = int(np.sum(pseudo_pvals <= alpha_prime))
    return min(max(k, 1), max_k)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ESTIMATORS: dict[EstimatorName, EstimatorFn] = {
    "higher_criticism": higher_criticism,
    "benjamini_hochberg": benjamini_hochberg,
    "bonferroni": bonferroni_correction,
}


def get_estimator(name: EstimatorName) -> EstimatorFn:
    """Look up an estimator function by name.

    Args:
        name: One of ``"higher_criticism"``, ``"benjamini_hochberg"``,
            ``"bonferroni"``.

    Returns:
        The corresponding estimator callable.

    Raises:
        ValueError: If ``name`` is not a known estimator.
    """
    if name not in ESTIMATORS:
        valid = ", ".join(f'"{k}"' for k in ESTIMATORS)
        raise ValueError(f"Unknown estimator {name!r}. Valid options: {valid}")
    return ESTIMATORS[name]
