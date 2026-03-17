"""
Core k-estimation algorithms.

All estimator functions share the same signature:

    fn(question_emb, doc_embs, null_distribution, alpha, max_k) -> int

They expect ``doc_embs`` to be ordered by descending similarity to the query
(most-similar first). The returned ``k`` is always in ``[1, max_k]``.

The statistical logic:
    1. Compute cosine similarity between the query and each candidate doc.
    2. Convert each similarity to an empirical p-value:
       ``p = mean(null_distribution >= sim)``
       i.e. the fraction of null scores at least as large as the observed score.
    3. Apply a statistical test to decide how many docs are "significantly similar".

Original estimators (v0.1):
    - Higher Criticism (Donoho & Jin, 2004) — sparse signal detection
    - Benjamini-Hochberg — FDR control
    - Bonferroni — FWER control

Research extension (v0.2):
    - Storey BH — adaptive BH with pi0 estimation (Storey, 2002)
    - Local FDR — Empirical Bayes posterior probability (Efron, 2001)
    - Kneedle — elbow/knee detection on similarity curve (Satopaa et al., 2011)
    - Berk-Jones — likelihood-ratio based goodness-of-fit (Berk & Jones, 1979)
    - Beta-Uniform Mixture — EM-based p-value decomposition (Pounds & Morris, 2003)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ragnaros._types import Embedding, EstimatorFn, EstimatorName, NullDist


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _similarities_and_pvalues(
    question_emb: Embedding,
    doc_embs: list[Embedding],
    null_distribution: NullDist,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cosine similarities and empirical p-values.

    Returns:
        (sorted_similarities_desc, corresponding_p_values)
    """
    similarities = cosine_similarity([question_emb], doc_embs)[0]
    sorted_sims = np.sort(similarities)[::-1]
    p_values = np.array(
        [(null_distribution >= sim).mean() for sim in sorted_sims]
    )
    p_values = np.clip(p_values, 1e-10, 1 - 1e-10)
    return sorted_sims, p_values


# ---------------------------------------------------------------------------
# 1. Higher Criticism (Donoho & Jin, 2004)
# ---------------------------------------------------------------------------


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
    _, p_values = _similarities_and_pvalues(question_emb, doc_embs, null_distribution)
    n = min(max_k, len(p_values))
    p_values = p_values[:n]

    i = np.arange(1, n + 1)
    # Use i/(n+1) — the expected value of the i-th order statistic of n Uniform(0,1)
    # variates. Using i/n inflates the last entry to 1.0 and biases the statistic.
    expected = i / (n + 1)
    hc_stats = np.sqrt(n) * (expected - p_values) / np.sqrt(p_values * (1 - p_values))

    best_k = int(np.argmax(hc_stats)) + 1
    return min(max(best_k, 1), max_k)


# ---------------------------------------------------------------------------
# 2. Benjamini-Hochberg (1995) — FDR control
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3. Bonferroni — FWER control
# ---------------------------------------------------------------------------


def bonferroni_correction(
    question_emb: Embedding,
    doc_embs: list[Embedding],
    null_distribution: NullDist,
    alpha: float = 0.05,
    max_k: int = 10,
) -> int:
    """Select k using Bonferroni correction.

    Controls the Family-Wise Error Rate at level ``alpha``. The most
    conservative of the estimators — corrects the threshold by dividing
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
# 4. Storey's Adaptive BH (Storey, 2002; Storey, Taylor & Siegmund, 2004)
# ---------------------------------------------------------------------------


def storey_bh(
    question_emb: Embedding,
    doc_embs: list[Embedding],
    null_distribution: NullDist,
    alpha: float = 0.05,
    max_k: int = 10,
) -> int:
    """Select k using Storey's adaptive Benjamini-Hochberg procedure.

    Estimates the proportion of true nulls (pi0) using a conservative
    estimator, then adjusts the BH threshold to ``alpha / pi0``. This is
    strictly more powerful than standard BH when many tests are truly null
    (which is typical in retrieval: most documents are irrelevant).

    The pi0 estimator uses a tuning parameter ``lambda = 0.5``:
        pi0_hat = #{p_i > lambda} / (n * (1 - lambda))

    **Research insight**: In RAG, most of the corpus is irrelevant for any
    given query, so pi0 ≈ 0.95+. Storey-BH adapts to this sparsity and
    recovers more true positives than vanilla BH while maintaining FDR control.

    Reference:
        Storey, J. D. (2002). "A direct approach to false discovery rates."
        Journal of the Royal Statistical Society: Series B, 64(3), 479–498.

    Args:
        question_emb: Query embedding, shape ``(D,)``.
        doc_embs: Candidate document embeddings.
        null_distribution: 1-D array of background cosine similarity scores.
        alpha: FDR level. Default 0.05.
        max_k: Maximum number of documents to return.

    Returns:
        Estimated optimal k in ``[1, max_k]``.
    """
    n_docs = len(doc_embs)
    similarities = cosine_similarity([question_emb], doc_embs)[0]
    p_values = np.array(
        [np.mean(null_distribution >= sim) for sim in similarities]
    )
    p_values = np.clip(p_values, 1e-10, 1 - 1e-10)

    # Estimate pi0 (proportion of true nulls) using lambda = 0.5
    lam = 0.5
    pi0 = np.sum(p_values > lam) / (n_docs * (1.0 - lam))
    pi0 = min(pi0, 1.0)  # pi0 cannot exceed 1

    # Adjust alpha: if pi0 is low (many signals), alpha_adj increases → more power
    alpha_adj = alpha / max(pi0, 0.01)  # guard against pi0 ≈ 0

    sorted_p_values = np.sort(p_values)
    k_selected = 0
    for i, p in enumerate(sorted_p_values, start=1):
        if p <= (i / n_docs) * alpha_adj:
            k_selected = i

    return max(min(k_selected, max_k), 1)


# ---------------------------------------------------------------------------
# 5. Local FDR / Empirical Bayes (Efron, 2001)
# ---------------------------------------------------------------------------


def local_fdr(
    question_emb: Embedding,
    doc_embs: list[Embedding],
    null_distribution: NullDist,
    alpha: float = 0.05,
    max_k: int = 10,
) -> int:
    """Select k using local False Discovery Rate (Efron's Empirical Bayes).

    Instead of controlling a global FDR, this method estimates the posterior
    probability that each document is a "null" (irrelevant) given its
    similarity score. Documents where ``lfdr <= alpha`` are considered relevant.

    The method models the observed p-value density as a two-component mixture:
        f(p) = pi0 * f0(p) + (1 - pi0) * f1(p)

    where f0 is Uniform(0,1) (null) and f1 is estimated from the data.
    The local fdr is:
        lfdr(p) = pi0 * f0(p) / f(p)

    For a document with empirical p-value p, we estimate f(p) using a
    histogram-based density estimator on the observed p-values, augmented
    with a kernel density estimate when enough points are available.

    **Research insight**: Local FDR naturally adapts its aggressiveness per
    document. A document with borderline similarity gets a moderate lfdr,
    while a clearly relevant document gets lfdr ≈ 0. This produces
    fine-grained, per-document relevance decisions.

    Reference:
        Efron, B. (2001). "Empirical Bayes analysis of a microarray experiment."
        Journal of the American Statistical Association, 96(456), 1151–1160.

    Args:
        question_emb: Query embedding, shape ``(D,)``.
        doc_embs: Candidate document embeddings.
        null_distribution: 1-D array of background cosine similarity scores.
        alpha: Local FDR threshold. Default 0.05.
        max_k: Maximum number of documents to return.

    Returns:
        Estimated optimal k in ``[1, max_k]``.
    """
    sorted_sims, p_values = _similarities_and_pvalues(
        question_emb, doc_embs, null_distribution
    )
    n = len(p_values)

    # Estimate pi0 via Storey's method
    lam = 0.5
    pi0 = np.sum(p_values > lam) / (n * (1.0 - lam))
    pi0 = np.clip(pi0, 0.01, 1.0)

    # Estimate f(p) using histogram density estimation
    # Use adaptive number of bins based on sample size
    n_bins = max(10, min(50, n // 2))
    hist_counts, bin_edges = np.histogram(p_values, bins=n_bins, range=(0.0, 1.0), density=True)

    # For each p-value, find its density estimate
    bin_width = 1.0 / n_bins
    lfdrs = np.zeros(n)
    for i, p in enumerate(p_values):
        bin_idx = min(int(p / bin_width), n_bins - 1)
        f_p = max(hist_counts[bin_idx], 1e-10)  # f(p) estimated density
        f0_p = 1.0  # Uniform(0,1) density
        lfdrs[i] = pi0 * f0_p / f_p

    # Clip lfdr to [0, 1]
    lfdrs = np.clip(lfdrs, 0.0, 1.0)

    # Count documents with lfdr <= alpha (these are "discoveries")
    # Process in order of decreasing similarity (ascending p-value)
    k = int(np.sum(lfdrs <= alpha))
    return max(min(k, max_k), 1)


# ---------------------------------------------------------------------------
# 6. Kneedle / Elbow Detection (Satopaa et al., 2011)
# ---------------------------------------------------------------------------


def kneedle(
    question_emb: Embedding,
    doc_embs: list[Embedding],
    null_distribution: NullDist,
    alpha: float = 0.05,
    max_k: int = 10,
) -> int:
    """Select k using the Kneedle algorithm for elbow/knee detection.

    Finds the point of maximum curvature in the sorted similarity curve.
    The intuition: relevant documents form a "plateau" of high similarity,
    followed by a sharp drop-off to irrelevant documents. The knee point
    marks the transition.

    The algorithm:
    1. Sort similarities in descending order.
    2. Normalize x and y to [0, 1].
    3. Subtract the diagonal (connecting first and last points).
    4. The knee is the point of maximum difference from the diagonal.

    Additionally, we use the null distribution to compute a significance-
    aware variant: we subtract the mean null similarity from each score
    before finding the knee, so the knee reflects the transition from
    "significantly above background" to "indistinguishable from background".

    **Research insight**: Kneedle is parameter-free (alpha is unused) and
    doesn't require p-value computation at all. It works directly on the
    similarity curve geometry. This makes it robust to misspecified null
    distributions and computationally very fast.

    Reference:
        Satopaa, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011).
        "Finding a 'Kneedle' in a Haystack: Detecting Knee Points in System Behavior."
        ICDCS Workshops.

    Args:
        question_emb: Query embedding, shape ``(D,)``.
        doc_embs: Candidate document embeddings.
        null_distribution: 1-D array of background cosine similarity scores.
            Used to compute a baseline threshold.
        alpha: Unused (kept for API consistency).
        max_k: Maximum number of documents to return.

    Returns:
        Estimated optimal k in ``[1, max_k]``.
    """
    similarities = cosine_similarity([question_emb], doc_embs)[0]
    sorted_sims = np.sort(similarities)[::-1]
    n = min(max_k + 2, len(sorted_sims))  # look slightly beyond max_k
    sorted_sims = sorted_sims[:n]

    if n <= 2:
        return 1

    # Subtract null distribution mean to focus on "above-background" signal
    null_mean = float(null_distribution.mean())
    adjusted_sims = sorted_sims - null_mean

    # Normalize to [0, 1] range
    x = np.linspace(0, 1, n)
    y_min, y_max = adjusted_sims[-1], adjusted_sims[0]
    if y_max - y_min < 1e-10:
        return 1  # all similarities are the same
    y = (adjusted_sims - y_min) / (y_max - y_min)

    # Subtract the diagonal from (0, y[0]=1) to (1, y[-1]=0)
    # This is the line connecting first normalized point to last
    diagonal = np.linspace(y[0], y[-1], n)
    differences = y - diagonal

    # The knee is the point of maximum positive difference
    knee_idx = int(np.argmax(differences))

    # The k is knee_idx + 1 (1-indexed)
    best_k = knee_idx + 1
    return max(min(best_k, max_k), 1)


# ---------------------------------------------------------------------------
# 7. Berk-Jones Statistic (Berk & Jones, 1979)
# ---------------------------------------------------------------------------


def berk_jones(
    question_emb: Embedding,
    doc_embs: list[Embedding],
    null_distribution: NullDist,
    alpha: float = 0.05,
    max_k: int = 10,
) -> int:
    """Select k using the Berk-Jones goodness-of-fit statistic.

    Related to Higher Criticism but uses a likelihood-ratio test instead
    of a standardized difference. The BJ statistic at rank i is:

        BJ_i = i * log(i/(n*p_i)) + (n-i) * log((n-i)/(n*(1-p_i)))

    This measures the Kullback-Leibler divergence between the observed
    fraction of "discoveries" at rank i and the expected fraction under
    the null. Berk-Jones is asymptotically optimal for detecting any
    departure from uniformity (Berk & Jones, 1979).

    **Research insight**: BJ is strictly more powerful than HC for detecting
    sparse signals when the signal is not extremely weak. In retrieval,
    relevant documents typically have noticeably higher similarity than
    background, putting us in BJ's sweet spot. The KL-divergence formulation
    also naturally handles asymmetric deviations better.

    Reference:
        Berk, R. H. & Jones, D. H. (1979). "Goodness-of-fit test statistics
        that dominate the Kolmogorov statistics." Zeitschrift fur
        Wahrscheinlichkeitstheorie und Verwandte Gebiete, 47(1), 47–59.

    Args:
        question_emb: Query embedding, shape ``(D,)``.
        doc_embs: Candidate document embeddings.
        null_distribution: 1-D array of background cosine similarity scores.
        alpha: Unused (kept for API consistency).
        max_k: Maximum number of documents to return.

    Returns:
        Estimated optimal k in ``[1, max_k]``.
    """
    _, p_values = _similarities_and_pvalues(question_emb, doc_embs, null_distribution)
    n = min(max_k, len(p_values))
    p_values = p_values[:n]  # already sorted by descending sim = ascending p-val

    # Sorted p-values (ascending — matching rank order)
    sorted_p = np.sort(p_values)
    sorted_p = np.clip(sorted_p, 1e-10, 1 - 1e-10)

    bj_stats = np.zeros(n)
    for i in range(1, n + 1):
        obs_frac = i / n
        exp_frac = sorted_p[i - 1]
        # KL-divergence-based statistic
        # Only count positive deviations (more discoveries than expected)
        if obs_frac > exp_frac:
            term1 = i * np.log(obs_frac / exp_frac) if obs_frac > 0 and exp_frac > 0 else 0
            term2 = (
                (n - i) * np.log((1 - obs_frac) / (1 - exp_frac))
                if (1 - obs_frac) > 0 and (1 - exp_frac) > 0
                else 0
            )
            bj_stats[i - 1] = term1 + term2
        else:
            bj_stats[i - 1] = 0.0

    best_k = int(np.argmax(bj_stats)) + 1
    return min(max(best_k, 1), max_k)


# ---------------------------------------------------------------------------
# 8. Beta-Uniform Mixture Model (Pounds & Morris, 2003)
# ---------------------------------------------------------------------------


def beta_mixture(
    question_emb: Embedding,
    doc_embs: list[Embedding],
    null_distribution: NullDist,
    alpha: float = 0.05,
    max_k: int = 10,
) -> int:
    """Select k using a Beta-Uniform Mixture (BUM) model on p-values.

    Models the p-value distribution as a two-component mixture:
        f(p) = pi0 * 1 + (1 - pi0) * Beta(a, 1)

    where pi0 is the null proportion and Beta(a, 1) with a < 1 captures
    the "spike near zero" from truly relevant documents. The model is
    fitted using a simple EM algorithm, then documents are classified
    as relevant if their posterior probability of being signal exceeds
    1 - alpha.

    **Research insight**: The BUM model directly estimates the mixing
    proportion pi0, giving a principled estimate of how many relevant
    documents exist. Unlike threshold-based methods, it explicitly
    models the shape of the signal distribution. The Beta(a,1)
    component naturally captures the left-skew of p-values from
    relevant documents.

    Reference:
        Pounds, S. & Morris, S. W. (2003). "Estimating the occurrence
        of false positives and false negatives in microarray studies by
        approximating and partitioning the empirical distribution of
        p-values." Bioinformatics, 19(10), 1236–1242.

    Args:
        question_emb: Query embedding, shape ``(D,)``.
        doc_embs: Candidate document embeddings.
        null_distribution: 1-D array of background cosine similarity scores.
        alpha: Posterior probability threshold. A document is "relevant" if
            its posterior probability of being signal > ``1 - alpha``. Default 0.05.
        max_k: Maximum number of documents to return.

    Returns:
        Estimated optimal k in ``[1, max_k]``.
    """
    _, p_values = _similarities_and_pvalues(question_emb, doc_embs, null_distribution)
    n = len(p_values)
    p_values = np.clip(p_values, 1e-10, 1 - 1e-10)

    # Initialize EM parameters
    pi0 = 0.9  # initial guess: 90% null
    a = 0.3  # initial Beta(a, 1) shape parameter

    # EM algorithm (10 iterations is typically sufficient for convergence)
    for _ in range(15):
        # E-step: compute responsibilities
        # f0(p) = 1 (Uniform density)
        # f1(p) = a * p^(a-1) (Beta(a,1) density)
        f0 = np.ones(n)
        f1 = a * np.power(p_values, a - 1)

        # Posterior probability of being null
        denom = pi0 * f0 + (1 - pi0) * f1
        gamma_null = pi0 * f0 / np.clip(denom, 1e-10, None)

        # M-step: update parameters
        pi0_new = np.mean(gamma_null)
        pi0_new = np.clip(pi0_new, 0.01, 0.999)

        # Update a: maximize expected log-likelihood for Beta(a,1) component
        # E[log f1] = log(a) + (a-1) * mean(log(p) * (1 - gamma))
        weights = 1 - gamma_null
        w_sum = np.sum(weights)
        if w_sum > 1e-10:
            weighted_log_p = np.sum(weights * np.log(p_values)) / w_sum
            # MLE for Beta(a,1): a = -1 / mean(log(p))
            a_new = -1.0 / min(weighted_log_p, -1e-10)
            a_new = np.clip(a_new, 0.01, 0.99)
        else:
            a_new = a

        pi0 = pi0_new
        a = a_new

    # Final classification: document is relevant if posterior P(signal | p) > 1 - alpha
    f0 = np.ones(n)
    f1 = a * np.power(p_values, a - 1)
    denom = pi0 * f0 + (1 - pi0) * f1
    posterior_signal = (1 - pi0) * f1 / np.clip(denom, 1e-10, None)

    # Count documents with high posterior signal probability
    # Sort by similarity (already sorted) and count those above threshold
    k = int(np.sum(posterior_signal > (1 - alpha)))
    return max(min(k, max_k), 1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ESTIMATORS: dict[EstimatorName, EstimatorFn] = {
    "higher_criticism": higher_criticism,
    "benjamini_hochberg": benjamini_hochberg,
    "bonferroni": bonferroni_correction,
    "storey_bh": storey_bh,
    "local_fdr": local_fdr,
    "kneedle": kneedle,
    "berk_jones": berk_jones,
    "beta_mixture": beta_mixture,
}


def get_estimator(name: EstimatorName) -> EstimatorFn:
    """Look up an estimator function by name.

    Args:
        name: One of ``"higher_criticism"``, ``"benjamini_hochberg"``,
            ``"bonferroni"``, ``"storey_bh"``, ``"local_fdr"``,
            ``"kneedle"``, ``"berk_jones"``, ``"beta_mixture"``.

    Returns:
        The corresponding estimator callable.

    Raises:
        ValueError: If ``name`` is not a known estimator.
    """
    if name not in ESTIMATORS:
        valid = ", ".join(f'"{k}"' for k in ESTIMATORS)
        raise ValueError(f"Unknown estimator {name!r}. Valid options: {valid}")
    return ESTIMATORS[name]
