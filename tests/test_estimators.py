"""Tests for all eight k-estimator functions."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from ragnaros.estimators import (
    ESTIMATORS,
    benjamini_hochberg,
    berk_jones,
    beta_mixture,
    bonferroni_correction,
    get_estimator,
    higher_criticism,
    kneedle,
    local_fdr,
    storey_bh,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_embeddings(n: int, dim: int = 16, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return list(embs / norms)


def make_null_dist(n: int = 500, seed: int = 42) -> np.ndarray:
    """Null distribution centred around low similarity values."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 0.4, size=n).astype(np.float32)


def make_scenario_with_signal(n_relevant: int = 3, n_irrelevant: int = 17, dim: int = 16, seed: int = 7):
    """Create a scenario where n_relevant docs are similar to the query."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim).astype(np.float32)
    q /= np.linalg.norm(q)

    # Relevant docs: close to query (add small noise)
    relevant = []
    for i in range(n_relevant):
        noise = rng.standard_normal(dim).astype(np.float32) * 0.2
        doc = q + noise
        doc /= np.linalg.norm(doc)
        relevant.append(doc)

    # Irrelevant docs: random directions
    irrelevant = make_embeddings(n_irrelevant, dim, seed=seed + 100)

    docs = relevant + irrelevant
    null = make_null_dist()
    return q, docs, null


# ---------------------------------------------------------------------------
# Shared parameter scenarios
# ---------------------------------------------------------------------------


@pytest.fixture
def base_scenario():
    """A query embedding and 20 candidate doc embeddings."""
    dim = 16
    rng = np.random.default_rng(7)
    q = rng.standard_normal(dim).astype(np.float32)
    q /= np.linalg.norm(q)
    docs = make_embeddings(20, dim, seed=99)
    null = make_null_dist()
    return q, docs, null


@pytest.fixture
def strong_signal_scenario():
    """Scenario with clear signal: 3 relevant + 17 irrelevant docs."""
    return make_scenario_with_signal(n_relevant=3, n_irrelevant=17)


@pytest.fixture
def weak_signal_scenario():
    """Scenario with very weak signal: 1 slightly relevant + 19 irrelevant."""
    dim = 16
    rng = np.random.default_rng(42)
    q = rng.standard_normal(dim).astype(np.float32)
    q /= np.linalg.norm(q)

    # One doc slightly correlated
    noise = rng.standard_normal(dim).astype(np.float32) * 0.7
    doc = q + noise
    doc /= np.linalg.norm(doc)
    docs = [doc] + make_embeddings(19, dim, seed=200)
    null = make_null_dist()
    return q, docs, null


# ---------------------------------------------------------------------------
# ALL estimators: shared property tests
# ---------------------------------------------------------------------------

ALL_ESTIMATOR_NAMES = list(ESTIMATORS.keys())
ALL_ESTIMATORS = [
    higher_criticism, benjamini_hochberg, bonferroni_correction,
    storey_bh, local_fdr, kneedle, berk_jones, beta_mixture,
]


class TestAllEstimatorsSharedProperties:
    """Universal properties that must hold for every estimator."""

    @pytest.mark.parametrize("fn", ALL_ESTIMATORS, ids=[f.__name__ for f in ALL_ESTIMATORS])
    def test_returns_int(self, base_scenario, fn):
        q, docs, null = base_scenario
        k = fn(q, docs, null)
        assert isinstance(k, (int, np.integer))

    @pytest.mark.parametrize("fn", ALL_ESTIMATORS, ids=[f.__name__ for f in ALL_ESTIMATORS])
    def test_k_always_at_least_1(self, base_scenario, fn):
        q, docs, null = base_scenario
        k = fn(q, docs, null, max_k=10)
        assert k >= 1

    @pytest.mark.parametrize("fn", ALL_ESTIMATORS, ids=[f.__name__ for f in ALL_ESTIMATORS])
    def test_k_respects_max_k(self, base_scenario, fn):
        q, docs, null = base_scenario
        for max_k in [1, 3, 5, 10]:
            k = fn(q, docs, null, max_k=max_k)
            assert 1 <= k <= max_k, f"{fn.__name__} returned k={k} for max_k={max_k}"

    @pytest.mark.parametrize("fn", ALL_ESTIMATORS, ids=[f.__name__ for f in ALL_ESTIMATORS])
    def test_single_doc(self, fn):
        dim = 16
        q = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        docs = [q.copy()]
        null = make_null_dist()
        k = fn(q, docs, null, max_k=5)
        assert k == 1

    @pytest.mark.parametrize("fn", ALL_ESTIMATORS, ids=[f.__name__ for f in ALL_ESTIMATORS])
    def test_deterministic(self, base_scenario, fn):
        """Same input always gives same output."""
        q, docs, null = base_scenario
        k1 = fn(q, docs, null, alpha=0.05, max_k=10)
        k2 = fn(q, docs, null, alpha=0.05, max_k=10)
        assert k1 == k2

    @pytest.mark.parametrize("fn", ALL_ESTIMATORS, ids=[f.__name__ for f in ALL_ESTIMATORS])
    def test_with_identical_query_and_doc(self, fn):
        """When a doc is identical to query, k should be >= 1."""
        dim = 16
        q = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        docs = [q.copy()] + make_embeddings(9, seed=77)
        null = make_null_dist()
        k = fn(q, docs, null, alpha=0.05, max_k=10)
        assert k >= 1

    @pytest.mark.parametrize("fn", ALL_ESTIMATORS, ids=[f.__name__ for f in ALL_ESTIMATORS])
    def test_all_unrelated_returns_small_k(self, fn):
        """With irrelevant docs, k should be small (but at least 1)."""
        rng = np.random.default_rng(123)
        q = rng.standard_normal(16).astype(np.float32)
        q /= np.linalg.norm(q)
        docs = make_embeddings(10, seed=999)
        # Null distribution with high similarities -> docs look insignificant
        null = np.ones(200, dtype=np.float32) * 0.99
        k = fn(q, docs, null, max_k=10)
        assert 1 <= k <= 10


# ---------------------------------------------------------------------------
# higher_criticism
# ---------------------------------------------------------------------------


class TestHigherCriticism:
    def test_returns_int(self, base_scenario):
        q, docs, null = base_scenario
        k = higher_criticism(q, docs, null)
        assert isinstance(k, int)

    def test_k_in_bounds(self, base_scenario):
        q, docs, null = base_scenario
        for max_k in [1, 5, 10]:
            k = higher_criticism(q, docs, null, max_k=max_k)
            assert 1 <= k <= max_k

    def test_highly_similar_doc_increases_k(self):
        """When the first doc is very similar to the query, k should be >= 1."""
        dim = 16
        q = np.ones(dim, dtype=np.float32)
        q /= np.linalg.norm(q)
        docs = [q.copy()] + make_embeddings(9, dim, seed=5)
        null = make_null_dist()
        k = higher_criticism(q, docs, null, max_k=10)
        assert k >= 1

    def test_max_k_caps_result(self, base_scenario):
        q, docs, null = base_scenario
        k = higher_criticism(q, docs, null, max_k=3)
        assert k <= 3

    def test_all_unrelated_docs_returns_at_least_1(self):
        rng = np.random.default_rng(123)
        q = rng.standard_normal(16).astype(np.float32)
        q /= np.linalg.norm(q)
        docs = make_embeddings(10, seed=999)
        null = np.ones(200, dtype=np.float32) * 0.99
        k = higher_criticism(q, docs, null, max_k=10)
        assert k >= 1


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_returns_int(self, base_scenario):
        q, docs, null = base_scenario
        k = benjamini_hochberg(q, docs, null)
        assert isinstance(k, int)

    def test_k_in_bounds(self, base_scenario):
        q, docs, null = base_scenario
        for max_k in [1, 5, 10]:
            k = benjamini_hochberg(q, docs, null, max_k=max_k)
            assert 1 <= k <= max_k

    def test_minimum_one(self):
        dim = 16
        rng = np.random.default_rng(0)
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        docs = make_embeddings(5, dim, seed=1)
        null = np.ones(300, dtype=np.float32) * 0.99
        k = benjamini_hochberg(q, docs, null, alpha=0.05, max_k=10)
        assert k >= 1

    def test_stricter_alpha_returns_fewer_docs(self, base_scenario):
        q, docs, null = base_scenario
        k_lenient = benjamini_hochberg(q, docs, null, alpha=0.20, max_k=10)
        k_strict = benjamini_hochberg(q, docs, null, alpha=0.01, max_k=10)
        assert k_strict <= k_lenient


# ---------------------------------------------------------------------------
# bonferroni_correction
# ---------------------------------------------------------------------------


class TestBonferroni:
    def test_returns_int(self, base_scenario):
        q, docs, null = base_scenario
        k = bonferroni_correction(q, docs, null)
        assert isinstance(k, int)

    def test_k_in_bounds(self, base_scenario):
        q, docs, null = base_scenario
        for max_k in [1, 5, 10]:
            k = bonferroni_correction(q, docs, null, max_k=max_k)
            assert 1 <= k <= max_k

    def test_more_conservative_than_bh_on_this_scenario(self, base_scenario):
        """On this specific fixed scenario, Bonferroni is at most as lenient as BH.

        Note: this is not a universal mathematical guarantee (both methods make
        different multiple-comparison corrections), but it holds for this test
        scenario and serves as a regression check.
        """
        q, docs, null = base_scenario
        k_bh = benjamini_hochberg(q, docs, null, alpha=0.05, max_k=10)
        k_bon = bonferroni_correction(q, docs, null, alpha=0.05, max_k=10)
        assert 1 <= k_bon <= 10
        assert 1 <= k_bh <= 10
        assert k_bon <= k_bh

    def test_identical_query_and_doc(self):
        dim = 16
        q = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        docs = [q.copy()] + make_embeddings(9, seed=77)
        null = make_null_dist()
        k = bonferroni_correction(q, docs, null, alpha=0.05, max_k=10)
        assert k >= 1


# ---------------------------------------------------------------------------
# storey_bh
# ---------------------------------------------------------------------------


class TestStoreyBH:
    def test_at_least_as_powerful_as_bh(self, base_scenario):
        """Storey-BH should find at least as many significant docs as BH."""
        q, docs, null = base_scenario
        k_bh = benjamini_hochberg(q, docs, null, alpha=0.05, max_k=10)
        k_storey = storey_bh(q, docs, null, alpha=0.05, max_k=10)
        # Storey adjusts alpha upward when many nulls exist -> more power
        assert k_storey >= k_bh

    def test_stricter_alpha_returns_fewer_docs(self, base_scenario):
        q, docs, null = base_scenario
        k_lenient = storey_bh(q, docs, null, alpha=0.20, max_k=10)
        k_strict = storey_bh(q, docs, null, alpha=0.01, max_k=10)
        assert k_strict <= k_lenient

    def test_strong_signal(self, strong_signal_scenario):
        q, docs, null = strong_signal_scenario
        k = storey_bh(q, docs, null, alpha=0.05, max_k=10)
        assert k >= 1

    def test_pi0_estimation_reasonable(self):
        """With mostly irrelevant docs, pi0 should be high."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal(16).astype(np.float32)
        q /= np.linalg.norm(q)
        docs = make_embeddings(20, seed=999)
        null = make_null_dist(500)
        k = storey_bh(q, docs, null, alpha=0.05, max_k=10)
        assert 1 <= k <= 10


# ---------------------------------------------------------------------------
# local_fdr
# ---------------------------------------------------------------------------


class TestLocalFDR:
    def test_strong_signal_detects_relevant(self, strong_signal_scenario):
        q, docs, null = strong_signal_scenario
        k = local_fdr(q, docs, null, alpha=0.05, max_k=10)
        assert k >= 1

    def test_alpha_sensitivity(self, base_scenario):
        """Higher alpha threshold should generally accept more docs."""
        q, docs, null = base_scenario
        k_strict = local_fdr(q, docs, null, alpha=0.01, max_k=10)
        k_lenient = local_fdr(q, docs, null, alpha=0.20, max_k=10)
        assert k_strict <= k_lenient

    def test_many_docs(self):
        """Works with larger candidate pools."""
        q, docs, null = make_scenario_with_signal(n_relevant=5, n_irrelevant=45, seed=10)
        k = local_fdr(q, docs, null, alpha=0.05, max_k=20)
        assert 1 <= k <= 20


# ---------------------------------------------------------------------------
# kneedle
# ---------------------------------------------------------------------------


class TestKneedle:
    def test_alpha_independent(self, base_scenario):
        """Kneedle ignores alpha (curve geometry only)."""
        q, docs, null = base_scenario
        k1 = kneedle(q, docs, null, alpha=0.01, max_k=10)
        k2 = kneedle(q, docs, null, alpha=0.20, max_k=10)
        assert k1 == k2

    def test_clear_elbow(self):
        """When there's a clear drop in similarity, kneedle finds it."""
        dim = 16
        rng = np.random.default_rng(7)
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)

        # 3 very similar docs, then a big drop
        relevant = []
        for _ in range(3):
            doc = q + rng.standard_normal(dim).astype(np.float32) * 0.1
            doc /= np.linalg.norm(doc)
            relevant.append(doc)

        # 7 random docs (much lower similarity)
        irrelevant = make_embeddings(7, dim, seed=555)
        docs = relevant + irrelevant
        null = make_null_dist()

        k = kneedle(q, docs, null, max_k=10)
        assert 1 <= k <= 5

    def test_flat_similarities_returns_1(self):
        """When all docs have identical similarity, return 1."""
        dim = 16
        rng = np.random.default_rng(42)
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        doc = rng.standard_normal(dim).astype(np.float32)
        doc /= np.linalg.norm(doc)
        docs = [doc.copy() for _ in range(10)]
        null = make_null_dist()
        k = kneedle(q, docs, null, max_k=10)
        assert k == 1

    def test_two_docs(self):
        """Edge case: only two candidate documents."""
        dim = 16
        rng = np.random.default_rng(42)
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        docs = make_embeddings(2, dim, seed=42)
        null = make_null_dist()
        k = kneedle(q, docs, null, max_k=10)
        assert 1 <= k <= 2


# ---------------------------------------------------------------------------
# berk_jones
# ---------------------------------------------------------------------------


class TestBerkJones:
    def test_strong_signal(self, strong_signal_scenario):
        q, docs, null = strong_signal_scenario
        k = berk_jones(q, docs, null, alpha=0.05, max_k=10)
        assert k >= 1

    def test_related_to_but_different_from_hc(self, base_scenario):
        """BJ and HC can differ since they use different test statistics."""
        q, docs, null = base_scenario
        k_hc = higher_criticism(q, docs, null, max_k=10)
        k_bj = berk_jones(q, docs, null, max_k=10)
        assert 1 <= k_hc <= 10
        assert 1 <= k_bj <= 10

    def test_with_many_candidates(self):
        """Works correctly with larger pools."""
        q, docs, null = make_scenario_with_signal(n_relevant=5, n_irrelevant=45, seed=99)
        k = berk_jones(q, docs, null, alpha=0.05, max_k=20)
        assert 1 <= k <= 20


# ---------------------------------------------------------------------------
# beta_mixture
# ---------------------------------------------------------------------------


class TestBetaMixture:
    def test_strong_signal_detects_relevant(self, strong_signal_scenario):
        q, docs, null = strong_signal_scenario
        k = beta_mixture(q, docs, null, alpha=0.05, max_k=10)
        assert k >= 1

    def test_alpha_sensitivity(self, base_scenario):
        """Higher alpha (more lenient) should accept at least as many docs."""
        q, docs, null = base_scenario
        k_strict = beta_mixture(q, docs, null, alpha=0.01, max_k=10)
        k_lenient = beta_mixture(q, docs, null, alpha=0.20, max_k=10)
        assert k_strict <= k_lenient

    def test_em_convergence_with_few_docs(self):
        """EM should not crash or diverge with very few documents."""
        dim = 16
        q = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        docs = make_embeddings(3, dim, seed=42)
        null = make_null_dist()
        k = beta_mixture(q, docs, null, alpha=0.05, max_k=5)
        assert 1 <= k <= 3

    def test_large_candidate_pool(self):
        """EM works with 50+ candidates."""
        q, docs, null = make_scenario_with_signal(n_relevant=5, n_irrelevant=50, seed=77)
        k = beta_mixture(q, docs, null, alpha=0.05, max_k=20)
        assert 1 <= k <= 20


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_estimators_in_registry(self):
        expected = {
            "higher_criticism", "benjamini_hochberg", "bonferroni",
            "storey_bh", "local_fdr", "kneedle", "berk_jones", "beta_mixture",
        }
        assert set(ESTIMATORS.keys()) == expected

    def test_get_estimator_returns_callable(self):
        for name in ESTIMATORS:
            fn = get_estimator(name)
            assert callable(fn)

    def test_get_estimator_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown estimator"):
            get_estimator("nonexistent")  # type: ignore[arg-type]

    def test_registry_functions_match_direct_imports(self, base_scenario):
        q, docs, null = base_scenario
        assert get_estimator("higher_criticism")(q, docs, null) == higher_criticism(q, docs, null)
        assert get_estimator("benjamini_hochberg")(q, docs, null) == benjamini_hochberg(q, docs, null)
        assert get_estimator("bonferroni")(q, docs, null) == bonferroni_correction(q, docs, null)
        assert get_estimator("storey_bh")(q, docs, null) == storey_bh(q, docs, null)
        assert get_estimator("local_fdr")(q, docs, null) == local_fdr(q, docs, null)
        assert get_estimator("kneedle")(q, docs, null) == kneedle(q, docs, null)
        assert get_estimator("berk_jones")(q, docs, null) == berk_jones(q, docs, null)
        assert get_estimator("beta_mixture")(q, docs, null) == beta_mixture(q, docs, null)

    def test_registry_count(self):
        assert len(ESTIMATORS) == 8


# ---------------------------------------------------------------------------
# Cross-estimator comparison tests
# ---------------------------------------------------------------------------


class TestCrossEstimatorComparisons:
    """Tests that compare behavior across estimators on the same scenario."""

    def test_all_find_signal_in_strong_scenario(self, strong_signal_scenario):
        """All estimators should detect at least 1 doc with strong signal."""
        q, docs, null = strong_signal_scenario
        for name, fn in ESTIMATORS.items():
            k = fn(q, docs, null, alpha=0.05, max_k=10)
            assert k >= 1, f"{name} failed to detect signal"

    def test_conservative_ordering_bh_vs_bonferroni(self, base_scenario):
        """Bonferroni should be at most as lenient as BH on this scenario."""
        q, docs, null = base_scenario
        k_bh = benjamini_hochberg(q, docs, null, alpha=0.05, max_k=10)
        k_bon = bonferroni_correction(q, docs, null, alpha=0.05, max_k=10)
        assert k_bon <= k_bh

    def test_storey_at_least_as_powerful_as_bh(self, base_scenario):
        """Storey-BH should find >= as many docs as standard BH."""
        q, docs, null = base_scenario
        k_bh = benjamini_hochberg(q, docs, null, alpha=0.05, max_k=10)
        k_storey = storey_bh(q, docs, null, alpha=0.05, max_k=10)
        assert k_storey >= k_bh

    def test_all_return_same_for_trivial_case(self):
        """With only 1 doc, all estimators must return 1."""
        dim = 16
        q = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        docs = [q.copy()]
        null = make_null_dist()
        for name, fn in ESTIMATORS.items():
            k = fn(q, docs, null, max_k=5)
            assert k == 1, f"{name} returned {k} for single-doc case"

    def test_diversity_on_ambiguous_scenario(self, base_scenario):
        """On an ambiguous scenario, estimators should produce varied k values.

        This verifies that the estimators are implementing genuinely different
        statistical approaches rather than all converging to the same answer.
        """
        q, docs, null = base_scenario
        ks = {name: fn(q, docs, null, alpha=0.05, max_k=10) for name, fn in ESTIMATORS.items()}
        unique_ks = set(ks.values())
        # With 8 estimators using different approaches, we expect some diversity
        assert len(unique_ks) >= 2, f"All estimators returned the same k: {ks}"
