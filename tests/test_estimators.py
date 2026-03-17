"""Tests for the three k-estimator functions."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from ragnaros.estimators import (
    ESTIMATORS,
    benjamini_hochberg,
    bonferroni_correction,
    get_estimator,
    higher_criticism,
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
        # First doc identical to query
        docs = [q.copy()] + make_embeddings(9, dim, seed=5)
        null = make_null_dist()
        k = higher_criticism(q, docs, null, max_k=10)
        assert k >= 1

    def test_single_doc(self):
        dim = 16
        q = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        docs = [q.copy()]
        null = make_null_dist()
        k = higher_criticism(q, docs, null, max_k=5)
        assert k == 1

    def test_max_k_caps_result(self, base_scenario):
        q, docs, null = base_scenario
        k = higher_criticism(q, docs, null, max_k=3)
        assert k <= 3

    def test_all_unrelated_docs_returns_at_least_1(self):
        """Even with irrelevant docs, k is at least 1."""
        rng = np.random.default_rng(123)
        q = rng.standard_normal(16).astype(np.float32)
        q /= np.linalg.norm(q)
        # All docs orthogonal to query
        docs = make_embeddings(10, seed=999)
        # Null distribution with high similarities (should make everything look insignificant)
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
        """Should always return at least 1."""
        dim = 16
        rng = np.random.default_rng(0)
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        docs = make_embeddings(5, dim, seed=1)
        # Null with very high similarities → all p-values near 1 → none significant
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
        # At minimum, both must be in valid range
        assert 1 <= k_bon <= 10
        assert 1 <= k_bh <= 10
        assert k_bon <= k_bh

    def test_identical_query_and_doc(self):
        """When a doc is identical to the query it should definitely be included."""
        dim = 16
        q = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        docs = [q.copy()] + make_embeddings(9, seed=77)
        null = make_null_dist()
        k = bonferroni_correction(q, docs, null, alpha=0.05, max_k=10)
        assert k >= 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_estimators_in_registry(self):
        assert set(ESTIMATORS.keys()) == {"higher_criticism", "benjamini_hochberg", "bonferroni"}

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
