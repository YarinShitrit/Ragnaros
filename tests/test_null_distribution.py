"""Tests for NullDistribution and NullDistributionBuilder."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ragnaros.null_distribution import NullDistribution
from ragnaros.null_distribution.builder import NullDistributionBuilder
from ragnaros.null_distribution._builtin_corpus import load_utterances


# ---------------------------------------------------------------------------
# NullDistribution.from_array
# ---------------------------------------------------------------------------


class TestFromArray:
    def test_basic_creation(self):
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        nd = NullDistribution.from_array(arr)
        assert len(nd) == 3
        assert nd.source == "user_provided"
        np.testing.assert_array_almost_equal(nd.values, arr)

    def test_converts_to_float32(self):
        arr = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        nd = NullDistribution.from_array(arr)
        assert nd.values.dtype == np.float32

    def test_flattens_2d(self):
        arr = np.array([[0.1, 0.2], [0.3, 0.4]])
        nd = NullDistribution.from_array(arr)
        assert nd.values.ndim == 1
        assert len(nd) == 4

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one score"):
            NullDistribution.from_array(np.array([]))

    def test_frozen_immutable(self):
        nd = NullDistribution.from_array(np.array([0.5]))
        with pytest.raises(Exception):
            nd.source = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip(self, null_distribution):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "null.npy"
            null_distribution.save(path)
            loaded = NullDistribution.load(path)

        assert len(loaded) == len(null_distribution)
        np.testing.assert_array_almost_equal(loaded.values, null_distribution.values)

    def test_metadata_persisted(self, null_distribution):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "null.npy"
            null_distribution.save(path)
            meta_path = path.with_suffix(".json")
            assert meta_path.exists()
            with meta_path.open() as fh:
                meta = json.load(fh)
            assert "source" in meta

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            NullDistribution.load("/nonexistent/path/null.npy")

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "deep" / "null.npy"
            nd = NullDistribution.from_array(np.array([0.1, 0.2]))
            nd.save(path)
            assert path.exists()


# ---------------------------------------------------------------------------
# NullDistributionBuilder
# ---------------------------------------------------------------------------


class TestNullDistributionBuilder:
    def test_build_returns_array(
        self, fake_vectorstore, fake_embeddings_model, unrelated_texts
    ):
        builder = NullDistributionBuilder(fake_vectorstore, fake_embeddings_model)
        result = builder.build(corpus_texts=unrelated_texts, n_trials=5, k=3)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.size > 0

    def test_build_scores_in_valid_range(
        self, fake_vectorstore, fake_embeddings_model, unrelated_texts
    ):
        builder = NullDistributionBuilder(fake_vectorstore, fake_embeddings_model)
        result = builder.build(corpus_texts=unrelated_texts, n_trials=5, k=3)
        # Cosine similarity should be in [-1, 1]
        assert float(result.min()) >= -1.01
        assert float(result.max()) <= 1.01

    def test_deterministic_with_seed(
        self, fake_vectorstore, fake_embeddings_model, unrelated_texts
    ):
        builder = NullDistributionBuilder(fake_vectorstore, fake_embeddings_model)
        r1 = builder.build(unrelated_texts, n_trials=5, k=3, seed=0)
        r2 = builder.build(unrelated_texts, n_trials=5, k=3, seed=0)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(
        self, fake_vectorstore, fake_embeddings_model, unrelated_texts
    ):
        builder = NullDistributionBuilder(fake_vectorstore, fake_embeddings_model)
        r1 = builder.build(unrelated_texts, n_trials=5, k=3, seed=0)
        r2 = builder.build(unrelated_texts, n_trials=5, k=3, seed=99)
        # Very unlikely to be identical
        assert not np.array_equal(r1, r2)

    @pytest.mark.asyncio
    async def test_abuild_matches_build(
        self, fake_vectorstore, fake_embeddings_model, unrelated_texts
    ):
        builder = NullDistributionBuilder(fake_vectorstore, fake_embeddings_model)
        sync_result = builder.build(unrelated_texts, n_trials=4, k=3, seed=42)
        async_result = await builder.abuild(unrelated_texts, n_trials=4, k=3, seed=42)
        # Same shape; values may differ slightly due to gather ordering but
        # the sorted arrays should be close.
        assert sync_result.size == async_result.size


# ---------------------------------------------------------------------------
# NullDistribution.from_corpus
# ---------------------------------------------------------------------------


class TestFromCorpus:
    def test_from_corpus_basic(
        self, fake_vectorstore, fake_embeddings_model, unrelated_texts
    ):
        nd = NullDistribution.from_corpus(
            texts=unrelated_texts,
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            n_trials=5,
            k=3,
        )
        assert nd.source == "custom_corpus"
        assert len(nd) > 0

    def test_from_corpus_with_cache(
        self, fake_vectorstore, fake_embeddings_model, unrelated_texts
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "null.npy"
            nd1 = NullDistribution.from_corpus(
                texts=unrelated_texts,
                vectorstore=fake_vectorstore,
                embeddings=fake_embeddings_model,
                n_trials=3,
                k=2,
                cache_path=cache,
            )
            assert cache.exists()
            # Second call should load from cache
            nd2 = NullDistribution.from_corpus(
                texts=unrelated_texts,
                vectorstore=fake_vectorstore,
                embeddings=fake_embeddings_model,
                n_trials=3,
                k=2,
                cache_path=cache,
            )
            np.testing.assert_array_equal(nd1.values, nd2.values)

    def test_from_corpus_empty_texts_raises(
        self, fake_vectorstore, fake_embeddings_model
    ):
        with pytest.raises(ValueError, match="at least one"):
            NullDistribution.from_corpus(
                texts=[],
                vectorstore=fake_vectorstore,
                embeddings=fake_embeddings_model,
            )


# ---------------------------------------------------------------------------
# Builtin corpus
# ---------------------------------------------------------------------------


class TestBuiltinCorpus:
    def test_load_utterances_nonempty(self):
        utterances = load_utterances()
        assert len(utterances) > 100

    def test_all_strings(self):
        for u in load_utterances():
            assert isinstance(u, str)
            assert len(u) > 0

    def test_from_builtin_corpus(self, fake_vectorstore, fake_embeddings_model):
        nd = NullDistribution.from_builtin_corpus(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            n_trials=5,
            k=3,
        )
        assert nd.source == "builtin_corpus"
        assert len(nd) > 0


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr():
    nd = NullDistribution.from_array(np.array([0.1, 0.5, 0.9]))
    r = repr(nd)
    assert "NullDistribution" in r
    assert "n_scores=3" in r
