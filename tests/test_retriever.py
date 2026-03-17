"""Tests for DynamicRetriever."""

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.documents import Document

from ragnaros.retriever import DynamicRetriever
from ragnaros._types import RetrievalMode


# ---------------------------------------------------------------------------
# Candidates mode (default)
# ---------------------------------------------------------------------------


class TestCandidatesMode:
    def test_returns_documents(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
            estimator="higher_criticism",
        )
        docs = retriever.invoke("What is the capital of France?")
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_returns_at_least_one_doc(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
        )
        docs = retriever.invoke("Paris France capital")
        assert len(docs) >= 1

    def test_respects_max_k(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
            max_k=3,
        )
        docs = retriever.invoke("France")
        assert len(docs) <= 3

    def test_all_three_estimators(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        for estimator in ["higher_criticism", "benjamini_hochberg", "bonferroni"]:
            retriever = DynamicRetriever.from_vectorstore(
                vectorstore=fake_vectorstore,
                embeddings=fake_embeddings_model,
                null_distribution=null_distribution,
                estimator=estimator,
            )
            docs = retriever.invoke("France capital")
            assert len(docs) >= 1, f"Estimator {estimator} returned no docs"

    def test_custom_estimator_callable(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        def always_three(q_emb, doc_embs, null_dist, alpha, max_k):
            return min(3, max_k)

        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
            estimator=always_three,
        )
        docs = retriever.invoke("France")
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_async_returns_documents(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
        )
        docs = await retriever.ainvoke("Paris France")
        assert isinstance(docs, list)
        assert len(docs) >= 1

    @pytest.mark.asyncio
    async def test_async_respects_max_k(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
            max_k=2,
        )
        docs = await retriever.ainvoke("France Paris Eiffel")
        assert len(docs) <= 2


# ---------------------------------------------------------------------------
# Exact mode
# ---------------------------------------------------------------------------


class TestExactMode:
    def test_returns_documents(
        self,
        fake_vectorstore,
        fake_embeddings_model,
        null_distribution,
        corpus_embs,
    ):
        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
            mode="exact",
            corpus_embeddings=corpus_embs,
        )
        docs = retriever.invoke("capital of France")
        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_exact_mode_requires_corpus_embeddings(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        with pytest.raises(ValueError, match="corpus_embeddings"):
            DynamicRetriever.from_vectorstore(
                vectorstore=fake_vectorstore,
                embeddings=fake_embeddings_model,
                null_distribution=null_distribution,
                mode="exact",
                corpus_embeddings=None,
            )

    def test_respects_max_k(
        self,
        fake_vectorstore,
        fake_embeddings_model,
        null_distribution,
        corpus_embs,
    ):
        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
            mode="exact",
            corpus_embeddings=corpus_embs,
            max_k=4,
        )
        docs = retriever.invoke("France Paris")
        assert len(docs) <= 4

    @pytest.mark.asyncio
    async def test_async_exact_mode(
        self,
        fake_vectorstore,
        fake_embeddings_model,
        null_distribution,
        corpus_embs,
    ):
        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
            mode="exact",
            corpus_embeddings=corpus_embs,
        )
        docs = await retriever.ainvoke("French history")
        assert len(docs) >= 1


# ---------------------------------------------------------------------------
# Factory validation
# ---------------------------------------------------------------------------


class TestFactory:
    def test_invalid_estimator_name(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        with pytest.raises(ValueError, match="Unknown estimator"):
            DynamicRetriever.from_vectorstore(
                vectorstore=fake_vectorstore,
                embeddings=fake_embeddings_model,
                null_distribution=null_distribution,
                estimator="does_not_exist",
            )

    def test_mode_enum_and_string_equivalent(
        self, fake_vectorstore, fake_embeddings_model, null_distribution
    ):
        r1 = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
            mode="candidates",
        )
        r2 = DynamicRetriever.from_vectorstore(
            vectorstore=fake_vectorstore,
            embeddings=fake_embeddings_model,
            null_distribution=null_distribution,
            mode=RetrievalMode.CANDIDATES,
        )
        assert r1.mode == r2.mode
