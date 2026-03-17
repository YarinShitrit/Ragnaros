"""
Shared pytest fixtures for RAGnaros tests.

All fixtures use deterministic, in-memory stubs — no OpenAI API calls,
no vector store persistence, no disk I/O (unless testing save/load).
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, List, Optional, Type

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


# ---------------------------------------------------------------------------
# Deterministic fake embeddings (inherits from LangChain Embeddings)
# ---------------------------------------------------------------------------


class FakeEmbeddings(Embeddings):
    """Deterministic embedding model that maps text → reproducible unit vector."""

    DIM = 16

    def _text_to_vector(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.DIM).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        return v.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._text_to_vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._text_to_vector(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------


CORPUS_TEXTS = [
    "Paris is the capital of France and a major European city.",
    "The Eiffel Tower stands 330 meters tall in the heart of Paris.",
    "French cuisine is famous for its baguettes, wine, and cheese.",
    "The River Seine flows through the centre of Paris.",
    "Napoleon Bonaparte was a French military and political leader.",
    "The Louvre is the world's largest art museum, located in Paris.",
    "France is a founding member of the European Union.",
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "Versailles Palace is located about 20 kilometres southwest of Paris.",
    "Charles de Gaulle Airport is the main international airport of Paris.",
]

UNRELATED_TEXTS = [
    "Did you catch the game last night?",
    "I need to pick up groceries after work.",
    "The weather was lovely this morning.",
    "Have you tried the new coffee shop downtown?",
    "I've been so tired lately.",
    "My cat knocked over my coffee again.",
    "What are you doing this weekend?",
    "I finally finished reading that novel.",
    "Can you believe it's already Friday?",
    "I think I need a vacation soon.",
]


@pytest.fixture
def corpus_texts() -> list[str]:
    return list(CORPUS_TEXTS)


@pytest.fixture
def unrelated_texts() -> list[str]:
    return list(UNRELATED_TEXTS)


# ---------------------------------------------------------------------------
# Synthetic embeddings and null distribution
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_embeddings_model() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def corpus_embs(fake_embeddings_model: FakeEmbeddings) -> np.ndarray:
    """All corpus document embeddings as a (N, D) array."""
    raw = fake_embeddings_model.embed_documents(CORPUS_TEXTS)
    return np.asarray(raw, dtype=np.float32)


@pytest.fixture
def null_dist_array(fake_embeddings_model: FakeEmbeddings, corpus_embs: np.ndarray) -> np.ndarray:
    """Null distribution built from unrelated texts vs corpus embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity

    scores: list[float] = []
    for text in UNRELATED_TEXTS:
        q_emb = fake_embeddings_model.embed_query(text)
        sims = cosine_similarity([q_emb], corpus_embs)[0]
        scores.extend(sims.tolist())
    return np.asarray(scores, dtype=np.float32)


@pytest.fixture
def null_distribution(null_dist_array: np.ndarray):
    from ragnaros.null_distribution import NullDistribution

    return NullDistribution.from_array(null_dist_array)


# ---------------------------------------------------------------------------
# Fake vector store (inherits from LangChain VectorStore)
# ---------------------------------------------------------------------------


class FakeVectorStore(VectorStore):
    """In-memory vector store backed by FakeEmbeddings."""

    def __init__(
        self,
        texts: list[str],
        embeddings_model: FakeEmbeddings,
    ) -> None:
        self._docs = [Document(page_content=t) for t in texts]
        self._embeddings_model = embeddings_model
        self._embs = np.asarray(
            embeddings_model.embed_documents(texts), dtype=np.float32
        )

    def _top_k(self, query: str, k: int) -> list[Document]:
        from sklearn.metrics.pairwise import cosine_similarity

        q_emb = np.asarray(self._embeddings_model.embed_query(query), dtype=np.float32)
        sims = cosine_similarity([q_emb], self._embs)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [self._docs[i] for i in idx]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return self._top_k(query, k)

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return self._top_k(query, k)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        q_emb = np.asarray(self._embeddings_model.embed_query(query), dtype=np.float32)
        sims = cosine_similarity([q_emb], self._embs)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [(self._docs[i], float(sims[i])) for i in idx]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Required abstract method — not used in tests."""
        raise NotImplementedError

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> FakeVectorStore:
        """Required abstract classmethod — not used in tests."""
        raise NotImplementedError


@pytest.fixture
def fake_vectorstore(fake_embeddings_model: FakeEmbeddings) -> FakeVectorStore:
    return FakeVectorStore(CORPUS_TEXTS, fake_embeddings_model)
