"""
DynamicRetriever: a LangChain-native retriever that uses statistical methods
to adaptively select the number of documents ``k`` per query.

Two retrieval modes
-------------------
``candidates`` (default, production-friendly)
    Fetches ``max_candidates`` documents from any LangChain ``VectorStore``,
    re-embeds them using the provided ``Embeddings`` model, computes cosine
    similarity, and applies the statistical estimator. Requires no
    pre-computation and works with Chroma, FAISS, Pinecone, Qdrant, etc.

``exact`` (research-equivalent)
    Requires the user to supply all corpus embeddings as a ``numpy.ndarray``
    (shape: N × D). Cosine similarity is computed against the entire corpus
    in-memory before calling the estimator. Matches the original research
    methodology exactly.

Swapping into a LangChain chain
--------------------------------
::

    from ragnaros import DynamicRetriever, NullDistribution

    null_dist = NullDistribution.from_builtin_corpus(
        vectorstore=my_store,
        embeddings=OpenAIEmbeddings(),
        cache_path="./null_dist.npy",
    )
    retriever = DynamicRetriever.from_vectorstore(
        vectorstore=my_store,
        embeddings=OpenAIEmbeddings(),
        null_distribution=null_dist,
        estimator="higher_criticism",
    )
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, PrivateAttr, model_validator
from sklearn.metrics.pairwise import cosine_similarity

from ragnaros._types import Embedding, EstimatorFn, EstimatorName, RetrievalMode
from ragnaros.estimators import get_estimator
from ragnaros.null_distribution import NullDistribution


# ---------------------------------------------------------------------------
# Internal strategy classes (not part of the public API)
# ---------------------------------------------------------------------------


class _ExactStrategy:
    """Uses pre-computed corpus embeddings for an exact all-corpus similarity search."""

    def __init__(self, corpus_embeddings: np.ndarray) -> None:
        self._corpus_embs = np.asarray(corpus_embeddings, dtype=np.float32)

    def get_candidate_embs(
        self,
        question_emb: Embedding,
    ) -> list[Embedding]:
        """Return ALL corpus embeddings sorted by similarity descending.

        Passing all embeddings to the estimator (rather than a pre-filtered
        top-k subset) matches the original research methodology and lets the
        statistical test operate on the full corpus distribution.
        """
        sims = cosine_similarity([question_emb], self._corpus_embs)[0]
        sorted_idx = np.argsort(sims)[::-1]
        return [self._corpus_embs[i] for i in sorted_idx]


class _CandidatesStrategy:
    """Fetches top candidates from vectorstore and re-embeds them at query time."""

    def __init__(self, vectorstore: VectorStore, embeddings: Embeddings) -> None:
        self._vectorstore = vectorstore
        self._embeddings = embeddings

    def get_candidate_embs(
        self,
        query: str,
        max_candidates: int,
    ) -> tuple[list[Embedding], list[Document]]:
        docs = self._vectorstore.similarity_search(query, k=max_candidates)
        if not docs:
            return [], []
        texts = [d.page_content for d in docs]
        raw_embs = self._embeddings.embed_documents(texts)
        embs = [np.asarray(e, dtype=np.float32) for e in raw_embs]
        return embs, docs

    async def aget_candidate_embs(
        self,
        query: str,
        max_candidates: int,
    ) -> tuple[list[Embedding], list[Document]]:
        docs = await self._vectorstore.asimilarity_search(query, k=max_candidates)
        if not docs:
            return [], []
        texts = [d.page_content for d in docs]
        raw_embs = await self._embeddings.aembed_documents(texts)
        embs = [np.asarray(e, dtype=np.float32) for e in raw_embs]
        return embs, docs


# ---------------------------------------------------------------------------
# Public retriever
# ---------------------------------------------------------------------------


class DynamicRetriever(BaseRetriever):
    """LangChain-compatible retriever with statistical dynamic-k selection.

    Extends ``langchain_core.retrievers.BaseRetriever`` so it can be used as a
    drop-in replacement in any LangChain chain (``RetrievalQA``, LCEL ``|``
    pipelines, agents, etc.).

    Args:
        vectorstore: Any populated LangChain ``VectorStore``.
        embeddings: The ``Embeddings`` model used to embed queries and (in
            candidates mode) candidate documents. **Must be the same model
            used to build the null distribution and to populate the store.**
        null_distribution: Pre-built :class:`~ragnaros.null_distribution.NullDistribution`.
        estimator: Estimator name (``"higher_criticism"``,
            ``"benjamini_hochberg"``, ``"bonferroni"``) or a custom callable
            with signature ``fn(question_emb, doc_embs, null_dist, alpha, max_k) -> int``.
        alpha: Statistical significance level. Default 0.05.
        max_k: Hard upper bound on the number of documents returned.
            Default 10.
        max_candidates: In *candidates* mode, the number of documents fetched
            from the vector store before applying the estimator. Default 50.
        mode: :class:`~ragnaros._types.RetrievalMode` — ``"candidates"``
            (default) or ``"exact"``.
        corpus_embeddings: Required when ``mode="exact"``. A ``numpy.ndarray``
            of shape ``(N, D)`` containing all corpus document embeddings.

    Example::

        retriever = DynamicRetriever.from_vectorstore(
            vectorstore=chroma_store,
            embeddings=OpenAIEmbeddings(),
            null_distribution=null_dist,
            estimator="higher_criticism",
            alpha=0.05,
            max_k=10,
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: VectorStore
    embeddings: Embeddings
    null_distribution: NullDistribution
    # Use Any so Pydantic doesn't try to resolve the EstimatorFn Callable type alias.
    # Actual type validation is done in _validate_and_init.
    estimator: Any = "higher_criticism"
    alpha: float = 0.05
    max_k: int = 10
    max_candidates: int = 50
    mode: RetrievalMode = RetrievalMode.CANDIDATES
    corpus_embeddings: Any = None  # np.ndarray | None — Any avoids Pydantic resolution issues

    # Private attributes — initialised in model_post_init
    _strategy: _ExactStrategy | _CandidatesStrategy = PrivateAttr()
    _estimator_fn: EstimatorFn = PrivateAttr()

    @model_validator(mode="after")
    def _validate_and_init(self) -> DynamicRetriever:
        if self.mode == RetrievalMode.EXACT and self.corpus_embeddings is None:
            raise ValueError(
                "corpus_embeddings must be provided when mode='exact'. "
                "Pass a numpy.ndarray of shape (N, D) containing all corpus embeddings, "
                "or switch to mode='candidates' to avoid pre-computation."
            )
        # Resolve estimator
        if callable(self.estimator) and not isinstance(self.estimator, str):
            self._estimator_fn = self.estimator  # type: ignore[assignment]
        else:
            self._estimator_fn = get_estimator(self.estimator)  # type: ignore[arg-type]

        # Build strategy
        if self.mode == RetrievalMode.EXACT:
            self._strategy = _ExactStrategy(self.corpus_embeddings)  # type: ignore[arg-type]
        else:
            self._strategy = _CandidatesStrategy(self.vectorstore, self.embeddings)

        return self

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_vectorstore(
        cls,
        vectorstore: VectorStore,
        embeddings: Embeddings,
        null_distribution: NullDistribution,
        *,
        mode: RetrievalMode | str = RetrievalMode.CANDIDATES,
        corpus_embeddings: np.ndarray | None = None,
        estimator: EstimatorName | EstimatorFn = "higher_criticism",
        alpha: float = 0.05,
        max_k: int = 10,
        max_candidates: int = 50,
    ) -> DynamicRetriever:
        """Convenience constructor.

        Args:
            vectorstore: Populated LangChain ``VectorStore``.
            embeddings: Embeddings model (same one used to build the store).
            null_distribution: Pre-built :class:`~ragnaros.null_distribution.NullDistribution`.
            mode: ``"candidates"`` (default) or ``"exact"``.
            corpus_embeddings: All corpus embeddings; required for ``mode="exact"``.
            estimator: Estimator name or custom callable.
            alpha: Significance level. Default 0.05.
            max_k: Max documents to return. Default 10.
            max_candidates: Candidates fetched in candidates mode. Default 50.

        Returns:
            A configured :class:`DynamicRetriever`.
        """
        return cls(
            vectorstore=vectorstore,
            embeddings=embeddings,
            null_distribution=null_distribution,
            mode=RetrievalMode(mode),
            corpus_embeddings=corpus_embeddings,
            estimator=estimator,
            alpha=alpha,
            max_k=max_k,
            max_candidates=max_candidates,
        )

    # ------------------------------------------------------------------
    # Core retrieval — synchronous
    # ------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        question_emb = np.asarray(
            self.embeddings.embed_query(query), dtype=np.float32
        )

        if self.mode == RetrievalMode.EXACT:
            strategy: _ExactStrategy = self._strategy  # type: ignore[assignment]
            doc_embs = strategy.get_candidate_embs(question_emb)
            if not doc_embs:
                return self.vectorstore.similarity_search(query, k=1)
            k = self._estimator_fn(
                question_emb,
                doc_embs,
                self.null_distribution.values,
                self.alpha,
                self.max_k,
            )
            return self.vectorstore.similarity_search(query, k=k)

        # Candidates mode
        strategy_c: _CandidatesStrategy = self._strategy  # type: ignore[assignment]
        doc_embs_list, candidates = strategy_c.get_candidate_embs(
            query, self.max_candidates
        )
        if not doc_embs_list:
            return []
        k = self._estimator_fn(
            question_emb,
            doc_embs_list,
            self.null_distribution.values,
            self.alpha,
            self.max_k,
        )
        return candidates[:k]

    # ------------------------------------------------------------------
    # Core retrieval — asynchronous
    # ------------------------------------------------------------------

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        question_emb = np.asarray(
            await self.embeddings.aembed_query(query), dtype=np.float32
        )
        loop = asyncio.get_running_loop()

        if self.mode == RetrievalMode.EXACT:
            # CPU-bound similarity computation — offload to thread pool.
            strategy: _ExactStrategy = self._strategy  # type: ignore[assignment]
            doc_embs = await loop.run_in_executor(
                None, strategy.get_candidate_embs, question_emb
            )
            if not doc_embs:
                return await self.vectorstore.asimilarity_search(query, k=1)
            k = await loop.run_in_executor(
                None,
                self._estimator_fn,
                question_emb,
                doc_embs,
                self.null_distribution.values,
                self.alpha,
                self.max_k,
            )
            return await self.vectorstore.asimilarity_search(query, k=k)

        # Candidates mode
        strategy_c: _CandidatesStrategy = self._strategy  # type: ignore[assignment]
        doc_embs_list, candidates = await strategy_c.aget_candidate_embs(
            query, self.max_candidates
        )
        if not doc_embs_list:
            return []
        k = await loop.run_in_executor(
            None,
            self._estimator_fn,
            question_emb,
            doc_embs_list,
            self.null_distribution.values,
            self.alpha,
            self.max_k,
        )
        return candidates[:k]


# Rebuild the model after class definition so Pydantic can resolve all forward
# references in the context of the fully-imported module.
DynamicRetriever.model_rebuild()
