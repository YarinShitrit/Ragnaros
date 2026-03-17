"""
NullDistributionBuilder: constructs the null distribution of cosine similarity
scores by comparing semantically unrelated texts against the target vector store.

The null distribution captures "how similar a random, unrelated query tends to
be to documents in this corpus". The statistical estimators use it as a
reference to decide whether a query-document similarity is significant.

Implementation notes
--------------------
- Only public LangChain APIs are used (``similarity_search``,
  ``aembed_documents``, etc.). No private ``._collection`` access.
- All similarity values are **cosine similarity** computed from raw embeddings
  so the metric is consistent with the estimators in ``ragnaros.estimators``.
- Candidate documents are fetched from the vector store, then re-embedded
  using the same ``Embeddings`` object. This means the null distribution is
  always comparable to the query-time cosine similarities in candidates mode.
- For exact mode (all corpus embeddings pre-loaded), pass ``corpus_embeddings``
  directly to ``NullDistribution.from_corpus`` instead of using this builder.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    from ragnaros._types import NullDist


class NullDistributionBuilder:
    """Builds a null distribution using a vector store and an embedding model.

    Args:
        vectorstore: Any LangChain ``VectorStore`` instance (Chroma, FAISS,
            Pinecone, Qdrant, …). Must be populated with your corpus documents.
        embeddings: Any LangChain ``Embeddings`` instance used to embed both
            the unrelated texts and the retrieved candidate documents. **Must
            be the same model used at query time.**
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        embeddings: Embeddings,
    ) -> None:
        self._vectorstore = vectorstore
        self._embeddings = embeddings

    # ------------------------------------------------------------------
    # Synchronous build
    # ------------------------------------------------------------------

    def build(
        self,
        corpus_texts: list[str],
        n_trials: int = 200,
        k: int = 20,
        show_progress: bool = False,
        seed: int = 42,
    ) -> NullDist:
        """Build the null distribution synchronously.

        Args:
            corpus_texts: List of unrelated texts to sample from (e.g. everyday
                conversation, random news headlines — anything semantically
                different from your vector store's content).
            n_trials: Number of random texts to sample. More trials → a
                smoother, more reliable null distribution. Default 200.
            k: Number of top documents to retrieve per sampled text. The top-k
                cosine similarities per trial are added to the distribution.
                Default 20.
            show_progress: Print a simple progress counter. Default ``False``.
            seed: Random seed for reproducibility. Default 42.

        Returns:
            1-D ``numpy.ndarray`` of ``float32`` cosine similarity scores
            (length ≤ ``n_trials * k``).
        """
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(corpus_texts), size=n_trials)
        sampled_texts = [corpus_texts[int(i)] for i in indices]

        if show_progress:
            print(f"[RAGnaros] Embedding {n_trials} unrelated texts…")

        # Batch-embed all sampled texts in a single call.
        query_embs: list[list[float]] = self._embeddings.embed_documents(sampled_texts)

        # Fetch top-k candidate documents for each sampled text.
        all_candidate_texts: list[str] = []
        trial_sizes: list[int] = []
        for idx, text in enumerate(sampled_texts):
            if show_progress and idx % 50 == 0:
                print(f"[RAGnaros] Fetching candidates: {idx}/{n_trials}")
            candidates = self._vectorstore.similarity_search(text, k=k)
            texts = [c.page_content for c in candidates]
            all_candidate_texts.extend(texts)
            trial_sizes.append(len(texts))

        if not all_candidate_texts:
            return np.array([], dtype=np.float32)

        if show_progress:
            print(f"[RAGnaros] Embedding {len(all_candidate_texts)} candidate texts…")

        # Batch-embed all candidate texts in a single call.
        all_candidate_embs: list[list[float]] = self._embeddings.embed_documents(
            all_candidate_texts
        )

        # Compute cosine similarities per trial.
        null_scores: list[float] = []
        idx = 0
        for q_emb, size in zip(query_embs, trial_sizes):
            if size > 0:
                cand_slice = all_candidate_embs[idx : idx + size]
                sims = cosine_similarity(
                    [np.asarray(q_emb)],
                    [np.asarray(e) for e in cand_slice],
                )[0]
                null_scores.extend(sims.tolist())
            idx += size

        if show_progress:
            print(
                f"[RAGnaros] Null distribution built — {len(null_scores)} scores."
            )

        return np.array(null_scores, dtype=np.float32)

    # ------------------------------------------------------------------
    # Asynchronous build
    # ------------------------------------------------------------------

    async def abuild(
        self,
        corpus_texts: list[str],
        n_trials: int = 200,
        k: int = 20,
        concurrency: int = 10,
        seed: int = 42,
    ) -> NullDist:
        """Build the null distribution asynchronously.

        Candidate retrieval calls are issued concurrently (up to
        ``concurrency`` simultaneous requests).

        Args:
            corpus_texts: List of unrelated texts to sample from.
            n_trials: Number of random texts to sample. Default 200.
            k: Documents retrieved per trial. Default 20.
            concurrency: Max simultaneous ``asimilarity_search`` calls.
                Default 10.
            seed: Random seed. Default 42.

        Returns:
            1-D ``numpy.ndarray`` of ``float32`` cosine similarity scores.
        """
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(corpus_texts), size=n_trials)
        sampled_texts = [corpus_texts[int(i)] for i in indices]

        # Batch-embed all sampled texts.
        query_embs: list[list[float]] = await self._embeddings.aembed_documents(
            sampled_texts
        )

        # Fetch candidates concurrently.
        semaphore = asyncio.Semaphore(concurrency)

        async def _fetch(text: str) -> list[str]:
            async with semaphore:
                candidates = await self._vectorstore.asimilarity_search(text, k=k)
                return [c.page_content for c in candidates]

        candidate_lists: list[list[str]] = await asyncio.gather(
            *[_fetch(t) for t in sampled_texts]
        )

        all_candidate_texts: list[str] = [
            t for texts in candidate_lists for t in texts
        ]

        if not all_candidate_texts:
            return np.array([], dtype=np.float32)

        # Batch-embed all candidates.
        all_candidate_embs: list[list[float]] = await self._embeddings.aembed_documents(
            all_candidate_texts
        )

        # Compute per-trial cosine similarities (CPU-bound, run in executor).
        trial_sizes = [len(t) for t in candidate_lists]

        def _compute_sims() -> list[float]:
            scores: list[float] = []
            idx = 0
            for q_emb, size in zip(query_embs, trial_sizes):
                if size > 0:
                    cand_slice = all_candidate_embs[idx : idx + size]
                    sims = cosine_similarity(
                        [np.asarray(q_emb)],
                        [np.asarray(e) for e in cand_slice],
                    )[0]
                    scores.extend(sims.tolist())
                idx += size
            return scores

        null_scores = await asyncio.get_event_loop().run_in_executor(
            None, _compute_sims
        )
        return np.array(null_scores, dtype=np.float32)
