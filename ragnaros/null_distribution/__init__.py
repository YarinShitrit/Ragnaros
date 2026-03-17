"""
NullDistribution: the reference distribution used by all k-estimators.

Quick-start examples
--------------------

**Mode 1 — user-provided array** (e.g. pre-computed in your own pipeline)::

    from ragnaros.null_distribution import NullDistribution
    null_dist = NullDistribution.from_array(my_scores_array)

**Mode 2 — auto-generate from built-in corpus** (no extra data required)::

    null_dist = NullDistribution.from_builtin_corpus(
        vectorstore=my_vectorstore,
        embeddings=OpenAIEmbeddings(),
        cache_path="./null_dist.npy",   # optional: saves and reuses
    )

**Mode 3 — user-supplied unrelated corpus**::

    null_dist = NullDistribution.from_corpus(
        texts=my_unrelated_sentences,
        vectorstore=my_vectorstore,
        embeddings=OpenAIEmbeddings(),
    )

**Mode 4 — load from disk** (after a previous ``save``)::

    null_dist = NullDistribution.load("./null_dist.npy")
    # later…
    null_dist.save("./null_dist.npy")
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ragnaros.null_distribution._builtin_corpus import load_utterances
from ragnaros.null_distribution.builder import NullDistributionBuilder

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    from ragnaros._types import NullDist


@dataclass(frozen=True)
class NullDistribution:
    """Immutable wrapper around a 1-D array of background similarity scores.

    The ``values`` array contains cosine similarity scores sampled from
    "unrelated query vs corpus document" pairs. The k-estimators use it
    as the reference distribution to compute empirical p-values:
    ``p = mean(values >= observed_similarity)``.

    Attributes:
        values: 1-D ``numpy.ndarray`` of ``float32`` similarity scores.
        source: Human-readable label of how this distribution was built.
        metadata: Arbitrary dict stored alongside the values (n_trials, model
            name, corpus size, etc.).
    """

    values: np.ndarray  # 1-D float32 array of background cosine similarity scores
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def from_array(arr: np.ndarray) -> NullDistribution:
        """Wrap a pre-computed similarity array.

        Args:
            arr: 1-D array of cosine similarity scores (float32 or float64).

        Returns:
            A :class:`NullDistribution` with ``source="user_provided"``.

        Raises:
            ValueError: If ``arr`` is empty or not 1-D.
        """
        arr = np.asarray(arr, dtype=np.float32).ravel()
        if arr.size == 0:
            raise ValueError("arr must contain at least one score.")
        return NullDistribution(
            values=arr,
            source="user_provided",
            metadata={"n_scores": int(arr.size)},
        )

    @staticmethod
    def from_builtin_corpus(
        vectorstore: VectorStore,
        embeddings: Embeddings,
        n_trials: int = 200,
        k: int = 20,
        cache_path: Path | str | None = None,
        show_progress: bool = False,
        seed: int = 42,
    ) -> NullDistribution:
        """Build from the library's bundled conversational corpus.

        If ``cache_path`` is provided and the file already exists, the cached
        distribution is loaded and returned without any API calls. After
        building, the result is saved to ``cache_path`` automatically.

        Args:
            vectorstore: Populated LangChain ``VectorStore``.
            embeddings: Same ``Embeddings`` model used to populate the store.
            n_trials: Random texts to sample from the built-in corpus.
                Default 200.
            k: Documents retrieved per trial. Default 20.
            cache_path: Optional path to a ``.npy`` file for caching.
            show_progress: Print progress to stdout. Default ``False``.
            seed: Random seed. Default 42.

        Returns:
            A :class:`NullDistribution` with ``source="builtin_corpus"``.
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                return NullDistribution.load(cache_path)

        texts = load_utterances()
        builder = NullDistributionBuilder(vectorstore, embeddings)
        scores = builder.build(
            corpus_texts=texts,
            n_trials=n_trials,
            k=k,
            show_progress=show_progress,
            seed=seed,
        )
        dist = NullDistribution(
            values=scores,
            source="builtin_corpus",
            metadata={
                "n_trials": n_trials,
                "k": k,
                "seed": seed,
                "n_scores": int(scores.size),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if cache_path is not None:
            dist.save(cache_path)
        return dist

    @staticmethod
    def from_corpus(
        texts: list[str],
        vectorstore: VectorStore,
        embeddings: Embeddings,
        n_trials: int = 200,
        k: int = 20,
        cache_path: Path | str | None = None,
        show_progress: bool = False,
        seed: int = 42,
    ) -> NullDistribution:
        """Build from a user-supplied list of unrelated texts.

        The texts should be semantically *unrelated* to your vector store's
        content. Good choices include everyday conversation snippets, random
        news headlines, or sentences from a different domain entirely.

        Args:
            texts: List of unrelated strings to sample from.
            vectorstore: Populated LangChain ``VectorStore``.
            embeddings: Same ``Embeddings`` model used to populate the store.
            n_trials: Random texts to sample. Default 200.
            k: Documents retrieved per trial. Default 20.
            cache_path: Optional path for caching/loading.
            show_progress: Print progress to stdout. Default ``False``.
            seed: Random seed. Default 42.

        Returns:
            A :class:`NullDistribution` with ``source="custom_corpus"``.
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                return NullDistribution.load(cache_path)

        if not texts:
            raise ValueError("texts must contain at least one string.")

        builder = NullDistributionBuilder(vectorstore, embeddings)
        scores = builder.build(
            corpus_texts=texts,
            n_trials=n_trials,
            k=k,
            show_progress=show_progress,
            seed=seed,
        )
        dist = NullDistribution(
            values=scores,
            source="custom_corpus",
            metadata={
                "n_trials": n_trials,
                "k": k,
                "seed": seed,
                "corpus_size": len(texts),
                "n_scores": int(scores.size),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if cache_path is not None:
            dist.save(cache_path)
        return dist

    @staticmethod
    def load(path: Path | str) -> NullDistribution:
        """Load a previously saved distribution from disk.

        Expects a ``.npy`` file for the scores and a sidecar ``.json`` for
        metadata (same file stem, same directory).

        Args:
            path: Path to the ``.npy`` file.

        Returns:
            The loaded :class:`NullDistribution`.

        Raises:
            FileNotFoundError: If the ``.npy`` file does not exist.
        """
        path = Path(path)
        # Normalise: if no extension is given, assume .npy (mirrors save behaviour).
        if path.suffix != ".npy":
            path = path.with_suffix(".npy")
        if not path.exists():
            raise FileNotFoundError(f"Null distribution file not found: {path}")

        values = np.load(str(path)).astype(np.float32)
        meta_path = path.with_suffix(".json")
        metadata: dict[str, Any] = {}
        source = "loaded"
        if meta_path.exists():
            with meta_path.open() as fh:
                saved = json.load(fh)
                source = saved.pop("source", "loaded")
                metadata = saved

        return NullDistribution(values=values, source=source, metadata=metadata)

    # ------------------------------------------------------------------
    # Async factory mirrors
    # ------------------------------------------------------------------

    @staticmethod
    async def afrom_builtin_corpus(
        vectorstore: VectorStore,
        embeddings: Embeddings,
        n_trials: int = 200,
        k: int = 20,
        cache_path: Path | str | None = None,
        concurrency: int = 10,
        seed: int = 42,
    ) -> NullDistribution:
        """Async version of :meth:`from_builtin_corpus`."""
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                return NullDistribution.load(cache_path)

        texts = load_utterances()
        builder = NullDistributionBuilder(vectorstore, embeddings)
        scores = await builder.abuild(
            corpus_texts=texts,
            n_trials=n_trials,
            k=k,
            concurrency=concurrency,
            seed=seed,
        )
        dist = NullDistribution(
            values=scores,
            source="builtin_corpus",
            metadata={
                "n_trials": n_trials,
                "k": k,
                "seed": seed,
                "n_scores": int(scores.size),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if cache_path is not None:
            dist.save(cache_path)
        return dist

    @staticmethod
    async def afrom_corpus(
        texts: list[str],
        vectorstore: VectorStore,
        embeddings: Embeddings,
        n_trials: int = 200,
        k: int = 20,
        cache_path: Path | str | None = None,
        concurrency: int = 10,
        seed: int = 42,
    ) -> NullDistribution:
        """Async version of :meth:`from_corpus`."""
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                return NullDistribution.load(cache_path)

        if not texts:
            raise ValueError("texts must contain at least one string.")

        builder = NullDistributionBuilder(vectorstore, embeddings)
        scores = await builder.abuild(
            corpus_texts=texts,
            n_trials=n_trials,
            k=k,
            concurrency=concurrency,
            seed=seed,
        )
        dist = NullDistribution(
            values=scores,
            source="custom_corpus",
            metadata={
                "n_trials": n_trials,
                "k": k,
                "seed": seed,
                "corpus_size": len(texts),
                "n_scores": int(scores.size),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if cache_path is not None:
            dist.save(cache_path)
        return dist

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save the distribution to disk.

        Writes two files:
        - ``<path>.npy``  — the scores array.
        - ``<path>.json`` — metadata + source label.

        Args:
            path: Destination path (e.g. ``"./null_dist.npy"``).
        """
        path = Path(path)
        # Normalise the path to always have the .npy extension before passing to
        # np.save. Without this, np.save silently appends ".npy", making the path
        # disagree with path.with_suffix(".json") for extension-less inputs.
        if path.suffix != ".npy":
            path = path.with_suffix(".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self.values)
        meta = {"source": self.source, **self.metadata}
        with path.with_suffix(".json").open("w") as fh:
            json.dump(meta, fh, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        return (
            f"NullDistribution(n_scores={len(self)}, source={self.source!r}, "
            f"mean={self.values.mean():.4f}, std={self.values.std():.4f})"
        )
