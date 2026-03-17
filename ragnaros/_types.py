from __future__ import annotations

from enum import Enum
from typing import Callable, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Core type aliases
# ---------------------------------------------------------------------------

Embedding = np.ndarray  # shape: (D,), dtype float32 or float64
NullDist = np.ndarray  # shape: (N,), dtype float32

EstimatorName = Literal[
    "higher_criticism",
    "benjamini_hochberg",
    "bonferroni",
    "storey_bh",
    "local_fdr",
    "kneedle",
    "berk_jones",
    "beta_mixture",
]
EstimatorFn = Callable[
    [
        "Embedding",  # question_emb
        "list[Embedding]",  # doc_embs (sorted by similarity descending)
        "NullDist",  # null_distribution
        float,  # alpha
        int,  # max_k
    ],
    int,
]


class RetrievalMode(str, Enum):
    """Controls how DynamicRetriever sources document embeddings at query time."""

    EXACT = "exact"
    """Pre-computed corpus embeddings provided by the user.
    Computes cosine similarity against ALL corpus vectors in-memory.
    Matches the original research methodology exactly."""

    CANDIDATES = "candidates"
    """No pre-computation required.
    Fetches top-max_candidates from the vector store, re-embeds them,
    then applies the statistical test. Works with any LangChain backend."""
