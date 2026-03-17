"""
RAGnaros: Adaptive k-selection for RAG pipelines.

Reduce token usage and improve accuracy by replacing fixed document retrieval
counts with statistically-grounded dynamic selection.

Quick start::

    from ragnaros import DynamicRetriever, NullDistribution

    # Build null distribution once (cached to disk)
    null_dist = NullDistribution.from_builtin_corpus(
        vectorstore=my_vectorstore,
        embeddings=my_embeddings,
        cache_path="./null_dist.npy",
    )

    # Create a dynamic retriever
    retriever = DynamicRetriever.from_vectorstore(
        vectorstore=my_vectorstore,
        embeddings=my_embeddings,
        null_distribution=null_dist,
        estimator="higher_criticism",  # or "benjamini_hochberg" / "bonferroni"
    )

    # Drop into any LangChain chain
    from langchain.chains import RetrievalQA
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

Research reference:
    Shitrit, Y. (2025). Toward Optimal Retrieval: Dynamic Document Retrieval
    in Vector-Based Search. M.Sc. Seminar, Reichman University.
"""

from ragnaros._types import EstimatorName, RetrievalMode
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
from ragnaros.null_distribution import NullDistribution
from ragnaros.retriever import DynamicRetriever

__version__ = "0.2.0"
__author__ = "Yarin Shitrit"

__all__ = [
    # Core
    "DynamicRetriever",
    "NullDistribution",
    # Estimators — original (v0.1)
    "higher_criticism",
    "benjamini_hochberg",
    "bonferroni_correction",
    # Estimators — research extension (v0.2)
    "storey_bh",
    "local_fdr",
    "kneedle",
    "berk_jones",
    "beta_mixture",
    # Registry
    "get_estimator",
    "ESTIMATORS",
    # Types
    "EstimatorName",
    "RetrievalMode",
]
