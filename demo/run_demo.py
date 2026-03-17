"""
RAGnaros Demo — Dynamic k-selection vs Fixed-k on HotPotQA

This demo:
1. Loads 500 HotPotQA questions with supporting facts as corpus
2. Builds a FAISS vector store with local sentence-transformers embeddings
3. Constructs a null distribution from unrelated queries
4. Compares Fixed-k (k=1, 3, 5, 7, 10) vs Dynamic estimators (HC, BH, Bonferroni)
5. Measures accuracy, token usage, cost, and generates visualizations

No API keys required — everything runs locally.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).parent
ASSETS_DIR = DEMO_DIR.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

N_CORPUS_QUESTIONS = 500   # HotPotQA questions to build the corpus from
N_EVAL_QUESTIONS = 100     # Questions to evaluate on
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, small local model
MAX_K = 10
ALPHA = 0.05
SEED = 42

# ---------------------------------------------------------------------------
# Local Embeddings wrapper (sentence-transformers → LangChain Embeddings)
# ---------------------------------------------------------------------------


class LocalEmbeddings(Embeddings):
    """LangChain-compatible wrapper around sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embs.tolist()

    def embed_query(self, text: str) -> list[float]:
        emb = self._model.encode([text], show_progress_bar=False, normalize_embeddings=True)
        return emb[0].tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


# ---------------------------------------------------------------------------
# Simple In-Memory Vector Store
# ---------------------------------------------------------------------------


class SimpleVectorStore(VectorStore):
    """Minimal in-memory vector store for the demo."""

    def __init__(self, docs: list[Document], embeddings_model: LocalEmbeddings) -> None:
        self._docs = docs
        self._embeddings_model = embeddings_model
        texts = [d.page_content for d in docs]
        print(f"  Embedding {len(texts)} documents...")
        self._embs = np.asarray(
            embeddings_model.embed_documents(texts), dtype=np.float32
        )
        print(f"  Done. Embedding shape: {self._embs.shape}")

    def _top_k(self, query: str, k: int) -> list[Document]:
        q_emb = np.asarray(self._embeddings_model.embed_query(query), dtype=np.float32)
        sims = cosine_similarity([q_emb], self._embs)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [self._docs[i] for i in idx]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return self._top_k(query, k)

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return self._top_k(query, k)

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        raise NotImplementedError

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Any, metadatas: Optional[List[dict]] = None, **kwargs: Any) -> "SimpleVectorStore":
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_hotpotqa(n: int = N_CORPUS_QUESTIONS, seed: int = SEED) -> tuple[list[str], list[str], list[str]]:
    """Load HotPotQA and return (questions, answers, supporting_facts_texts).

    Each supporting fact text is a concatenation of the sentences that serve
    as evidence for the answer.
    """
    print(f"\nLoading HotPotQA dataset (first {n} examples)...")
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "fullwiki", split="train", trust_remote_code=True)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    subset = ds.select(indices.tolist())

    questions = []
    answers = []
    corpus_texts = []

    for row in subset:
        q = row["question"]
        a = row["answer"]
        # Build context from the supporting facts
        context_parts = []
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            text = f"{title}: {' '.join(sentences)}"
            context_parts.append(text)

        if context_parts and a:
            questions.append(q)
            answers.append(a)
            corpus_texts.append("\n".join(context_parts))

    print(f"  Loaded {len(questions)} questions with supporting facts")
    return questions, answers, corpus_texts


# ---------------------------------------------------------------------------
# Simple answer function (no LLM — extractive baseline)
# ---------------------------------------------------------------------------


def find_answer_in_docs(question: str, docs: list[Document], ground_truth: str) -> str:
    """Simple extractive answering: check if ground truth appears in retrieved docs.

    This is a proxy for LLM-based answering — if the correct information is in
    the retrieved documents, we assume the LLM would answer correctly.
    Returns the ground truth if found in docs, otherwise returns a wrong answer.
    """
    combined = " ".join(d.page_content.lower() for d in docs)
    gt_lower = ground_truth.lower().strip()

    # Check if the answer (or key parts of it) appear in retrieved docs
    if gt_lower in combined:
        return ground_truth
    # Check partial match (for multi-word answers, check if most words appear)
    gt_words = gt_lower.split()
    if len(gt_words) > 1:
        matches = sum(1 for w in gt_words if w in combined)
        if matches / len(gt_words) >= 0.7:
            return ground_truth
    return "WRONG_ANSWER"


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    method: str
    accuracy: float
    mean_k: float
    median_k: float
    total_tokens: int
    estimated_cost: float
    acc_per_dollar: float
    k_values: list[int]
    k_dist: Counter


def evaluate_fixed_k(
    k: int,
    questions: list[str],
    answers: list[str],
    vectorstore: SimpleVectorStore,
) -> EvalResult:
    correct = 0
    total_tokens = 0
    k_values = []

    for q, a in zip(questions, answers):
        docs = vectorstore.similarity_search(q, k=k)
        result = find_answer_in_docs(q, docs, a)
        if result == a:
            correct += 1
        # Estimate tokens: question + all doc contents + answer
        input_text = q + " " + " ".join(d.page_content for d in docs)
        total_tokens += int(len(input_text.split()) * 1.3) + 1
        total_tokens += int(len(result.split()) * 1.3) + 1
        k_values.append(k)

    n = len(questions)
    acc = correct / n
    # GPT-4o-mini pricing
    cost = total_tokens * 1.10 / 1_000_000
    apd = acc / cost if cost > 0 else 0

    return EvalResult(
        method=f"fixed_k={k}",
        accuracy=acc,
        mean_k=float(k),
        median_k=float(k),
        total_tokens=total_tokens,
        estimated_cost=cost,
        acc_per_dollar=apd,
        k_values=k_values,
        k_dist=Counter(k_values),
    )


def evaluate_dynamic(
    estimator_name: str,
    questions: list[str],
    answers: list[str],
    vectorstore: SimpleVectorStore,
    embeddings: LocalEmbeddings,
    null_dist: Any,
    max_k: int = MAX_K,
    alpha: float = ALPHA,
) -> EvalResult:
    from ragnaros import DynamicRetriever

    retriever = DynamicRetriever.from_vectorstore(
        vectorstore=vectorstore,
        embeddings=embeddings,
        null_distribution=null_dist,
        estimator=estimator_name,
        alpha=alpha,
        max_k=max_k,
        max_candidates=50,
    )

    correct = 0
    total_tokens = 0
    k_values = []

    for q, a in zip(questions, answers):
        docs = retriever.invoke(q)
        k_used = len(docs)
        result = find_answer_in_docs(q, docs, a)
        if result == a:
            correct += 1
        input_text = q + " " + " ".join(d.page_content for d in docs)
        total_tokens += int(len(input_text.split()) * 1.3) + 1
        total_tokens += int(len(result.split()) * 1.3) + 1
        k_values.append(k_used)

    n = len(questions)
    acc = correct / n
    cost = total_tokens * 1.10 / 1_000_000
    apd = acc / cost if cost > 0 else 0

    return EvalResult(
        method=estimator_name,
        accuracy=acc,
        mean_k=float(np.mean(k_values)),
        median_k=float(np.median(k_values)),
        total_tokens=total_tokens,
        estimated_cost=cost,
        acc_per_dollar=apd,
        k_values=k_values,
        k_dist=Counter(k_values),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def generate_visualizations(
    results: list[EvalResult],
    null_values: np.ndarray,
    real_sims: np.ndarray,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir.mkdir(exist_ok=True)

    # --- Color palette ---
    DYNAMIC_COLORS = {
        "higher_criticism": "#e74c3c",
        "benjamini_hochberg": "#2ecc71",
        "bonferroni": "#3498db",
    }
    FIXED_COLOR = "#95a5a6"

    def get_color(name: str) -> str:
        for key, color in DYNAMIC_COLORS.items():
            if key in name.lower():
                return color
        return FIXED_COLOR

    # ================================================================
    # 1. Cost vs Accuracy
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    fixed = [r for r in results if r.method.startswith("fixed_k=")]
    dynamic = [r for r in results if not r.method.startswith("fixed_k=")]

    if fixed:
        fixed_sorted = sorted(fixed, key=lambda r: r.estimated_cost)
        ax.plot(
            [r.estimated_cost for r in fixed_sorted],
            [r.accuracy for r in fixed_sorted],
            color=FIXED_COLOR, linewidth=2, linestyle="--", zorder=1, label="Fixed-k",
        )
        ax.scatter(
            [r.estimated_cost for r in fixed_sorted],
            [r.accuracy for r in fixed_sorted],
            color=FIXED_COLOR, s=80, zorder=2, edgecolors="white", linewidth=0.5,
        )
        for r in fixed_sorted:
            k_val = r.method.split("=")[-1]
            ax.annotate(
                f"k={k_val}", (r.estimated_cost, r.accuracy),
                textcoords="offset points", xytext=(6, 6), fontsize=8, color=FIXED_COLOR,
            )

    for r in dynamic:
        color = get_color(r.method)
        label = {"higher_criticism": "Higher Criticism", "benjamini_hochberg": "Benjamini-Hochberg", "bonferroni": "Bonferroni"}.get(r.method, r.method)
        ax.scatter(
            r.estimated_cost, r.accuracy,
            color=color, s=160, zorder=3, label=label, edgecolors="white", linewidth=1,
        )
        ax.annotate(
            label, (r.estimated_cost, r.accuracy),
            textcoords="offset points", xytext=(10, -5), fontsize=10,
            color=color, fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
        )

    ax.set_xlabel("Estimated Cost (USD per 100 questions)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. Cost — Fixed-k vs. Dynamic Estimators", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "cost_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'cost_vs_accuracy.png'}")

    # ================================================================
    # 2. k Distribution for dynamic methods
    # ================================================================
    if dynamic:
        fig, ax = plt.subplots(figsize=(10, 6))
        all_ks = sorted(set(k for r in dynamic for k in r.k_dist.keys()))
        n_methods = len(dynamic)
        x = np.arange(len(all_ks))
        width = 0.8 / max(n_methods, 1)

        for i, r in enumerate(dynamic):
            counts = [r.k_dist.get(k, 0) for k in all_ks]
            total = sum(counts)
            fractions = [c / total if total else 0 for c in counts]
            offset = (i - n_methods / 2 + 0.5) * width
            label = {"higher_criticism": "Higher Criticism", "benjamini_hochberg": "Benjamini-Hochberg", "bonferroni": "Bonferroni"}.get(r.method, r.method)
            ax.bar(x + offset, fractions, width=width, label=label, color=get_color(r.method), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in all_ks])
        ax.set_xlabel("k (documents retrieved)", fontsize=12)
        ax.set_ylabel("Fraction of queries", fontsize=12)
        ax.set_title("Distribution of k Values per Dynamic Method", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / "k_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / 'k_distribution.png'}")

    # ================================================================
    # 3. Null vs Real similarity distribution
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null_values, bins=50, density=True, alpha=0.6, color="#3498db", label="Null (unrelated queries)")
    ax.hist(real_sims, bins=50, density=True, alpha=0.6, color="#e74c3c", label="Real (relevant queries)")
    try:
        from scipy.stats import gaussian_kde
        xr = np.linspace(min(null_values.min(), real_sims.min()) - 0.05, max(null_values.max(), real_sims.max()) + 0.05, 300)
        ax.plot(xr, gaussian_kde(null_values)(xr), color="#2980b9", linewidth=2)
        ax.plot(xr, gaussian_kde(real_sims)(xr), color="#c0392b", linewidth=2)
    except ImportError:
        pass
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Null vs. Real Similarity Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "null_vs_real.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'null_vs_real.png'}")

    # ================================================================
    # 4. Accuracy per Dollar bar chart
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_results = sorted(results, key=lambda r: r.acc_per_dollar)
    names = []
    for r in sorted_results:
        label = {"higher_criticism": "Higher Criticism", "benjamini_hochberg": "Benjamini-Hochberg", "bonferroni": "Bonferroni"}.get(r.method, r.method)
        names.append(label)
    values = [r.acc_per_dollar for r in sorted_results]
    colors = [get_color(r.method) for r in sorted_results]
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Accuracy per Dollar", fontsize=12)
    ax.set_title("Cost Efficiency — Accuracy per Dollar", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'efficiency.png'}")

    # ================================================================
    # 5. Token savings comparison
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    # Compare tokens: use max fixed-k as baseline
    max_fixed = max(fixed, key=lambda r: r.total_tokens) if fixed else None
    if max_fixed:
        baseline_tokens = max_fixed.total_tokens
        all_results = sorted(results, key=lambda r: r.total_tokens, reverse=True)
        names = []
        savings = []
        colors_bar = []
        for r in all_results:
            label = {"higher_criticism": "Higher Criticism", "benjamini_hochberg": "Benjamini-Hochberg", "bonferroni": "Bonferroni"}.get(r.method, r.method)
            names.append(label)
            savings.append(r.total_tokens)
            colors_bar.append(get_color(r.method))
        y = np.arange(len(names))
        bars = ax.barh(y, savings, color=colors_bar, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Estimated Tokens (per 100 questions)", fontsize=12)
        ax.set_title("Token Usage Comparison", fontsize=14, fontweight="bold")
        # Add percentage labels
        for i, (bar, r) in enumerate(zip(bars, all_results)):
            pct = (1 - r.total_tokens / baseline_tokens) * 100
            if pct > 0:
                ax.text(bar.get_width() + baseline_tokens * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"-{pct:.0f}%", va="center", fontsize=9, color=get_color(r.method), fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / "token_savings.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / 'token_savings.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  RAGnaros Demo — Dynamic k-selection vs Fixed-k")
    print("=" * 60)

    # 1. Load data
    questions, answers, corpus_texts = load_hotpotqa(N_CORPUS_QUESTIONS, SEED)

    # Split: use first N_EVAL_QUESTIONS for evaluation, rest for corpus
    eval_questions = questions[:N_EVAL_QUESTIONS]
    eval_answers = answers[:N_EVAL_QUESTIONS]

    # Build corpus from ALL supporting facts
    print(f"\nBuilding corpus from {len(corpus_texts)} supporting fact sets...")
    docs = [Document(page_content=text) for text in corpus_texts]

    # 2. Initialize embeddings
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embeddings = LocalEmbeddings(EMBEDDING_MODEL)

    # 3. Build vector store
    print("\nBuilding vector store...")
    vectorstore = SimpleVectorStore(docs, embeddings)

    # 4. Build null distribution
    print("\nBuilding null distribution...")
    from ragnaros import NullDistribution

    # Use built-in unrelated corpus for null distribution
    null_dist = NullDistribution.from_builtin_corpus(
        vectorstore=vectorstore,
        embeddings=embeddings,
        n_trials=200,
        k=20,
    )
    print(f"  Null distribution: {len(null_dist.values)} similarity scores")
    print(f"  Mean: {null_dist.values.mean():.4f}, Std: {null_dist.values.std():.4f}")

    # 5. Collect real similarities for visualization
    print("\nCollecting real query similarities...")
    real_sims = []
    for q in eval_questions[:50]:
        q_emb = np.asarray(embeddings.embed_query(q), dtype=np.float32)
        top_docs = vectorstore.similarity_search(q, k=10)
        doc_embs = np.asarray(embeddings.embed_documents([d.page_content for d in top_docs]), dtype=np.float32)
        sims = cosine_similarity([q_emb], doc_embs)[0]
        real_sims.extend(sims.tolist())
    real_sims_arr = np.asarray(real_sims, dtype=np.float32)

    # 6. Run evaluations
    print("\n" + "=" * 60)
    print("  Running evaluations...")
    print("=" * 60)

    results: list[EvalResult] = []

    # Fixed-k baselines
    for k in [1, 3, 5, 7, 10]:
        print(f"\n  Evaluating fixed k={k}...")
        t0 = time.time()
        r = evaluate_fixed_k(k, eval_questions, eval_answers, vectorstore)
        elapsed = time.time() - t0
        results.append(r)
        print(f"    Accuracy: {r.accuracy:.1%} | Tokens: {r.total_tokens:,} | Cost: ${r.estimated_cost:.4f} | Time: {elapsed:.1f}s")

    # Dynamic estimators
    for est in ["higher_criticism", "benjamini_hochberg", "bonferroni"]:
        print(f"\n  Evaluating {est}...")
        t0 = time.time()
        r = evaluate_dynamic(est, eval_questions, eval_answers, vectorstore, embeddings, null_dist)
        elapsed = time.time() - t0
        results.append(r)
        print(f"    Accuracy: {r.accuracy:.1%} | Mean k: {r.mean_k:.1f} | Tokens: {r.total_tokens:,} | Cost: ${r.estimated_cost:.4f} | Time: {elapsed:.1f}s")

    # 7. Print summary table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<25} {'Accuracy':>10} {'Mean k':>8} {'Tokens':>10} {'Cost':>10} {'Acc/$':>10}")
    print("-" * 75)
    for r in results:
        label = {"higher_criticism": "Higher Criticism", "benjamini_hochberg": "Benjamini-Hochberg", "bonferroni": "Bonferroni"}.get(r.method, r.method)
        print(f"{label:<25} {r.accuracy:>9.1%} {r.mean_k:>8.1f} {r.total_tokens:>10,} ${r.estimated_cost:>8.4f} {r.acc_per_dollar:>10.1f}")

    # Token savings
    max_fixed = max((r for r in results if r.method.startswith("fixed_k=")), key=lambda r: r.total_tokens)
    print(f"\n  Token savings vs {max_fixed.method}:")
    for r in results:
        if not r.method.startswith("fixed_k="):
            saving = (1 - r.total_tokens / max_fixed.total_tokens) * 100
            label = {"higher_criticism": "Higher Criticism", "benjamini_hochberg": "Benjamini-Hochberg", "bonferroni": "Bonferroni"}.get(r.method, r.method)
            print(f"    {label}: {saving:.1f}% fewer tokens")

    # 8. Generate visualizations
    print("\n" + "=" * 60)
    print("  Generating visualizations...")
    print("=" * 60)
    generate_visualizations(results, null_dist.values, real_sims_arr, ASSETS_DIR)

    # 9. Save results as JSON for README
    results_json = []
    for r in results:
        results_json.append({
            "method": r.method,
            "accuracy": round(r.accuracy, 4),
            "mean_k": round(r.mean_k, 1),
            "median_k": round(r.median_k, 1),
            "total_tokens": r.total_tokens,
            "estimated_cost": round(r.estimated_cost, 4),
            "acc_per_dollar": round(r.acc_per_dollar, 1),
        })
    results_path = DEMO_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
