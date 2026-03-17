"""
RAGnaros Demo — Dynamic k-selection vs Fixed-k on HotPotQA

This demo:
1. Loads 500 HotPotQA questions with supporting facts as corpus
2. Builds an in-memory vector store with local sentence-transformers embeddings
3. Constructs a null distribution from unrelated queries
4. Compares Fixed-k (k=1, 3, 5, 7, 10) vs ALL 8 dynamic estimators
5. Measures accuracy, token usage, cost, and generates visualizations
6. Produces comparison heatmap and sensitivity analysis plots

No API keys required — everything runs locally.

Estimators tested:
  - Original (v0.1): Higher Criticism, Benjamini-Hochberg, Bonferroni
  - Research extension (v0.2): Storey-BH, Local FDR, Kneedle, Berk-Jones, Beta Mixture
"""

from __future__ import annotations

import json
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

# All estimators to test
ESTIMATOR_NAMES = [
    "higher_criticism",
    "benjamini_hochberg",
    "bonferroni",
    "storey_bh",
    "local_fdr",
    "kneedle",
    "berk_jones",
    "beta_mixture",
]

# Pretty display names
DISPLAY_NAMES = {
    "higher_criticism": "Higher Criticism",
    "benjamini_hochberg": "Benjamini-Hochberg",
    "bonferroni": "Bonferroni",
    "storey_bh": "Storey-BH",
    "local_fdr": "Local FDR",
    "kneedle": "Kneedle",
    "berk_jones": "Berk-Jones",
    "beta_mixture": "Beta Mixture",
}


# ---------------------------------------------------------------------------
# Local Embeddings wrapper (sentence-transformers -> LangChain Embeddings)
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
    """Load HotPotQA and return (questions, answers, supporting_facts_texts)."""
    print(f"\nLoading HotPotQA dataset (first {n} examples)...")
    from datasets import load_dataset

    try:
        ds = load_dataset("hotpot_qa", "fullwiki", split="train")
    except Exception:
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    subset = ds.select(indices.tolist())

    questions = []
    answers = []
    corpus_texts = []

    for row in subset:
        q = row["question"]
        a = row["answer"]
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
# Simple answer function (no LLM -- extractive baseline)
# ---------------------------------------------------------------------------


def find_answer_in_docs(question: str, docs: list[Document], ground_truth: str) -> str:
    """Simple extractive answering: check if ground truth appears in retrieved docs."""
    combined = " ".join(d.page_content.lower() for d in docs)
    gt_lower = ground_truth.lower().strip()

    if gt_lower in combined:
        return ground_truth
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
        input_text = q + " " + " ".join(d.page_content for d in docs)
        total_tokens += int(len(input_text.split()) * 1.3) + 1
        total_tokens += int(len(result.split()) * 1.3) + 1
        k_values.append(k)

    n = len(questions)
    acc = correct / n
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
    sample_query_emb: np.ndarray,
    sample_doc_embs: list[np.ndarray],
    null_dist_obj: Any,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir.mkdir(exist_ok=True)

    # Color palette
    DYNAMIC_COLORS = {
        "higher_criticism": "#e74c3c",
        "benjamini_hochberg": "#2ecc71",
        "bonferroni": "#3498db",
        "storey_bh": "#9b59b6",
        "local_fdr": "#e67e22",
        "kneedle": "#1abc9c",
        "berk_jones": "#f39c12",
        "beta_mixture": "#e91e63",
    }
    FIXED_COLOR = "#95a5a6"

    def get_color(name: str) -> str:
        for key, color in DYNAMIC_COLORS.items():
            if key in name.lower():
                return color
        return FIXED_COLOR

    def get_display_name(method: str) -> str:
        return DISPLAY_NAMES.get(method, method)

    # ================================================================
    # 1. Cost vs Accuracy
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

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

    for i, r in enumerate(dynamic):
        color = get_color(r.method)
        label = get_display_name(r.method)
        ax.scatter(
            r.estimated_cost, r.accuracy,
            color=color, s=160, zorder=3, label=label, edgecolors="white", linewidth=1,
        )

    ax.set_xlabel("Estimated Cost (USD per 100 questions)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. Cost — All Estimators", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "cost_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: cost_vs_accuracy.png")

    # ================================================================
    # 2. k Distribution for dynamic methods
    # ================================================================
    if dynamic:
        fig, ax = plt.subplots(figsize=(14, 7))
        all_ks = sorted(set(k for r in dynamic for k in r.k_dist.keys()))
        n_methods = len(dynamic)
        x = np.arange(len(all_ks))
        width = 0.8 / max(n_methods, 1)

        for i, r in enumerate(dynamic):
            counts = [r.k_dist.get(k, 0) for k in all_ks]
            total = sum(counts)
            fractions = [c / total if total else 0 for c in counts]
            offset = (i - n_methods / 2 + 0.5) * width
            ax.bar(x + offset, fractions, width=width, label=get_display_name(r.method),
                   color=get_color(r.method), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in all_ks])
        ax.set_xlabel("k (documents retrieved)", fontsize=12)
        ax.set_ylabel("Fraction of queries", fontsize=12)
        ax.set_title("Distribution of k Values — All Dynamic Methods", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / "k_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: k_distribution.png")

    # ================================================================
    # 3. Null vs Real similarity distribution
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null_values, bins=50, density=True, alpha=0.6, color="#3498db", label="Null (unrelated queries)")
    ax.hist(real_sims, bins=50, density=True, alpha=0.6, color="#e74c3c", label="Real (relevant queries)")
    try:
        from scipy.stats import gaussian_kde
        xr = np.linspace(
            min(null_values.min(), real_sims.min()) - 0.05,
            max(null_values.max(), real_sims.max()) + 0.05, 300,
        )
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
    print(f"  Saved: null_vs_real.png")

    # ================================================================
    # 4. Accuracy per Dollar bar chart
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.5)))
    sorted_results = sorted(results, key=lambda r: r.acc_per_dollar)
    names = [get_display_name(r.method) for r in sorted_results]
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
    print(f"  Saved: efficiency.png")

    # ================================================================
    # 5. Token savings comparison
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.5)))
    max_fixed = max(fixed, key=lambda r: r.total_tokens) if fixed else None
    if max_fixed:
        baseline_tokens = max_fixed.total_tokens
        all_results = sorted(results, key=lambda r: r.total_tokens, reverse=True)
        names = [get_display_name(r.method) for r in all_results]
        savings = [r.total_tokens for r in all_results]
        colors_bar = [get_color(r.method) for r in all_results]
        y = np.arange(len(names))
        bars = ax.barh(y, savings, color=colors_bar, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Estimated Tokens (per 100 questions)", fontsize=12)
        ax.set_title("Token Usage Comparison", fontsize=14, fontweight="bold")
        for bar_item, r in zip(bars, all_results):
            pct = (1 - r.total_tokens / baseline_tokens) * 100
            if pct > 0:
                ax.text(bar_item.get_width() + baseline_tokens * 0.01,
                        bar_item.get_y() + bar_item.get_height() / 2,
                        f"-{pct:.0f}%", va="center", fontsize=9,
                        color=get_color(r.method), fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / "token_savings.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: token_savings.png")

    # ================================================================
    # 6. Comparison Heatmap (NEW)
    # ================================================================
    fig, ax = plt.subplots(figsize=(max(10, len(results) * 1.0), 5))
    metrics_names = ["Accuracy", "Mean k", "Cost ($)", "Acc/$"]
    raw_data = np.zeros((4, len(results)))
    for j, r in enumerate(results):
        raw_data[0, j] = r.accuracy
        raw_data[1, j] = r.mean_k
        raw_data[2, j] = r.estimated_cost
        raw_data[3, j] = r.acc_per_dollar

    # Normalize to [0,1] per metric; invert "lower is better" rows
    normed = np.zeros_like(raw_data)
    invert_rows = {1, 2}  # mean_k and cost: lower is better
    for i in range(4):
        row = raw_data[i]
        rmin, rmax = row.min(), row.max()
        if rmax - rmin < 1e-10:
            normed[i] = 0.5
        else:
            normed[i] = (row - rmin) / (rmax - rmin)
        if i in invert_rows:
            normed[i] = 1 - normed[i]

    im = ax.imshow(normed, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    method_labels = [get_display_name(r.method) for r in results]
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Accuracy ↑", "Mean k ↓", "Cost ($) ↓", "Acc/$ ↑"], fontsize=10)
    for i in range(4):
        for j in range(len(results)):
            val = raw_data[i, j]
            if i == 0:
                txt = f"{val:.1%}"
            elif i == 2:
                txt = f"${val:.4f}"
            elif i == 3:
                txt = f"{val:.0f}"
            else:
                txt = f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_title("Estimator Comparison Heatmap", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Normalised Score", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: comparison_heatmap.png")

    # ================================================================
    # 7. Sensitivity Analysis — alpha vs k (NEW)
    # ================================================================
    from ragnaros.estimators import ESTIMATORS as EST_REGISTRY

    fig, ax = plt.subplots(figsize=(12, 7))
    alphas = np.linspace(0.001, 0.20, 40)
    for est_name in ESTIMATOR_NAMES:
        fn = EST_REGISTRY[est_name]
        ks = [fn(sample_query_emb, sample_doc_embs, null_values, alpha=a, max_k=MAX_K)
              for a in alphas]
        ax.plot(alphas, ks, label=get_display_name(est_name),
                color=get_color(est_name), linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Alpha (significance level)", fontsize=12)
    ax.set_ylabel("Selected k", fontsize=12)
    ax.set_title("Sensitivity Analysis: How Alpha Affects k Selection", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.set_ylim(0, MAX_K + 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: sensitivity_alpha.png")

    # ================================================================
    # 8. Estimator Family Grouping (NEW)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Group 1: Hypothesis Testing (HC, BH, Bonferroni, Storey-BH)
    group1 = ["higher_criticism", "benjamini_hochberg", "bonferroni", "storey_bh"]
    # Group 2: Bayesian / Model-Based (Local FDR, Beta Mixture)
    group2 = ["local_fdr", "beta_mixture"]
    # Group 3: Geometric / Non-Parametric (Kneedle, Berk-Jones)
    group3 = ["kneedle", "berk_jones"]

    groups = [
        ("Hypothesis Testing", group1),
        ("Bayesian / Model-Based", group2),
        ("Geometric / Non-Parametric", group3),
    ]

    for ax_i, (title, members) in zip(axes, groups):
        member_results = [r for r in results if r.method in members]
        if not member_results:
            continue
        names_g = [get_display_name(r.method) for r in member_results]
        accs = [r.accuracy for r in member_results]
        costs = [r.estimated_cost for r in member_results]
        colors_g = [get_color(r.method) for r in member_results]

        ax_i.bar(range(len(names_g)), accs, color=colors_g, alpha=0.85, edgecolor="white")
        ax_i.set_xticks(range(len(names_g)))
        ax_i.set_xticklabels(names_g, rotation=30, ha="right", fontsize=9)
        ax_i.set_ylabel("Accuracy", fontsize=11)
        ax_i.set_title(title, fontsize=12, fontweight="bold")
        ax_i.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_i.set_ylim(0, 1.0)
        ax_i.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Add cost labels on bars
        for j, (acc_val, cost_val) in enumerate(zip(accs, costs)):
            ax_i.text(j, acc_val + 0.01, f"${cost_val:.4f}", ha="center", fontsize=8, color="#555")

    fig.suptitle("Accuracy by Estimator Family (with cost labels)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "estimator_families.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: estimator_families.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  RAGnaros v0.2 Demo — 8 Estimators vs Fixed-k on HotPotQA")
    print("=" * 70)

    # 1. Load data
    questions, answers, corpus_texts = load_hotpotqa(N_CORPUS_QUESTIONS, SEED)

    eval_questions = questions[:N_EVAL_QUESTIONS]
    eval_answers = answers[:N_EVAL_QUESTIONS]

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
    sample_query_emb = None
    sample_doc_embs = []
    for idx, q in enumerate(eval_questions[:50]):
        q_emb = np.asarray(embeddings.embed_query(q), dtype=np.float32)
        top_docs = vectorstore.similarity_search(q, k=10)
        doc_embs = [np.asarray(e, dtype=np.float32)
                    for e in embeddings.embed_documents([d.page_content for d in top_docs])]
        sims = cosine_similarity([q_emb], doc_embs)[0]
        real_sims.extend(sims.tolist())
        # Save first query for sensitivity analysis
        if idx == 0:
            sample_query_emb = q_emb
            sample_doc_embs = doc_embs
    real_sims_arr = np.asarray(real_sims, dtype=np.float32)

    # 6. Run evaluations
    print("\n" + "=" * 70)
    print("  Running evaluations...")
    print("=" * 70)

    results: list[EvalResult] = []

    # Fixed-k baselines
    for k in [1, 3, 5, 7, 10]:
        print(f"\n  Evaluating fixed k={k}...")
        t0 = time.time()
        r = evaluate_fixed_k(k, eval_questions, eval_answers, vectorstore)
        elapsed = time.time() - t0
        results.append(r)
        print(f"    Accuracy: {r.accuracy:.1%} | Tokens: {r.total_tokens:,} | "
              f"Cost: ${r.estimated_cost:.4f} | Time: {elapsed:.1f}s")

    # ALL dynamic estimators
    for est in ESTIMATOR_NAMES:
        display = DISPLAY_NAMES.get(est, est)
        print(f"\n  Evaluating {display}...")
        t0 = time.time()
        r = evaluate_dynamic(est, eval_questions, eval_answers, vectorstore, embeddings, null_dist)
        elapsed = time.time() - t0
        results.append(r)
        print(f"    Accuracy: {r.accuracy:.1%} | Mean k: {r.mean_k:.1f} | "
              f"Tokens: {r.total_tokens:,} | Cost: ${r.estimated_cost:.4f} | Time: {elapsed:.1f}s")

    # 7. Print summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<25} {'Accuracy':>10} {'Mean k':>8} {'Tokens':>10} {'Cost':>10} {'Acc/$':>10}")
    print("-" * 75)

    # Sort by accuracy per dollar for the table
    for r in results:
        label = DISPLAY_NAMES.get(r.method, r.method)
        print(f"{label:<25} {r.accuracy:>9.1%} {r.mean_k:>8.1f} "
              f"{r.total_tokens:>10,} ${r.estimated_cost:>8.4f} {r.acc_per_dollar:>10.1f}")

    # Token savings
    max_fixed = max((r for r in results if r.method.startswith("fixed_k=")),
                    key=lambda r: r.total_tokens)
    print(f"\n  Token savings vs {max_fixed.method}:")
    for r in results:
        if not r.method.startswith("fixed_k="):
            saving = (1 - r.total_tokens / max_fixed.total_tokens) * 100
            label = DISPLAY_NAMES.get(r.method, r.method)
            print(f"    {label}: {saving:.1f}% fewer tokens")

    # Find best estimator
    dynamic_results = [r for r in results if not r.method.startswith("fixed_k=")]
    best_acc = max(dynamic_results, key=lambda r: r.accuracy)
    best_eff = max(dynamic_results, key=lambda r: r.acc_per_dollar)
    cheapest = min(dynamic_results, key=lambda r: r.estimated_cost)

    print(f"\n  Best accuracy:    {DISPLAY_NAMES.get(best_acc.method, best_acc.method)} "
          f"({best_acc.accuracy:.1%})")
    print(f"  Best efficiency:  {DISPLAY_NAMES.get(best_eff.method, best_eff.method)} "
          f"({best_eff.acc_per_dollar:.0f} acc/$)")
    print(f"  Cheapest:         {DISPLAY_NAMES.get(cheapest.method, cheapest.method)} "
          f"(${cheapest.estimated_cost:.4f})")

    # 8. Generate visualizations
    print("\n" + "=" * 70)
    print("  Generating visualizations...")
    print("=" * 70)
    generate_visualizations(
        results, null_dist.values, real_sims_arr,
        sample_query_emb, sample_doc_embs, null_dist,
        ASSETS_DIR,
    )

    # 9. Save results as JSON
    results_json = []
    for r in results:
        results_json.append({
            "method": r.method,
            "display_name": DISPLAY_NAMES.get(r.method, r.method),
            "accuracy": round(r.accuracy, 4),
            "mean_k": round(r.mean_k, 1),
            "median_k": round(r.median_k, 1),
            "total_tokens": r.total_tokens,
            "estimated_cost": round(r.estimated_cost, 4),
            "acc_per_dollar": round(r.acc_per_dollar, 1),
            "k_distribution": dict(r.k_dist),
        })
    results_path = DEMO_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    print("\n" + "=" * 70)
    print("  Demo complete! All 8 estimators evaluated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
