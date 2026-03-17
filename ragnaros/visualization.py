"""
Visualization utilities for RAGnaros evaluation results.

All functions return a ``matplotlib.figure.Figure`` and accept an optional
``ax`` argument so they can be embedded in existing notebook layouts.

``matplotlib`` is an optional dependency::

    pip install ragnaros[viz]

If matplotlib is not installed, calling any function raises ``ImportError``
with a helpful installation hint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ragnaros._types import NullDist
    from ragnaros.evaluation.harness import MethodResult


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install ragnaros[viz]"
        ) from e


# ---------------------------------------------------------------------------
# Colour palette (matches the research notebook aesthetics)
# ---------------------------------------------------------------------------

_DYNAMIC_COLORS = {
    "higher_criticism": "#e74c3c",
    "benjamini_hochberg": "#2ecc71",
    "bonferroni": "#3498db",
    "storey_bh": "#9b59b6",
    "local_fdr": "#e67e22",
    "kneedle": "#1abc9c",
    "berk_jones": "#f39c12",
    "beta_mixture": "#e91e63",
}
_FIXED_COLOR = "#95a5a6"


def _get_color(method_name: str) -> str:
    for key, color in _DYNAMIC_COLORS.items():
        if key in method_name.lower():
            return color
    return _FIXED_COLOR


# ---------------------------------------------------------------------------
# 1. Cost vs Accuracy scatter
# ---------------------------------------------------------------------------


def cost_accuracy_plot(
    results: list[MethodResult],
    highlight_methods: list[str] | None = None,
    ax: Axes | None = None,
    title: str = "Accuracy vs. Cost",
) -> Figure:
    """Scatter plot of cost (x-axis) vs accuracy (y-axis) for all methods.

    Fixed-k methods appear in gray; dynamic methods are coloured. Highlighted
    methods receive a text annotation with an arrow.

    Args:
        results: List of :class:`~ragnaros.evaluation.harness.MethodResult`.
        highlight_methods: Method names to annotate on the plot. If ``None``,
            all dynamic methods are highlighted.
        ax: Optional existing ``Axes`` to draw on.
        title: Plot title. Default ``"Accuracy vs. Cost"``.

    Returns:
        The ``matplotlib.figure.Figure`` containing the plot.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9, 6))

    # Separate fixed-k and dynamic results
    fixed = [r for r in results if r.method_name.startswith("fixed_k=")]
    dynamic = [r for r in results if not r.method_name.startswith("fixed_k=")]

    # Plot fixed-k as a line + scatter
    if fixed:
        fixed_sorted = sorted(fixed, key=lambda r: r.total_cost_usd)
        ax_.plot(
            [r.total_cost_usd for r in fixed_sorted],
            [r.accuracy for r in fixed_sorted],
            color=_FIXED_COLOR,
            linewidth=1.5,
            linestyle="--",
            zorder=1,
            label="Fixed-k",
        )
        ax_.scatter(
            [r.total_cost_usd for r in fixed_sorted],
            [r.accuracy for r in fixed_sorted],
            color=_FIXED_COLOR,
            s=60,
            zorder=2,
        )
        # Annotate a few k values
        for r in fixed_sorted[::3]:
            k_val = r.method_name.split("=")[-1]
            ax_.annotate(
                f"k={k_val}",
                (r.total_cost_usd, r.accuracy),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=_FIXED_COLOR,
            )

    # Plot dynamic methods
    if highlight_methods is None:
        highlight_methods = [r.method_name for r in dynamic]

    for r in dynamic:
        color = _get_color(r.method_name)
        ax_.scatter(
            r.total_cost_usd,
            r.accuracy,
            color=color,
            s=120,
            zorder=3,
            label=r.method_name,
        )
        if r.method_name in (highlight_methods or []):
            ax_.annotate(
                r.method_name,
                (r.total_cost_usd, r.accuracy),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
                color=color,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
            )

    ax_.set_xlabel("Estimated Cost (USD)", fontsize=11)
    ax_.set_ylabel("Accuracy", fontsize=11)
    ax_.set_title(title, fontsize=13, fontweight="bold")
    ax_.yaxis.set_major_formatter(
        __import__("matplotlib").ticker.PercentFormatter(xmax=1.0)
    )
    ax_.legend(loc="lower right", fontsize=9)
    ax_.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. k distribution bar chart
# ---------------------------------------------------------------------------


def k_distribution_plot(
    results: list[MethodResult],
    ax: Axes | None = None,
    title: str = "k Distribution per Method",
) -> Figure:
    """Bar chart showing the distribution of k values used per dynamic method.

    Args:
        results: List of :class:`~ragnaros.evaluation.harness.MethodResult`.
            Fixed-k methods are skipped automatically.
        ax: Optional existing ``Axes``.
        title: Plot title.

    Returns:
        The ``matplotlib.figure.Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    dynamic = [r for r in results if not r.method_name.startswith("fixed_k=")]
    if not dynamic:
        raise ValueError("No dynamic-k methods found in results.")

    # Find the full range of k values across all methods.
    all_ks = sorted(
        set(k for r in dynamic for k in r.k_dist.keys())
    )

    n_methods = len(dynamic)
    x = np.arange(len(all_ks))
    width = 0.8 / max(n_methods, 1)

    fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 5))

    for i, r in enumerate(dynamic):
        counts = [r.k_dist.get(k, 0) for k in all_ks]
        total = sum(counts)
        fractions = [c / total if total else 0 for c in counts]
        offset = (i - n_methods / 2 + 0.5) * width
        ax_.bar(
            x + offset,
            fractions,
            width=width,
            label=r.method_name,
            color=_get_color(r.method_name),
            alpha=0.85,
        )

    ax_.set_xticks(x)
    ax_.set_xticklabels([str(k) for k in all_ks])
    ax_.set_xlabel("k (documents retrieved)", fontsize=11)
    ax_.set_ylabel("Fraction of queries", fontsize=11)
    ax_.set_title(title, fontsize=13, fontweight="bold")
    ax_.yaxis.set_major_formatter(
        __import__("matplotlib").ticker.PercentFormatter(xmax=1.0)
    )
    ax_.legend(fontsize=9)
    ax_.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Null vs real similarity distributions
# ---------------------------------------------------------------------------


def null_vs_real_plot(
    null_distribution: NullDist,
    real_similarities: list[float] | NullDist,
    labels: tuple[str, str] = ("Null (unrelated)", "Real (relevant)"),
    ax: Axes | None = None,
    title: str = "Null vs. Real Similarity Distribution",
    bins: int = 50,
) -> Figure:
    """Histogram overlay comparing the null distribution to query-time similarities.

    Use this to visually verify that the null distribution is well-separated
    from real query similarities.

    Args:
        null_distribution: :class:`~ragnaros.null_distribution.NullDistribution`
            or raw 1-D array of background similarity scores.
        real_similarities: Similarity scores from actual queries (pass the
            cosine similarities from a real evaluation run).
        labels: Legend labels for the two distributions.
        ax: Optional existing ``Axes``.
        title: Plot title.
        bins: Histogram bins. Default 50.

    Returns:
        The ``matplotlib.figure.Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    # Accept both NullDistribution objects and raw arrays
    null_vals = (
        null_distribution.values
        if hasattr(null_distribution, "values")
        else np.asarray(null_distribution)
    )
    real_vals = np.asarray(real_similarities)

    fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9, 5))

    ax_.hist(
        null_vals,
        bins=bins,
        density=True,
        alpha=0.6,
        color="#3498db",
        label=labels[0],
    )
    ax_.hist(
        real_vals,
        bins=bins,
        density=True,
        alpha=0.6,
        color="#e74c3c",
        label=labels[1],
    )

    # Overlay KDE if scipy is available
    try:
        from scipy.stats import gaussian_kde

        x = np.linspace(
            min(null_vals.min(), real_vals.min()) - 0.05,
            max(null_vals.max(), real_vals.max()) + 0.05,
            300,
        )
        ax_.plot(x, gaussian_kde(null_vals)(x), color="#2980b9", linewidth=2)
        ax_.plot(x, gaussian_kde(real_vals)(x), color="#c0392b", linewidth=2)
    except ImportError:
        pass

    ax_.set_xlabel("Cosine Similarity", fontsize=11)
    ax_.set_ylabel("Density", fontsize=11)
    ax_.set_title(title, fontsize=13, fontweight="bold")
    ax_.legend(fontsize=10)
    ax_.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Accuracy-per-dollar comparison bar chart
# ---------------------------------------------------------------------------


def efficiency_plot(
    results: list[MethodResult],
    ax: Axes | None = None,
    title: str = "Accuracy per Dollar",
) -> Figure:
    """Horizontal bar chart of accuracy-per-dollar for all methods.

    Args:
        results: List of :class:`~ragnaros.evaluation.harness.MethodResult`.
        ax: Optional existing ``Axes``.
        title: Plot title.

    Returns:
        The ``matplotlib.figure.Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    sorted_results = sorted(results, key=lambda r: r.accuracy_per_dollar)
    names = [r.method_name for r in sorted_results]
    values = [r.accuracy_per_dollar for r in sorted_results]
    colors = [_get_color(n) for n in names]

    fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(
        figsize=(9, max(4, len(names) * 0.5))
    )

    y = np.arange(len(names))
    ax_.barh(y, values, color=colors, alpha=0.85)
    ax_.set_yticks(y)
    ax_.set_yticklabels(names, fontsize=10)
    ax_.set_xlabel("Accuracy per Dollar", fontsize=11)
    ax_.set_title(title, fontsize=13, fontweight="bold")
    ax_.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Estimator comparison heatmap
# ---------------------------------------------------------------------------


def comparison_heatmap(
    results: list[MethodResult],
    metrics: list[str] | None = None,
    ax: Axes | None = None,
    title: str = "Estimator Comparison",
) -> Figure:
    """Heatmap comparing all methods across multiple metrics.

    Columns are methods, rows are metrics. Cell values are normalised
    to [0, 1] within each row, and the colour intensity reflects performance
    (higher is better for all metrics).

    Args:
        results: List of :class:`~ragnaros.evaluation.harness.MethodResult`.
        metrics: Metric names to include. Defaults to
            ``["accuracy", "mean_k", "total_cost_usd", "accuracy_per_dollar"]``.
            ``mean_k`` and ``total_cost_usd`` are inverted so "lower is better"
            becomes "higher is better" in the heatmap.
        ax: Optional existing ``Axes``.
        title: Plot title.

    Returns:
        The ``matplotlib.figure.Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    if metrics is None:
        metrics = ["accuracy", "mean_k", "total_cost_usd", "accuracy_per_dollar"]

    names = [r.method_name for r in results]
    # Build raw data matrix: rows = metrics, cols = methods
    raw = np.zeros((len(metrics), len(results)))
    for j, r in enumerate(results):
        for i, m in enumerate(metrics):
            raw[i, j] = getattr(r, m)

    # Normalise each row to [0, 1] — invert "lower is better" metrics
    invert = {"mean_k", "total_cost_usd"}
    normed = np.zeros_like(raw)
    for i, m in enumerate(metrics):
        row = raw[i]
        rmin, rmax = row.min(), row.max()
        if rmax - rmin < 1e-10:
            normed[i] = 0.5
        else:
            normed[i] = (row - rmin) / (rmax - rmin)
        if m in invert:
            normed[i] = 1 - normed[i]

    fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(
        figsize=(max(8, len(names) * 1.2), max(4, len(metrics) * 0.8))
    )

    im = ax_.imshow(normed, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Labels
    ax_.set_xticks(range(len(names)))
    ax_.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax_.set_yticks(range(len(metrics)))

    display_names = {
        "accuracy": "Accuracy ↑",
        "mean_k": "Mean k ↓",
        "total_cost_usd": "Cost ($) ↓",
        "accuracy_per_dollar": "Acc/$ ↑",
    }
    ax_.set_yticklabels([display_names.get(m, m) for m in metrics], fontsize=10)

    # Annotate cells with raw values
    for i in range(len(metrics)):
        for j in range(len(names)):
            val = raw[i, j]
            if metrics[i] == "accuracy":
                txt = f"{val:.1%}"
            elif metrics[i] == "total_cost_usd":
                txt = f"${val:.4f}"
            elif metrics[i] == "accuracy_per_dollar":
                txt = f"{val:.1f}"
            else:
                txt = f"{val:.2f}"
            ax_.text(j, i, txt, ha="center", va="center", fontsize=8, fontweight="bold")

    ax_.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax_, label="Normalised Score", shrink=0.8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Sensitivity analysis plot (alpha vs k)
# ---------------------------------------------------------------------------


def sensitivity_plot(
    question_emb,
    doc_embs: list,
    null_distribution,
    estimator_names: list[str] | None = None,
    alpha_range: tuple[float, float] = (0.001, 0.20),
    n_points: int = 30,
    max_k: int = 10,
    ax: Axes | None = None,
    title: str = "Sensitivity Analysis: Alpha vs. Selected k",
) -> Figure:
    """Line plot showing how each estimator's k changes as alpha varies.

    Useful for understanding estimator stability and choosing a good alpha.

    Args:
        question_emb: Query embedding.
        doc_embs: Candidate document embeddings.
        null_distribution: NullDistribution object or raw array.
        estimator_names: Estimator names to plot. Defaults to all registered.
        alpha_range: (min_alpha, max_alpha) range to sweep.
        n_points: Number of alpha values to evaluate. Default 30.
        max_k: Maximum k. Default 10.
        ax: Optional existing ``Axes``.
        title: Plot title.

    Returns:
        The ``matplotlib.figure.Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    from ragnaros.estimators import ESTIMATORS

    null_vals = (
        null_distribution.values
        if hasattr(null_distribution, "values")
        else np.asarray(null_distribution)
    )

    if estimator_names is None:
        estimator_names = list(ESTIMATORS.keys())

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)

    fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6))

    for name in estimator_names:
        fn = ESTIMATORS[name]
        ks = [fn(question_emb, doc_embs, null_vals, alpha=a, max_k=max_k) for a in alphas]
        ax_.plot(alphas, ks, label=name, color=_get_color(name), linewidth=2, marker="o", markersize=3)

    ax_.set_xlabel("Alpha (significance level)", fontsize=11)
    ax_.set_ylabel("Selected k", fontsize=11)
    ax_.set_title(title, fontsize=13, fontweight="bold")
    ax_.legend(fontsize=9, loc="upper left")
    ax_.set_ylim(0, max_k + 1)
    ax_.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
