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
