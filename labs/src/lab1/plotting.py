"""Plotting helpers for quantisation and pruning sweep results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# Shared style constants

_COLORS = {
    "ptq": "#2196F3",
    "qat": "#FF5722",
    "random": "#4CAF50",
    "l1-norm": "#9C27B0",
}
_MARKERS = {
    "ptq": "o",
    "qat": "s",
    "random": "o",
    "l1-norm": "s",
}


def _save_and_show(fig: plt.Figure, save_path: Path | None) -> None:
    """Optionally save a figure then display it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [saved] {save_path}")
    plt.show()

# Task 1 plot
def plot_quantization_sweep(
    results: list[dict],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Task 1 plot: PTQ and QAT accuracy vs fixed-point width.

    Parameters
    ----------
    results : list[dict]
        Output of run_quantization_sweep.
    save_path : Path, optional
        If given, the figure is saved as a PNG.
    """
    sorted_results = sorted(results, key=lambda r: r["width"])
    widths = [r["width"] for r in sorted_results]
    ptq = [r["ptq_accuracy"] for r in sorted_results]
    qat = [r["qat_accuracy"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(widths, ptq, f"{_MARKERS['ptq']}-", color=_COLORS["ptq"],
            label="PTQ (post-quantisation)")
    ax.plot(widths, qat, f"{_MARKERS['qat']}-", color=_COLORS["qat"],
            label="QAT (quantisation-aware training)")

    ax.set_xlabel("Fixed-point width (bits)", fontsize=13)
    ax.set_ylabel("IMDb Accuracy", fontsize=13)
    ax.set_title("Task 1: Accuracy vs Fixed-Point Width",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(widths)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_and_show(fig, save_path)
    return fig

# Task 2 plot
def plot_pruning_sweep(
    results: list[dict],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Task 2 plot: post-finetune accuracy vs sparsity, one curve per method.

    Parameters
    ----------
    results : list[dict]
        Output of run_pruning_sweep.
    save_path : Path, optional
        If given, the figure is saved as a PNG.
    """
    methods = sorted(set(r["method"] for r in results))

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        subset = sorted(
            [r for r in results if r["method"] == method],
            key=lambda r: r["sparsity"],
        )
        sparsities = [r["sparsity"] for r in subset]
        accs = [r["post_finetune_accuracy"] for r in subset]

        ax.plot(
            sparsities, accs,
            f"{_MARKERS.get(method, 'o')}-",
            color=_COLORS.get(method),
            label=method,
        )

    ax.set_xlabel("Sparsity", fontsize=13)
    ax.set_ylabel("IMDb Accuracy (post fine-tune)", fontsize=13)
    ax.set_title("Task 2: Accuracy vs Sparsity (Random vs L1-Norm)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_and_show(fig, save_path)
    return fig
