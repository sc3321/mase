"""Visualisation functions and study analysis report."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna

from .analysis import (
    best_so_far_curve,
    best_so_far_curves_by_precision,
    get_completed_trials,
    get_dominant_type,
    get_failed_trials,
    print_trial_detail,
)
from .constants import get_persistence_root
from .search import load_study


PRECISION_COLORS: dict[str, str] = {
    "Linear": "#9E9E9E",
    "LinearInteger": "#2196F3",
    "LinearMinifloatDenorm": "#FF9800",
    "LinearMinifloatIEEE": "#FF5722",
    "LinearLog": "#795548",
    "LinearBlockFP": "#4CAF50",
    "LinearBlockMinifloat": "#00BCD4",
    "LinearBlockLog": "#607D8B",
    "LinearBinary": "#9C27B0",
    "LinearBinaryScaling": "#E91E63",
    "LinearBinaryResidualSign": "#673AB7",
}


def plot_curves(
    curves: dict[str, tuple[list[int], list[float]]],
    title: str = "Search Progress",
    xlabel: str = "Trial",
    ylabel: str = "Best Accuracy So Far",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot one or more cumulative-max accuracy curves on the same axes."""
    fig, ax = plt.subplots(figsize=figsize)

    for label, (xs, ys) in curves.items():
        color = PRECISION_COLORS.get(label)
        ax.plot(
            xs, ys,
            marker="o", markersize=3, linewidth=1.5,
            label=label, color=color,
        )

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    plt.show()
    return fig


def plot_trial_values(
    study: optuna.Study,
    title: str = "Per-Trial Accuracy",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Scatter plot of every trial's accuracy, coloured by dominant precision."""
    trials = get_completed_trials(study)

    fig, ax = plt.subplots(figsize=(10, 6))

    groups: dict[str, tuple[list[int], list[float]]] = {}
    for t in trials:
        dom = get_dominant_type(t)
        groups.setdefault(dom, ([], []))
        groups[dom][0].append(t.number)
        groups[dom][1].append(t.value)

    for label in sorted(groups):
        xs, ys = groups[label]
        color = PRECISION_COLORS.get(label)
        ax.scatter(xs, ys, label=label, color=color, alpha=0.7, s=30)

    ax.set_xlabel("Trial", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    plt.show()
    return fig


def plot_accuracy_per_trial(
    study: optuna.Study,
    title: str = "Accuracy per Trial",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 5),
    color: str = "#2196F3",
) -> plt.Figure:
    """Line-and-scatter plot of each trial's final accuracy."""
    trials = get_completed_trials(study)

    numbers = [t.number for t in trials]
    accuracies = [t.value for t in trials]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        numbers, accuracies,
        color=color, linewidth=1.0, alpha=0.5, zorder=1,
    )
    ax.scatter(
        numbers, accuracies,
        color=color, s=24, zorder=2, label="trial accuracy",
    )

    mean_acc = np.mean(accuracies)
    median_acc = np.median(accuracies)
    ax.axhline(
        mean_acc, color="#FF5722", linestyle="--", linewidth=1.2,
        label=f"mean = {mean_acc:.4f}",
    )
    ax.axhline(
        median_acc, color="#4CAF50", linestyle="--", linewidth=1.2,
        label=f"median = {median_acc:.4f}",
    )

    ax.set_xlabel("Trial", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    plt.show()
    return fig


def plot_precision_comparison(
    studies: dict[str, optuna.Study],
    title: str = "Precision Comparison",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot one cummax curve per study (one study per precision type)."""
    curves = {}
    for label, study in studies.items():
        curves[label] = best_so_far_curve(study)
    return plot_curves(curves, title=title, save_path=save_path)


def plot_accuracy_histogram(
    study: optuna.Study,
    title: str = "Accuracy Distribution",
    bins: int = 20,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Histogram of per-trial accuracy values."""
    trials = get_completed_trials(study)
    values = [t.value for t in trials]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=bins, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(
        np.mean(values), color="#FF5722", linestyle="--", linewidth=1.5,
        label=f"mean={np.mean(values):.4f}",
    )
    ax.axvline(
        np.median(values), color="#4CAF50", linestyle="--", linewidth=1.5,
        label=f"median={np.median(values):.4f}",
    )
    ax.set_xlabel("Accuracy", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_precision_boxplot(
    study: optuna.Study,
    title: str = "Accuracy by Precision Type",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Box plot of accuracy grouped by dominant precision type."""
    trials = get_completed_trials(study)

    groups: dict[str, list[float]] = {}
    for t in trials:
        dom = get_dominant_type(t)
        groups.setdefault(dom, []).append(t.value)

    sorted_labels = sorted(groups, key=lambda k: np.median(groups[k]))
    sorted_data = [groups[k] for k in sorted_labels]
    sorted_colors = [PRECISION_COLORS.get(k, "#9E9E9E") for k in sorted_labels]

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_labels) * 1.2), 6))
    bp = ax.boxplot(
        sorted_data,
        patch_artist=True,
        labels=sorted_labels,
        widths=0.6,
    )
    for patch, color in zip(bp["boxes"], sorted_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def _load_trial_json(study_name: str, trial_number: int) -> dict | None:
    """Load the persisted JSON for a single trial, or *None* if missing."""
    path = get_persistence_root() / study_name / "trials" / f"trial_{trial_number}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_training_loss_curves(
    study: optuna.Study,
    title: str = "Training Loss per Trial",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 6),
    alpha: float | None = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """Overlay every trial's per-step training loss on a single plot."""
    trials = get_completed_trials(study)

    fig, ax = plt.subplots(figsize=figsize)
    colormap = plt.get_cmap(cmap)
    plotted = 0

    for idx, t in enumerate(trials):
        data = _load_trial_json(study.study_name, t.number)
        if data is None or not data.get("log_history"):
            continue

        steps, losses = [], []
        for entry in data["log_history"]:
            if "loss" in entry and "step" in entry:
                steps.append(entry["step"])
                losses.append(entry["loss"])

        if not steps:
            continue

        n = len(trials)
        a = alpha if alpha is not None else max(0.08, min(0.5, 4.0 / n))
        color = colormap(idx / max(len(trials) - 1, 1))

        ax.plot(
            steps, losses,
            color=color, alpha=a, linewidth=0.9,
            label=f"Trial {t.number}" if n <= 20 else None,
        )
        plotted += 1

    if plotted == 0:
        ax.text(
            0.5, 0.5,
            "No training log history found.\n"
            "Re-run the search to start recording per-step losses.",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if plotted > 0 and len(trials) <= 20:
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    elif plotted > 0:
        sm = plt.cm.ScalarMappable(
            cmap=colormap,
            norm=plt.Normalize(
                vmin=trials[0].number, vmax=trials[-1].number,
            ),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Trial number", fontsize=11)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    plt.show()
    return fig


def plot_training_loss_by_precision(
    study: optuna.Study,
    title: str = "Training Loss by Precision Type",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 6),
    alpha: float | None = None,
) -> plt.Figure:
    """Overlay per-step training loss curves, coloured by dominant precision type."""
    trials = get_completed_trials(study)

    fig, ax = plt.subplots(figsize=figsize)
    plotted = 0
    legend_added: set[str] = set()

    for t in trials:
        data = _load_trial_json(study.study_name, t.number)
        if data is None or not data.get("log_history"):
            continue

        steps, losses = [], []
        for entry in data["log_history"]:
            if "loss" in entry and "step" in entry:
                steps.append(entry["step"])
                losses.append(entry["loss"])
        if not steps:
            continue

        n = len(trials)
        a = alpha if alpha is not None else max(0.10, min(0.55, 5.0 / n))
        dom = get_dominant_type(t)
        color = PRECISION_COLORS.get(dom, "#9E9E9E")

        label = dom if dom not in legend_added else None
        ax.plot(
            steps, losses,
            color=color, alpha=a, linewidth=0.9, label=label,
        )
        legend_added.add(dom)
        plotted += 1

    if plotted == 0:
        ax.text(
            0.5, 0.5,
            "No training log history found.\n"
            "Re-run the search to start recording per-step losses.",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if legend_added:
        ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    plt.show()
    return fig


def plot_precision_trial_count(
    study: optuna.Study,
    title: str = "Trials per Precision Type",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Horizontal bar chart showing trial count per precision type."""
    trials = get_completed_trials(study)

    counts: dict[str, int] = {}
    for t in trials:
        dom = get_dominant_type(t)
        counts[dom] = counts.get(dom, 0) + 1

    sorted_labels = sorted(counts, key=lambda k: counts[k])
    sorted_counts = [counts[k] for k in sorted_labels]
    sorted_colors = [PRECISION_COLORS.get(k, "#9E9E9E") for k in sorted_labels]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(sorted_labels, sorted_counts, color=sorted_colors, edgecolor="white")

    for bar, cnt in zip(bars, sorted_counts):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", fontsize=10,
        )

    ax.set_xlabel("Number of Trials", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    plt.show()
    return fig


def analyse_study(
    study_or_name: optuna.Study | str,
    save_dir: Path | str | None = None,
    top_k: int = 5,
) -> optuna.Study:
    """One-call study analysis: load, print analytics, and generate plots."""
    if isinstance(study_or_name, str):
        print(f"Loading study '{study_or_name}' ...")
        study = load_study(study_or_name)
    else:
        study = study_or_name

    trials = get_completed_trials(study)
    failed = get_failed_trials(study)
    n_total = len(study.trials)

    if not trials:
        print("No completed trials — nothing to analyse.")
        return study

    values = [t.value for t in trials]
    ranked = sorted(trials, key=lambda t: t.value, reverse=True)

    w = 72
    print(f"\n{'=' * w}")
    print(f" Study Analysis: {study.study_name} ".center(w))
    print(f"{'=' * w}")
    print(f"  Total trials:     {n_total}")
    print(f"  Completed:        {len(trials)}")
    print(f"  Failed / pruned:  {len(failed)}")
    if failed:
        exceptions = [
            t.user_attrs.get("exception", "unknown")
            for t in failed if t.user_attrs.get("exception")
        ]
        if exceptions:
            unique_exc = Counter(exceptions)
            print(f"  Unique errors:    {len(unique_exc)}")
            for exc, cnt in unique_exc.most_common(3):
                short = exc[:80] + ("..." if len(exc) > 80 else "")
                print(f"    x{cnt}: {short}")

    print(f"\n{'-' * w}")
    print("  Accuracy Statistics")
    print(f"{'-' * w}")
    print(f"  Mean:    {np.mean(values):.4f}")
    print(f"  Median:  {np.median(values):.4f}")
    print(f"  Std:     {np.std(values):.4f}")
    print(f"  Min:     {np.min(values):.4f}")
    print(f"  Max:     {np.max(values):.4f}")
    q25, q75 = np.percentile(values, [25, 75])
    print(f"  Q25:     {q25:.4f}")
    print(f"  Q75:     {q75:.4f}")

    type_groups: dict[str, list[float]] = {}
    for t in trials:
        dom = get_dominant_type(t)
        type_groups.setdefault(dom, []).append(t.value)

    print(f"\n{'-' * w}")
    print("  Per-Precision Breakdown")
    print(f"{'-' * w}")
    print(f"  {'Type':<28} {'#':>4}  {'Mean':>7}  {'Best':>7}  {'Worst':>7}")
    print(f"  {'-' * 60}")
    for prec in sorted(
        type_groups,
        key=lambda k: np.mean(type_groups[k]),
        reverse=True,
    ):
        vals = type_groups[prec]
        print(
            f"  {prec:<28} {len(vals):>4}"
            f"  {np.mean(vals):>7.4f}"
            f"  {np.max(vals):>7.4f}"
            f"  {np.min(vals):>7.4f}"
        )

    show_k = min(top_k, len(ranked))
    print(f"\n{'-' * w}")
    print(f"  Top {show_k} Trials")
    print(f"{'-' * w}")
    for t in ranked[:show_k]:
        print_trial_detail(t, f"#{t.number}")

    worst = [t for t in ranked[-show_k:] if t.value > 0]
    if worst:
        print(f"{'-' * w}")
        print(f"  Bottom {len(worst)} Trials (excluding accuracy=0)")
        print(f"{'-' * w}")
        for t in reversed(worst):
            print_trial_detail(t, f"#{t.number}")

    print(f"{'=' * w}\n")

    def _sp(name: str) -> Path | None:
        if save_dir is None:
            return None
        return Path(save_dir) / f"{study.study_name}_{name}.png"

    print("Generating plots ...\n")
    print("-- Per-Trial Plots --")

    plot_accuracy_per_trial(
        study,
        title=f"{study.study_name}: Accuracy per Trial",
        save_path=_sp("acc_per_trial"),
    )

    plot_curves(
        {study.study_name: best_so_far_curve(study)},
        title=f"{study.study_name}: Best Accuracy So Far",
        save_path=_sp("cummax"),
    )

    plot_trial_values(
        study,
        title=f"{study.study_name}: Per-Trial Accuracy (by precision)",
        save_path=_sp("scatter"),
    )

    plot_accuracy_histogram(
        study,
        title=f"{study.study_name}: Accuracy Distribution",
        save_path=_sp("histogram"),
    )

    plot_training_loss_curves(
        study,
        title=f"{study.study_name}: Training Loss per Trial",
        save_path=_sp("loss_per_trial"),
    )

    if len(type_groups) > 1:
        print("\n-- Per-Precision Plots --")

        plot_precision_trial_count(
            study,
            title=f"{study.study_name}: Trials per Precision",
            save_path=_sp("precision_counts"),
        )

        plot_curves(
            best_so_far_curves_by_precision(study),
            title=f"{study.study_name}: Best Accuracy by Precision",
            save_path=_sp("cummax_by_precision"),
        )

        plot_precision_boxplot(
            study,
            title=f"{study.study_name}: Accuracy by Precision",
            save_path=_sp("boxplot"),
        )

        plot_training_loss_by_precision(
            study,
            title=f"{study.study_name}: Training Loss by Precision",
            save_path=_sp("loss_by_precision"),
        )

    if save_dir is not None:
        print(f"\nAll plots saved to: {Path(save_dir).resolve()}")

    return study
