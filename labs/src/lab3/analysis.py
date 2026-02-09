"""Trial extraction, cumulative-max curves, formatting, and printing."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import optuna
from optuna.trial import TrialState

from .search import load_study, make_storage  # noqa: F401 (re-exported)


def get_completed_trials(
    study: optuna.Study,
) -> list[optuna.trial.FrozenTrial]:
    """Return completed trials sorted by trial number."""
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    trials.sort(key=lambda t: t.number)
    return trials


def get_failed_trials(
    study: optuna.Study,
) -> list[optuna.trial.FrozenTrial]:
    """Return trials that raised an exception or were pruned."""
    return [
        t for t in study.trials
        if t.state in (TrialState.FAIL, TrialState.PRUNED)
    ]


def _cummax(values: list[float]) -> list[float]:
    """Running (cumulative) maximum of *values*."""
    out: list[float] = []
    best = float("-inf")
    for v in values:
        best = max(best, v if v is not None else float("-inf"))
        out.append(best)
    return out


def best_so_far_curve(
    study: optuna.Study,
) -> tuple[list[int], list[float]]:
    """Extract the cumulative-max accuracy curve. Returns (trial_numbers, cummax_accuracy)."""
    trials = get_completed_trials(study)
    numbers = [t.number for t in trials]
    values = [t.value for t in trials]
    return numbers, _cummax(values)


def get_trial_layer_types(
    trial: optuna.trial.FrozenTrial,
) -> list[str]:
    """Return the list of layer-type names chosen in *trial*."""
    return [v for k, v in trial.params.items() if k.endswith("__type")]


def get_dominant_type(
    trial: optuna.trial.FrozenTrial,
    *,
    exclude_linear: bool = False,
    min_fraction: float = 0.0,
) -> str:
    """Return the most-common layer type for a trial."""
    types = get_trial_layer_types(trial)
    if not types:
        return "unknown"

    counts = Counter(types)

    if exclude_linear:
        counts.pop("Linear", None)
        if not counts:
            return "Linear"

    dominant, n = counts.most_common(1)[0]
    if min_fraction > 0:
        total = sum(counts.values())
        if total > 0 and (n / total) < float(min_fraction):
            return "mixed"

    return dominant


def best_so_far_curves_by_precision(
    study: optuna.Study,
) -> dict[str, tuple[list[int], list[float]]]:
    """Return one cummax curve per dominant precision type."""
    trials = get_completed_trials(study)

    groups: dict[str, list[tuple[int, float]]] = {}
    for t in trials:
        dom = get_dominant_type(t)
        groups.setdefault(dom, []).append((t.number, t.value))

    curves: dict[str, tuple[list[int], list[float]]] = {}
    for precision, entries in sorted(groups.items()):
        entries.sort(key=lambda e: e[0])
        numbers = [e[0] for e in entries]
        values = [e[1] for e in entries]
        curves[precision] = (numbers, _cummax(values))

    return curves


def format_trial_params(trial: optuna.trial.FrozenTrial) -> str:
    """Return a multi-line, human-readable string of a trial's layer config."""
    layers: dict[str, dict[str, Any]] = {}
    for k, v in sorted(trial.params.items()):
        if "__type" in k:
            prefix = k.rsplit("__type", 1)[0]
            layers.setdefault(prefix, {})["type"] = v
        else:
            parts = k.rsplit("__", 1)
            if len(parts) == 2:
                prefix, param_name = parts
                layers.setdefault(prefix, {})[param_name] = v

    lines: list[str] = []
    for prefix in sorted(layers):
        info = layers[prefix]
        layer_type = info.pop("type", "?")
        readable = prefix.replace("__", ".")
        if info:
            param_str = ", ".join(f"{k}={v}" for k, v in sorted(info.items()))
            lines.append(f"    {readable}: {layer_type}  ({param_str})")
        else:
            lines.append(f"    {readable}: {layer_type}")
    return "\n".join(lines)


def print_study_summary(study: optuna.Study) -> None:
    """Print a compact summary of a completed study."""
    trials = get_completed_trials(study)
    if not trials:
        print("No completed trials.")
        return

    best = study.best_trial
    print(f"\nStudy: {study.study_name}")
    print(f"  Completed trials: {len(trials)}")
    print(f"  Best trial:       #{best.number}  (accuracy={best.value:.4f})")

    dom = get_dominant_type(best)
    print(f"  Best dominant:    {dom}")

    type_counts = Counter(get_dominant_type(t) for t in trials)
    print("  Trials by type:   " + ", ".join(
        f"{t}({n})" for t, n in sorted(type_counts.items())
    ))
    print()


def print_trial_detail(
    trial: optuna.trial.FrozenTrial,
    label: str,
) -> None:
    """Print a detailed block for a single trial."""
    types = get_trial_layer_types(trial)
    counts = Counter(types)
    mix_str = ", ".join(f"{t}({n})" for t, n in sorted(counts.items()))
    dominant = get_dominant_type(trial)

    print(f"  {label}:")
    print(f"    Trial #:    {trial.number}")
    print(f"    Accuracy:   {trial.value:.4f}")
    print(f"    Dominant:   {dominant}")
    print(f"    Layer mix:  {mix_str}")

    exc = trial.user_attrs.get("exception")
    if exc:
        print(f"    Exception:  {exc}")

    param_block = format_trial_params(trial)
    if param_block:
        print("    Layers:")
        for line in param_block.splitlines():
            print(f"  {line}")
    print()
