"""Sweep runners for quantisation (Task 1) and pruning (Task 2)."""

from __future__ import annotations

import copy
from pathlib import Path

import chop.passes as passes

from .constants import CHECKPOINT
from .configs import make_quant_config, make_prune_config
from .io_utils import save_json, load_json
from .model import build_masegraph, evaluate_model


# Task 1 - Quantisation width sweep

def run_quantization_sweep(
    widths: list[int],
    dataset,
    tokenizer,
    save_dir: Path,
    num_epochs: int = 1,
    checkpoint: str = CHECKPOINT,
) -> list[dict]:
    """
    For each fixed-point width, quantise a fresh BERT model, evaluate PTQ
    accuracy, fine-tune (QAT), then evaluate again.

    Parameters
    ----------
    widths : 
        Bit-widths to sweep, e.g. list(range(4, 33)).
    save_dir : 
        Google-Drive (or local) directory for results & checkpoints.

    Returns
    -------
        One entry per width with keys width, ptq_accuracy,
        qat_accuracy, ptq_metrics, qat_metrics, quant_config.
    """
    save_dir = Path(save_dir)
    partial_path = save_dir / "quantization_sweep_partial.json"

    # Resume from partial results if they exist
    results: list[dict] = []
    if partial_path.exists():
        results = load_json(partial_path)
        done_widths = {r["width"] for r in results}
        widths = [w for w in widths if w not in done_widths]
        print(f"Resuming: {len(results)} done, {len(widths)} remaining")

    for i, width in enumerate(widths):
        print(f"\n{'=' * 60}")
        print(f"[Task 1] Width {width}  ({i + 1}/{len(widths)})")
        print(f"{'=' * 60}")

        config = make_quant_config(width)
        mg = build_masegraph(checkpoint)
        mg, _ = passes.quantize_transform_pass(
            mg, pass_args=copy.deepcopy(config),
        )

        ptq_acc, qat_acc, ptq_metrics, qat_metrics = evaluate_model(
            mg, dataset, tokenizer, num_epochs=num_epochs,
        )

        print(f"  PTQ accuracy: {ptq_acc:.4f}")
        print(f"  QAT accuracy: {qat_acc:.4f}")

        results.append({
            "width": width,
            "ptq_accuracy": ptq_acc,
            "qat_accuracy": qat_acc,
            "ptq_metrics": ptq_metrics,
            "qat_metrics": qat_metrics,
            "quant_config": config,
        })

        # Persist after every iteration (crash-proof)
        save_json(results, partial_path)

    # Final consolidated save
    final_path = save_dir / "quantization_sweep_results.json"
    save_json(results, final_path)

    # Persist the best quant config so Task 2 can load it independently
    if results:
        best = max(results, key=lambda r: r["qat_accuracy"])
        save_json(best["quant_config"], save_dir / "best_quant_config.json")
        save_json(
            {
                "best_width": best["width"],
                "best_qat_accuracy": best["qat_accuracy"],
                "best_ptq_accuracy": best["ptq_accuracy"],
            },
            save_dir / "best_quant_summary.json",
        )

    return results


# Task 2 - Pruning sparsity sweep

def run_pruning_sweep(
    sparsities: list[float],
    methods: list[str],
    dataset,
    tokenizer,
    save_dir: Path,
    best_quant_config: dict | None = None,
    num_epochs: int = 1,
    checkpoint: str = CHECKPOINT,
) -> list[dict]:
    """
    For each (method, sparsity) pair, build a fresh model, optionally
    quantise it with best_quant_config (from Task 1), prune, evaluate,
    fine-tune, then evaluate again.

    Parameters
    ----------
    sparsities : 
        Sparsity levels to sweep
    methods :
        Pruning strategies
    best_quant_config : 
        If provided, the model is quantised before pruning (replicating the
        best setup from Task 1).

    Returns
    -------
        One entry per (method, sparsity) with accuracy and config metadata.
    """
    save_dir = Path(save_dir)
    partial_path = save_dir / "pruning_sweep_partial.json"

    # Resume from partial results if they exist
    results: list[dict] = []
    if partial_path.exists():
        results = load_json(partial_path)
        done_keys = {(r["method"], r["sparsity"]) for r in results}
    else:
        done_keys = set()

    total = len(sparsities) * len(methods)
    count = 0

    for method in methods:
        for sparsity in sparsities:
            count += 1

            if (method, sparsity) in done_keys:
                print(f"  [skip] {method} @ {sparsity:.1f}  (already done)")
                continue

            print(f"\n{'=' * 60}")
            print(f"[Task 2] {method} | sparsity={sparsity:.1f}  ({count}/{total})")
            print(f"{'=' * 60}")

            mg = build_masegraph(checkpoint)

            # Optionally quantise first (best config from Task 1)
            if best_quant_config is not None:
                mg, _ = passes.quantize_transform_pass(
                    mg, pass_args=copy.deepcopy(best_quant_config),
                )

            prune_cfg = make_prune_config(sparsity, method)
            mg, _ = passes.prune_transform_pass(mg, pass_args=prune_cfg)

            pre_acc, post_acc, pre_metrics, post_metrics = evaluate_model(
                mg, dataset, tokenizer, num_epochs=num_epochs,
            )

            print(f"  Pre-finetune accuracy:  {pre_acc:.4f}")
            print(f"  Post-finetune accuracy: {post_acc:.4f}")

            results.append({
                "method": method,
                "sparsity": sparsity,
                "pre_finetune_accuracy": pre_acc,
                "post_finetune_accuracy": post_acc,
                "pre_finetune_metrics": pre_metrics,
                "post_finetune_metrics": post_metrics,
                "prune_config": prune_cfg,
                "quant_config": best_quant_config,
            })

            # Persist after every iteration (crash-proof)
            save_json(results, partial_path)

    # Final consolidated save
    final_path = save_dir / "pruning_sweep_results.json"
    save_json(results, final_path)
    return results


# Result loaders & analysis

def load_quantization_results(save_dir: Path) -> list[dict]:
    save_dir = Path(save_dir)
    final = save_dir / "quantization_sweep_results.json"
    partial = save_dir / "quantization_sweep_partial.json"

    if final.exists():
        print(f"  [load] {final}")
        return load_json(final)
    if partial.exists():
        print(f"  [load partial] {partial}")
        return load_json(partial)

    raise FileNotFoundError(
        f"No quantisation results found in {save_dir}. "
        "Run run_quantization_sweep first."
    )


def get_best_result(
    results: list[dict],
    metric: str = "qat_accuracy",
) -> dict:
    """Return the result dict with the highest value of metric."""
    return max(results, key=lambda r: r[metric])
