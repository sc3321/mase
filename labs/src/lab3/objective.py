"""Optuna objective closure - ties model construction, training, and eval."""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import optuna

from chop.tools import get_trainer

from .constants import get_persistence_root
from .model_builder import make_model_constructor
from .quant_configs import build_quant_config as _default_build_quant_config


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)


def _study_dir(study_name: str) -> Path:
    """Return <persistence_root>/<study_name>/."""
    d = get_persistence_root() / study_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model_for_trial(
    trial: optuna.trial.Trial,
    model: nn.Module,
    tag: str = "final",
) -> str:
    out_dir = _study_dir(trial.study.study_name) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"trial_{trial.number}_{tag}.pt"

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, model_path)
    trial.set_user_attr(f"model_path_{tag}", str(model_path))
    return str(model_path)


def save_trial_result(
    trial: optuna.trial.Trial,
    accuracy: float,
    metrics: dict | None = None,
    exception: str | None = None,
    log_history: list[dict] | None = None,
) -> str:
    out_dir = _study_dir(trial.study.study_name) / "trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trial_{trial.number}.json"

    record = {
        "trial_number": trial.number,
        "accuracy": accuracy,
        "params": dict(trial.params),
        "metrics": metrics,
        "log_history": log_history,
        "exception": exception,
    }
    _save_json(record, path)
    return str(path)


def make_objective(
    dataset,
    tokenizer,
    base_model: nn.Module,
    layer_map: dict[str, type],
    search_space: dict,
    num_train_epochs: int = 1,
    build_quant_config_fn=None,
    *,
    construct_model_fn=None,
    constructor_factory=make_model_constructor,
    enable_pruning: bool = True,
    prune_metric: str = "eval_accuracy",
):
    """Return an objective(trial) -> float closure ready for Optuna."""
    if build_quant_config_fn is None:
        build_quant_config_fn = _default_build_quant_config

    if construct_model_fn is None:
        construct_model = constructor_factory(
            base_model, layer_map, search_space, build_quant_config_fn,
        )
    else:
        construct_model = construct_model_fn

    def _smoke_test(model: nn.Module) -> None:
        """Quick forward + backward pass to catch errors early."""
        device = next(model.parameters()).device
        dummy_ids = torch.zeros(2, 8, dtype=torch.long, device=device)
        dummy_mask = torch.ones(2, 8, dtype=torch.long, device=device)
        dummy_labels = torch.zeros(2, dtype=torch.long, device=device)

        model.train()
        out = model(
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            labels=dummy_labels,
        )
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    def objective(trial: optuna.trial.Trial) -> float:
        trainer = None
        model = None
        acc = 0.0
        metrics = None
        log_history = None
        exception_str = None
        try:
            model = construct_model(trial)
            _smoke_test(model)

            trainer = get_trainer(
                model=model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=num_train_epochs,
            )

            if enable_pruning:
                try:
                    from optuna.integration import HuggingFacePruningCallback
                    if hasattr(trainer, "add_callback"):
                        trainer.add_callback(
                            HuggingFacePruningCallback(trial, prune_metric)
                        )
                except Exception:
                    pass

            trainer.train()

            if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
                log_history = list(trainer.state.log_history)

            metrics = trainer.evaluate()

            save_model_for_trial(trial, trainer.model, tag="final")

            acc = float(metrics.get("eval_accuracy", 0.0))
            if acc == 0.0:
                print(
                    f"[Trial {trial.number}] WARNING: accuracy=0.0, "
                    f"metrics keys={list(metrics.keys())}"
                )
            return acc

        except Exception as e:
            print(f"\n{'=' * 60}")
            print(f"[Trial {trial.number}] EXCEPTION: {e!r}")
            traceback.print_exc()
            print(f"{'=' * 60}\n")
            exception_str = repr(e)
            trial.set_user_attr("exception", exception_str)
            return 0.0

        finally:
            try:
                save_trial_result(
                    trial,
                    accuracy=acc,
                    metrics=metrics,
                    exception=exception_str,
                    log_history=log_history,
                )
            except Exception:
                pass

            del trainer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective
