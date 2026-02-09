"""Model constructor factory - builds a quantized BERT for each Optuna trial."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import optuna
import re
from typing import Callable, Optional


from chop.tools.utils import deepsetattr

from .constants import sanitize_prefix
from .layers import QuantLinearAdapter, ctor_accepts_arg
from .quant_configs import build_quant_config as _default_build_quant_config


def pretty_print_trial(
    trial_number: int,
    layer_choices: list[dict],
) -> None:
    """Print a readable summary of per-layer choices for one trial."""
    header = f" Trial {trial_number} – Layer Configuration "
    w = max(72, len(header) + 4)
    print(f"\n{'=' * w}")
    print(header.center(w))
    print(f"{'=' * w}")

    for i, lc in enumerate(layer_choices, 1):
        tag = lc["type"]
        if tag == "Linear":
            tag += "  (unchanged)"
        print(f"\n  [{i}] {lc['name']}")
        print(f"      Type:  {tag}")
        print(
            f"      Shape: in={lc['shape'][0]}, "
            f"out={lc['shape'][1]}, bias={lc['shape'][2]}"
        )
        cfg = lc.get("config")
        if cfg:
            groups: dict[str, list[tuple[str, Any]]] = {}
            for k, v in cfg.items():
                prefix_key = k.split("_", 1)[0]
                groups.setdefault(prefix_key, []).append((k, v))
            for _grp, params in groups.items():
                print(f"      {', '.join(f'{k}={v}' for k, v in params)}")

    types = [lc["type"] for lc in layer_choices]
    counts = Counter(types)
    summary = ", ".join(f"{t}({n})" for t, n in sorted(counts.items()))
    print(f"\n  Summary: {len(layer_choices)} layers -> {summary}")
    print(f"{'=' * w}\n")


def make_model_constructor(
    base_model: nn.Module,
    layer_map: dict[str, type],
    search_space: dict,
    build_quant_config_fn=None,
):
    """Return a construct_model(trial) -> nn.Module closure."""
    if build_quant_config_fn is None:
        build_quant_config_fn = _default_build_quant_config

    layer_choices_key = "linear_layer_choices"

    def construct_model(trial: optuna.trial.Trial) -> nn.Module:
        trial_model = deepcopy(base_model)
        layer_choices: list[dict] = []

        for name, layer in list(trial_model.named_modules()):
            if not isinstance(layer, torch.nn.Linear):
                continue

            layer_cls_name = trial.suggest_categorical(
                f"{sanitize_prefix(name)}__type",
                search_space[layer_choices_key],
            )
            new_layer_cls = layer_map[layer_cls_name]

            choice_info: dict = {
                "name": name,
                "type": layer_cls_name,
                "shape": (
                    layer.in_features,
                    layer.out_features,
                    layer.bias is not None,
                ),
            }

            if new_layer_cls is torch.nn.Linear:
                layer_choices.append(choice_info)
                continue

            kwargs: dict[str, Any] = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None,
            }

            config = None
            if ctor_accepts_arg(new_layer_cls, "config"):
                config = build_quant_config_fn(
                    trial, layer_cls_name, prefix=name,
                )
                kwargs["config"] = config

            choice_info["config"] = config
            layer_choices.append(choice_info)

            inner = new_layer_cls(**kwargs)

            with torch.no_grad():
                inner.weight.copy_(layer.weight)
                if (
                    layer.bias is not None
                    and getattr(inner, "bias", None) is not None
                ):
                    inner.bias.copy_(layer.bias)

            if layer_cls_name == "LinearBinaryResidualSign":
                new_layer = QuantLinearAdapter(
                    inner,
                    out_features=layer.out_features,
                    force_2d_input=True,
                    auto_retry=True,
                )
            else:
                new_layer = inner


            deepsetattr(trial_model, name, new_layer)

        pretty_print_trial(trial.number, layer_choices)
        return trial_model

    return construct_model

_ENCODER_LAYER_RE = re.compile(r"\.encoder\.layer\.(\d+)\.")

def bert_linear_group(
    module_name: str,
    *,
    num_hidden_layers: int | None = None,
    stage_mode: str = "none",
) -> Optional[str]:
    """Map a BERT-style Linear module path to a group name."""
    name = module_name

    # Head / classifier-ish stuff
    if ".pooler.dense" in name:
        return "head.pooler"
    if name.endswith("classifier") or ".classifier" in name:
        return "head.classifier"

    # Encoder layer index (if present)
    m = _ENCODER_LAYER_RE.search(name)
    layer_idx = int(m.group(1)) if m else None

    stage = None
    if stage_mode == "3stage" and (layer_idx is not None) and num_hidden_layers:
        third = max(1, int(num_hidden_layers) // 3)
        if layer_idx < third:
            stage = "early"
        elif layer_idx < 2 * third:
            stage = "mid"
        else:
            stage = "late"

    def with_stage(tag: str) -> str:
        return f"enc.{stage}.{tag}" if stage else f"enc.{tag}"

    # Attention projections
    if ".attention.self.query" in name or ".attention.self.key" in name or ".attention.self.value" in name:
        return with_stage("attn_qkv")

    if ".attention.output.dense" in name:
        return with_stage("attn_out")

    if ".intermediate.dense" in name:
        return with_stage("ffn_in")

    if re.search(r"\.encoder\.layer\.\d+\.output\.dense$", name):
        return with_stage("ffn_out")

    return "other.linear"


def pretty_print_grouped_trial(
    trial_number: int,
    group_choices: dict[str, dict[str, Any]],
    *,
    group_layer_counts: dict[str, int] | None = None,
) -> None:
    header = f" Trial {trial_number} – GROUPED Layer Configuration "
    w = max(72, len(header) + 4)
    print(f"\n{'=' * w}")
    print(header.center(w))
    print(f"{'=' * w}")

    for group in sorted(group_choices):
        info = group_choices[group]
        layer_type = info["type"]
        cfg = info.get("config")

        count_str = ""
        if group_layer_counts and group in group_layer_counts:
            count_str = f"  (layers={group_layer_counts[group]})"

        print(f"\n  Group: {group}{count_str}")
        print(f"    Type: {layer_type}")

        if cfg:
            # compact one-liner config print
            items = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
            print(f"    Config: {items}")

    print(f"\n{'=' * w}\n")


def make_grouped_model_constructor(
    base_model: nn.Module,
    layer_map: dict[str, type],
    search_space: dict,
    build_quant_config_fn=None,
    *,
    stage_mode: str = "none",
    group_fn: Callable[..., Optional[str]] = bert_linear_group,
    force_full_precision_groups: set[str] | None = None,
):
    """Grouped variant of make_model_constructor with one config per group."""
    if build_quant_config_fn is None:
        build_quant_config_fn = _default_build_quant_config

    if force_full_precision_groups is None:
        force_full_precision_groups = {"head.classifier"}

    layer_choices_key = "linear_layer_choices"
    num_hidden_layers = getattr(getattr(base_model, "config", None), "num_hidden_layers", None)

    def construct_model(trial: optuna.trial.Trial) -> nn.Module:
        trial_model = deepcopy(base_model)

        linear_entries: list[tuple[str, nn.Linear, str]] = []
        group_counts: dict[str, int] = {}

        for name, layer in list(trial_model.named_modules()):
            if not isinstance(layer, torch.nn.Linear):
                continue

            group = group_fn(
                name,
                num_hidden_layers=num_hidden_layers,
                stage_mode=stage_mode,
            ) or "other.linear"

            linear_entries.append((name, layer, group))
            group_counts[group] = group_counts.get(group, 0) + 1

        group_choices: dict[str, dict[str, Any]] = {}
        for group in sorted(set(g for _, _, g in linear_entries)):
            if group in force_full_precision_groups:
                group_choices[group] = {"type": "Linear", "config": None}
                continue

            layer_cls_name = trial.suggest_categorical(
                f"{sanitize_prefix(group)}__type",
                search_space[layer_choices_key],
            )

            config = None
            new_layer_cls = layer_map[layer_cls_name]
            if new_layer_cls is not torch.nn.Linear and ctor_accepts_arg(new_layer_cls, "config"):
                config = build_quant_config_fn(trial, layer_cls_name, prefix=group)

            group_choices[group] = {"type": layer_cls_name, "config": config}

        layer_choices: list[dict] = []
        for name, layer, group in linear_entries:
            choice = group_choices[group]
            layer_cls_name = choice["type"]
            new_layer_cls = layer_map[layer_cls_name]

            choice_info: dict = {
                "name": name,
                "group": group,
                "type": layer_cls_name,
                "shape": (layer.in_features, layer.out_features, layer.bias is not None),
                "config": choice.get("config"),
            }
            layer_choices.append(choice_info)

            if new_layer_cls is torch.nn.Linear:
                continue

            kwargs: dict[str, Any] = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None,
            }

            if ctor_accepts_arg(new_layer_cls, "config"):
                kwargs["config"] = choice.get("config")

            inner = new_layer_cls(**kwargs)

            with torch.no_grad():
                inner.weight.copy_(layer.weight)
                if layer.bias is not None and getattr(inner, "bias", None) is not None:
                    inner.bias.copy_(layer.bias)

            if layer_cls_name == "LinearBinaryResidualSign":
                new_layer = QuantLinearAdapter(
                    inner,
                    out_features=layer.out_features,
                    force_2d_input=True,
                    auto_retry=True,
                )
            else:
                new_layer = inner

            deepsetattr(trial_model, name, new_layer)

        pretty_print_grouped_trial(
            trial.number,
            group_choices,
            group_layer_counts=group_counts,
        )
        return trial_model

    return construct_model
