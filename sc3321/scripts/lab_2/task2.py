#!/usr/bin/env python3

import os
from pathlib import Path

import optuna
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForSequenceClassification

from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
from chop.nn.modules import Identity
from chop.pipelines import CompressionPipeline
from chop import MaseGraph

# -------------------------
# User config
# -------------------------
checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

MAX_TRIALS = 50
EPOCHS_PRE_COMPRESSION = 1
EPOCHS_POST_COMPRESSION = 1

# -------------------------
# Search + compression configs
# -------------------------
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choice": ["linear", "identity"],
}

quantization_config = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

pruning_config = {
    "weight": {"sparsity": 0.5, "method": "l1-norm", "scope": "local"},
    "activation": {"sparsity": 0.5, "method": "l1-norm", "scope": "local"},
}

# -------------------------
# Helpers
# -------------------------
def get_outdir() -> Path:
    try:
        root = Path(__file__).resolve().parents[3]
    except NameError:
        root = Path.cwd()
    outdir = root / "sc3321" / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def running_best(study: optuna.Study):
    best = []
    cur = float("-inf")
    for t in study.trials:
        if t.value is None:
            continue
        cur = max(cur, float(t.value))
        best.append(cur)
    return best


def construct_model(trial: optuna.Trial):
    config = AutoConfig.from_pretrained(checkpoint)

    config.num_hidden_layers = trial.suggest_categorical("num_layers", search_space["num_layers"])
    config.num_attention_heads = trial.suggest_categorical("num_heads", search_space["num_heads"])
    config.hidden_size = trial.suggest_categorical("hidden_size", search_space["hidden_size"])
    config.intermediate_size = trial.suggest_categorical(
        "intermediate_size", search_space["intermediate_size"]
    )

    model = AutoModelForSequenceClassification.from_config(config)

    choice = trial.suggest_categorical("linear_layer_choice", search_space["linear_layer_choice"])
    if choice == "identity":
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
                deepsetattr(model, name, Identity())

    return model


def make_objective(
    *,
    do_compress: bool,
    do_post_compress_train: bool,
    epochs_pre: int = EPOCHS_PRE_COMPRESSION,
    epochs_post: int = EPOCHS_POST_COMPRESSION,
):
    def objective(trial: optuna.Trial) -> float:
        model = construct_model(trial)

        trainer_pre = get_trainer(
            model=model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=epochs_pre,
        )
        trainer_pre.train()
        
        if not do_compress:
            eval_results = trainer_pre.evaluate()
            trial.set_user_attr("model", model)
            return float(eval_results["eval_accuracy"])
        
        model = model.to("cpu")
        mg = MaseGraph(model)
        pipe = CompressionPipeline()
        mg, _ = pipe(
            mg,
            pass_args={
                "quantize_transform_pass": quantization_config,
                "prune_transform_pass": pruning_config,
            },
        )

        compressed_model = mg.model

        trainer_post = get_trainer(
            model=compressed_model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=(epochs_post if do_post_compress_train else 0),
        )

        if do_post_compress_train and epochs_post > 0:
            trainer_post.train()

        eval_results = trainer_post.evaluate()
        trial.set_user_attr("model", compressed_model)
        return float(eval_results["eval_accuracy"])

    return objective


# -------------------------
# Execution
# -------------------------
torch.manual_seed(0)

OUTDIR = get_outdir()

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

from optuna.samplers import TPESampler
base_sampler = TPESampler(seed=0)

# (1) Baseline NAS
study_baseline = optuna.create_study(direction="maximize", sampler=base_sampler)
study_baseline.optimize(
    make_objective(do_compress=False, do_post_compress_train=False),
    n_trials=MAX_TRIALS,
)

# (2) Compression-aware, no retraining
study_comp_no_retrain = optuna.create_study(direction="maximize", sampler=TPESampler(seed=0))
study_comp_no_retrain.optimize(
    make_objective(do_compress=True, do_post_compress_train=False),
    n_trials=MAX_TRIALS,
)

# (3) Compression-aware, with retraining
study_comp_with_retrain = optuna.create_study(direction="maximize", sampler=TPESampler(seed=0))
study_comp_with_retrain.optimize(
    make_objective(do_compress=True, do_post_compress_train=True),
    n_trials=MAX_TRIALS,
)

# -------------------------
# Plot (save only)
# -------------------------
b1 = running_best(study_baseline)
b2 = running_best(study_comp_no_retrain)
b3 = running_best(study_comp_with_retrain)

plt.figure()
plt.plot(range(1, len(b1) + 1), b1, label="Task 1 baseline (no compression)")
plt.plot(range(1, len(b2) + 1), b2, label="Compression-aware (no post-compression training)")
plt.plot(range(1, len(b3) + 1), b3, label="Compression-aware (+ post-compression training)")
plt.xlabel("Number of trials")
plt.ylabel("Best accuracy so far")
plt.legend()
plt.grid(True)

fig_path = OUTDIR / "task2_three_curves_running_best.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved figure to: {fig_path}")

