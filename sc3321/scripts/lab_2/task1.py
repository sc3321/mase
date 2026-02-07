checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"
import platform
import pandas as pd
from pathlib import Path
from chop.tools import get_tokenized_dataset
import os
import optuna
import matplotlib.pyplot as plt

dataset, tokenizer = get_tokenized_dataset(
        dataset = dataset_name,
        checkpoint=tokenizer_checkpoint,
        return_tokenizer=True,
)

import torch.nn as nn
from chop.nn.modules import Identity

search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choice": ["linear", "identity"],  # <-- strings, not classes
}


from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr

def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)

    config.num_hidden_layers = trial.suggest_categorical("num_layers", search_space["num_layers"])
    config.num_attention_heads = trial.suggest_categorical("num_heads", search_space["num_heads"])
    config.hidden_size = trial.suggest_categorical("hidden_size", search_space["hidden_size"])
    config.intermediate_size = trial.suggest_categorical("intermediate_size", search_space["intermediate_size"])

    model = AutoModelForSequenceClassification.from_config(config)

    choice = trial.suggest_categorical("linear_layer_choice", search_space["linear_layer_choice"])
    if choice == "identity":
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
                deepsetattr(model, name, Identity())

    return model


from chop.tools import get_trainer


def objective(trial):

    # Define the model
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # Set the model as an attribute so we can fetch it later
    trial.set_user_attr("model", model)

    return eval_results["eval_accuracy"]

# Task 1

MAX_TRIALS = 50
ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "sc3321" / "outputs" 
os.makedirs(OUTDIR, exist_ok=True)
from optuna.samplers import GridSampler, TPESampler

samplers = {
    "Grid": GridSampler(search_space),   # you must define search_space
    "TPE": TPESampler(seed=0),
}

def running_best(study: optuna.Study):
    best = []
    cur = float("-inf")
    for t in study.trials:
        if t.value is None:
            continue
        cur = max(cur, t.value)
        best.append(cur)
    return best

plt.figure()

for name, sampler in samplers.items():
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=MAX_TRIALS, timeout=60 * 60 * 24)

    y = running_best(study)
    x = range(1, len(y) + 1)
    plt.plot(x, y, label=name)

plt.xlabel("Number of trials")
plt.ylabel("Best accuracy so far")
plt.legend()

# outpath = os.path.join(OUTDIR, "sampler_comparison.png")
outpath = OUTDIR / "sampler_comparison.png"
plt.tight_layout()
plt.savefig(outpath, dpi=300)
plt.close()

#Task 2


from chop.pipelines import CompressionPipeline
from chop import MaseGraph






