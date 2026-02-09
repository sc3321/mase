"""Global constants, reproducibility seeds, and shared utilities."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TOKENIZER_CHECKPOINT = "bert-base-uncased"
CHECKPOINT  = "prajjwal1/bert-tiny"

DATASET_NAME = "imdb"
HF_INPUT_NAMES = ["input_ids", "attention_mask", "labels"]

ARTIFACT_ROOT = Path("./artifacts").resolve()
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

GDRIVE_PROJECT = "mase_optuna"


def get_persistence_root() -> Path:
    """Return a crash-safe root directory for all persisted artifacts."""
    try:
        import google.colab  # noqa: F401
        from google.colab import drive

        mount_point = Path("/content/drive")
        if not (mount_point / "MyDrive").exists():
            drive.mount(str(mount_point))

        root = mount_point / "MyDrive" / GDRIVE_PROJECT
        root.mkdir(parents=True, exist_ok=True)
        return root
    except ImportError:
        return ARTIFACT_ROOT


def sanitize_prefix(prefix: str) -> str:
    """Convert a dotted module path to an Optuna-safe parameter name."""
    return prefix.replace(".", "__").replace("/", "__")
