
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def mount_gdrive(mount_point: str = "/content/drive") -> None:
    from google.colab import drive          # type: ignore[import]
    drive.mount(mount_point, force_remount=False)


def make_save_dir(project: str = "mase_bert_experiments") -> Path:
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("/content/drive/MyDrive") / project / f"run_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[new run] {save_dir}")
    return save_dir

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


def save_json(obj: Any, path: Path) -> None:
    """Write obj as pretty-printed JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)
    print(f"  [saved] {path}")


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
