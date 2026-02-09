"""Optuna study creation, storage, and search execution."""
from __future__ import annotations

import optuna
from optuna.samplers import TPESampler

from .constants import SEED, get_persistence_root


def make_storage(study_name: str) -> str:
    """Return an SQLite Optuna storage URL for study_name."""
    db_dir = get_persistence_root() / "optuna"
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_dir / f'{study_name}.db'}"

def make_tpe_sampler(seed: int = SEED) -> TPESampler:
    """Create a TPE sampler with multivariate+group if supported."""
    try:
        return TPESampler(
            seed=seed,
            multivariate=True,
            group=True,
            n_startup_trials=15,
        )
    except TypeError:
        return TPESampler(seed=seed)


def run_search(
    objective,
    name: str = "search",
    sampler=None,
    direction: str = "maximize",
    n_trials: int = 50,
    timeout: int = 60 * 60,
    delete_existing: bool = False,
) -> optuna.Study:
    """Create or resume an Optuna study and run objective for up to n_trials total."""
    storage = make_storage(name)

    if delete_existing:
        try:
            optuna.delete_study(study_name=name, storage=storage)
            print(f"Deleted existing study '{name}'.")
        except Exception:
            pass

    if sampler is None:
        sampler = make_tpe_sampler(SEED)

    study = optuna.create_study(
        direction=direction,
        study_name=name,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    existing = len(study.trials)
    remaining = max(0, n_trials - existing)

    if existing > 0:
        best_val = study.best_value if study.best_trial else "N/A"
        print(
            f"[resume] Study '{name}': {existing} trials already done "
            f"(best={best_val}), running {remaining} more."
        )

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, timeout=timeout)
    else:
        print(f"[skip] Study '{name}' already has {existing}/{n_trials} trials.")

    return study


def load_study(name: str) -> optuna.Study:
    """Load an existing Optuna study by name from the default SQLite storage."""
    storage = make_storage(name)
    return optuna.load_study(study_name=name, storage=storage)
