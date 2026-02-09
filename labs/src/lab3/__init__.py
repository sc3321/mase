"""Mixed-precision quantisation search for BERT."""

from .constants import (
    SEED, CHECKPOINT, DATASET_NAME, ARTIFACT_ROOT,
    sanitize_prefix, get_persistence_root,
)
from .layers import get_layer_map, QuantLinearAdapter, ctor_accepts_arg
from .search_space import get_search_space, restrict_search_space
from .quant_configs import build_quant_config
from .data import prep_env, load_dataset_and_tokenizer
from .model_builder import make_model_constructor
from .objective import make_objective, save_model_for_trial, save_trial_result
from .search import run_search, load_study, make_storage
from .analysis import (
    get_completed_trials,
    get_failed_trials,
    best_so_far_curve,
    best_so_far_curves_by_precision,
    get_trial_layer_types,
    get_dominant_type,
    print_study_summary,
)
from .plotting import (
    plot_curves,
    plot_trial_values,
    plot_accuracy_per_trial,
    plot_precision_comparison,
    plot_accuracy_histogram,
    plot_precision_boxplot,
    plot_training_loss_curves,
    plot_training_loss_by_precision,
    plot_precision_trial_count,
    analyse_study,
)

__all__ = [
    "SEED", "CHECKPOINT", "DATASET_NAME", "ARTIFACT_ROOT",
    "sanitize_prefix", "get_persistence_root",
    "get_layer_map", "QuantLinearAdapter", "ctor_accepts_arg",
    "get_search_space", "restrict_search_space",
    "build_quant_config",
    "prep_env", "load_dataset_and_tokenizer",
    "make_model_constructor",
    "make_objective", "save_model_for_trial", "save_trial_result",
    "run_search", "load_study", "make_storage",
    "get_completed_trials", "get_failed_trials",
    "best_so_far_curve", "best_so_far_curves_by_precision",
    "get_trial_layer_types", "get_dominant_type", "print_study_summary",
    "plot_curves", "plot_trial_values", "plot_accuracy_per_trial",
    "plot_precision_comparison", "plot_accuracy_histogram",
    "plot_precision_boxplot", "plot_training_loss_curves",
    "plot_training_loss_by_precision", "plot_precision_trial_count",
    "analyse_study",
]
