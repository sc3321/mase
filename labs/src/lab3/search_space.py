"""Search-space definitions for the Optuna mixed-precision search."""
from __future__ import annotations

from .layers import get_layer_map


def get_search_space() -> dict:
    """Default search space covering all precisions."""
    return {
        "linear_layer_choices": list(get_layer_map().keys()),
        "widths": [8, 16, 32],
        "frac_widths": [2, 4, 8],
        "exponent_widths": [3, 4, 5],
        "exponent_biases": [3, 7, 15],
        "log_exponent_biases": [7, 15, 31],
        "exponent_bias_widths": [4, 5, 6],
        "block_sizes": [8, 16, 32],
        "skip_first_dim_choices": [True],
        "binary_stochastic_choices": [False, True],
        "binary_bipolar_choices": [True],
        "binary_training_choices": [True],
        "data_in_levels": [2, 3],
        "data_in_residual_sign": [True, False],
    }




def restrict_search_space(
    search_space: dict,
    layer_choices: list[str],
) -> dict:
    """Return a copy of search_space with linear_layer_choices restricted."""
    restricted = dict(search_space)
    restricted["linear_layer_choices"] = list(layer_choices)
    return restricted
