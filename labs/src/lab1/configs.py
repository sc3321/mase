"""Config builders for quantisation and pruning passes."""

from __future__ import annotations


def make_quant_config(width: int, frac_width: int | None = None) -> dict:
    """
    Build a quantisation config that sets every Linear layer to width-bit
    fixed-point with fractional width frac_width (default width // 2).
    """
    if frac_width is None:
        frac_width = width // 2
    return {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {
            "config": {
                "name": "integer",
                "data_in_width": width,
                "data_in_frac_width": frac_width,
                "weight_width": width,
                "weight_frac_width": frac_width,
                "bias_width": width,
                "bias_frac_width": frac_width,
            }
        },
    }


def make_prune_config(sparsity: float, method: str = "l1-norm") -> dict:
    return {
        "weight": {
            "sparsity": sparsity,
            "method": method,
            "scope": "local",
        },
        "activation": {
            "sparsity": sparsity,
            "method": method,
            "scope": "local",
        },
    }
