"""Per-layer quantisation config builders for Optuna trials."""
from __future__ import annotations

from .constants import sanitize_prefix
from .search_space import get_search_space


BASE_RES_SIGN_CONFIG: dict = {
    "bypass": False,
    "data_in_stochastic": False,
    "weight_stochastic": False,
    "bias_stochastic": False,
    "data_in_bipolar": True,
    "weight_bipolar": True,
    "bias_bipolar": True,
    "binary_training": True,
    "data_in_levels": 2,
    "data_in_residual_sign": True,
}


def _p(prefix: str, key: str) -> str:
    """Build a sanitised Optuna param name from a dotted module path."""
    return f"{sanitize_prefix(prefix)}__{key}"


def _cat(trial, name: str, choices, default):
    """Suggest a categorical, falling back to *default* if choices is empty."""
    if choices is None:
        return default
    if not isinstance(choices, (list, tuple)) or len(choices) == 0:
        return default
    return trial.suggest_categorical(name, choices)


def _block_shape_for_linear(block_size: int, *, kind: str) -> list[int]:
    """Return list-shaped block dimensions for chop block quantizers."""
    bs = int(block_size)
    if kind == "bias":
        return [bs]
    return [bs, bs]  # weight or data_in

def _derive_exponent_bias(exponent_width: int) -> int:
    """Derive the standard exponent bias for an IEEE-like format."""
    ew = int(exponent_width)
    if ew <= 0:
        return 0
    return (2 ** (ew - 1)) - 1


def _fixed_point_config(trial, prefix: str, ss: dict) -> dict:
    w = _cat(trial, _p(prefix, "width"), ss["widths"], 8)
    f = _cat(trial, _p(prefix, "frac_width"), ss["frac_widths"], 4)
    return {
        "data_in_width": w, "data_in_frac_width": f,
        "weight_width": w, "weight_frac_width": f,
        "bias_width": w, "bias_frac_width": f,
        "floor": False,
    }


def _minifloat_config(trial, prefix: str, ss: dict) -> dict:
    w = _cat(trial, _p(prefix, "width"), ss["widths"], 8)
    ew = _cat(trial, _p(prefix, "exp_width"), ss["exponent_widths"], 4)
    default_bias = (2 ** (int(ew) - 1)) - 1 if int(ew) > 0 else 0
    eb = _derive_exponent_bias(int(ew))
    return {
        "weight_width": w, "weight_exponent_width": ew, "weight_exponent_bias": eb,
        "data_in_width": w, "data_in_exponent_width": ew, "data_in_exponent_bias": eb,
        "bias_width": w, "bias_exponent_width": ew, "bias_exponent_bias": eb,
    }



def _log_config(trial, prefix: str, ss: dict) -> dict:
    w = _cat(trial, _p(prefix, "width"), ss["widths"], 8)
    eb = _cat(trial, _p(prefix, "exp_bias"), ss["exponent_biases"], 15)
    return {
        "weight_width": w, "weight_exponent_bias": eb,
        "data_in_width": w, "data_in_exponent_bias": eb,
        "bias_width": w, "bias_exponent_bias": eb,
    }


def _block_fp_config(trial, prefix: str, ss: dict) -> dict:
    w = _cat(trial, _p(prefix, "width"), ss["widths"], 8)
    ew = _cat(trial, _p(prefix, "exp_width"), ss["exponent_widths"], 5)
    bs = _cat(trial, _p(prefix, "block_size"), ss["block_sizes"], 16)
    eb = _derive_exponent_bias(int(ew))
    skip_first = True
    return {
        "weight_width": w,
        "weight_exponent_width": ew,
        "weight_exponent_bias": eb,
        "weight_block_size": _block_shape_for_linear(bs, kind="weight"),

        "data_in_width": w,
        "data_in_exponent_width": ew,
        "data_in_exponent_bias": eb,
        "data_in_block_size": _block_shape_for_linear(bs, kind="data_in"),
        "data_in_skip_first_dim": skip_first,

        "bias_width": w,
        "bias_exponent_width": ew,
        "bias_exponent_bias": eb,
        "bias_block_size": _block_shape_for_linear(bs, kind="bias"),
    }



def _block_minifloat_config(trial, prefix: str, ss: dict) -> dict:
    w = _cat(trial, _p(prefix, "width"), ss["widths"], 8)
    ew = _cat(trial, _p(prefix, "exp_width"), ss["exponent_widths"], 4)
    ebw = _cat(trial, _p(prefix, "exp_bias_width"), ss["exponent_bias_widths"], 5)
    bs = _cat(trial, _p(prefix, "block_size"), ss["block_sizes"], 16)
    skip_first = True
    return {
        "weight_width": w, "weight_exponent_width": ew, "weight_exponent_bias_width": ebw,
        "weight_block_size": _block_shape_for_linear(bs, kind="weight"),
        "data_in_width": w, "data_in_exponent_width": ew, "data_in_exponent_bias_width": ebw,
        "data_in_block_size": _block_shape_for_linear(bs, kind="data_in"),
        "data_in_skip_first_dim": skip_first,
        "bias_width": w, "bias_exponent_width": ew, "bias_exponent_bias_width": ebw,
        "bias_block_size": _block_shape_for_linear(bs, kind="bias"),
    }


def _block_log_config(trial, prefix: str, ss: dict) -> dict:
    w = _cat(trial, _p(prefix, "width"), ss["widths"], 8)
    ebw = _cat(trial, _p(prefix, "exp_bias_width"), ss["exponent_bias_widths"], 5)
    bs = _cat(trial, _p(prefix, "block_size"), ss["block_sizes"], 16)
    skip_first = True
    return {
        "weight_width": w, "weight_exponent_bias_width": ebw,
        "weight_block_size": _block_shape_for_linear(bs, kind="weight"),
        "data_in_width": w, "data_in_exponent_bias_width": ebw,
        "data_in_block_size": _block_shape_for_linear(bs, kind="data_in"),
        "data_in_skip_first_dim": skip_first,
        "bias_width": w, "bias_exponent_bias_width": ebw,
        "bias_block_size": _block_shape_for_linear(bs, kind="bias"),
    }


def _binary_config(trial, prefix: str, ss: dict) -> dict:
    w_stoch = _cat(trial, _p(prefix, "weight_stochastic"),
                   ss["binary_stochastic_choices"], False)
    return {"weight_stochastic": w_stoch, "weight_bipolar": True}


def _binary_scaling_config(trial, prefix: str, ss: dict) -> dict:
    x_stoch = _cat(trial, _p(prefix, "data_in_stochastic"),
                   ss["binary_stochastic_choices"], False)
    w_stoch = _cat(trial, _p(prefix, "weight_stochastic"),
                   ss["binary_stochastic_choices"], False)
    b_stoch = _cat(trial, _p(prefix, "bias_stochastic"),
                   ss["binary_stochastic_choices"], False)
    return {
        "bypass": False,
        "data_in_stochastic": x_stoch,
        "weight_stochastic": w_stoch,
        "bias_stochastic": b_stoch,
        "data_in_bipolar": True,
        "weight_bipolar": True,
        "bias_bipolar": True,
        "binary_training": True,
    }


def _residual_sign_config(trial, prefix: str, ss: dict) -> dict:
    levels = _cat(trial, _p(prefix, "data_in_levels"),
                  ss["data_in_levels"], 2)
    res_sign = _cat(trial, _p(prefix, "data_in_residual_sign"),
                    ss["data_in_residual_sign"], True)
    x_stoch = _cat(trial, _p(prefix, "data_in_stochastic"),
                   ss["binary_stochastic_choices"], False)
    w_stoch = _cat(trial, _p(prefix, "weight_stochastic"),
                   ss["binary_stochastic_choices"], False)
    b_stoch = _cat(trial, _p(prefix, "bias_stochastic"),
                   ss["binary_stochastic_choices"], False)
    return {
        **BASE_RES_SIGN_CONFIG,
        "data_in_levels": levels,
        "data_in_residual_sign": res_sign,
        "data_in_stochastic": x_stoch,
        "weight_stochastic": w_stoch,
        "bias_stochastic": b_stoch,
        "data_in_bipolar": True,
        "weight_bipolar": True,
        "bias_bipolar": True,
        "binary_training": True,
    }


_CONFIG_DISPATCH: dict[str, callable] = {
    "LinearInteger": _fixed_point_config,
    "LinearMinifloatDenorm": _minifloat_config,
    "LinearMinifloatIEEE": _minifloat_config,
    "LinearLog": _log_config,
    "LinearBlockFP": _block_fp_config,
    "LinearBlockMinifloat": _block_minifloat_config,
    "LinearBlockLog": _block_log_config,
    "LinearBinary": _binary_config,
    "LinearBinaryScaling": _binary_scaling_config,
    "LinearBinaryResidualSign": _residual_sign_config,
}


def build_quant_config(
    trial,
    layer_type: str,
    prefix: str,
    search_space: dict | None = None,
) -> dict:
    """Build a quantisation config dict for layer_type."""
    if search_space is None:
        search_space = get_search_space()
    builder = _CONFIG_DISPATCH.get(layer_type, _fixed_point_config)
    return builder(trial, prefix, search_space)
