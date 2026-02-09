"""Quantized-layer registry, adapter module, and constructor introspection."""
from __future__ import annotations

import inspect
from inspect import Parameter

import torch
import torch.nn as nn

from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearLog,
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearBlockLog,
    LinearBinary,
    LinearBinaryScaling,
    LinearBinaryResidualSign,
)
import math

# Fix BlockMinifloatQuantize.backward() signature for PyTorch autograd compatibility
from chop.nn.quantizers.block_minifloat import BlockMinifloatQuantize as _BMQ

@staticmethod
def _bm_backward(ctx, grad_output):
    return grad_output, None, None, None, None, None

_BMQ.backward = _bm_backward
del _BMQ, _bm_backward


def get_layer_map() -> dict[str, type]:
    """Map of human-readable names to quantized Linear classes."""
    return {
        "Linear": torch.nn.Linear,
        "LinearInteger": LinearInteger,
        "LinearMinifloatDenorm": LinearMinifloatDenorm,
        "LinearMinifloatIEEE": LinearMinifloatIEEE,
        "LinearLog": LinearLog,
        "LinearBlockFP": LinearBlockFP,
        "LinearBlockMinifloat": LinearBlockMinifloat,
        "LinearBlockLog": LinearBlockLog,
        "LinearBinary": LinearBinary,
        "LinearBinaryScaling": LinearBinaryScaling,
        "LinearBinaryResidualSign": LinearBinaryResidualSign,
    }


def ctor_accepts_arg(
    cls: type,
    arg_name: str,
    *,
    allow_varkw: bool = False,
) -> bool:
    """Return True if cls.__init__ accepts arg_name."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if arg_name in params:
        return True
    if allow_varkw:
        return any(p.kind == Parameter.VAR_KEYWORD for p in params.values())
    return False


class QuantLinearAdapter(nn.Module):
    """Shape and sparsity adapter for quantized linear layers."""

    def __init__(
        self,
        inner: nn.Module,
        out_features: int,
        force_3d_input: bool = False,
        force_2d_input: bool = False,
        auto_retry: bool = True,
    ):
        super().__init__()
        if force_3d_input and force_2d_input:
            raise ValueError("force_3d_input and force_2d_input cannot both be True.")

        self.inner = inner
        self.out_features = int(out_features)
        self.force_3d_input = bool(force_3d_input)
        self.force_2d_input = bool(force_2d_input)
        self.auto_retry = bool(auto_retry)

    @property
    def weight(self):
        return getattr(self.inner, "weight", None)

    @property
    def bias(self):
        return getattr(self.inner, "bias", None)

    def _call_inner(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)

    def _to_2d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        return x.reshape(-1, x.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_shape = (*x.shape[:-1], self.out_features)

        x_in = x
        squeezed_seq = False

        if self.force_2d_input:
            x_in = self._to_2d(x_in)
        elif self.force_3d_input and x_in.dim() == 2:
            x_in = x_in.unsqueeze(1)
            squeezed_seq = True

        try:
            y = self._call_inner(x_in)
        except AssertionError:
            if not self.auto_retry:
                raise
            y = self._call_inner(self._to_2d(x))
        except RuntimeError:
            if not self.auto_retry:
                raise
            if x.dim() >= 3:
                y = self._call_inner(self._to_2d(x))
            elif x.dim() == 2:
                y = self._call_inner(x.unsqueeze(1))
                squeezed_seq = True
            else:
                raise

        if getattr(y, "is_sparse", False):
            y = y.to_dense()

        if squeezed_seq and y.dim() == 3 and y.shape[1] == 1:
            y = y.squeeze(1)

        if tuple(y.shape) != tuple(target_shape):
            if y.dim() == 2 and len(target_shape) == 3:
                B, S = target_shape[0], target_shape[1]
                if y.shape[0] == B * S and y.shape[1] == self.out_features:
                    y = y.reshape(B, S, self.out_features)
                elif y.numel() == B * S * self.out_features:
                    y = y.reshape(B, S, self.out_features)
                else:
                    raise RuntimeError(f"Cannot reshape {tuple(y.shape)} -> {target_shape}")
            elif y.numel() == math.prod(target_shape):
                y = y.reshape(*target_shape)
            else:
                raise RuntimeError(f"Cannot reshape {tuple(y.shape)} -> {target_shape}")

        return y
