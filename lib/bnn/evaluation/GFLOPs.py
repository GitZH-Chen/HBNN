from lib.bnn.BMLR import BMLR
from lib.poincare.layers.PMLR import (
    PoincareMLR,
    BusemannPoincareMLR,
    UnidirectionalPoincareMLR,
)
from lib.lorentz.layers.LMLR import LorentzMLR
import torch.nn as nn

"""
GFLOPs Estimator for Hyperbolic/EUclidean MLR layers.

Compares the forward-pass FLOPs (in GFLOPs) for:
- Euclidean MLR: `nn.Linear` (reference)
- Poincaré MLR: `PoincareMLR`, `BusemannPoincareMLR`, `UnidirectionalPoincareMLR`
  vs. `BMLR` with metric="poincare"
- Lorentz MLR: `LorentzMLR` vs. `BMLR` with metric="lorentz"

Curvature notes (K < 0):
- Lorentz model uses parameter `k` such that `k = -1/K`.
- Poincaré ball uses parameter `c` such that `c = -K` and `k = K` internally.

Typical usage of these MLRs appears in `image_classification/models/classifier.py`.
This module provides fast, closed-form FLOPs estimates without running a forward pass.
The counts are approximate but sized to the dominant operations per layer.
"""


def _gflops_from_flops(flops: float) -> float:
    return float(flops) / 1e9


def gflops_linear(in_features: int, out_features: int, batch_size: int, bias: bool = True) -> float:
    """Estimate GFLOPs for a standard `nn.Linear` forward.

    - MatMul: ~2 * B * in_features * out_features (mul+add)
    - Bias add: ~B * out_features
    """
    flops = 2 * batch_size * in_features * out_features
    if bias:
        flops += batch_size * out_features
    return _gflops_from_flops(flops)


def gflops_bmlr(layer: BMLR, batch_size: int) -> float:
    """Estimate GFLOPs for BMLR.

    Dominant cost per sample:
    - Poincaré metric: a matrix product `x @ v` of size (B, d) x (d, C) -> ~2*B*d*C
    - Lorentz metric: a matrix product on spatial part (d) similarly dominates.
    Additional elementwise ops (logs/divs/scales) are lower order and added as a
    small constant per class.
    """
    d = layer.d
    C = layer.C
    # dominant dot-products
    flops = 2 * batch_size * d * C
    # small per-(sample,class) scalar ops (log/div/add/mul): ~12
    flops += 12 * batch_size * C
    # once-per-class small overheads
    flops += 4 * d * C
    return _gflops_from_flops(flops)


def gflops_unidirectional_poincare(layer: UnidirectionalPoincareMLR, batch_size: int) -> float:
    """Estimate GFLOPs for UnidirectionalPoincareMLR.

    Dominant cost: matrix multiply (B, d) x (d, C) for `matmul(rcx, z_unit)`.
    Add small per-(sample,class) elementwise overhead.
    """
    d = layer.feat_dim
    C = layer.num_outcome
    flops = 2 * batch_size * d * C
    flops += 12 * batch_size * C
    return _gflops_from_flops(flops)


def gflops_poincare_mlr(layer: PoincareMLR, batch_size: int) -> float:
    """Estimate GFLOPs for PoincareMLR (Ganea et al. Eq. 25).

    Implementation evaluates per-class hyperplane distance with gyro-ops.
    Per (sample, class) cost is O(d) with a moderate constant. We approximate as:
    ~ (12 * d) flops per (sample, class) plus small scalar overheads.
    """
    d = layer.feat_dim
    C = layer.num_outcome
    flops = batch_size * C * (12 * d + 24)
    return _gflops_from_flops(flops)


def gflops_busemann_poincare(layer: BusemannPoincareMLR, batch_size: int) -> float:
    """Estimate GFLOPs for BusemannPoincareMLR.

    Uses a per-class loop with Möbius add, norms, and logs. Similar O(d) per
    (sample, class), with a slightly higher constant vs. PoincareMLR.
    """
    d = layer.feat_dim
    C = layer.num_outcome
    flops = batch_size * C * (14 * d + 32)
    return _gflops_from_flops(flops)


def gflops_lorentz_mlr(layer: LorentzMLR, batch_size: int) -> float:
    """Estimate GFLOPs for LorentzMLR.

    Dominant cost per (sample, class): inner product on spatial dims of x and z,
    i.e., ~2*d_s (mul+add). Remaining scalar ops are low order.
    """
    C = layer.a.shape[0]
    d_spatial = layer.z.shape[1]
    flops = batch_size * C * (2 * d_spatial + 20)
    return _gflops_from_flops(flops)


def count_gflops(module, batch_size: int) -> float:
    """Dispatch GFLOPs estimate for a decoder/MLR module.

    Supported modules:
    - nn.Linear
    - BMLR
    - PoincareMLR
    - BusemannPoincareMLR
    - UnidirectionalPoincareMLR
    - LorentzMLR
    """
    if isinstance(module, nn.Linear):
        return gflops_linear(module.in_features, module.out_features, batch_size, module.bias is not None)
    if isinstance(module, BMLR):
        return gflops_bmlr(module, batch_size)
    if isinstance(module, UnidirectionalPoincareMLR):
        return gflops_unidirectional_poincare(module, batch_size)
    if isinstance(module, PoincareMLR):
        return gflops_poincare_mlr(module, batch_size)
    if isinstance(module, BusemannPoincareMLR):
        return gflops_busemann_poincare(module, batch_size)
    if isinstance(module, LorentzMLR):
        return gflops_lorentz_mlr(module, batch_size)
    raise TypeError(f"Unsupported module type for GFLOPs estimation: {type(module).__name__}")
