"""
Benchmark efficiency of MLR layers in Euclidean, Poincaré, and Lorentz geometries.

Measures average forward-pass time for the following layers (in order):
- Euclidean MLR: nn.Linear(d, C)
- Poincaré MLR: PoincareMLR(d, C, ball)
- Unidirectional Poincaré MLR: UnidirectionalPoincareMLR(d, C, ball)
- Busemann Poincaré MLR: BusemannPoincareMLR(d, C, ball)
- Poincaré BMLR: BMLR(metric="poincare", K=ball.k)
- Lorentz MLR: LorentzMLR(manifold, d+1, C)
- Lorentz BMLR: BMLR(metric="lorentz", K=K)

Inputs are generated on the appropriate manifold:
- Euclidean: x_e ∈ R^{B×d}
- Poincaré: x_p = exp_0(ε) ∈ B_k^{B×d}
- Lorentz:  x_l = exp_0([0, ε]) ∈ H_K^{B×(d+1)}

Arguments
- --batches: number of timed iterations (default: 200)
- --batch-size: batch size per forward (default: 512)
- --dim: feature dimension d (default: 512)
- --classes: number of classes C (default: 10)
- --warmup: warmup iterations before timing (default: 50)
- --dtype: float32 | float16 | bfloat16 (default: float32)
- --K: negative curvature K (default: -1.0)

Notes
- Uses GPU when available. Synchronizes CUDA around timers.
- Runs in torch.no_grad() and eval() for speed and consistency.
"""

import time
from typing import Dict, Tuple

import torch
import torch.nn as nn

from lib.bnn.BMLR import BMLR
from lib.poincare.layers.PMLR import (
    PoincareMLR,
    BusemannPoincareMLR,
    UnidirectionalPoincareMLR,
)
from lib.lorentz.layers.LMLR import LorentzMLR
from lib.geoopt.manifolds.stereographic import PoincareBall
from lib.lorentz.manifold import CustomLorentz


def _get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _parse_dtype(name: str, device: torch.device) -> torch.dtype:
    # Only two fast/stable choices are supported
    name = name.lower()
    if name == "float64":
        return torch.float64
    return torch.float32


@torch.no_grad()
def _benchmark_module(module: nn.Module, x: torch.Tensor, iters: int, warmup: int) -> float:
    device = x.device
    module.eval()

    # Warmup
    for _ in range(warmup):
        _ = module(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = module(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    return (t1 - t0) / max(1, iters)


def _make_inputs(
    batch_size: int,
    dim: int,
    K: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, PoincareBall, CustomLorentz]:
    # Poincaré ball with c = -K (K < 0)
    ball = PoincareBall(c=-K, learnable=False)
    # Lorentz manifold with k = -1/K
    manifold = CustomLorentz(k=-1.0 / K, learnable=False)

    # Base Euclidean features
    eps = 0.1
    x_e = torch.randn(batch_size, dim, device=device, dtype=dtype) * eps

    # Poincaré points via expmap at origin
    x_p = ball.expmap0(x_e.clone())

    # Lorentz points: pad a zero time-like tangent then expmap at origin
    x_tan = nn.functional.pad(x_e.clone(), (1, 0), value=0.0)
    x_l = manifold.expmap0(x_tan)

    return x_e, x_p, x_l, ball, manifold


def _make_layers(
    dim: int,
    classes: int,
    K: float,
    ball: PoincareBall,
    manifold: CustomLorentz,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, nn.Module]:
    layers: Dict[str, nn.Module] = {}

    # Euclidean baseline
    layers["Euclidean MLR"] = nn.Linear(dim, classes, bias=True)

    # Poincaré family
    layers["Poincaré MLR"] = PoincareMLR(dim, classes, ball)
    layers["Unidir Poincaré MLR"] = UnidirectionalPoincareMLR(dim, classes, bias=True, ball=ball)
    layers["Busemann Poincaré MLR"] = BusemannPoincareMLR(dim, classes, ball)
    layers["Poincaré BMLR"] = BMLR(n_classes=classes, dim=dim, K=ball.k, metric="poincare")

    # Lorentz family
    layers["Lorentz MLR"] = LorentzMLR(manifold, num_features=dim + 1, num_classes=classes)
    layers["Lorentz BMLR"] = BMLR(n_classes=classes, dim=dim, K=K, metric="lorentz")

    # Move to device/dtype
    for k, m in layers.items():
        m.to(device=device, dtype=dtype)
    return layers


def run_benchmark(
    batches: int = 200,
    batch_size: int = 512,
    dim: int = 512,
    classes: int = 10,
    warmup: int = 50,
    dtype_name: str = "float32",
    K: float = -1.0,
):
    device = _get_device()
    dtype = _parse_dtype(dtype_name, device)

    x_e, x_p, x_l, ball, manifold = _make_inputs(batch_size, dim, K, device, dtype)
    layers = _make_layers(dim, classes, K, ball, manifold, device, dtype)

    # Fixed output order
    order = [
        "Euclidean MLR",
        "Poincaré MLR",
        "Unidir Poincaré MLR",
        "Busemann Poincaré MLR",
        "Poincaré BMLR",
        "Lorentz MLR",
        "Lorentz BMLR",
    ]

    # Map layer to its input tensor
    inputs = {
        "Euclidean MLR": x_e,
        "Poincaré MLR": x_p,
        "Unidir Poincaré MLR": x_p,
        "Busemann Poincaré MLR": x_p,
        "Poincaré BMLR": x_p,
        "Lorentz MLR": x_l,
        "Lorentz BMLR": x_l,
    }

    print(f"Device: {device.type}, dtype: {str(dtype).split('.')[-1]}, K={K}")
    print(f"Batches: {batches}, Batch size: {batch_size}, d={dim}, C={classes}, Warmup: {warmup}")
    print()
    header = f"{'Layer':30s} {'ms/iter':>12s} {'samples/s':>12s}"
    print(header)
    print("-" * len(header))

    for name in order:
        mod = layers[name]
        x = inputs[name]
        sec = _benchmark_module(mod, x, iters=batches, warmup=warmup)
        ms = sec * 1e3
        sps = batch_size / sec if sec > 0 else float("inf")
        print(f"{name:30s} {ms:12.3f} {sps:12.1f}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Efficiency benchmark for (hyperbolic) MLR layers")
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--classes", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"]) 
    p.add_argument("--K", type=float, default=-1.0, help="Negative curvature (K<0)")
    args = p.parse_args()

    run_benchmark(
        batches=args.batches,
        batch_size=args.batch_size,
        dim=args.dim,
        classes=args.classes,
        warmup=args.warmup,
        dtype_name=args.dtype,
        K=args.K,
    )
