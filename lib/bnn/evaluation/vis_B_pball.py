"""Utility script to visualize Busemann functions on the Poincaré ball."""

import math
from typing import Tuple

import matplotlib.pyplot as plt
import torch

from ..Geometry import constantcurvature

#TODO: please use double

def _sample_geodesic(model: constantcurvature.Stereographic,
                     direction: torch.Tensor,
                     t_limits: Tuple[float, float],
                     steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the parameter values, points, and Busemann evaluations along gamma_v(t)."""
    start, end = t_limits
    with torch.no_grad():
        ts = torch.linspace(start, end, steps=steps, dtype=model.K.dtype, device=model.K.device)
        tangent = ts.unsqueeze(-1) * direction
        points = model.exp0(tangent)
        values = model.busemann(direction, points)
    return ts, points, values


def main():
    """Visualize the Busemann function along a geodesic in the Poincaré ball."""
    model = constantcurvature.Stereographic(K=-1.0).to(device="cpu", dtype=torch.float64)
    dtype = model.K.dtype
    device = model.K.device

    direction = torch.tensor([1,0], dtype=dtype, device=device)
    direction = direction / direction.norm()

    t_span = (-1.0, 1.0)
    steps = 40000
    ts, points, busemann_vals = _sample_geodesic(model, direction, t_span, steps)

    radius = float(1.0 / math.sqrt(-model.K.item()))
    ts_np = ts.cpu().numpy()
    points_np = points.cpu().numpy()
    busemann_np = busemann_vals.cpu().numpy()

    fig, (ax_ball, ax_line) = plt.subplots(1, 2, figsize=(12, 5))

    # Encode Busemann values as colors to show how they vary along the geodesic.
    scatter = ax_ball.scatter(points_np[:, 0], points_np[:, 1], c=busemann_np, cmap="coolwarm", s=14)
    # Draw the boundary of the Poincaré ball for reference.
    disk = plt.Circle((0.0, 0.0), radius, color="black", fill=False, linewidth=1.2)
    ax_ball.add_patch(disk)
    ax_ball.set_aspect("equal", "box")
    ax_ball.set_title("Geodesic samples in the Poincare ball")
    ax_ball.set_xlabel("x1")
    ax_ball.set_ylabel("x2")
    fig.colorbar(scatter, ax=ax_ball, label="B(v, gamma(t))")

    ax_line.plot(ts_np, busemann_np, color="tab:blue", linewidth=2.0)
    ax_line.axvline(0.0, color="tab:gray", linestyle="--", linewidth=1.0)
    ax_line.set_title("Busemann value along gamma_v(t)")
    ax_line.set_xlabel("t")
    ax_line.set_ylabel("B(v, gamma(t))")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
