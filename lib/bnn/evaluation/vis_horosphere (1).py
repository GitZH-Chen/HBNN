import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..Geometry.constantcurvature.stereographic import Stereographic


def _contour_levels(values: np.ndarray, bins: int = 9) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("No finite Busemann values available for contouring.")
    lower, upper = np.quantile(finite, [0.05, 0.95])
    if not np.isfinite(lower) or not np.isfinite(upper):
        lower, upper = finite.min(), finite.max()
    if np.isclose(lower, upper):
        upper = lower + 1e-3
    locator = MaxNLocator(nbins=bins)
    return locator.tick_values(lower, upper)


def main():
    K = -1.0
    grid_size = 400
    margin = 1e-4

    model = Stereographic(K=K)
    dtype = model.K.dtype
    device = model.K.device
    radius = float(1.0 / np.sqrt(-K))

    direction = torch.tensor([1.0, 0], dtype=dtype, device=device)
    direction = direction / direction.norm()
    base_point = torch.tensor([0.35 * radius, -0.18 * radius], dtype=dtype, device=device)

    lin = torch.linspace(-radius + margin, radius - margin, steps=grid_size, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(lin, lin, indexing="xy")
    grid = torch.stack((xx, yy), dim=-1)

    # Only evaluate Busemann levels inside the ball to avoid the boundary blow-up.
    mask = grid.pow(2).sum(dim=-1) < (radius - margin) ** 2
    interior_points = grid[mask]

    busemann_values = torch.full(mask.shape, torch.nan, dtype=dtype, device=device)
    busemann_values[mask] = model.busemann(direction, interior_points)

    # Left-translate horospheres by p via gyro addition to realize B(v, ⊖p ⊕ x).
    shifted_points = model.gyroadd(model.gyroinv(base_point), interior_points)
    shifted_values = torch.full_like(busemann_values, torch.nan)
    shifted_values[mask] = model.busemann(direction, shifted_points)

    x_np = xx.cpu().numpy()
    y_np = yy.cpu().numpy()
    busemann_np = busemann_values.cpu().numpy()
    shifted_np = shifted_values.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    levels1 = _contour_levels(busemann_np)
    levels2 = _contour_levels(shifted_np)

    cs1 = axes[0].contour(x_np, y_np, busemann_np, levels=levels1, cmap="viridis")
    cs2 = axes[1].contour(x_np, y_np, shifted_np, levels=levels2, cmap="plasma")
    axes[0].clabel(cs1, fmt="B={:.2f}", fontsize=7)
    axes[1].clabel(cs2, fmt="B={:.2f}", fontsize=7)

    direction_np = direction.cpu().numpy()
    ideal_point = direction_np / np.sqrt(-K)
    base_point_np = base_point.cpu().numpy()

    # Show the geodesic towards the ideal boundary point for reference.
    axes[0].plot([0.0, ideal_point[0]], [0.0, ideal_point[1]], ls="--", lw=1.0, color="tab:gray", alpha=0.8)
    axes[0].scatter([ideal_point[0]], [ideal_point[1]], s=25, color="black", zorder=4)

    # Mark the translation anchor p used in the shifted level sets.
    axes[1].scatter([base_point_np[0]], [base_point_np[1]], s=35, color="tab:red", label="p", zorder=4)
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)

    for ax in axes:
        disk = plt.Circle((0.0, 0.0), radius, color="black", fill=False, linewidth=1.1)
        ax.add_patch(disk)
        ax.set_aspect("equal", "box")
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    axes[0].set_title(r"$B(v, x) = c$")
    axes[1].set_title(r"$B(v, \ominus p \oplus x) = c$")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
