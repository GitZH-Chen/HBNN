import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from typing import Iterable, Optional

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[3]
    if project_root.exists():
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)

from lib.bnn.Geometry.constantcurvature.stereographic import Stereographic
from lib.bnn.Geometry.constantcurvature.hyperboloid import Hyperboloid

# Single-column, horizontal 1×2 figure size (inches)
FIGSIZE_1X2_SINGLE = (8, 3.5)


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


def _int_ticks(values):
    ticks = []
    for val in values:
        tick = int(round(float(val)))
        if tick not in ticks:
            ticks.append(tick)
    return ticks

def visualize_horospheres(
    K: float,
    cs_poincare: Optional[Iterable[float]],
    vs,
    grid_size: int = 400,
    span_scale: float = 1.0,
    cs_lorentz: Optional[Iterable[float]] = None,
):
    """Visualize horospheres (level sets of Busemann functions) in 2D.

    - Left: Poincaré ball model. Plots level sets B(v, x) = c inside the ball and the boundary circle.
    - Right: Lorentz/Hyperboloid model. Plots level sets B(v, x) = c over the spatial plane
      (x1, x2) via the embedding x = (t, x1, x2), and overlays a reference hyperbola.

    Args:
        K: Negative curvature (K < 0).
        cs_poincare: Iterable[float] of level constants for Poincaré contours.
        cs_lorentz: Iterable[float] (optional) for Lorentz contours. If ``None``,
            the same levels as ``cs_poincare`` are used.
        vs: Iterable of unit direction vectors v in R^2 (Euclidean unit). Each v defines B^v.
        grid_size: Resolution of the evaluation grid along each axis.
    """
    if not (K < 0):
        raise ValueError("K must be negative for hyperbolic models.")

    # Instantiate models
    ball = Stereographic(K=K)
    lor = Hyperboloid(K=K)
    dtype = ball.K.dtype
    device = ball.K.device

    radius = float(1.0 / np.sqrt(-K))
    margin = 1e-4

    # Build grids
    lin_ball = torch.linspace(-radius + margin, radius - margin, steps=grid_size, dtype=dtype, device=device)
    xx_b, yy_b = torch.meshgrid(lin_ball, lin_ball, indexing="xy")
    grid_ball = torch.stack((xx_b, yy_b), dim=-1)
    mask_ball = grid_ball.pow(2).sum(dim=-1) < (radius - margin) ** 2

    # Spatial grid for Lorentz (no boundary; use same window for scale consistency)
    span = radius * float(span_scale)
    lin_lor = torch.linspace(-span, span, steps=grid_size, dtype=dtype, device=device)
    xx_l, yy_l = torch.meshgrid(lin_lor, lin_lor, indexing="xy")
    grid_space = torch.stack((xx_l, yy_l), dim=-1)
    points_lor = lor.add_time(grid_space)

    # Prepare figure (single-column horizontal 1×2) with explicit GridSpec margins
    fig = plt.figure(figsize=FIGSIZE_1X2_SINGLE)
    gs = fig.add_gridspec(
        1,
        2,
        left=0.06,
        right=0.98,
        top=0.92,
        bottom=0.15,
        wspace=0,
    )
    left_ax = fig.add_subplot(gs[0, 0])
    right_ax = fig.add_subplot(gs[0, 1])

    # Colors per direction
    colors = plt.cm.tab10.colors

    # Convert to numpy grids for contour calls
    x_b_np = xx_b.cpu().numpy()
    y_b_np = yy_b.cpu().numpy()
    x_l_np = xx_l.cpu().numpy()
    y_l_np = yy_l.cpu().numpy()

    # Ensure lists
    cs_p_list = list(cs_poincare) if cs_poincare is not None else []
    cs_l_list = cs_p_list if cs_lorentz is None else list(cs_lorentz)
    vs = [np.asarray(v, dtype=float) for v in vs]

    # Poincaré ball: contour B(v, x) = c for each v
    for i, v in enumerate(vs):
        v_t = torch.tensor(v, dtype=dtype, device=device)
        v_t = v_t / (v_t.norm().clamp_min(1e-12))
        vals = torch.full(mask_ball.shape, torch.nan, dtype=dtype, device=device)
        with torch.no_grad():
            vals[mask_ball] = ball.busemann(v_t, grid_ball[mask_ball])
        vals_np = vals.cpu().numpy()
        try:
            cs_set = cs_p_list if cs_p_list else _contour_levels(vals_np)
            cs_handle = left_ax.contour(
                x_b_np,
                y_b_np,
                vals_np,
                levels=cs_set,
                colors=[colors[i % len(colors)]],
            )
            if i == 0:
                left_ax.clabel(cs_handle, inline=True, fmt="B^v(x)={:.2f}", fontsize=7)
        except Exception:
            pass

    # Boundary of the ball
    disk = plt.Circle((0.0, 0.0), radius, color="black", fill=False, linewidth=1.1)
    left_ax.add_patch(disk)
    left_ax.set_aspect("equal", "box")
    left_ax.set_xlim(-radius, radius)
    left_ax.set_ylim(-radius, radius)
    left_ax.set_title("Poincaré")
    left_ax.set_xlabel(r"$x_1$")
    left_ax.set_ylabel(r"$x_2$")
    x_min, x_max = left_ax.get_xlim()
    y_min, y_max = left_ax.get_ylim()
    left_ax.set_xticks(_int_ticks((x_min, 0.5 * (x_min + x_max), x_max)))
    left_ax.set_yticks(_int_ticks((y_min, 0.5 * (y_min + y_max), y_max)))
    for spine in left_ax.spines.values():
        spine.set_visible(False)
    left_ax.axhline(0.0, color="black", linewidth=0.9, zorder=0)
    left_ax.axvline(0.0, color="black", linewidth=0.9, zorder=0)

    # Lorentz/Hyperboloid: B(v, x(t, s)) over spatial grid
    for i, v in enumerate(vs):
        v_t = torch.tensor(v, dtype=dtype, device=device)
        v_t = v_t / (v_t.norm().clamp_min(1e-12))
        with torch.no_grad():
            vals_l = lor.busemann(v_t, points_lor)
        vals_l_np = vals_l.cpu().numpy()
        try:
            cs_set = cs_l_list if cs_l_list else _contour_levels(vals_l_np)
            cs_handle = right_ax.contour(
                x_l_np,
                y_l_np,
                vals_l_np,
                levels=cs_set,
                colors=[colors[i % len(colors)]],
            )
            if i == 0:
                right_ax.clabel(cs_handle, inline=True, fmt="B^v(x)={:.2f}", fontsize=7)
        except Exception:
            pass

    # Overlay a reference hyperbola curve (unit hyperbola scaled by 1/sqrt(-K))
    # Here we draw x^2 - y^2 = (1/sqrt(-K))^2 in the (x1,x2)-plane as a visual cue.
    ys = np.linspace(-span, span, 512)
    a2 = radius ** 2
    xs_pos = np.sqrt(ys ** 2 + a2)
    right_ax.plot(xs_pos, ys, "k--", linewidth=1.0, alpha=0.7)
    right_ax.plot(-xs_pos, ys, "k--", linewidth=1.0, alpha=0.7)
    right_ax.set_aspect("equal", "box")
    right_ax.set_xlim(-span, span)
    right_ax.set_ylim(-span, span)
    right_ax.set_title("Lorentz")
    right_ax.set_xlabel(r"$(x_s)_1$")
    right_ax.set_ylabel(r"$(x_s)_2$")
    rx_min, rx_max = right_ax.get_xlim()
    ry_min, ry_max = right_ax.get_ylim()
    right_ax.set_xticks(_int_ticks((rx_min, 0.5 * (rx_min + rx_max), rx_max)))
    right_ax.set_yticks(_int_ticks((ry_min, 0.5 * (ry_min + ry_max), ry_max)))

    output_path = Path("horosphere_plot.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.show()


def main():
    verbose = False
    frontsize = 14
    plt.rcParams.update(
        {
            "font.size": frontsize,
            "axes.titlesize": frontsize,
            "axes.labelsize": frontsize-2,
            "xtick.labelsize": frontsize-2,
            "ytick.labelsize": frontsize-2,
            "legend.fontsize": frontsize-2,
            "figure.titlesize": frontsize,
        }
    )
    span_scale = 2
    K = -1.0
    # c_min, c_max, c_step = -1.0, 3.0, 0.5
    # num_levels = int(round((c_max - c_min) / c_step)) + 1
    # cs = np.linspace(c_min, c_max, num=num_levels, dtype=float).tolist()
    cs_poincare = np.linspace(-1, 2, num=10, dtype=float).tolist() #[-1, 1, 1.5, 2]
    cs_lorentz = np.linspace(1, 2, num=10, dtype=float).tolist()  # Set to a list (e.g., [-1, 0.5, 1.0]) for custom Lorentz levels
    vs = [
        [1, 0],
        # [-1, 0],
    ]
    grid_size = 400

    vs = [np.asarray(v, dtype=float) for v in vs]
    vs = [v / (np.linalg.norm(v) + 1e-12) for v in vs]
    lor = Hyperboloid(K=K)
    ball = Stereographic(K=K)
    dtype = lor.K.dtype
    device = lor.K.device

    radius = float(1.0 / np.sqrt(-K))
    margin = 1e-4
    span = 2.0 * radius * float(span_scale)
    resolution = min(200, grid_size)

    fig = plt.figure(figsize=FIGSIZE_1X2_SINGLE)
    gs = fig.add_gridspec(
        1,
        2,
        left=0.06,
        right=0.98,
        top=0.92,
        bottom=0.15,
        wspace=0.03,
    )
    left_ax = fig.add_subplot(gs[0, 0])
    right_ax = fig.add_subplot(gs[0, 1], projection="3d")
    # Ensure comparable drawing area to the 2D subplot
    try:
        right_ax.set_box_aspect((1, 1, 1))
        right_ax.set_proj_type("ortho")
        # Reduce label padding to avoid extra space requirement
        right_ax.xaxis.labelpad = 2
        right_ax.yaxis.labelpad = 2
        right_ax.zaxis.labelpad = 2
    except Exception:
        pass

    lin_ball = torch.linspace(
        -radius + margin, radius - margin, steps=grid_size, dtype=dtype, device=device
    )
    xx_b, yy_b = torch.meshgrid(lin_ball, lin_ball, indexing="xy")
    grid_ball = torch.stack((xx_b, yy_b), dim=-1)
    mask_ball = grid_ball.pow(2).sum(dim=-1) < (radius - margin) ** 2
    x_b_np = xx_b.detach().cpu().numpy()
    y_b_np = yy_b.detach().cpu().numpy()

    cs_p_list = list(cs_poincare) if cs_poincare is not None else []
    cs_l_list = cs_p_list if cs_lorentz is None else list(cs_lorentz)
    default_colors = ["tab:red", "tab:blue", "tab:green"]
    if len(vs) <= len(default_colors):
        horosphere_colors = default_colors[: len(vs)]
    else:
        cmap = plt.cm.get_cmap("tab10", len(vs) - len(default_colors))
        extra_colors = [cmap(i) for i in range(len(vs) - len(default_colors))]
        horosphere_colors = default_colors + extra_colors

    for i, (v_np, color) in enumerate(zip(vs, horosphere_colors)):
        v_t = torch.tensor(v_np, dtype=dtype, device=device)
        v_t = v_t / v_t.norm().clamp_min(1e-12)
        vals = torch.full(mask_ball.shape, torch.nan, dtype=dtype, device=device)
        with torch.no_grad():
            vals[mask_ball] = ball.busemann(v_t, grid_ball[mask_ball])
        vals_np = vals.detach().cpu().numpy()
        levels = cs_p_list if cs_p_list else _contour_levels(vals_np)
        try:
            contour_set = left_ax.contour(
                x_b_np,
                y_b_np,
                vals_np,
                levels=levels,
                colors=[color],
                linestyles="-",
            )
            if verbose and i == 0:
                left_ax.clabel(
                    contour_set,
                    inline=True,
                    fmt=lambda val: f"$B={val:.1f}$",
                    fontsize=7,
                )
        except Exception:
            continue

    disk = plt.Circle((0.0, 0.0), radius, color="black", fill=False, linewidth=1.1)
    left_ax.add_patch(disk)
    left_ax.set_aspect("equal", "box")
    left_ax.set_xlim(-radius, radius)
    left_ax.set_ylim(-radius, radius)
    left_ax.set_title("Poincaré")
    left_ax.set_xlabel(r"$x_1$")
    left_ax.set_ylabel(r"$x_2$")
    lx_min, lx_max = left_ax.get_xlim()
    ly_min, ly_max = left_ax.get_ylim()
    left_ax.set_xticks(_int_ticks((lx_min, 0.5 * (lx_min + lx_max), lx_max)))
    left_ax.set_yticks(_int_ticks((ly_min, 0.5 * (ly_min + ly_max), ly_max)))

    lin = torch.linspace(-span, span, steps=resolution, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(lin, lin, indexing="xy")
    base_space = torch.stack((xx, yy), dim=-1)
    with torch.no_grad():
        hyperboloid_points = lor.add_time(base_space)

    points_np = hyperboloid_points.detach().cpu().numpy()
    t_np = points_np[..., 0]
    x_np = points_np[..., 1]
    y_np = points_np[..., 2]
    z_origin = radius
    z_min, z_max = max(z_origin, t_np.min()), t_np.max()

    busemann_fields = []
    level_values = None
    for v_np in vs:
        v_t = torch.tensor(v_np, dtype=dtype, device=device)
        v_t = v_t / v_t.norm().clamp_min(1e-12)
        with torch.no_grad():
            busemann_vals = lor.busemann(v_t, hyperboloid_points)
        busemann_np = busemann_vals.detach().cpu().numpy()
        busemann_fields.append(busemann_np)
        if level_values is None:
            if cs_l_list:
                level_values = np.asarray(cs_l_list, dtype=float)
            else:
                level_values = _contour_levels(busemann_np)

    if level_values is None or level_values.size == 0:
        raise RuntimeError("No contour levels available for Lorentz horospheres.")

    level_values = np.asarray(level_values, dtype=float)
    level_values.sort()

    surface_field = busemann_fields[0]
    norm = Normalize(vmin=surface_field.min(), vmax=surface_field.max())
    facecolors = plt.cm.viridis(norm(surface_field))
    if facecolors.ndim == 3:
        facecolors[..., -1] = 1.0

    right_ax.plot_surface(
        x_np,
        y_np,
        t_np,
        facecolors=facecolors,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    sqrt_negK = float(np.sqrt(-K))
    placed_level_labels = set()
    u_span = span * 6.0
    u_resolution = 2500
    for v_np, color in zip(vs, horosphere_colors):
        v_norm = np.linalg.norm(v_np)
        if v_norm < 1e-12:
            continue
        v_dir = v_np / v_norm
        v_perp = np.array([-v_dir[1], v_dir[0]], dtype=float)
        v_perp_norm = np.linalg.norm(v_perp)
        if v_perp_norm < 1e-12:
            v_perp = np.array([0.0, 1.0], dtype=float)
        else:
            v_perp = v_perp / v_perp_norm

        u_vals = np.linspace(-u_span, u_span, num=u_resolution)
        step = u_vals[1] - u_vals[0]

        for level in level_values:
            c_const = np.exp(level * sqrt_negK) / sqrt_negK
            alpha = (u_vals ** 2 - (1.0 / K) - c_const ** 2) / (2.0 * c_const)
            x_t_vals = alpha + c_const
            x_s_vals = np.outer(alpha, v_dir) + np.outer(u_vals, v_perp)

            spatial_norm = np.linalg.norm(x_s_vals, axis=1)
            expected_t = np.sqrt((1.0 / -K) + spatial_norm ** 2)
            tol_spatial = 1e-6
            tol_time = 1e-3
            mask = (
                (np.abs(x_s_vals[:, 0]) <= span + tol_spatial)
                & (np.abs(x_s_vals[:, 1]) <= span + tol_spatial)
                & (x_t_vals >= 0.0)
                & (np.abs(x_t_vals - expected_t) <= tol_time)
            )
            if not np.any(mask):
                continue

            u_sel = u_vals[mask]
            x_t_sel = x_t_vals[mask]
            x_s_sel = x_s_vals[mask]

            if u_sel.size < 2:
                continue

            breaks = np.where(np.abs(np.diff(u_sel)) > step * 1.5)[0]
            segments = np.split(np.arange(u_sel.size), breaks + 1)

            for segment_idx in segments:
                if segment_idx.size < 2:
                    continue

                xs_seg = x_s_sel[segment_idx]
                xt_seg = x_t_sel[segment_idx]

                right_ax.plot(
                    xs_seg[:, 0],
                    xs_seg[:, 1],
                    xt_seg,
                    color=color,
                    linestyle="-",
                    alpha=1.0,
                    zorder=10,
                )

                if verbose and level not in placed_level_labels:
                    mid = segment_idx[segment_idx.size // 2]
                    label_point = np.array(
                        [x_s_sel[mid, 0], x_s_sel[mid, 1], x_t_sel[mid]], dtype=float
                    )
                    right_ax.text(
                        label_point[0],
                        label_point[1],
                        label_point[2],
                        f"$B={level:.1f}$",
                        color=color,
                        fontsize=8,
                        ha="center",
                        va="bottom",
                    )
                    placed_level_labels.add(level)

    right_ax.set_xlim(-span, span)
    right_ax.set_ylim(-span, span)
    right_ax.set_zlim(z_origin, z_max)
    right_ax.set_xlabel(r"$(x_s)_1$")
    right_ax.set_ylabel(r"$(x_s)_2$")
    right_ax.set_zlabel(r"$x_t$")
    right_ax.set_title("Lorentz")
    right_ax.view_init(elev=25.0, azim=-60.0)
    r3d_x_min, r3d_x_max = right_ax.get_xlim()
    r3d_y_min, r3d_y_max = right_ax.get_ylim()
    r3d_z_min, r3d_z_max = right_ax.get_zlim()
    right_ax.set_xticks(_int_ticks((r3d_x_min, 0.5 * (r3d_x_min + r3d_x_max), r3d_x_max)))
    right_ax.set_yticks(_int_ticks((r3d_y_min, 0.5 * (r3d_y_min + r3d_y_max), r3d_y_max)))
    right_ax.set_zticks(_int_ticks((r3d_z_min, 0.5 * (r3d_z_min + r3d_z_max), r3d_z_max)))
    right_ax.grid(False)

    output_path = Path("horosphere.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
