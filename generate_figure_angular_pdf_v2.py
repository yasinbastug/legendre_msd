"""
Generate the receiver-surface angular-density figure for consecutive TS taps.

Conceptual setup:
    The transmitter emits a batch of molecules at t = 0. The same emission is
    observed by consecutive symbol windows of length TS (default: five taps,
    through 4 Ts - 5 Ts):

        tap 0 : t in [0,    TS)
        tap 1 : t in [TS,   2 TS)
        tap 2 : t in [2 TS, 3 TS)
        tap 3 : t in [3 TS, 4 TS)
        tap 4 : t in [4 TS, 5 TS)

    The right panel shows the receiver-surface angular density h(cos(theta))
    of the absorbed-molecule direction, plotted versus theta. This is the
    density with respect to z = cos(theta), equivalently proportional to the
    surface/solid-angle density f_Omega(theta), not the marginal density
    p_theta(theta). Therefore, no sin(theta) factor is applied and the curve
    may be positive at theta = 0.

    The left panel is a 2-D side-view schematic of the geometry with the
    actual absorbed-molecule positions projected onto the xz plane and
    color-coded by tap.

The hits all come from the cached single-emission distribution file built
by main.experiment_save_hits_indexed, so this script does not need to run
any new Brownian simulation.
"""

from __future__ import annotations

import argparse
import glob
import sys
import zipfile
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SIM_DIR = SCRIPT_DIR / "simulations"
DEFAULT_OUT = SCRIPT_DIR.parent / "plots" / "figure_angular_surface_density_5taps.pdf"

sys.path.insert(0, str(SCRIPT_DIR))
from legendre import load_hits_file  # noqa: E402
from main import (  # noqa: E402
    experiment_save_hits_indexed,
    float_tag,
    raw_tag,
)

# Publication palette (matches generate_figures_v3.py detector colors + extras).
TAP_COLORS = (
    "#2B6E46",  # muted green
    "#1F4E79",  # deep blue
    "#7F3B2E",  # muted brick
    "#0E7490",  # teal
    "#595959",  # charcoal
    "#9A5B13",  # muted ochre
)

# Later taps are slightly more transparent / thinner so early taps stay dominant.
TAP_SCATTER_ALPHAS = (0.90, 0.85, 0.80, 0.75, 0.70, 0.65)
TAP_LINE_WIDTHS = (3.0, 2.6, 2.3, 2.0, 1.8, 1.6)

# Typography and axis-size controls.
SCHEMATIC_TEXT_FS = 20
AXIS_LABEL_FS = 22
TICK_LABEL_FS = 18
LEGEND_FS = 17
AXIS_SPINE_LW = 1.9
TICK_WIDTH = 1.8
TICK_LENGTH = 8


def compute_z_cos_theta(xyz: np.ndarray) -> np.ndarray:
    """Compute z = cos(theta) from the +z Tx axis at each hit position."""
    norms = np.linalg.norm(xyz, axis=1)
    return np.clip(xyz[:, 2] / np.maximum(norms, 1e-15), -1.0, 1.0)


def reflected_gaussian_kde_z(
    values_z: np.ndarray,
    grid_z: np.ndarray,
    bw_z: float,
) -> np.ndarray:
    """Boundary-corrected Gaussian KDE for h(z) on z in [-1, 1].

    This estimates the density of z = cos(theta), which is proportional to
    the receiver-surface density with respect to solid angle. It does NOT
    multiply by sin(theta), so the plotted curve may be positive at theta = 0.
    """
    if len(values_z) == 0:
        return np.zeros_like(grid_z)

    # Reflect samples at the two boundaries of [-1, 1].
    sample = np.concatenate([values_z, -2.0 - values_z, 2.0 - values_z])
    diffs = (grid_z[None, :] - sample[:, None]) / bw_z
    kernel = np.exp(-0.5 * diffs * diffs) / (np.sqrt(2.0 * np.pi) * bw_z)
    density = kernel.sum(axis=0) / float(len(values_z))

    # Numerical renormalization over z in [-1, 1].
    area = np.trapz(density, grid_z)
    if area > 0.0:
        density = density / area

    return density


def tap_interval_label(k: int) -> str:
    """Legend label for tap k using symbolic T_s intervals."""
    if k == 0:
        return r"$0 - T_s$"
    if k == 1:
        return r"$T_s - 2T_s$"
    return rf"${k}T_s - {k + 1}T_s$"


def render_geometry_panel(
    ax,
    xyz: np.ndarray,
    masks: list[np.ndarray],
    colors: tuple[str, ...],
    alphas: tuple[float, ...],
    *,
    distance: float,
    radius: float,
    max_hits_per_tap: int,
    rng: np.random.Generator,
    rotation_deg: float = 0.0,
    scatter_fraction: float = 0.5,
) -> None:
    """2-D side-view schematic: Tx point + arrow + Rx sphere outline + colored hits."""

    # Visual layout only: Rx size is set by display_distance / radius.
    # Original physical layout ~2.5; previous enlarged layout ~1.55.
    # Use an in-between ratio so the circle is larger than the first version
    # but smaller than the last enlarged one.
    display_radius = float(radius)
    display_distance = display_radius * 2.05
    _ = distance  # physical Tx-Rx distance is not used for schematic layout

    # 1. Draw Rx background fill behind points.
    rx_base = mpatches.Circle(
        (0.0, 0.0),
        display_radius,
        fill=True,
        facecolor="#f8f9fa",
        edgecolor="none",
        zorder=1,
    )
    ax.add_patch(rx_base)

    # Calculate rotation mathematics.
    theta = np.radians(rotation_deg)
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    tx_x = display_distance * cos_th
    tx_y = display_distance * sin_th

    # 2. Make the Tx node more distinct and rotated.
    ax.plot(
        [tx_x],
        [tx_y],
        marker="o",
        color="#2c3e50",
        markersize=8,
        zorder=5,
    )
    ax.text(
        tx_x,
        tx_y + 0.55 * display_radius / 5.0,
        "Tx",
        fontsize=SCHEMATIC_TEXT_FS,
        fontweight="bold",
        color="#2c3e50",
        ha="center",
        va="bottom",
        zorder=5,
    )

    # 3. Emission arrow between Tx and Rx surface (medium length).
    tip_r = display_radius + 0.12 * display_radius
    tail_r = display_distance - 0.12 * display_radius
    tip_x = tip_r * cos_th
    tip_y = tip_r * sin_th
    tail_x = tail_r * cos_th
    tail_y = tail_r * sin_th

    ax.annotate(
        "",
        xy=(tip_x, tip_y),
        xytext=(tail_x, tail_y),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.4,head_length=0.55",
            lw=1.8,
            color="#555555",
        ),
        zorder=2,
    )

    # 4. Scatter the hits with a thin white edge.
    # Mixed randomly to avoid color dominance.
    all_x = []
    all_y = []
    all_rgba = []

    for mask, color, alpha in zip(masks, colors, alphas):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue

        # Cap first, then randomly keep a fraction so the sphere is less crowded.
        n_cap = min(idx.size, max_hits_per_tap)
        frac = float(np.clip(scatter_fraction, 0.0, 1.0))
        n_keep = max(1, int(round(n_cap * frac)))
        idx = rng.choice(idx, size=n_keep, replace=False)

        all_x.extend(xyz[idx, 2])
        all_y.extend(xyz[idx, 0])

        rgba_color = mcolors.to_rgba(color, alpha)
        all_rgba.extend([rgba_color] * len(idx))

    if all_x:
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_rgba = np.array(all_rgba)

        # Apply 2-D rotation matrix to points.
        rot_x = all_x * cos_th - all_y * sin_th
        rot_y = all_x * sin_th + all_y * cos_th

        shuffle_idx = rng.permutation(len(rot_x))

        scat = ax.scatter(
            rot_x[shuffle_idx],
            rot_y[shuffle_idx],
            s=50,  # Point size unchanged.
            c=all_rgba[shuffle_idx],
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

        # Mask the scatter points so they never bleed outside the base circle.
        scat.set_clip_path(rx_base)

    # 5. Draw a crisp Rx border on top of the points.
    rx_border = mpatches.Circle(
        (0.0, 0.0),
        display_radius,
        fill=False,
        edgecolor="#343a40",
        lw=2.0,
        zorder=4,
    )
    ax.add_patch(rx_border)

    # Add the Rx text label with a subtle white glow/stroke for legibility.
    ax.text(
        0.0,
        0.0,
        "Rx",
        fontsize=SCHEMATIC_TEXT_FS,
        fontweight="bold",
        color="#495057",
        ha="center",
        va="center",
        zorder=5,
        path_effects=[pe.withStroke(linewidth=4, foreground="white")],
    )

    # Viewport: slightly more padding than the enlarged version so Rx is mid-size.
    ax.invert_xaxis()
    ax.set_aspect("equal")
    margin = display_radius * 0.28

    # Adjust Y limits dynamically to accommodate the rotated Tx position.
    max_y = max(display_radius + margin, abs(tx_y) + 1.1)

    ax.set_xlim(display_distance + margin, -(display_radius + margin))
    ax.set_ylim(-max_y, max_y)
    ax.axis("off")


def render_surface_density_panel(
    ax,
    z_per_tap: list[np.ndarray],
    colors: tuple[str, ...],
    line_widths: tuple[float, ...],
    *,
    bw_z: float,
) -> None:
    """Plot h(cos(theta)) versus theta for each tap.

    This is density with respect to z = cos(theta), equivalently proportional
    to surface/solid-angle density f_Omega. It is not the marginal p_theta,
    so no sin(theta) factor is applied.
    """
    theta_grid_deg = np.linspace(0.0, 180.0, 721)
    grid_z_for_plot = np.cos(np.radians(theta_grid_deg))

    # Use an increasing z grid for KDE and integration, then interpolate onto
    # theta increasing from 0 to 180 deg.
    kde_grid_z = np.linspace(-1.0, 1.0, 721)

    # Add subtle background grid for readability.
    ax.grid(True, linestyle="--", alpha=0.4, zorder=1)

    for k, (z_k, color, lw) in enumerate(zip(z_per_tap, colors, line_widths)):
        h_z_increasing = reflected_gaussian_kde_z(z_k, kde_grid_z, bw_z=bw_z)
        h_z_for_plot = np.interp(grid_z_for_plot, kde_grid_z, h_z_increasing)

        ax.plot(
            theta_grid_deg,
            h_z_for_plot,
            color=color,
            lw=lw,
            alpha=0.95,
            label=tap_interval_label(k),
            zorder=3,
        )

    ax.set_xlim(0.0, 180.0)
    ax.set_xticks([0, 60, 120, 180])

    ax.set_xlabel(r"Angle $\theta$ [deg]", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(r"Surface angular density", fontsize=AXIS_LABEL_FS)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FS,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        direction="in",
        top=True,
        right=True,
    )

    # Remove y-axis numeric values but keep the axis label.
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_SPINE_LW)

    legend = ax.legend(
        loc="upper right",
        fontsize=LEGEND_FS if len(z_per_tap) <= 3 else max(13, LEGEND_FS - 3),
        frameon=True,
        handlelength=2.0,
        borderpad=0.5,
        labelspacing=0.35 if len(z_per_tap) > 3 else 0.5,
    )
    legend.get_frame().set_linewidth(AXIS_SPINE_LW)


def is_readable_torch_pt(path: Path, *, min_bytes: int = 1024) -> bool:
    """Return True if path looks like a complete PyTorch zip archive (.pt)."""
    path = Path(path)

    if not path.is_file():
        return False

    if path.stat().st_size < min_bytes:
        return False

    try:
        with zipfile.ZipFile(path, "r") as archive:
            if archive.testzip() is not None:
                return False
        return True
    except (zipfile.BadZipFile, OSError):
        return False


def distribution_candidates(
    sim_dir: Path,
    *,
    distance: float,
    radius: float,
    diffusion_coef: float,
    step_time: float,
    memory: float,
    n_emit_distribution: int,
) -> list[Path]:
    """Candidate distribution .pt files."""
    sim_dir = Path(sim_dir)
    raw = raw_tag(distance, radius, diffusion_coef, step_time, memory)
    preferred = sim_dir / f"Hits_distribution_{raw}_N{n_emit_distribution}.pt"

    paths: list[Path] = []

    if preferred.exists():
        paths.append(preferred)

    paths.extend(Path(p) for p in glob.glob(str(sim_dir / f"Hits_distribution_{raw}_N*.pt")))
    paths.extend(Path(p) for p in glob.glob(str(sim_dir / f"Hits_distribution_{raw}_TS*_MT*_N*.pt")))
    paths.extend(Path(p) for p in glob.glob(str(sim_dir / f"Hits_distribution_d_{float_tag(distance)}*.pt")))

    return sorted({p.resolve() for p in paths})


def resolve_hits_file(args: argparse.Namespace) -> Path:
    """Pick a readable distribution file, or regenerate if requested."""
    sim_dir = Path(args.sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    if args.hits_file:
        explicit = Path(args.hits_file).expanduser()
        if not explicit.is_absolute():
            explicit = (Path.cwd() / explicit).resolve()
        candidates = [explicit]
    else:
        candidates = distribution_candidates(
            sim_dir,
            distance=args.distance,
            radius=args.radius,
            diffusion_coef=args.diffusion_coef,
            step_time=args.step_time,
            memory=args.memory,
            n_emit_distribution=args.n_emit_distribution,
        )

    for path in candidates:
        if is_readable_torch_pt(path):
            return path

        if path.exists():
            print(f"Skipping corrupt/incomplete hits file ({path.stat().st_size:,} bytes): {path}")

    preferred = sim_dir / (
        f"Hits_distribution_{raw_tag(args.distance, args.radius, args.diffusion_coef, args.step_time, args.memory)}"
        f"_N{args.n_emit_distribution}.pt"
    )

    if args.regenerate or not candidates:
        print(f"Generating distribution hits file: {preferred}")
        return experiment_save_hits_indexed(
            filename=str(preferred),
            radius=args.radius,
            total_time=args.memory,
            step_time=args.step_time,
            diffusion_coef=args.diffusion_coef,
            distance=args.distance,
            nof_molecules=args.n_emit_distribution,
            batch_size=args.brownian_batch_size,
            override=args.regenerate,
        )

    raise FileNotFoundError(
        "No readable Hits_distribution_*.pt file found.\n"
        f"  sim_dir: {sim_dir}\n"
        f"  tried: {len(candidates)} path(s)\n\n"
        "The error 'failed finding central directory' means the .pt file is truncated "
        "(copy interrupted, disk full, or simulation killed mid-write).\n\n"
        "Fix options:\n"
        f"  1. Re-run with --regenerate to rebuild: {preferred}\n"
        "  2. Copy a complete file from another machine into sim_dir/\n"
        "  3. Pass --hits-file /path/to/valid/Hits_distribution_....pt"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--hits-file",
        default=None,
        help="Distribution .pt file (default: auto-find under --sim-dir).",
    )
    p.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    p.add_argument("--ts", type=float, default=0.2)
    p.add_argument(
        "--n-taps",
        type=int,
        default=5,
        help="Number of consecutive TS taps to plot (max %d; default through 4 Ts-5 Ts)."
        % len(TAP_COLORS),
    )
    p.add_argument("--memory", type=float, default=5.0, help="Brownian horizon for distribution file tag [s].")
    p.add_argument("--distance", type=float, default=12.5)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--diffusion-coef", type=float, default=79.4)
    p.add_argument("--step-time", type=float, default=1e-4)
    p.add_argument("--n-emit-distribution", type=int, default=1_000_000)
    p.add_argument("--brownian-batch-size", type=int, default=500_000)

    p.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of the distribution .pt if missing or corrupt.",
    )
    p.add_argument(
        "--max-hits-per-tap",
        type=int,
        default=300,
        help="Cap on schematic scatter points per tap after fraction sampling.",
    )
    p.add_argument(
        "--scatter-fraction",
        type=float,
        default=0.5,
        help="Fraction of molecules randomly kept for the schematic scatter (default 1/2).",
    )
    p.add_argument(
        "--kde-bw-z",
        type=float,
        default=0.08,
        help="KDE bandwidth for z = cos(theta).",
    )
    p.add_argument(
        "--rotation-deg",
        type=float,
        default=20.0,
        help="Rotate Tx relative to Rx [deg].",
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_taps > len(TAP_COLORS):
        raise ValueError(f"--n-taps={args.n_taps} exceeds the {len(TAP_COLORS)} configured tap colors.")

    total_window = args.n_taps * args.ts
    if total_window > args.memory:
        raise ValueError(
            f"n_taps * ts = {total_window:g} s exceeds --memory = {args.memory:g} s. "
            "Increase --memory or reduce --n-taps / --ts."
        )

    hits_path = resolve_hits_file(args)
    print(f"Using hits file: {hits_path}")

    hits = load_hits_file(str(hits_path))
    t = hits["t"].astype(np.float64)
    xyz = hits["xyz"].astype(np.float64)

    keep = (t >= 0.0) & (t < total_window)
    t = t[keep]
    xyz = xyz[keep]

    # --- SCALE THE PHYSICAL HITS TO MATCH THE DRAWN RADIUS ---
    # Normalizes the simulated vectors and expands them to exactly `args.radius`
    # so they map accurately to the edge of the visual receiver boundary.
    hit_norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = (xyz / np.maximum(hit_norms, 1e-15)) * args.radius
    # ---------------------------------------------------------

    z_all = compute_z_cos_theta(xyz)

    masks: list[np.ndarray] = []
    z_per_tap: list[np.ndarray] = []

    print(f"TS = {args.ts} s, {args.n_taps} taps -> total window = {total_window:g} s")
    for k in range(args.n_taps):
        low = k * args.ts
        high = (k + 1) * args.ts
        mask = (t >= low) & (t < high)

        masks.append(mask)
        z_per_tap.append(z_all[mask])

        print(f"  tap {k}: {int(mask.sum()):>8,d} hits in [{low:g}, {high:g}) s")

    colors = TAP_COLORS[: args.n_taps]
    alphas = TAP_SCATTER_ALPHAS[: args.n_taps]
    line_widths = TAP_LINE_WIDTHS[: args.n_taps]
    rng = np.random.default_rng(args.seed)

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.08},
    )

    render_geometry_panel(
        ax_left,
        xyz,
        masks,
        colors,
        alphas,
        distance=args.distance,
        radius=args.radius,
        max_hits_per_tap=args.max_hits_per_tap,
        rng=rng,
        rotation_deg=args.rotation_deg,
        scatter_fraction=args.scatter_fraction,
    )

    render_surface_density_panel(
        ax_right,
        z_per_tap,
        colors,
        line_widths,
        bw_z=args.kde_bw_z,
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.95,
        bottom=0.16,
        wspace=0.08,
    )

    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()

    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out, format="pdf", bbox_inches="tight")

    png_out = out.with_suffix(".png")
    fig.savefig(png_out, dpi=200, bbox_inches="tight")

    plt.close(fig)

    print(f"\nSaved {out}")
    print(f"Saved {png_out}")


if __name__ == "__main__":
    main()




