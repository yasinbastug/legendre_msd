"""Run every detection experiment of the paper and its revision.

Each sweep varies one quantity at the paper operating point (N_Tx = 100,
T_s = 0.2 s, L = 10, gamma = 1e-6, epsilon = 1e-12, B = 100, K by the 75% F_hit
rule, no environmental noise), runs all five detectors, and appends one row per
detector and point to experiments/<prefix>_<sweep>__<options>/summary.csv.
Finished points are skipped on a rerun, so interrupted sweeps resume.

Paper figures (prefix "paper"):
    ntx          BER vs. N_Tx = 25..250 (Fig. 2) with P01/P10 curves (Fig. 4)
    ts           BER vs. T_s = 0.2..0.5 s (Fig. 3)

Revision experiments (prefix "reviewer"):
    noise        environmental noise lambda_env / lambda_0, uniform on the Rx
                 surface and in time; detectors know lambda_env  (R2-3, R3-1)
    L, gamma,    Legendre order, ridge penalty, numerical floor,
    floor, B     number of arrival-time bins                     (R2-2, R3-3)
    fit          held-out accuracy of the Legendre density vs. L (R2-1)
    memory, K    detector memory as F_hit fraction / in taps      (R1-7)
    tail         ISI beyond the detector memory ignored (0) or
                 modelled as uniform background (1)              (R1-7)
    chmem,       channel memory in seconds / as F_hit fraction,
    chtarget     detector memory fixed by --detector-taps        (R1-7)
    direction    error in the assumed Tx direction [deg]         (R2-4)
    distance,    matched geometry: each value gets its own simulated
    radius,      channel and library (10 s channel, K <= 18, ISI tail
    diffusion    as uniform background)

Channel vs. detector memory: by default both follow the 75% F_hit rule, so the
simulated channel ends where the detector memory ends (as in the submission).
--channel-memory SECONDS or --channel-target FRACTION simulate a longer channel;
--tail-background then adds the expected count of the unmodelled ISI tail to
lambda_env. Hit pools covering the channel are simulated automatically.

Examples:
    python experiments.py ntx                                   # Figs. 2 and 4
    python experiments.py ts --channel-target 0.9 --tail-background
    python experiments.py noise --n-bits 2000 --full-device cpu # quick check
    python experiments.py plot noise                            # replot a sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as gf
from legendre import (
    DEFAULT_EPS_FLOOR,
    _trapz,
    fit_legendre_ls,
    legendre_basis,
    load_hits_file,
    load_legendre_library,
    truncate_legendre_library,
    unit_rows,
    with_eps_floor,
)
from main import (
    DEFAULT_AXIS,
    add_background_noise,
    ensure_legendre_library,
    find_compatible_bit_pool_file,
    find_compatible_distribution_file,
    float_tag,
    pool_horizon,
    generate_bit_sequence,
    generate_received_symbols_from_indexed_impulse,
    parse_args as parse_channel_args,
)


# Paper operating point.
N_TV_BINS = 20

# Values shown in the plots, when a subset of the swept values is wanted.
PLOT_VALUES = {"L": (2, 6, 10, 20)}
# Detectors that do not depend on the swept parameter, drawn as plain reference lines.
REFERENCE_DETECTORS = {"L": ("fixed_threshold", "viterbi")}

PAPER = {
    "L": 10, "gamma": 1e-6, "floor": DEFAULT_EPS_FLOOR, "B": 100, "memory": 0.75, "direction": 0.0,
    "distance": 12.5, "radius": 5.0, "diffusion": 79.4, "chtarget": 0.75,
}

# Sweeps that produce the paper figures (folder prefix "paper" instead of "reviewer").
PAPER_SWEEPS = ("ntx", "ts")

# Geometry sweeps change the physical channel: sweep name -> main.py channel attribute.
GEOMETRY_SWEEPS = {"distance": "distance", "radius": "radius", "diffusion": "diffusion_coef"}
GEOMETRY_CHANNEL_MEMORY = 10.0   # seconds of simulated ISI per geometry
GEOMETRY_MAX_TAPS = 18           # trellis budget: 2^17 states

SWEEPS = {
    # name: (column, default values, x-axis label, log-x)
    "ntx": ("N_Tx", tuple(float(n) for n in range(25, 251, 25)), r"$N_{\mathrm{Tx}}$", False),
    "ts": ("ts", (0.2, 0.3, 0.4, 0.5), r"Symbol duration $T_s$ [s]", False),
    "noise": ("noise_ratio", (0.0, 0.25, 0.5, 1.0, 2.0, 4.0), r"$\lambda_{\mathrm{env}} / \lambda_0$", False),
    "L": ("L", (2, 4, 6, 8, 10, 14, 20), r"Legendre order $L$", False),
    "gamma": ("gamma", (0.0, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0), r"Ridge penalty $\gamma$", True),
    "floor": ("floor", (1e-12, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1), r"Numerical floor $\epsilon$", True),
    "B": ("B", (10, 25, 50, 100, 200), r"Number of time bins $B$", True),
    "memory": ("memory_target", (0.5, 0.6, 0.65, 0.7, 0.75), r"$F_{\mathrm{hit}}$ fraction defining memory", False),
    "direction": ("direction_error_deg", (0.0, 5.0, 10.0, 20.0, 45.0, 90.0), r"Tx direction error [deg]", False),
    "tail": ("tail_background", (0.0, 1.0), r"ISI tail beyond $K$", False),
    "K": ("detector_taps", tuple(float(k) for k in range(1, 21)), r"Detector memory $K$ [taps]", False),
    "distance": ("distance_um", (10.0, 12.5, 15.0, 17.5), r"Tx-Rx distance $d$ [$\mu$m]", False),
    "radius": ("radius_um", (3.0, 4.0, 5.0, 6.0, 7.0), r"Rx radius $r$ [$\mu$m]", False),
    "chmem": ("channel_memory_s", (7.5, 10.0, 15.0, 20.0, 30.0), r"Channel memory [s]", False),
    "chtarget": ("channel_fhit_target", (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9),
                 r"$F_{\mathrm{hit}}$ fraction covered by the channel", False),
    "diffusion": ("diffusion_um2_s", (40.0, 60.0, 79.4, 120.0, 160.0),
                  r"Diffusion coefficient $D$ [$\mu$m$^2$/s]", False),
}


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep", choices=[*SWEEPS, "fit", "plot"])
    p.add_argument("plot_sweep", nargs="?", choices=list(SWEEPS) + ["fit"], help="sweep to replot (with 'plot')")
    p.add_argument("--values", type=float, nargs="+", help="override the swept values")
    p.add_argument("--n-bits", type=int, default=100_000)
    p.add_argument("--n-tx", type=int, default=100)
    p.add_argument("--ts", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--noise-ratio", type=float, default=0.0,
                   help="lambda_env / lambda_0 applied in non-noise sweeps")
    p.add_argument("--detectors", nargs="+", choices=gf.ALL_DETECTORS, default=list(gf.ALL_DETECTORS))
    p.add_argument("--full-device", default="auto",
                   help="torch device for the full Legendre-Viterbi detector (auto: cuda if available, else cpu)")
    p.add_argument("--max-viterbi-taps", type=int, default=25)
    p.add_argument("--channel-memory", type=float, default=None,
                   help="simulated ISI horizon in seconds (default: the paper's 75%% F_hit point)")
    p.add_argument("--channel-target", type=float, default=None,
                   help="simulated ISI horizon as the F_hit fraction it covers, per T_s (e.g. 0.9)")
    p.add_argument("--detector-taps", type=int, default=None,
                   help="fix the detector memory K in taps (default: the 75%% F_hit rule)")
    p.add_argument("--tail-background", action="store_true",
                   help="model the ISI tail beyond the detector memory as uniform background")
    p.add_argument("--threshold-curves", action="store_true",
                   help="also store P01/P10 threshold curves (always on for ntx)")
    p.add_argument("--n-emit-distribution", type=int, default=1_000_000,
                   help="molecules simulated for the Legendre library")
    p.add_argument("--tag", default="", help="extra suffix for the experiment folder")
    return p.parse_args(argv)


def channel_memory(cli: argparse.Namespace, args: argparse.Namespace) -> float:
    """Simulated ISI horizon in seconds for args.ts (see module docstring)."""
    if cli.channel_memory is not None:
        return float(cli.channel_memory)
    target = cli.channel_target if cli.channel_target is not None else PAPER["memory"]
    memory, _, _ = gf.compute_memory_for_args_ts(args, args.ts, float(target))
    return float(memory)


def channel_args(cli: argparse.Namespace, ts: float | None = None, **overrides) -> argparse.Namespace:
    """Channel/library args from main.py at the paper operating point."""
    args = parse_channel_args([])
    args.sim_dir = str(gf.resolve_project_path(args.sim_dir))
    args.legendre_dir = str(gf.resolve_project_path(args.legendre_dir))
    args.experiments_dir = str(gf.resolve_project_path(args.experiments_dir))
    args.ts = float(cli.ts if ts is None else ts)
    gf.configure_args(args, ts=args.ts, memory=channel_memory(cli, args), n_bits=cli.n_bits)
    args.n_emit_bit = cli.n_tx
    args.n_emit_distribution = cli.n_emit_distribution
    # One pool serves every point of a sweep, so size it for the largest N_Tx.
    args.bit_pool_capacity = cli.n_bits * int(getattr(cli, "max_n_tx", cli.n_tx))
    args.leg_l = PAPER["L"]
    args.leg_lam = PAPER["gamma"]
    args.t_rel_bins = PAPER["B"]
    args.detectors = cli.detectors
    args.full_device = cli.full_device
    args.detector_axis = DEFAULT_AXIS
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def detector_taps(args: argparse.Namespace, target: float, channel_taps: int) -> int:
    """Taps for which F_hit reaches `target` of its asymptote, capped by the channel."""
    _, taps, _ = gf.compute_memory_for_args_ts(args, args.ts, float(target))
    return min(int(taps), int(channel_taps))


def tail_rate(full_lib: dict, n_tx: int, p_one: float, taps: int) -> float:
    """Expected count per symbol from the channel taps the detector does not model."""
    return float(p_one) * float(n_tx) * float(np.sum(full_lib["tap_prob"][int(taps):]))


def build_library(args: argparse.Namespace, memory_taps: int) -> dict:
    return load_legendre_library(str(ensure_legendre_library(args, memory_taps)))


def lambda_0(leglib: dict, n_tx: int) -> float:
    """Expected current-symbol count lambda_S = N_Tx p_cap^(0)."""
    return float(n_tx) * float(leglib["tap_prob"][0])


def experiment_dir(cli: argparse.Namespace, sweep: str, create: bool = True) -> Path:
    prefix = "paper" if sweep in PAPER_SWEEPS else "reviewer"
    tag = (
        f"{prefix}_{sweep}__ts{float_tag(cli.ts)}__ntx{cli.n_tx}__nbits{cli.n_bits}"
        f"__seed{cli.seed}__noise{float_tag(cli.noise_ratio)}"
    )
    if cli.channel_memory is not None:
        tag += f"__chmem{float_tag(cli.channel_memory)}"
    if cli.channel_target is not None:
        tag += f"__cht{float_tag(cli.channel_target)}"
    if cli.detector_taps is not None:
        tag += f"__K{cli.detector_taps}"
    if cli.tail_background:
        tag += "__tailbg"
    if cli.tag:
        tag += f"__{cli.tag}"
    path = gf.PROJECT_ROOT / "experiments" / tag
    return gf.maybe_make_out_dir(path) if create else path


# ---------------------------------------------------------------------
# Detection sweeps
# ---------------------------------------------------------------------


def load_channel(
    cli: argparse.Namespace, args: argparse.Namespace, bits: np.ndarray, tag: str, min_library_taps: int = 0,
    n_tx: int | None = None,
) -> dict:
    """Simulated channel for `args`: received symbols, library, lambda_0.

    The library covers max(channel taps, min_library_taps), so a detector whose
    memory exceeds a short channel still models (non-existent) late ISI taps.
    """
    n_tx = int(cli.n_tx if n_tx is None else n_tx)
    channel_taps = gf.memory_taps_from_ts(args.memory, args.ts)
    library_taps = max(channel_taps, int(min_library_taps))
    pool_file = find_compatible_bit_pool_file(args)
    if (pool_horizon(pool_file) or 0.0) + 1e-9 < args.memory:
        raise ValueError(
            f"{pool_file.name} only simulates {pool_horizon(pool_file)} s, "
            f"but the channel memory is {args.memory} s."
        )
    print(
        f"[{tag}] d={args.distance} r={args.radius} D={args.diffusion_coef}: channel memory "
        f"{args.memory} s = {channel_taps} taps, pool {pool_file.name}"
    )
    received = generate_received_symbols_from_indexed_impulse(
        hit_file=pool_file,
        bits=bits,
        n_emit_bit=n_tx,
        memory_taps=channel_taps,
        ts=args.ts,
    )
    # Library over the whole channel (or the detector memory, if longer);
    # detectors get it truncated to their memory.
    lib_args = argparse.Namespace(**vars(args))
    lib_args.memory = library_taps * args.ts
    full_lib = build_library(lib_args, library_taps)
    return {
        "pool_file": pool_file,
        "channel_taps": channel_taps,
        "received_clean": received,
        "full_lib": full_lib,
        "lambda_0": lambda_0(full_lib, n_tx),
    }


def run_detection_sweep(cli: argparse.Namespace) -> Path:
    sweep = cli.sweep
    column, defaults, _, _ = SWEEPS[sweep]
    values = tuple(cli.values) if cli.values else defaults
    geometry = sweep in GEOMETRY_SWEEPS
    # Sweeps whose points each need their own simulated channel.
    per_point_channel = geometry or sweep in ("chmem", "chtarget", "ts", "ntx")
    if sweep in ("chmem", "chtarget"):
        # Longest channel first: its pools are then reused by the shorter ones.
        values = tuple(sorted(values, reverse=True))
    if sweep == "ntx":
        cli.max_n_tx = int(max(values))
    out_dir = experiment_dir(cli, sweep)
    summary_file = out_dir / "summary.csv"
    threshold_curves = bool(cli.threshold_curves) or sweep == "ntx"

    done = set()
    if summary_file.exists():
        done = set(pd.read_csv(summary_file)[column].round(12))
    if all(round(float(v), 12) in done for v in values):
        print(f"[{sweep}] all {len(values)} points already in {summary_file}")
        return summary_file

    base = channel_args(cli)
    bits = generate_bit_sequence(cli.n_bits, base.p_one, cli.seed)
    # Sweeps that change the channel per point load it inside the loop.
    shared = None if per_point_channel else load_channel(cli, base, bits, sweep)

    (out_dir / "manifest.json").write_text(json.dumps({
        "sweep": sweep,
        "column": column,
        "values": list(values),
        "paper_operating_point": PAPER,
        "ts": base.ts,
        "channel_memory": base.memory,
        "channel_memory_taps": None if per_point_channel else shared["channel_taps"],
        "paper_detector_taps": None if per_point_channel else detector_taps(base, PAPER["memory"], shared["channel_taps"]),
        "fixed_detector_taps": cli.detector_taps,
        "geometry_max_detector_taps": GEOMETRY_MAX_TAPS if geometry else None,
        "channel_target": cli.channel_target,
        "tail_background": bool(cli.tail_background),
        "threshold_curves": threshold_curves,
        "n_tx": cli.n_tx,
        "n_bits": cli.n_bits,
        "seed": cli.seed,
        "lambda_0": None if per_point_channel else shared["lambda_0"],
        "noise_ratio_other_sweeps": cli.noise_ratio,
        "detectors": cli.detectors,
        "max_viterbi_taps": cli.max_viterbi_taps,
        "bit_pool": None if per_point_channel else shared["pool_file"].name,
    }, indent=2))

    for value in values:
        if round(float(value), 12) in done:
            print(f"[{sweep}] {column}={value}: already in summary.csv, skipping")
            continue

        n_tx = int(value) if sweep == "ntx" else int(cli.n_tx)
        args = channel_args(cli, ts=float(value) if sweep == "ts" else None)
        args.n_emit_bit = n_tx
        if geometry:
            setattr(args, GEOMETRY_SWEEPS[sweep], float(value))
        elif sweep == "chmem":
            args.memory = float(value)
            gf.memory_taps_from_ts(args.memory, args.ts)  # must be a whole number of symbols
        elif sweep == "chtarget":
            _, channel_taps_target, _ = gf.compute_memory_for_args_ts(args, args.ts, float(value))
            args.memory = channel_taps_target * args.ts
        channel = (
            load_channel(cli, args, bits, sweep, min_library_taps=cli.detector_taps or 0, n_tx=n_tx)
            if per_point_channel else shared
        )
        channel_taps = channel["channel_taps"]
        full_lib = channel["full_lib"]
        lam0 = channel["lambda_0"]

        leglib = full_lib
        noise_ratio = cli.noise_ratio
        memory_taps = detector_taps(args, PAPER["memory"], channel_taps)
        if geometry:
            memory_taps = min(memory_taps, GEOMETRY_MAX_TAPS)
        if cli.detector_taps is not None:
            memory_taps = min(int(cli.detector_taps), int(full_lib["memory_taps"]))
        model_tail = bool(cli.tail_background)

        if sweep == "noise":
            noise_ratio = float(value)
        elif sweep == "L":
            args.leg_l = int(value)
            leglib = build_library(args, channel_taps)
        elif sweep == "gamma":
            args.leg_lam = float(value)
            leglib = build_library(args, channel_taps)
        elif sweep == "floor":
            leglib = with_eps_floor(full_lib, float(value))
        elif sweep == "B":
            args.t_rel_bins = int(value)
            leglib = build_library(args, channel_taps)
        elif sweep == "memory":
            memory_taps = detector_taps(args, float(value), channel_taps)
        elif sweep == "K":
            memory_taps = min(int(value), channel_taps)
        elif sweep == "direction":
            delta = np.deg2rad(float(value))
            args.detector_axis = np.array([0.0, np.sin(delta), np.cos(delta)])
        elif sweep == "tail":
            model_tail = bool(value)

        if memory_taps < int(leglib["memory_taps"]):
            leglib = truncate_legendre_library(leglib, memory_taps)

        lambda_env = noise_ratio * lam0
        lambda_tail = tail_rate(full_lib, n_tx, args.p_one, memory_taps) if model_tail else 0.0
        args.legendre_lambda_env = lambda_env + lambda_tail
        args.viterbi_lambda_env = lambda_env + lambda_tail
        received = add_background_noise(
            channel["received_clean"], lambda_env=lambda_env, ts=args.ts, radius=args.radius, seed=cli.seed + 1,
        )

        print(
            f"[{sweep}] {column}={value}  lambda_0={lam0:.3f}  lambda_env={lambda_env:.3f}  "
            f"modelled tail={lambda_tail:.3f}  detector taps={memory_taps}/{channel_taps}"
        )
        t0 = time.perf_counter()
        rows = gf.run_one_detector_scenario(
            args=args,
            bits=bits,
            received_symbol_hits=received,
            leglib=leglib,
            n_tx=n_tx,
            physical_memory_taps=memory_taps,
            max_viterbi_memory_taps=cli.max_viterbi_taps,
            max_strong_viterbi_memory_taps=cli.max_viterbi_taps,
            save_symbol_csvs=False,
            add_threshold_curves=threshold_curves,
            out_dir=out_dir,
            scenario_tag=sweep,
        )
        df = pd.DataFrame(rows)
        df.insert(0, column, value)
        df.insert(1, "lambda_env", lambda_env)
        df.insert(2, "lambda_0", lam0)
        df.insert(3, "detector_memory_taps", memory_taps)
        df.insert(4, "channel_memory_taps", channel_taps)
        df.insert(5, "lambda_tail_modelled", lambda_tail)
        for name, const in (("N_Tx", n_tx), ("ts", args.ts)):
            if name not in df.columns:
                df[name] = const
        df.to_csv(summary_file, mode="a", header=not summary_file.exists(), index=False)
        print(gf.operating_points(df)[["detector", "BER", "P_miss", "P_FA", "runtime_seconds"]]
              .to_string(index=False))
        print(f"[{sweep}] point done in {gf.format_duration(time.perf_counter() - t0)}")

    return summary_file


# ---------------------------------------------------------------------
# Approximation accuracy (no detection)
# ---------------------------------------------------------------------


def floored_density(coeff: np.ndarray, z: np.ndarray, mu_grid: np.ndarray, eps: float) -> np.ndarray:
    """Eq. (14): max(eps, h_L) renormalized on [-1, 1]."""
    L = len(coeff) - 1
    raw = np.maximum(legendre_basis(z, L) @ coeff, eps)
    norm = _trapz(np.maximum(legendre_basis(mu_grid, L) @ coeff, eps), mu_grid)
    return raw / norm


def run_fit_accuracy(cli: argparse.Namespace) -> Path:
    """Held-out accuracy of the Legendre angular density vs. L (and gamma).

    The distribution pool is split in half at random. For every (tap, time bin)
    the density is fitted on the training half exactly as in the library
    (Eqs. 12-14) and scored on the test half by
      - mean held-out log-likelihood of z per molecule, and
      - total-variation distance to the test-half histogram,
    both averaged over (tap, bin) with weights equal to the test molecule count.
    The total-variation distance uses N_TV_BINS coarse z-bins so that it is not
    dominated by the sampling noise of ~100 test molecules per cell. Reference
    rows: a 160-bin histogram of the training half with add-1/2 smoothing, whose
    TV to the test half is also the sampling-noise floor of the TV metric.
    """
    out_dir = experiment_dir(cli, "fit")
    summary_file = out_dir / "summary.csv"
    args = channel_args(cli)
    memory_taps = gf.memory_taps_from_ts(args.memory, args.ts)
    hits = load_hits_file(str(find_compatible_distribution_file(args)))
    t_abs, xyz = hits["t"], hits["xyz"]
    keep = (t_abs >= 0.0) & (t_abs < args.memory)
    t_abs = t_abs[keep]
    z_all = np.clip(unit_rows(xyz[keep]) @ DEFAULT_AXIS, -1.0, 1.0)

    rng = np.random.default_rng(cli.seed)
    train = rng.random(len(t_abs)) < 0.5

    n_mu = args.mu_bins
    mu_edges = np.linspace(-1.0, 1.0, n_mu + 1)
    mu_grid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = mu_edges[1] - mu_edges[0]
    B = PAPER["B"]
    tap = np.minimum((t_abs // args.ts).astype(np.int64), memory_taps - 1)
    b = np.clip(((t_abs - tap * args.ts) / args.ts * B).astype(np.int64), 0, B - 1)

    L_values = [int(v) for v in (cli.values or (0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 30))]
    gamma_values = (0.0, 1e-6, 1e-2, 1.0, 10.0)
    tv_edges = np.linspace(-1.0, 1.0, N_TV_BINS + 1)

    # Group molecules by (tap, bin) once.
    cell = tap * B + b
    order = np.argsort(cell, kind="stable")
    bounds = np.searchsorted(cell[order], np.arange(memory_taps * B + 1))

    def hist_density(z, pseudo=0.0):
        h, _ = np.histogram(z, bins=mu_edges)
        h = h + pseudo
        return h / max(h.sum() * dmu, 1e-15)

    def coarse_mass(dens_grid):
        """Mass of a mu-grid density in each coarse TV bin."""
        k = np.clip(np.searchsorted(tv_edges, mu_grid, side="right") - 1, 0, N_TV_BINS - 1)
        return np.bincount(k, weights=dens_grid * dmu, minlength=N_TV_BINS)

    def score(estimator: str, L: int | None, gamma: float) -> dict:
        tot_w = ll_sum = tv_sum = 0.0
        for c in range(memory_taps * B):
            idx = order[bounds[c]:bounds[c + 1]]
            z_tr, z_te = z_all[idx[train[idx]]], z_all[idx[~train[idx]]]
            if len(z_tr) == 0 or len(z_te) == 0:
                continue
            p_tr = hist_density(z_tr)
            p_te = hist_density(z_te)
            if L is None:
                p_sm = hist_density(z_tr, pseudo=0.5)
                k = np.clip(np.searchsorted(mu_edges, z_te, side="right") - 1, 0, n_mu - 1)
                dens_te = p_sm[k]
                dens_grid = p_sm
            else:
                coeff = fit_legendre_ls(p_mu=p_tr, mu_grid=mu_grid, L=L, lam=gamma)
                dens_te = floored_density(coeff, z_te, mu_grid, DEFAULT_EPS_FLOOR)
                dens_grid = floored_density(coeff, mu_grid, mu_grid, DEFAULT_EPS_FLOOR)
            w = len(z_te)
            tot_w += w
            ll_sum += float(np.sum(np.log(dens_te)))
            tv_sum += w * 0.5 * float(np.sum(np.abs(coarse_mass(dens_grid) - coarse_mass(p_te))))
        row = {
            "estimator": estimator,
            "L": np.nan if L is None else L,
            "gamma": gamma,
            "heldout_loglik_per_molecule": ll_sum / tot_w,
            "weighted_tv_distance": tv_sum / tot_w,
            "n_test_molecules": int(tot_w),
        }
        print(row)
        return row

    rows = [score(f"histogram_{n_mu}", None, np.nan)]
    rows += [score("legendre", L, gamma) for gamma in gamma_values for L in L_values]
    df = pd.DataFrame(rows)
    df.to_csv(summary_file, index=False)
    return summary_file


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------


def _finish_axes(ax, fig, out_file: Path, *, legend_ncol: int = 3) -> None:
    import matplotlib.pyplot as plt

    gf.apply_publication_style(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=legend_ncol,
            frameon=False, fontsize=12, handlelength=2.2, columnspacing=1.2,
        )
    fig.tight_layout()
    fig.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_file)


def _ber_yaxis(ax, ber) -> None:
    """Log BER axis with plain decimal ticks."""
    import matplotlib.pyplot as plt

    ax.set_yscale("log")
    lo, hi = float(np.min(ber)), float(np.max(ber))
    yticks = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5)
              if lo / 1.15 <= t <= hi * 1.15]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:g}" for t in yticks])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(lo / 1.15, hi * 1.15)


DEFAULT_TAGS = {"ts0p2", "ntx100", "nbits100000", "seed2026", "noise0"}


def plot_name(summary_file: Path) -> str:
    """Experiment folder name without the parts that equal the defaults.

    reviewer_noise__ts0p2__ntx100__nbits100000__seed2026__noise0 -> reviewer_noise
    reviewer_chtarget__ts0p5__...__K7__tailbg -> reviewer_chtarget__ts0p5__K7__tailbg
    """
    head, *parts = summary_file.parent.name.split("__")
    return "__".join([head, *(p for p in parts if p not in DEFAULT_TAGS)])


def plot_sweep(summary_file: Path, sweep: str) -> Path:
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_file)
    plots_dir = gf.maybe_make_out_dir(gf.PROJECT_ROOT / "plots")
    out_file = plots_dir / f"{plot_name(summary_file)}.pdf"
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    line = dict(linewidth=2.8, markersize=8, markeredgewidth=1.2)

    if sweep == "fit":
        leg = df[df["estimator"] == "legendre"]
        ref = df[df["estimator"] != "legendre"]["heldout_loglik_per_molecule"]
        # gamma = 0 coincides with the paper's gamma = 1e-6, so it is not drawn.
        colors = {1e-6: "#1F4E79", 1e-2: "#2B6E46", 1.0: "#6E4B8B", 10.0: "#7F3B2E"}
        markers = {1e-6: "o", 1e-2: "D", 1.0: "P", 10.0: "^"}
        for gamma, color in colors.items():
            g = leg[np.isclose(leg["gamma"], gamma)].sort_values("L")
            if g.empty:
                continue
            label = rf"$\gamma = 10^{{{int(round(np.log10(gamma)))}}}$" + (" (paper)" if gamma == PAPER["gamma"] else "")
            ax.plot(g["L"], g["heldout_loglik_per_molecule"], color=color, marker=markers[gamma],
                    linestyle="-", label=label, **line)
        if len(ref):
            ax.axhline(float(ref.iloc[0]), color="k", ls=":", lw=2, label="Smoothed histogram")
        ax.axvline(PAPER["L"], color="0.55", lw=1.2, ls="--")
        ax.set_xlabel(r"Legendre order $L$")
        ax.set_ylabel("Held-out log-likelihood")
        _finish_axes(ax, fig, out_file, legend_ncol=3)
        return out_file

    column, _, xlabel, logx = SWEEPS[sweep]
    df = gf.operating_points(df)
    if sweep in PLOT_VALUES:
        df = df[df[column].isin(PLOT_VALUES[sweep])]

    if sweep == "tail":
        # Reference: the paper's channel, truncated at the detector memory (no tail).
        ref_file = next(iter(sorted((gf.PROJECT_ROOT / "experiments").glob(
            "reviewer_noise__ts0p2__ntx100__nbits100000__seed2026__noise0/summary.csv"))), None)
        cats = []
        if ref_file is not None:
            ref = gf.operating_points(pd.read_csv(ref_file))
            ref = ref[np.isclose(ref["noise_ratio"], 0.0)].assign(**{column: -1.0})
            df = pd.concat([ref, df], ignore_index=True)
            cats.append((-1.0, "No tail\n(channel = $K$)"))
        cats += [(0.0, "Tail\nignored"), (1.0, "Tail as\nbackground")]
        pos = {v: i for i, (v, _) in enumerate(cats)}
        for detector in gf.DETECTOR_ORDER:
            g = df[df["detector"] == detector]
            g = g[g[column].isin(pos)].sort_values(column)
            if g.empty:
                continue
            style = gf.detector_style(detector)
            ax.plot([pos[v] for v in g[column]], g["BER"], label=gf.detector_label(detector),
                    color=style["color"], marker=style["marker"], linestyle=style["linestyle"], **line)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c for _, c in cats], fontsize=13)
        ax.set_xlim(-0.3, len(cats) - 0.7)
        _ber_yaxis(ax, df[df[column].isin(pos)]["BER"])
        ax.set_ylabel("BER")
        _finish_axes(ax, fig, out_file)
        return out_file

    for detector in gf.DETECTOR_ORDER:
        g = df[df["detector"] == detector].sort_values(column)
        if g.empty:
            continue
        style = gf.detector_style(detector)
        if detector in REFERENCE_DETECTORS.get(sweep, ()):
            ax.axhline(float(g["BER"].mean()), label=gf.detector_label(detector),
                       color=style["color"], linestyle=style["linestyle"], linewidth=line["linewidth"])
            continue
        ax.plot(g[column].to_numpy(dtype=float), g["BER"], label=gf.detector_label(detector),
                color=style["color"], marker=style["marker"], linestyle=style["linestyle"], **line)

    values = np.sort(df[column].unique().astype(float))
    if sweep == "L":
        ax.set_xticks(list(PLOT_VALUES["L"]))
        ax.set_xticklabels([str(v) for v in PLOT_VALUES["L"]])
        ax.set_xlim(1.5, 20.5)
    elif sweep == "K":
        ax.set_xticks(range(1, 21))
        ax.set_xticklabels([str(k) for k in range(1, 21)], fontsize=12)
        ax.set_xlim(0.5, 20.5)
        manifest = summary_file.parent / "manifest.json"
        if sweep == "K" and manifest.exists():
            ax.axvline(json.loads(manifest.read_text()).get("paper_detector_taps", np.nan),
                       color="0.55", lw=1.2, ls="--")
    elif logx and np.any(values <= 0):
        # gamma = 0 is shown on a symmetric-log axis, linear around zero.
        positive = values[values > 0]
        ax.set_xscale("symlog", linthresh=float(positive.min()), linscale=0.6)
        ax.set_xlim(left=0)
        ticks = [0.0] + [v for v in positive if np.isclose(np.log10(v) % 2, 0) or v == positive.min()]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0"] + [rf"$10^{{{int(round(np.log10(v)))}}}$" for v in ticks[1:]])
        ax.minorticks_off()
    elif logx:
        ax.set_xscale("log")
        if sweep == "B":
            ax.set_xticks(values)
            ax.set_xticklabels([f"{int(v)}" for v in values])
            ax.minorticks_off()
    else:
        # Drop tick labels that would crowd the previous one.
        span = values.max() - values.min()
        ticks = [v for i, v in enumerate(values) if i == 0 or v - values[i - 1] >= 0.08 * span]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{round(v, 2):g}" for v in ticks])

    if sweep in PAPER:
        ax.axvline(PAPER[sweep], color="0.55", lw=1.2, ls="--")
    _ber_yaxis(ax, df["BER"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("BER")
    _finish_axes(ax, fig, out_file)
    return out_file


def plot_chtarget_comparison(cli: argparse.Namespace) -> Path:
    """Tail ignored (dashed, open markers) vs. tail as background (solid) for one T_s."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    flags = {False: "ignored", True: "background"}
    frames = {}
    for tail in flags:
        cli.tail_background = tail
        f = experiment_dir(cli, "chtarget", create=False) / "summary.csv"
        if f.exists():
            frames[tail] = gf.operating_points(pd.read_csv(f))
    if not frames:
        raise SystemExit("no chtarget results for these options")

    out_file = gf.maybe_make_out_dir(gf.PROJECT_ROOT / "plots") / f"reviewer_chtarget_ts{float_tag(cli.ts)}_compare.pdf"
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    detectors = ("viterbi", "legendre", "legendre_viterbi_full")
    for tail, df in frames.items():
        for detector in detectors:
            g = df[df["detector"] == detector].sort_values("channel_fhit_target")
            style = gf.detector_style(detector)
            ax.plot(g["channel_fhit_target"], g["BER"], color=style["color"], marker=style["marker"],
                    linestyle="-" if tail else "--", linewidth=2.4, markersize=7,
                    markerfacecolor=style["color"] if tail else "white", markeredgewidth=1.4)
    all_ber = pd.concat(frames.values())
    _ber_yaxis(ax, all_ber[all_ber["detector"].isin(detectors)]["BER"])
    values = np.sort(all_ber["channel_fhit_target"].unique())
    ax.set_xticks(values)
    ax.set_xticklabels([f"{round(v, 2):g}" for v in values])
    ax.axvline(PAPER["chtarget"], color="0.55", lw=1.2, ls="--")
    ax.set_xlabel(r"$F_{\mathrm{hit}}$ fraction covered by the channel")
    ax.set_ylabel("BER")
    handles = [Line2D([], [], color=gf.detector_style(d)["color"], marker=gf.detector_style(d)["marker"],
                      linestyle="-", label=gf.detector_label(d)) for d in detectors]
    handles += [Line2D([], [], color="k", linestyle="--", marker="o", markerfacecolor="white", label="Tail ignored"),
                Line2D([], [], color="k", linestyle="-", marker="o", label="Tail as background")]
    gf.apply_publication_style(ax)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, fontsize=12, handlelength=2.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_file)
    return out_file


def main() -> None:
    cli = parse_cli()
    if cli.sweep in GEOMETRY_SWEEPS or cli.plot_sweep in GEOMETRY_SWEEPS:
        if cli.channel_memory is None:
            cli.channel_memory = GEOMETRY_CHANNEL_MEMORY
        cli.tail_background = True
    if cli.sweep == "plot" and cli.plot_sweep == "chtarget" and cli.tag == "compare":
        cli.tag = ""
        plot_chtarget_comparison(cli)
        return
    if cli.sweep == "plot":
        if not cli.plot_sweep:
            raise SystemExit("usage: experiments.py plot <sweep> [same options as the run]")
        plot_sweep(experiment_dir(cli, cli.plot_sweep, create=False) / "summary.csv", cli.plot_sweep)
        return
    summary = run_fit_accuracy(cli) if cli.sweep == "fit" else run_detection_sweep(cli)
    plot_sweep(summary, cli.sweep)


if __name__ == "__main__":
    main()
