"""Run one shared experiment and compare Legendre vs. count-Viterbi detection."""

from __future__ import annotations

import argparse
from pathlib import Path
import glob
import math
import time

import numpy as np

from legendre import (
    build_legendre_library_from_distribution_file,
    legendre_detect,
    load_hits_file,
    load_legendre_library,
    make_legendre_library_tag,
)
from legendre_viterbi import legendre_viterbi_detect
from viterbi import viterbi_detect


DEFAULT_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

PROJECT_ROOT = "./"
DEFAULT_SIM_DIR = PROJECT_ROOT + "simulations"
DEFAULT_LEGENDRE_DIR = PROJECT_ROOT + "legendre_files"
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT + "experiments"


def float_tag(x: float, ndigits: int = 8) -> str:
    return f"{float(x):.{ndigits}g}".replace("-", "m").replace(".", "p").replace("+", "")


def raw_tag(distance: float, radius: float, diffusion_coef: float, step_time: float, memory: float) -> str:
    return (
        f"d_{float_tag(distance)}"
        f"_r{float_tag(radius)}"
        f"_D{float_tag(diffusion_coef)}"
        f"_dt{float_tag(step_time)}"
        f"_MEM{float_tag(memory)}"
    )


def generate_bit_sequence(n_bits: int, p_one: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, p_one, size=n_bits).astype(np.int8)


def generate_received_symbols_from_indexed_impulse(
    hit_file: str | Path,
    bits: np.ndarray,
    n_emit_bit: int,
    memory_taps: int,
    ts: float,
) -> list[dict[str, np.ndarray | int]]:
    """Build symbol observations from one indexed Brownian hit pool."""
    hits = load_hits_file(str(hit_file))
    t_imp = hits["t"].astype(np.float64)
    xyz_imp = hits["xyz"].astype(np.float64)
    idx_imp = hits["idx"]
    if idx_imp is None:
        raise ValueError("Bit-pool file must contain molecule indices under key 'idx'.")

    bits = np.asarray(bits, dtype=np.int8)
    needed = len(bits) * int(n_emit_bit)
    if hits["nof_molecules"] is not None and needed > hits["nof_molecules"]:
        raise ValueError(
            f"Need {needed} indexed molecules for {len(bits)} bits × {n_emit_bit}, "
            f"but the pool only contains {hits['nof_molecules']}."
        )

    received_symbol_hits = []
    for n in range(len(bits)):
        collected_t_rel, collected_xyz, collected_source_bit, collected_tap = [], [], [], []
        for tap in range(memory_taps):
            k = n - tap
            if k < 0 or bits[k] == 0:
                continue
            idx_low = k * int(n_emit_bit)
            idx_high = (k + 1) * int(n_emit_bit)
            t_low = tap * ts
            t_high = (tap + 1) * ts
            mask = (
                (idx_imp >= idx_low)
                & (idx_imp < idx_high)
                & (t_imp >= t_low)
                & (t_imp < t_high)
            )
            selected = np.flatnonzero(mask)
            if len(selected) == 0:
                continue
            collected_t_rel.append(t_imp[selected] - t_low)
            collected_xyz.append(xyz_imp[selected])
            collected_source_bit.extend([k] * len(selected))
            collected_tap.extend([tap] * len(selected))

        if collected_t_rel:
            t_rel_n = np.concatenate(collected_t_rel).astype(np.float64)
            xyz_n = np.vstack(collected_xyz).astype(np.float64)
        else:
            t_rel_n = np.empty(0, dtype=np.float64)
            xyz_n = np.empty((0, 3), dtype=np.float64)

        received_symbol_hits.append(
            {
                "rx_symbol": n,
                "t_rel": t_rel_n,
                "xyz": xyz_n,
                "source_bit": np.asarray(collected_source_bit, dtype=np.int64),
                "tap": np.asarray(collected_tap, dtype=np.int64),
            }
        )
    return received_symbol_hits



def segment_sphere_intersections_with_tau(p0, p1, radius):
    d = p1 - p0
    a = (d * d).sum(dim=1)
    b = 2.0 * (p0 * d).sum(dim=1)
    c = (p0 * p0).sum(dim=1) - radius * radius
    disc = b * b - 4.0 * a * c
    import torch
    disc = torch.clamp(disc, min=0.0)
    sqrt_disc = torch.sqrt(disc)
    denom = 2.0 * torch.clamp(a, min=1e-12)
    tau = torch.clamp((-b - sqrt_disc) / denom, 0.0, 1.0)
    return p0 + tau.unsqueeze(1) * d, tau


def experiment_save_hits_indexed(
    filename,
    radius,
    total_time,
    step_time,
    diffusion_coef,
    distance,
    nof_molecules,
    device=None,
    override=False,
    batch_size=500_000,
):
    """Simulate indexed Brownian first hits and save them as a .pt pool."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Generating Brownian hit files requires PyTorch.") from exc

    filename = str(filename)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if Path(filename).exists() and not override:
        return Path(filename)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    radius_t = torch.as_tensor(radius, device=device, dtype=torch.float32)
    diffusion_coef_t = torch.as_tensor(diffusion_coef, device=device, dtype=torch.float32)
    nof_steps = int(math.ceil(float(total_time) / float(step_time)))
    sigma = torch.sqrt(2.0 * diffusion_coef_t * float(step_time))
    idx_out, times_out, pos_out = [], [], []
    total_hits = 0
    started = time.time()
    print(f"Generating Brownian hits: {filename}")
    print(f"  molecules={int(nof_molecules):,}, steps={nof_steps:,}, batch_size={int(batch_size):,}, device={device}")

    for start_idx in range(0, int(nof_molecules), int(batch_size)):
        end_idx = min(start_idx + int(batch_size), int(nof_molecules))
        n_chunk = end_idx - start_idx
        positions = torch.zeros((n_chunk, 3), device=device, dtype=torch.float32)
        positions[:, 2] = float(distance)
        alive = torch.ones(n_chunk, device=device, dtype=torch.bool)
        mol_idx = torch.arange(start_idx, end_idx, device=device, dtype=torch.int64)
        with torch.no_grad():
            for step in range(nof_steps):
                if not alive.any():
                    break
                prev = positions.clone()
                positions.add_(torch.randn_like(positions) * sigma)
                new_hits = (positions.pow(2).sum(dim=1) <= radius_t * radius_t) & alive
                if new_hits.any():
                    hit_pos, tau = segment_sphere_intersections_with_tau(prev[new_hits], positions[new_hits], radius_t)
                    t_hit = (float(step) + tau) * float(step_time)
                    valid = t_hit < float(total_time)
                    if valid.any():
                        idx_out.append(mol_idx[new_hits][valid].detach().cpu())
                        times_out.append(t_hit[valid].detach().cpu())
                        pos_out.append(hit_pos[valid].detach().cpu())
                        total_hits += int(valid.sum().item())
                    alive[new_hits] = False
                if step % 5000 == 0:
                    print(
                        f"  chunk {start_idx}:{end_idx}, step {step}/{nof_steps}, "
                        f"hits={total_hits}, elapsed={time.time() - started:.1f}s"
                    )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if total_hits == 0:
        data = {
            "idx": torch.empty(0, dtype=torch.int64),
            "t": torch.empty(0, dtype=torch.float32),
            "xyz": torch.empty((0, 3), dtype=torch.float32),
            "nof_molecules": int(nof_molecules),
        }
    else:
        idx_cat = torch.cat(idx_out).to(torch.int64)
        order = torch.argsort(idx_cat)
        data = {
            "idx": idx_cat[order],
            "t": torch.cat(times_out).to(torch.float32)[order],
            "xyz": torch.cat(pos_out).to(torch.float32)[order],
            "nof_molecules": int(nof_molecules),
        }
    torch.save(data, filename)
    print(f"Saved Brownian hit file with {total_hits:,} hits: {filename}")
    return Path(filename)

def find_compatible_bit_pool_file(args: argparse.Namespace) -> Path:
    preferred = Path(args.sim_dir) / (
        f"Hits_bit_pool_indexed_{raw_tag(args.distance, args.radius, args.diffusion_coef, args.step_time, args.memory)}"
        f"_N{args.bit_pool_capacity}.pt"
    )
    if preferred.exists():
        return preferred

    candidates = []
    raw = raw_tag(args.distance, args.radius, args.diffusion_coef, args.step_time, args.memory)
    candidates.extend(glob.glob(str(Path(args.sim_dir) / f"Hits_bit_pool_indexed_{raw}_N*.pt")))
    candidates.extend(glob.glob(str(Path(args.sim_dir) / f"Hits_bit_pool_indexed_{raw}_TS*_MT*_N*.pt")))
    candidates.extend(glob.glob(str(Path(args.sim_dir) / f"Hits_bit_pool_indexed_d_{float_tag(args.distance)}*.pt")))
    if not candidates:
        print(f"No compatible indexed bit-pool found. Generating: {preferred}")
        return experiment_save_hits_indexed(
            filename=preferred,
            radius=args.radius,
            total_time=args.memory,
            step_time=args.step_time,
            diffusion_coef=args.diffusion_coef,
            distance=args.distance,
            nof_molecules=max(args.bit_pool_capacity, args.n_bits * args.n_emit_bit),
            batch_size=args.brownian_batch_size,
        )
    return Path(sorted(set(candidates))[-1])


def find_compatible_distribution_file(args: argparse.Namespace) -> Path:
    preferred = Path(args.sim_dir) / (
        f"Hits_distribution_{raw_tag(args.distance, args.radius, args.diffusion_coef, args.step_time, args.memory)}"
        f"_N{args.n_emit_distribution}.pt"
    )
    if preferred.exists():
        return preferred

    candidates = []
    raw = raw_tag(args.distance, args.radius, args.diffusion_coef, args.step_time, args.memory)
    candidates.extend(glob.glob(str(Path(args.sim_dir) / f"Hits_distribution_{raw}_N*.pt")))
    candidates.extend(glob.glob(str(Path(args.sim_dir) / f"Hits_distribution_{raw}_TS*_MT*_N*.pt")))
    candidates.extend(glob.glob(str(Path(args.sim_dir) / f"Hits_distribution_d_{float_tag(args.distance)}*.pt")))
    if not candidates:
        print(f"No compatible raw distribution file found. Generating: {preferred}")
        return experiment_save_hits_indexed(
            filename=preferred,
            radius=args.radius,
            total_time=args.memory,
            step_time=args.step_time,
            diffusion_coef=args.diffusion_coef,
            distance=args.distance,
            nof_molecules=args.n_emit_distribution,
            batch_size=args.brownian_batch_size,
        )
    return Path(sorted(set(candidates))[-1])


def ensure_legendre_library(args: argparse.Namespace, memory_taps: int) -> Path:
    tag = make_legendre_library_tag(
        ts=args.ts,
        memory_taps=memory_taps,
        memory=args.memory,
        distance=args.distance,
        radius=args.radius,
        diffusion_coef=args.diffusion_coef,
        step_time=args.step_time,
    )
    legendre_file = Path(args.legendre_dir) / (
        f"legendre_{tag}_Ndist{args.n_emit_distribution}_L{args.leg_l}"
        f"_mu{args.mu_bins}_trel{args.t_rel_bins}_lam{float_tag(args.leg_lam)}.npz"
    )
    if legendre_file.exists():
        return legendre_file

    distribution_file = find_compatible_distribution_file(args)

    build_legendre_library_from_distribution_file(
        hit_file=str(distribution_file),
        out_file=str(legendre_file),
        n_emit_distribution=args.n_emit_distribution,
        memory=args.memory,
        ts=args.ts,
        memory_taps=memory_taps,
        n_mu_bins=args.mu_bins,
        n_t_bins=args.t_rel_bins,
        L=args.leg_l,
        lam=args.leg_lam,
        tx_axis=DEFAULT_AXIS,
    )
    return legendre_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-bits", type=int, default=100)
    p.add_argument("--n-emit-bit", type=int, default=250)
    p.add_argument("--p-one", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ts", type=float, default=0.1)
    p.add_argument("--memory", type=float, default=5.0)
    p.add_argument("--radius", type=float, default=5.0)
    p.add_argument("--distance", type=float, default=12.5)
    p.add_argument("--diffusion-coef", type=float, default=79.4)
    p.add_argument("--step-time", type=float, default=1e-4)
    p.add_argument("--n-emit-distribution", type=int, default=1_000_000)
    p.add_argument("--bit-pool-capacity", type=int, default=10_000_000)
    p.add_argument("--brownian-batch-size", type=int, default=500_000)
    p.add_argument("--leg-l", type=int, default=10)
    p.add_argument("--leg-lam", type=float, default=1e-6)
    p.add_argument("--mu-bins", type=int, default=160)
    p.add_argument("--t-rel-bins", type=int, default=100)
    p.add_argument("--legendre-lambda-env", type=float, default=1.0)
    p.add_argument("--viterbi-lambda-env", type=float, default=0.0)
    p.add_argument(
        "--viterbi-memory-taps",
        type=int,
        default=None,
        help="Number of earliest taps used by count-Viterbi; defaults to min(full memory taps, 20).",
    )
    p.add_argument("--sim-dir", default=str(DEFAULT_SIM_DIR))
    p.add_argument("--legendre-dir", default=str(DEFAULT_LEGENDRE_DIR))
    p.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    memory_taps = int(round(args.memory / args.ts))
    if not np.isclose(memory_taps * args.ts, args.memory):
        raise ValueError("memory must be an integer multiple of ts")

    bits = generate_bit_sequence(args.n_bits, args.p_one, args.seed)
    bit_pool_file = find_compatible_bit_pool_file(args)

    received = generate_received_symbols_from_indexed_impulse(
        hit_file=bit_pool_file,
        bits=bits,
        n_emit_bit=args.n_emit_bit,
        memory_taps=memory_taps,
        ts=args.ts,
    )
    legendre_file = ensure_legendre_library(args, memory_taps)
    leglib = load_legendre_library(str(legendre_file))

    viterbi_memory_taps = (
        int(args.viterbi_memory_taps)
        if args.viterbi_memory_taps is not None
        else min(memory_taps, 20)
    )

    legendre_df, legendre_metrics = legendre_detect(
        received_symbol_hits=received,
        bits=bits,
        leglib=leglib,
        n_emit_bit=args.n_emit_bit,
        axis=DEFAULT_AXIS,
        p_one=args.p_one,
        lambda_env=args.legendre_lambda_env,
    )
    viterbi_df, viterbi_metrics, _ = viterbi_detect(
        received_symbol_hits=received,
        bits=bits,
        tap_prob=leglib["tap_prob"],
        n_emit_bit=args.n_emit_bit,
        p_one=args.p_one,
        lambda_env=args.viterbi_lambda_env,
        memory_taps=viterbi_memory_taps,
    )
    hybrid_df, hybrid_metrics, _ = legendre_viterbi_detect(
        received_symbol_hits=received,
        bits=bits,
        leglib=leglib,
        tap_prob=leglib["tap_prob"],
        n_emit_bit=args.n_emit_bit,
        p_one=args.p_one,
        axis=DEFAULT_AXIS,
        lambda_env_count=args.viterbi_lambda_env,
        lambda_env_shape=args.legendre_lambda_env,
        memory_taps=viterbi_memory_taps,
        expected_prev_p1=args.p_one,
    )

    print(f"bits: {args.n_bits}, emitted molecules per active bit: {args.n_emit_bit}")
    print(f"active bits: {int(bits.sum())}")
    print(f"Legendre library: {legendre_file}")
    print(f"Viterbi taps used: {viterbi_memory_taps} / {memory_taps}")
    print("\nDetector comparison")
    print("method      BER       errors   P_miss    P_FA")
    print(
        f"legendre    {legendre_metrics['BER']:<9.6g} {legendre_metrics['errors']:<8d}"
        f" {legendre_metrics['P_miss']:<9.6g} {legendre_metrics['P_FA']:<9.6g}"
    )
    print(
        f"viterbi     {viterbi_metrics['BER']:<9.6g} {viterbi_metrics['errors']:<8d}"
        f" {viterbi_metrics['P_miss']:<9.6g} {viterbi_metrics['P_FA']:<9.6g}"
    )
    print(
        f"leg+vit     {hybrid_metrics['BER']:<9.6g} {hybrid_metrics['errors']:<8d}"
        f" {hybrid_metrics['P_miss']:<9.6g} {hybrid_metrics['P_FA']:<9.6g}"
    )

    legendre_df.to_csv(Path(args.sim_dir) / "legendre_detection.csv", index=False)
    viterbi_df.to_csv(Path(args.sim_dir) / "viterbi_detection.csv", index=False)
    hybrid_df.to_csv(Path(args.sim_dir) / "legendre_viterbi_detection.csv", index=False)


if __name__ == "__main__":
    main()
