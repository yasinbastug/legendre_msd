"""Legendre-library construction and expected-ISI Legendre detection."""

import math
import os
from typing import Iterable

import numpy as np
import pandas as pd
EPS = 1e-300


def make_legendre_library_tag(ts, memory_taps, memory, distance, radius, diffusion_coef, step_time):
    def tag(x):
        s=f"{float(x):.8g}".replace('-', 'm').replace('.', 'p').replace('+','')
        return s
    return f"d_{tag(distance)}_r{tag(radius)}_D{tag(diffusion_coef)}_dt{tag(step_time)}_MEM{tag(memory)}_TS{tag(ts)}_MT{int(memory_taps)}"

def normalize_axis(a):
    """
    Normalize a 3D direction vector.
    """
    a = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(a)

    if n < 1e-15:
        raise ValueError("Cannot normalize near-zero axis vector.")

    return a / n

def unit_rows(X):
    """
    Convert rows of X into unit vectors.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must have shape (N, 3).")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)

    return X / norms

def legendre_basis(mu, L):
    """
    Returns matrix P where P[:, ell] = P_ell(mu).
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    N = mu.shape[0]

    P = np.empty((N, L + 1), dtype=np.float64)
    P[:, 0] = 1.0

    if L >= 1:
        P[:, 1] = mu

    for ell in range(1, L):
        P[:, ell + 1] = (
            ((2 * ell + 1) * mu * P[:, ell] - ell * P[:, ell - 1])
            / (ell + 1)
        )

    return P

def fit_legendre_ls(p_mu, mu_grid, L, lam):
    X = legendre_basis(mu_grid, L)

    penalty = np.array(
        [0.0] + [ell * ell for ell in range(1, L + 1)],
        dtype=np.float64,
    )

    A = X.T @ X + lam * np.diag(penalty)
    b = X.T @ p_mu

    return np.linalg.solve(A, b)

def project_pdf_and_scale_from_coeffs(coeff, mu_grid, L, eps_floor=1e-12):
    X = legendre_basis(mu_grid, L)

    raw = X @ coeff
    raw = np.maximum(raw, eps_floor)

    Z = np.trapz(raw, mu_grid)

    if Z <= 0:
        raw = np.ones_like(mu_grid)
        Z = np.trapz(raw, mu_grid)

    return float(1.0 / max(Z, 1e-15))

def load_hits_file(hit_file):
    if hit_file.endswith(".pt"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading .pt hit files requires PyTorch. Install torch in this environment first."
            ) from exc
        data = torch.load(hit_file, map_location="cpu")

        if "t" not in data or "xyz" not in data:
            raise ValueError(f"{hit_file} must contain keys 't' and 'xyz'.")

        t = data["t"].detach().cpu().numpy().astype(np.float64)
        xyz = data["xyz"].detach().cpu().numpy().astype(np.float64)

        idx = None
        if "idx" in data:
            idx = data["idx"].detach().cpu().numpy().astype(np.int64)

        nof_molecules = None
        if "nof_molecules" in data:
            nof_molecules = int(data["nof_molecules"])

        return {
            "t": t,
            "xyz": xyz,
            "idx": idx,
            "nof_molecules": nof_molecules,
        }

    raise ValueError("hit_file must be a .pt file.")

def build_legendre_library_from_distribution_file(
    hit_file,
    out_file,
    n_emit_distribution,
    memory,
    ts,
    memory_taps,
    n_mu_bins=160,
    n_t_bins=100,
    L=10,
    lam=1e-6,
    tx_axis=(0.0, 0.0, 1.0),
):
    hits = load_hits_file(hit_file)

    t_abs = hits["t"].astype(np.float64)
    xyz = hits["xyz"].astype(np.float64)

    tx_axis = normalize_axis(tx_axis)

    keep = (t_abs >= 0.0) & (t_abs < memory)

    t_abs = t_abs[keep]
    xyz = xyz[keep]

    if len(t_abs) == 0:
        raise ValueError("No received molecules found inside MEMORY horizon.")

    p_hat = unit_rows(xyz)
    mu = np.clip(p_hat @ tx_axis, -1.0, 1.0)

    mu_edges = np.linspace(-1.0, 1.0, n_mu_bins + 1)
    mu_grid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = float(mu_edges[1] - mu_edges[0])

    # Relative time inside one symbol interval [0, TS)
    t_edges = np.linspace(0.0, ts, n_t_bins + 1)

    coeffs = np.zeros((memory_taps, n_t_bins, L + 1), dtype=np.float64)
    scales = np.ones((memory_taps, n_t_bins), dtype=np.float64)

    # rel_mass[tap, j] = P(relative-time bin j | hit in tap)
    rel_mass = np.zeros((memory_taps, n_t_bins), dtype=np.float64)

    # tap_prob[tap] = P(hit in tap per emitted molecule)
    tap_prob = np.zeros(memory_taps, dtype=np.float64)
    tap_counts = np.zeros(memory_taps, dtype=np.int64)

    for tap in range(memory_taps):
        low = tap * ts
        high = (tap + 1) * ts

        mask_tap = (t_abs >= low) & (t_abs < high)

        t_tap = t_abs[mask_tap] - low
        mu_tap = mu[mask_tap]

        count_tap = len(t_tap)
        tap_counts[tap] = count_tap
        tap_prob[tap] = count_tap / float(n_emit_distribution)

        if count_tap == 0:
            rel_mass[tap, :] = 1.0 / n_t_bins
            coeffs[tap, :, 0] = 0.5
            scales[tap, :] = 1.0
            continue

        t_counts, _ = np.histogram(t_tap, bins=t_edges)
        rel_mass[tap, :] = t_counts.astype(np.float64) / float(count_tap)

        for j in range(n_t_bins):
            tj_low = t_edges[j]
            tj_high = t_edges[j + 1]

            mask_j = (t_tap >= tj_low) & (t_tap < tj_high)
            mu_j = mu_tap[mask_j]

            if len(mu_j) == 0:
                coeffs[tap, j, 0] = 0.5
                scales[tap, j] = 1.0
                continue

            hist_mu, _ = np.histogram(mu_j, bins=mu_edges)

            p_mu = hist_mu.astype(np.float64) / max(hist_mu.sum() * dmu, 1e-15)

            c = fit_legendre_ls(
                p_mu=p_mu,
                mu_grid=mu_grid,
                L=L,
                lam=lam,
            )

            s = project_pdf_and_scale_from_coeffs(
                coeff=c,
                mu_grid=mu_grid,
                L=L,
            )

            coeffs[tap, j, :] = c
            scales[tap, j] = s

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    np.savez_compressed(
        out_file,
        memory=float(memory),
        ts=float(ts),
        memory_taps=int(memory_taps),
        n_emit_distribution=int(n_emit_distribution),
        L=int(L),
        lam=float(lam),
        mu_edges=mu_edges,
        t_edges=t_edges,
        coeffs=coeffs,
        scales=scales,
        rel_mass=rel_mass,
        tap_prob=tap_prob,
        tap_counts=tap_counts,
        tx_axis=np.asarray(tx_axis, dtype=np.float64),
        source_hit_file=str(hit_file),
        hyperparam_tag=str(make_legendre_tag()) if "make_legendre_tag" in globals() else "",
    )

    print("\n[OK] Saved Legendre library:")
    print(" ", out_file)
    print("Tap counts:")
    print(" ", tap_counts)
    print("Tap probabilities:")
    print(" ", tap_prob)
    print("Total hit probability inside MEMORY:", float(np.sum(tap_prob)))

    return out_file

def load_legendre_library(npz_file):
    """
    Loads the learned Legendre distribution library.

    Expected keys:
        memory
        ts
        memory_taps
        n_emit_distribution
        L
        lam
        mu_edges
        t_edges
        coeffs
        scales
        rel_mass
        tap_prob
        tap_counts
        tx_axis
        source_hit_file
        hyperparam_tag
    """
    if not os.path.exists(npz_file):
        raise FileNotFoundError(
            "Legendre library file does not exist:\n"
            f"  {npz_file}\n\n"
            "You need to build the Legendre library first, or point "
            "LEGENDRE_FILE to an existing .npz library."
        )

    Z = np.load(npz_file, allow_pickle=True)

    required_keys = [
        "memory",
        "ts",
        "memory_taps",
        "n_emit_distribution",
        "L",
        "lam",
        "mu_edges",
        "t_edges",
        "coeffs",
        "scales",
        "rel_mass",
        "tap_prob",
        "tap_counts",
        "tx_axis",
        "source_hit_file",
    ]

    missing = [k for k in required_keys if k not in Z.files]

    if missing:
        raise ValueError(
            "Legendre library is missing required keys:\n"
            f"  {missing}\n\n"
            f"File: {npz_file}"
        )

    return {
        "memory": float(Z["memory"]),
        "ts": float(Z["ts"]),
        "memory_taps": int(Z["memory_taps"]),
        "n_emit_distribution": int(Z["n_emit_distribution"]),
        "L": int(Z["L"]),
        "lam": float(Z["lam"]),
        "mu_edges": Z["mu_edges"].astype(np.float64),
        "t_edges": Z["t_edges"].astype(np.float64),
        "coeffs": Z["coeffs"].astype(np.float64),
        "scales": Z["scales"].astype(np.float64),
        "rel_mass": Z["rel_mass"].astype(np.float64),
        "tap_prob": Z["tap_prob"].astype(np.float64),
        "tap_counts": Z["tap_counts"].astype(np.int64),
        "tx_axis": Z["tx_axis"].astype(np.float64),
        "source_hit_file": str(Z["source_hit_file"]),
        "hyperparam_tag": str(Z["hyperparam_tag"]) if "hyperparam_tag" in Z.files else "",
    }

def log_poisson_pmf(k, lam):
    """
    log Poisson(k; lambda)
    """
    k = int(k)
    lam = float(lam)

    if k < 0:
        return -np.inf

    lam = max(lam, 1e-15)

    return k * math.log(lam) - lam - math.lgamma(k + 1)

def time_bin_indices_from_t_rel(t_rel, t_edges):
    """
    Converts continuous relative arrival times t_rel into discrete
    arrival-time bin indices.

    Python bin indices:
        0, ..., B-1
    """
    t_rel = np.asarray(t_rel, dtype=np.float64)
    t_edges = np.asarray(t_edges, dtype=np.float64)

    B = len(t_edges) - 1

    b = np.searchsorted(t_edges, t_rel, side="right") - 1
    b = np.clip(b, 0, B - 1).astype(np.int64)

    return b

def legendre_surface_density_bin(
    leglib,
    tap,
    bin_idx,
    xyz_obs,
    axis=None,
):
    """
    Computes surface density:

        f_Omega(u | a, tap, b)
        =
        hbar_L(a^T u | tap, b) / (2*pi)
    """
    if len(xyz_obs) == 0:
        return np.empty(0, dtype=np.float64)

    if axis is None:
        axis = leglib["tx_axis"]

    axis = normalize_axis(axis)

    p_hat = unit_rows(xyz_obs)
    z = np.clip(p_hat @ axis, -1.0, 1.0)

    L = int(leglib["L"])
    P = legendre_basis(z, L)

    coeffs = leglib["coeffs"][tap]      # [B, L+1]
    scales = leglib["scales"][tap]      # [B]

    raw = np.sum(P * coeffs[bin_idx], axis=1)

    hbar = scales[bin_idx] * np.maximum(raw, 1e-12)

    f_omega = hbar / (2.0 * np.pi)

    return np.maximum(f_omega, EPS)

def tap_joint_density_bin(
    leglib,
    tap,
    t_rel,
    xyz_obs,
    axis=None,
):
    """
    Computes joint per-molecule density for one tap:

        f_tap(u,b)
        =
        m_b(tap) * f_Omega(u | tap, b)

    This is normalized over:
        S^2 x arrival bins
    """
    if len(t_rel) == 0:
        return np.empty(0, dtype=np.float64)

    t_edges = leglib["t_edges"]
    bin_idx = time_bin_indices_from_t_rel(t_rel, t_edges)

    m_b = leglib["rel_mass"][tap, bin_idx]

    f_omega = legendre_surface_density_bin(
        leglib=leglib,
        tap=tap,
        bin_idx=bin_idx,
        xyz_obs=xyz_obs,
        axis=axis,
    )

    return np.maximum(m_b * f_omega, EPS)

def uniform_q0_density_bin(
    leglib,
    t_rel,
    xyz_obs,
):
    """
    Uniform environmental/background density:

        q_env(u,b) = 1 / (4*pi*B)
    """
    n = len(t_rel)

    if n == 0:
        return np.empty(0, dtype=np.float64)

    B = len(leglib["t_edges"]) - 1
    q = 1.0 / (4.0 * np.pi * B)

    return np.full(n, q, dtype=np.float64)

def expected_isi_tap_rates(
    leglib,
    n_emit_bit,
    current_bit_hypothesis,
    expected_prev_p1=0.5,
):
    """
    Returns expected received count contribution from each tap.

    current_bit_hypothesis:
        0 for H0
        1 for H1

    tap 0:
        current bit contribution:
            H0: 0
            H1: N_EMIT_BIT * tap_prob[0]

    taps 1..M-1:
        expected previous-bit ISI contribution:
            N_EMIT_BIT * expected_prev_p1 * tap_prob[tap]
    """
    tap_prob = np.asarray(leglib["tap_prob"], dtype=np.float64)
    M = len(tap_prob)

    rates = np.zeros(M, dtype=np.float64)

    if int(current_bit_hypothesis) == 1:
        rates[0] = float(n_emit_bit) * tap_prob[0]
    elif int(current_bit_hypothesis) == 0:
        rates[0] = 0.0
    else:
        raise ValueError("current_bit_hypothesis must be 0 or 1.")

    for tap in range(1, M):
        rates[tap] = float(n_emit_bit) * float(expected_prev_p1) * tap_prob[tap]

    return rates

def expected_isi_total_rate(
    leglib,
    n_emit_bit,
    current_bit_hypothesis,
    expected_prev_p1=0.5,
    lambda_env=0.0,
):
    """
    Expected total received molecule count under H0/H1.
    """
    rates = expected_isi_tap_rates(
        leglib=leglib,
        n_emit_bit=n_emit_bit,
        current_bit_hypothesis=current_bit_hypothesis,
        expected_prev_p1=expected_prev_p1,
    )

    return float(np.sum(rates) + float(lambda_env))

def expected_isi_mixture_density(
    leglib,
    t_rel,
    xyz_obs,
    n_emit_bit,
    current_bit_hypothesis,
    axis=None,
    expected_prev_p1=0.5,
    lambda_env=0.0,
):
    """
    Conditional per-molecule density under H0 or H1.

    Uses rate-weighted mixture:

        f_H(u,b)
        =
        [lambda_env q_env(u,b)
         +
         sum_tap lambda_tap f_tap(u,b)]
        /
        [lambda_env + sum_tap lambda_tap]

    where:
        lambda_tap = expected received count from that tap.
    """
    n = len(t_rel)

    if n == 0:
        return np.empty(0, dtype=np.float64)

    rates = expected_isi_tap_rates(
        leglib=leglib,
        n_emit_bit=n_emit_bit,
        current_bit_hypothesis=current_bit_hypothesis,
        expected_prev_p1=expected_prev_p1,
    )

    lambda_env = float(lambda_env)
    total_rate = float(np.sum(rates) + lambda_env)

    if total_rate <= 0.0:
        return uniform_q0_density_bin(
            leglib=leglib,
            t_rel=t_rel,
            xyz_obs=xyz_obs,
        )

    mixture = np.zeros(n, dtype=np.float64)

    if lambda_env > 0.0:
        q_env = uniform_q0_density_bin(
            leglib=leglib,
            t_rel=t_rel,
            xyz_obs=xyz_obs,
        )
        mixture += lambda_env * q_env

    for tap, rate in enumerate(rates):
        if rate <= 0.0:
            continue

        f_tap = tap_joint_density_bin(
            leglib=leglib,
            tap=tap,
            t_rel=t_rel,
            xyz_obs=xyz_obs,
            axis=axis,
        )

        mixture += rate * f_tap

    mixture /= total_rate

    return np.maximum(mixture, EPS)

def expected_isi_loglik(
    leglib,
    t_rel,
    xyz_obs,
    n_emit_bit,
    current_bit_hypothesis,
    axis=None,
    expected_prev_p1=0.5,
    lambda_env=0.0,
    use_count_likelihood=True,
    use_shape_likelihood=True,
):
    """
    Log-likelihood under H0 or H1 with expected ISI.

    H0:
        current_bit_hypothesis = 0

    H1:
        current_bit_hypothesis = 1

    Count:
        N_rx ~ Poisson(lambda_H)

    Shape/time:
        observed molecules are iid from the expected-ISI mixture density.
    """
    n_rx = int(len(t_rel))

    ll_count = 0.0
    ll_samples = 0.0

    lambda_H = expected_isi_total_rate(
        leglib=leglib,
        n_emit_bit=n_emit_bit,
        current_bit_hypothesis=current_bit_hypothesis,
        expected_prev_p1=expected_prev_p1,
        lambda_env=lambda_env,
    )

    if use_count_likelihood:
        ll_count = log_poisson_pmf(
            k=n_rx,
            lam=lambda_H,
        )

    if use_shape_likelihood and n_rx > 0:
        p = expected_isi_mixture_density(
            leglib=leglib,
            t_rel=t_rel,
            xyz_obs=xyz_obs,
            n_emit_bit=n_emit_bit,
            current_bit_hypothesis=current_bit_hypothesis,
            axis=axis,
            expected_prev_p1=expected_prev_p1,
            lambda_env=lambda_env,
        )

        ll_samples = float(np.sum(np.log(np.maximum(p, EPS))))

    return {
        "ell": float(ll_count + ll_samples),
        "ell_count": float(ll_count),
        "ell_samples": float(ll_samples),
        "lambda": float(lambda_H),
        "n_rx": int(n_rx),
    }

def expected_isi_llr_known_axis(
    leglib,
    t_rel,
    xyz_obs,
    n_emit_bit,
    axis=None,
    expected_prev_p1=0.5,
    lambda_env=0.0,
    use_count_likelihood=True,
    use_shape_likelihood=True,
):
    """
    Expected-ISI likelihood-ratio detector:

        LLR = log p(D | current bit = 1, expected ISI)
              -
              log p(D | current bit = 0, expected ISI)
    """
    h1 = expected_isi_loglik(
        leglib=leglib,
        t_rel=t_rel,
        xyz_obs=xyz_obs,
        n_emit_bit=n_emit_bit,
        current_bit_hypothesis=1,
        axis=axis,
        expected_prev_p1=expected_prev_p1,
        lambda_env=lambda_env,
        use_count_likelihood=use_count_likelihood,
        use_shape_likelihood=use_shape_likelihood,
    )

    h0 = expected_isi_loglik(
        leglib=leglib,
        t_rel=t_rel,
        xyz_obs=xyz_obs,
        n_emit_bit=n_emit_bit,
        current_bit_hypothesis=0,
        axis=axis,
        expected_prev_p1=expected_prev_p1,
        lambda_env=lambda_env,
        use_count_likelihood=use_count_likelihood,
        use_shape_likelihood=use_shape_likelihood,
    )

    return {
        "llr": float(h1["ell"] - h0["ell"]),

        "ell1": float(h1["ell"]),
        "ell0": float(h0["ell"]),

        "count_llr": float(h1["ell_count"] - h0["ell_count"]),
        "sample_llr": float(h1["ell_samples"] - h0["ell_samples"]),

        "ell1_count": float(h1["ell_count"]),
        "ell0_count": float(h0["ell_count"]),

        "ell1_samples": float(h1["ell_samples"]),
        "ell0_samples": float(h0["ell_samples"]),

        "lambda_h1": float(h1["lambda"]),
        "lambda_h0": float(h0["lambda"]),

        "n_rx": int(h1["n_rx"]),
    }

def binary_detection_metrics(y_true, y_hat):
    y_true = np.asarray(y_true, dtype=np.int8)
    y_hat = np.asarray(y_hat, dtype=np.int8)
    errors = int(np.sum(y_hat != y_true))
    return {
        "BER": float(errors / len(y_true)),
        "errors": errors,
        "P_miss": float(np.mean(y_hat[y_true == 1] == 0)) if np.any(y_true == 1) else np.nan,
        "P_FA": float(np.mean(y_hat[y_true == 0] == 1)) if np.any(y_true == 0) else np.nan,
        "n_bits": int(len(y_true)),
    }


def legendre_detect(
    received_symbol_hits,
    bits,
    leglib,
    n_emit_bit,
    axis=(0.0, 0.0, 1.0),
    p_one=0.5,
    expected_prev_p1=0.5,
    lambda_env=1.0,
    use_count_likelihood=True,
    use_shape_likelihood=True,
    llr_threshold=None,
):
    """Run expected-ISI Legendre likelihood detection over one bit sequence."""
    bits = np.asarray(bits, dtype=np.int8)
    axis = normalize_axis(axis)
    if llr_threshold is None:
        llr_threshold = math.log((1.0 - float(p_one)) / max(float(p_one), 1e-15))

    rows = []
    for n, obs in enumerate(received_symbol_hits):
        result = expected_isi_llr_known_axis(
            leglib=leglib,
            t_rel=obs["t_rel"],
            xyz_obs=obs["xyz"],
            n_emit_bit=n_emit_bit,
            axis=axis,
            expected_prev_p1=expected_prev_p1,
            lambda_env=lambda_env,
            use_count_likelihood=use_count_likelihood,
            use_shape_likelihood=use_shape_likelihood,
        )
        detected = int(result["llr"] > llr_threshold)
        rows.append({
            "bit_index": n,
            "true_bit": int(bits[n]),
            "detected_bit": detected,
            "correct": bool(detected == int(bits[n])),
            "n_rx": int(result["n_rx"]),
            "expected_isi_llr": float(result["llr"]),
            "count_llr": float(result["count_llr"]),
            "sample_llr": float(result["sample_llr"]),
            "lambda_h0": float(result["lambda_h0"]),
            "lambda_h1": float(result["lambda_h1"]),
        })

    df = pd.DataFrame(rows)
    metrics = binary_detection_metrics(df["true_bit"], df["detected_bit"])
    return df, metrics
