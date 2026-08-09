"""
Full maximum-likelihood Legendre-Viterbi detector with PyTorch/CUDA
branch-metric parallelization.

This module preserves the same finite-memory, binned Poisson point-process
branch metric as the original strong detector:

    mu_branch(u,b) = lambda_env q_env(u,b)
                     + sum_{m=0}^{K-1} x_{n-m} mu_m(u,b)

    gamma_n = log P(x_n) - Lambda_branch
              + sum_i log mu_branch(u_i, b_i)

No pruning, no reduced-state approximation, and no hybrid shape LLR are used.
This variant uses torch.float32 on the PyTorch path as requested, with TF32
disabled so CUDA matmul does not silently use lower-precision tensor cores.

The speedup comes from computing all K-bit branch-mask metrics on the GPU,
precomputing branch masks once instead of rebuilding them every symbol, avoiding
CUDA synchronization inside the metric loop, and updating the Viterbi states in
parallel. The Viterbi recursion is still exact for the implemented model and
still scales as O(N_symbols * 2^K * N_rx).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss, legval

try:
    import torch
except Exception:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def binary_detection_metrics(true_bits: np.ndarray, detected_bits: np.ndarray) -> Dict[str, float]:
    """Compute BER, false-alarm probability, and missed-detection probability."""
    true_bits = np.asarray(true_bits, dtype=np.int8)
    detected_bits = np.asarray(detected_bits, dtype=np.int8)

    if true_bits.shape != detected_bits.shape:
        raise ValueError("true_bits and detected_bits must have the same shape.")

    errors = detected_bits != true_bits
    ber = float(np.mean(errors)) if len(true_bits) else float("nan")

    idx0 = true_bits == 0
    idx1 = true_bits == 1

    p_1_given_0 = float(np.mean(detected_bits[idx0] == 1)) if np.any(idx0) else float("nan")
    p_0_given_1 = float(np.mean(detected_bits[idx1] == 0)) if np.any(idx1) else float("nan")

    return {
        "BER": ber,
        "P_1_given_0": p_1_given_0,
        "P_0_given_1": p_0_given_1,
        "n_bits": int(len(true_bits)),
        "n_errors": int(np.sum(errors)),
    }


def normalize_axis(axis: Sequence[float]) -> np.ndarray:
    """Return a unit-norm 3D axis vector."""
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError("axis must be a length-3 vector.")
    norm = np.linalg.norm(axis)
    if norm <= 0:
        raise ValueError("axis must have nonzero norm.")
    return axis / norm


def hitting_cdf(t: np.ndarray, d: float, r: float, diffusion_coeff: float) -> np.ndarray:
    """
    Absorbing spherical receiver hitting CDF:

        F_hit(t,d,r) = (r/d) erfc((d-r)/sqrt(4Dt)).

    t may be scalar or array. For t <= 0, the value is set to 0.
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t, dtype=np.float64)

    positive = t > 0
    if np.any(positive):
        arg = (d - r) / np.sqrt(4.0 * diffusion_coeff * t[positive])
        erfc_vec = np.vectorize(math.erfc)
        out[positive] = (r / d) * erfc_vec(arg)

    return out


def make_tap_bin_prob_from_fhit(
    K: int,
    B: int,
    Ts: float,
    d: float,
    r: float,
    diffusion_coeff: float,
) -> np.ndarray:
    """
    Construct tap-bin hitting probabilities from F_hit differences.

    tap_bin_prob[m,b] =
        F_hit(m Ts + (b+1) Delta) - F_hit(m Ts + b Delta),

    where Delta = Ts / B.
    """
    if K < 1:
        raise ValueError("K must be at least 1.")
    if B < 1:
        raise ValueError("B must be at least 1.")
    if Ts <= 0:
        raise ValueError("Ts must be positive.")

    delta = Ts / B
    edges = np.arange(K * B + 1, dtype=np.float64) * delta
    F = hitting_cdf(edges, d=d, r=r, diffusion_coeff=diffusion_coeff)
    probs = np.diff(F).reshape(K, B)

    # Guard against tiny numerical negatives from floating-point roundoff.
    probs = np.maximum(probs, 0.0)
    return probs


def _extract_bins_and_z(
    obs: Dict,
    Ts: float,
    B: int,
    axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract within-symbol bin indices and z = a^T u values from one symbol observation.

    Expected obs format:
        obs["xyz"]   : array-like, shape (N_rx, 3), absorption positions or directions
        obs["t_rel"] : array-like, shape (N_rx,), arrival times relative to current slot

    Optional:
        obs["bin"] or obs["bins"] may be supplied instead of t_rel.
        If bins are 1-based, they are automatically converted to 0-based.
    """
    xyz = np.asarray(obs.get("xyz", []), dtype=np.float64).reshape(-1, 3)
    n_rx = xyz.shape[0]

    if n_rx == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    norms = np.linalg.norm(xyz, axis=1)
    if np.any(norms <= 0):
        raise ValueError("All xyz rows must have nonzero norm.")

    u = xyz / norms[:, None]
    z = np.clip(u @ axis, -1.0, 1.0)

    if "bin" in obs:
        b_idx = np.asarray(obs["bin"], dtype=np.int64)
    elif "bins" in obs:
        b_idx = np.asarray(obs["bins"], dtype=np.int64)
    else:
        t_rel = np.asarray(obs["t_rel"], dtype=np.float64)
        if t_rel.shape[0] != n_rx:
            raise ValueError("obs['t_rel'] must have the same length as obs['xyz'].")
        b_idx = np.floor(t_rel / Ts * B).astype(np.int64)

    if b_idx.shape[0] != n_rx:
        raise ValueError("Bin array must have the same length as obs['xyz'].")

    # Convert likely 1-based bin indices to 0-based.
    if b_idx.size and b_idx.min() >= 1 and b_idx.max() <= B:
        b_idx = b_idx - 1

    b_idx = np.clip(b_idx, 0, B - 1)
    return b_idx, z


# ---------------------------------------------------------------------
# Learned Legendre library
# ---------------------------------------------------------------------


@dataclass
class FullLegendreLibrary:
    """
    Learned tap-specific Legendre library.

    coeffs:
        Array of shape (K, B, L+1).
        coeffs[m,b,l] is the coefficient of P_l(z) for tap m and bin b.

    tap_bin_prob:
        Array of shape (K, B).
        tap_bin_prob[m,b] is the probability that one emitted molecule is
        absorbed in tap m and bin b.

    eps_floor:
        Numerical floor used to ensure nonnegative truncated Legendre density.

    quad_order:
        Gauss-Legendre quadrature order used to renormalize floored densities.
    """

    coeffs: np.ndarray
    tap_bin_prob: np.ndarray
    eps_floor: float = 1e-12
    quad_order: int = 256

    def __post_init__(self) -> None:
        self.coeffs = np.asarray(self.coeffs, dtype=np.float64)
        self.tap_bin_prob = np.asarray(self.tap_bin_prob, dtype=np.float64)

        if self.coeffs.ndim != 3:
            raise ValueError("coeffs must have shape (K, B, L+1).")
        if self.tap_bin_prob.ndim != 2:
            raise ValueError("tap_bin_prob must have shape (K, B).")
        if self.coeffs.shape[:2] != self.tap_bin_prob.shape:
            raise ValueError("coeffs and tap_bin_prob must agree in K and B.")
        if self.eps_floor <= 0:
            raise ValueError("eps_floor must be positive.")
        if self.quad_order < 8:
            raise ValueError("quad_order should be at least 8.")

        self.K = int(self.coeffs.shape[0])
        self.B = int(self.coeffs.shape[1])
        self.L = int(self.coeffs.shape[2] - 1)

        self.tap_bin_prob = np.maximum(self.tap_bin_prob, 0.0)

        self._norm = self._compute_floor_normalizers()

    def _compute_floor_normalizers(self) -> np.ndarray:
        """
        Compute

            int_{-1}^{1} max(eps, h_L(z | m,b)) dz

        for each tap and bin.
        """
        nodes, weights = leggauss(self.quad_order)
        norm = np.zeros((self.K, self.B), dtype=np.float64)

        for m in range(self.K):
            for b in range(self.B):
                h = legval(nodes, self.coeffs[m, b, :])
                h_floor = np.maximum(self.eps_floor, h)
                val = float(np.sum(weights * h_floor))
                if not np.isfinite(val) or val <= 0:
                    raise ValueError(f"Invalid density normalizer at tap {m}, bin {b}.")
                norm[m, b] = val

        return norm

    def surface_density(self, m: int, b_idx: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Evaluate surface density

            f_m,b(u) = (1 / 2pi) hbar_L(a^T u | m,b)

        for one tap m and arrays of bin indices and z values.
        """
        if m < 0 or m >= self.K:
            raise IndexError("tap index m out of range.")

        b_idx = np.asarray(b_idx, dtype=np.int64)
        z = np.asarray(z, dtype=np.float64)

        if b_idx.shape != z.shape:
            raise ValueError("b_idx and z must have the same shape.")

        out = np.empty_like(z, dtype=np.float64)

        for b in np.unique(b_idx):
            if b < 0 or b >= self.B:
                raise IndexError("bin index out of range.")
            mask = b_idx == b
            h = legval(z[mask], self.coeffs[m, b, :])
            h_floor = np.maximum(self.eps_floor, h)
            h_bar = h_floor / self._norm[m, b]
            out[mask] = h_bar / (2.0 * math.pi)

        return out

    def truncated(self, K: int) -> "FullLegendreLibrary":
        """Return a copy using only the first K taps."""
        if K < 1 or K > self.K:
            raise ValueError(f"K must satisfy 1 <= K <= {self.K}.")
        return FullLegendreLibrary(
            coeffs=self.coeffs[:K].copy(),
            tap_bin_prob=self.tap_bin_prob[:K].copy(),
            eps_floor=self.eps_floor,
            quad_order=self.quad_order,
        )


def build_library_from_arrays(
    coeffs: np.ndarray,
    tap_bin_prob: np.ndarray,
    eps_floor: float = 1e-12,
    quad_order: int = 256,
) -> FullLegendreLibrary:
    """
    Convenience wrapper.

    coeffs shape:
        (K, B, L+1)

    tap_bin_prob shape:
        (K, B)
    """
    return FullLegendreLibrary(
        coeffs=coeffs,
        tap_bin_prob=tap_bin_prob,
        eps_floor=eps_floor,
        quad_order=quad_order,
    )


# ---------------------------------------------------------------------
# PyTorch/CUDA full-ML detector
# ---------------------------------------------------------------------


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for full_ml_legendre_viterbi_detect. "
            "Install torch, or use the original NumPy-only detector."
        )


def _resolve_torch_device(
    device: Optional[Union[str, "torch.device"]],
    require_cuda: bool,
) -> "torch.device":
    _require_torch()
    assert torch is not None

    if device is None:
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
        else:
            if require_cuda:
                raise RuntimeError("CUDA was required, but torch.cuda.is_available() is False.")
            resolved = torch.device("cpu")
    else:
        resolved = torch.device(device)
        if require_cuda and resolved.type != "cuda":
            raise ValueError("require_cuda=True requires device='cuda' or another CUDA device.")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() is False.")

    return resolved


def _configure_torch_for_float32_accuracy() -> None:
    """Disable lower-precision matmul modes while using torch.float32."""
    _require_torch()
    assert torch is not None

    # TF32 can change float32 matmul results on NVIDIA Ampere+ GPUs. Disable it
    # so torch.float32 means IEEE-style FP32 matmul, not TF32 tensor-core math.
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    # Do not force torch.use_deterministic_algorithms(True): it can disable or
    # slow operations unnecessarily. The code below uses explicit comparisons
    # for tie handling and does not use random operations.


def _precompute_observations_numpy(
    received_symbol_hits: List[Dict],
    leglib: FullLegendreLibrary,
    n_emit_bit: float,
    Ts: float,
    axis: np.ndarray,
    lambda_env_count: float,
    env_bin_prob_arr: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Precompute tap_mu_hits and env_mu_hits in NumPy float64."""
    K = leglib.K
    B = leglib.B
    n_symbols = len(received_symbol_hits)

    tap_mu_hits: List[np.ndarray] = []
    env_mu_hits: List[np.ndarray] = []
    counts = np.zeros(n_symbols, dtype=np.int64)

    for obs in received_symbol_hits:
        b_idx, z = _extract_bins_and_z(obs, Ts=Ts, B=B, axis=axis)
        n_rx = len(z)
        counts[len(tap_mu_hits)] = n_rx

        tap_mu = np.zeros((K, n_rx), dtype=np.float64)
        for m in range(K):
            if n_rx == 0:
                continue
            surf_density = leglib.surface_density(m=m, b_idx=b_idx, z=z)
            tap_mu[m, :] = (
                float(n_emit_bit)
                * leglib.tap_bin_prob[m, b_idx]
                * surf_density
            )

        if n_rx == 0:
            env_mu = np.zeros(0, dtype=np.float64)
        else:
            # Uniform environmental surface density over the sphere.
            env_mu = (
                float(lambda_env_count)
                * env_bin_prob_arr[b_idx]
                * (1.0 / (4.0 * math.pi))
            )

        tap_mu_hits.append(tap_mu)
        env_mu_hits.append(env_mu)

    return tap_mu_hits, env_mu_hits, counts


def _default_branch_chunk(
    n_rx: int,
    n_branches: int,
    target_temp_bytes: int,
    bytes_per_value: int,
) -> int:
    """Select a branch chunk size based on the main temporary mu matrix."""
    if n_rx <= 0:
        return n_branches
    # Main temporary is mu[chunk, n_rx]. For torch.float32, bytes_per_value=4.
    chunk = int(target_temp_bytes // max(bytes_per_value * n_rx, 1))
    return max(1, min(n_branches, chunk))


def full_ml_legendre_viterbi_detect(
    received_symbol_hits: List[Dict],
    bits: Optional[Sequence[int]],
    leglib: FullLegendreLibrary,
    n_emit_bit: float,
    Ts: float,
    p_one: float = 0.5,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    memory_taps: Optional[int] = None,
    lambda_env_count: float = 0.0,
    env_bin_prob: Optional[Sequence[float]] = None,
    use_bit_prior: bool = True,
    initial_state: int = 0,
    max_states: int = 2_000_000,
    *,
    device: Optional[Union[str, "torch.device"]] = None,
    require_cuda: bool = False,
    branch_chunk: Optional[int] = None,
    target_temp_bytes: int = 4 * 1024 * 1024 * 1024,
    preload_observations_to_device: bool = True,
    backpointer_storage: str = "device",
    precompute_branch_bits: bool = True,
    strict_intensity_check: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict[str, float]], Dict]:
    """
    Full maximum-likelihood Legendre-Viterbi detector using PyTorch parallelism.

    This function keeps the same finite-memory binned PPP branch metric as the
    original strong detector. It parallelizes over the 2^K K-bit branch masks
    and over the 2^(K-1) Viterbi states using torch.float32.

    Parameters
    ----------
    received_symbol_hits:
        List of observations, one per symbol.
        Each observation should contain obs["xyz"] and obs["t_rel"], or
        obs["xyz"] and obs["bin"] / obs["bins"].

    bits:
        True bits, used only for metrics. Pass None if unknown.

    leglib:
        FullLegendreLibrary containing tap-specific Legendre coefficients and
        tap-bin hitting probabilities.

    n_emit_bit:
        Number of molecules emitted for an active bit.

    Ts:
        Symbol duration.

    p_one:
        Prior probability P(x_n = 1).

    axis:
        Known Tx direction from the Rx center.

    memory_taps:
        Number of channel taps K. If None, all taps in leglib are used.

    lambda_env_count:
        Expected environmental-noise molecule count per symbol interval.

    env_bin_prob:
        Optional length-B distribution over time bins for environmental noise.
        If None and lambda_env_count > 0, a uniform bin distribution is used.

    use_bit_prior:
        If False, the bit prior term is omitted.

    initial_state:
        Initial trellis state. Usually 0, corresponding to zero pre-history.

    max_states:
        Safety limit for the number of Viterbi states, 2^(K-1).

    device:
        Torch device. Use "cuda" for GPU. If None, CUDA is used when available;
        otherwise CPU torch is used unless require_cuda=True.

    require_cuda:
        If True, raise an error instead of silently falling back to CPU.

    branch_chunk:
        Number of K-bit branch masks to score at once. Leave None for automatic
        memory-based chunking. Lower this if CUDA memory is limited.

    target_temp_bytes:
        Approximate target size for the main temporary mu matrix when
        branch_chunk is None. Default is 4 GiB.

    preload_observations_to_device:
        If True, copies all per-symbol tap/env intensity arrays to the torch
        device before Viterbi starts. Faster on GPU when memory allows. If False,
        each symbol is copied just before it is scored.

    backpointer_storage:
        "device" keeps backpointers on the torch device until traceback.
        "cpu" stores them in NumPy each symbol to save GPU memory.

    precompute_branch_bits:
        If True, precompute the full K-bit branch-mask matrix once on the
        selected torch device and reuse it for every symbol/chunk. This is
        usually much faster than reconstructing masks inside the Viterbi loop.
        It does not change the detector metric.

    strict_intensity_check:
        If True, exactly preserves the original invalid-branch rule: any branch
        with a nonpositive or nonfinite intensity at any observed hit receives
        -inf metric. Keep True unless you have independently verified all
        intensities are strictly positive and finite.

    Returns
    -------
    df:
        Per-symbol detection results.

    metrics:
        BER and error probabilities if bits are supplied, else None.

    result:
        Dictionary containing detected bits, path metric, state path, and
        configuration details.
    """
    _require_torch()
    assert torch is not None
    _configure_torch_for_float32_accuracy()

    if Ts <= 0:
        raise ValueError("Ts must be positive.")
    if n_emit_bit < 0:
        raise ValueError("n_emit_bit must be nonnegative.")
    if lambda_env_count < 0:
        raise ValueError("lambda_env_count must be nonnegative.")
    if target_temp_bytes < 1024:
        raise ValueError("target_temp_bytes is too small.")
    if backpointer_storage not in {"device", "cpu"}:
        raise ValueError("backpointer_storage must be 'device' or 'cpu'.")

    dev = _resolve_torch_device(device=device, require_cuda=require_cuda)
    dtype = torch.float32

    if memory_taps is not None:
        leglib = leglib.truncated(int(memory_taps))

    K = int(leglib.K)
    B = int(leglib.B)
    axis_arr = normalize_axis(axis)

    n_symbols = len(received_symbol_hits)

    if bits is not None:
        bits_arr = np.asarray(bits, dtype=np.int8)
        if len(bits_arr) != n_symbols:
            raise ValueError("bits and received_symbol_hits must have the same length.")
    else:
        bits_arr = None

    state_len = K - 1
    n_states = 1 << state_len
    n_branches = 1 << K

    if n_states > int(max_states):
        raise MemoryError(
            f"Full ML Legendre-Viterbi would require {n_states:,} states for K={K}. "
            "Reduce memory_taps or increase max_states intentionally."
        )

    if initial_state < 0 or initial_state >= n_states:
        raise ValueError("initial_state is out of range.")

    p_one_clip = min(max(float(p_one), 1e-15), 1.0 - 1e-15)
    log_p1 = math.log(p_one_clip)
    log_p0 = math.log(1.0 - p_one_clip)

    if not use_bit_prior:
        log_p1 = 0.0
        log_p0 = 0.0

    if env_bin_prob is None:
        env_bin_prob_arr = np.full(B, 1.0 / B, dtype=np.float64)
    else:
        env_bin_prob_arr = np.asarray(env_bin_prob, dtype=np.float64)
        if env_bin_prob_arr.shape != (B,):
            raise ValueError("env_bin_prob must have shape (B,).")
        if np.any(env_bin_prob_arr < 0):
            raise ValueError("env_bin_prob must be nonnegative.")
        total = float(np.sum(env_bin_prob_arr))
        if total <= 0:
            raise ValueError("env_bin_prob must have positive sum.")
        env_bin_prob_arr = env_bin_prob_arr / total

    lambda_tap_np = float(n_emit_bit) * np.sum(leglib.tap_bin_prob, axis=1)

    tap_mu_hits_np, env_mu_hits_np, counts = _precompute_observations_numpy(
        received_symbol_hits=received_symbol_hits,
        leglib=leglib,
        n_emit_bit=float(n_emit_bit),
        Ts=float(Ts),
        axis=axis_arr,
        lambda_env_count=float(lambda_env_count),
        env_bin_prob_arr=env_bin_prob_arr,
    )

    lambda_tap = torch.as_tensor(lambda_tap_np, dtype=dtype, device=dev)
    branch_ids = torch.arange(n_branches, dtype=torch.long, device=dev)
    tap_indices = torch.arange(K, dtype=torch.long, device=dev)

    # Base branch terms depend only on the K-bit branch mask:
    # bit 0 = current bit, bit m = x_{n-m}.
    # Precompute the K-bit branch-mask matrix once. The previous version
    # reconstructed this matrix inside every symbol/chunk, which is expensive
    # and also creates many short-lived CUDA allocations. Reusing this tensor
    # preserves the exact same branch metric.
    if precompute_branch_bits:
        branch_bits_all: Optional["torch.Tensor"] = (
            ((branch_ids[:, None] >> tap_indices[None, :]) & 1).to(dtype)
        )
        lambda_by_branch = branch_bits_all.matmul(lambda_tap)
    else:
        branch_bits_all = None
        branch_bits_for_lambda = ((branch_ids[:, None] >> tap_indices[None, :]) & 1).to(dtype)
        lambda_by_branch = branch_bits_for_lambda.matmul(lambda_tap)
        del branch_bits_for_lambda

    current_bit_by_branch = (branch_ids & 1).to(torch.long)
    prior_by_branch = torch.where(
        current_bit_by_branch == 1,
        torch.tensor(log_p1, dtype=dtype, device=dev),
        torch.tensor(log_p0, dtype=dtype, device=dev),
    )
    branch_base = prior_by_branch - (float(lambda_env_count) + lambda_by_branch)

    if state_len > 0:
        new_states = torch.arange(n_states, dtype=torch.long, device=dev)
        prev0 = new_states >> 1
        oldest_mask_state = 1 << (state_len - 1)
        prev1 = prev0 | oldest_mask_state
        branch0 = new_states
        branch1 = new_states | (1 << (K - 1))
    else:
        new_states = torch.zeros(1, dtype=torch.long, device=dev)
        prev0 = prev1 = branch0 = branch1 = new_states

    if preload_observations_to_device:
        tap_mu_hits_t = [torch.as_tensor(x, dtype=dtype, device=dev) for x in tap_mu_hits_np]
        env_mu_hits_t = [torch.as_tensor(x, dtype=dtype, device=dev) for x in env_mu_hits_np]
    else:
        tap_mu_hits_t = []
        env_mu_hits_t = []

    def get_symbol_tensors(n: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if preload_observations_to_device:
            return tap_mu_hits_t[n], env_mu_hits_t[n]
        return (
            torch.as_tensor(tap_mu_hits_np[n], dtype=dtype, device=dev),
            torch.as_tensor(env_mu_hits_np[n], dtype=dtype, device=dev),
        )

    def score_all_branch_masks(tap_mu: "torch.Tensor", env_mu: "torch.Tensor", n_rx: int) -> "torch.Tensor":
        """
        Compute gamma[branch_mask] for all 2^K K-bit branch masks.

        gamma[mask] = log P(current_bit) - Lambda(mask)
                      + sum_i log(env_mu_i + sum_m mask_m tap_mu[m,i])

        All PyTorch path computations use torch.float32 as requested. No
        approximation/pruning/reduced-state detector is used.
        """
        gamma = branch_base.clone()

        if n_rx == 0:
            return gamma

        if branch_chunk is None:
            chunk = _default_branch_chunk(
                n_rx=n_rx,
                n_branches=n_branches,
                target_temp_bytes=int(target_temp_bytes),
                bytes_per_value=torch.empty((), dtype=dtype).element_size(),
            )
        else:
            chunk = max(1, min(n_branches, int(branch_chunk)))

        neg_inf = torch.tensor(-math.inf, dtype=dtype, device=dev)

        for start in range(0, n_branches, chunk):
            stop = min(start + chunk, n_branches)

            # FP32 subset-sum over taps, using full K-bit branch masks.
            # If precompute_branch_bits=True, this avoids reconstructing the
            # mask tensor for every symbol/chunk.
            if branch_bits_all is not None:
                bits_chunk = branch_bits_all[start:stop, :]
            else:
                ids = branch_ids[start:stop]
                bits_chunk = ((ids[:, None] >> tap_indices[None, :]) & 1).to(dtype)

            mu = bits_chunk.matmul(tap_mu)
            mu = mu + env_mu.unsqueeze(0)

            if strict_intensity_check:
                # No .item() / Python if here: that would synchronize CUDA on
                # every chunk. torch.where preserves the original invalid-branch
                # rule while staying fully on device.
                valid = torch.all((mu > 0.0) & torch.isfinite(mu), dim=1)
                vals_raw = torch.log(mu).sum(dim=1)
                vals = torch.where(valid, vals_raw, neg_inf)
            else:
                # Faster path if the caller has verified all intensities are
                # strictly positive and finite. log(0) still naturally gives
                # -inf. This does not alter the likelihood formula, but skips
                # validation work.
                vals = torch.log(mu).sum(dim=1)

            gamma[start:stop] = gamma[start:stop] + vals

        return gamma

    dp_prev = torch.full((n_states,), -math.inf, dtype=dtype, device=dev)
    dp_prev[int(initial_state)] = 0.0

    if state_len == 0:
        # One-state trellis; each symbol chooses between branch mask 0 and 1.
        back_bits = np.zeros(n_symbols, dtype=np.int8)

        for n in range(n_symbols):
            tap_mu, env_mu = get_symbol_tensors(n)
            gamma = score_all_branch_masks(tap_mu, env_mu, int(counts[n]))
            score0 = dp_prev[0] + gamma[0]
            score1 = dp_prev[0] + gamma[1]
            take1 = bool((score1 > score0).item())
            if take1:
                dp_prev[0] = score1
                back_bits[n] = 1
            else:
                dp_prev[0] = score0
                back_bits[n] = 0

        detected_bits = back_bits.copy()
        state_path = np.zeros(n_symbols, dtype=np.int64)
        best_final_state = 0
        best_final_metric = float(dp_prev[0].item())

    else:
        if backpointer_storage == "device":
            back_choice_t = torch.empty((n_symbols, n_states), dtype=torch.bool, device=dev)
            back_choice_np: Optional[np.ndarray] = None
        else:
            back_choice_t = None  # type: ignore[assignment]
            back_choice_np = np.empty((n_symbols, n_states), dtype=np.bool_)

        for n in range(n_symbols):
            tap_mu, env_mu = get_symbol_tensors(n)
            gamma = score_all_branch_masks(tap_mu, env_mu, int(counts[n]))

            score0 = dp_prev[prev0] + gamma[branch0]
            score1 = dp_prev[prev1] + gamma[branch1]

            # Match the original tie behavior: choose predecessor 0 on ties.
            take_prev1 = score1 > score0
            dp_prev = torch.where(take_prev1, score1, score0)

            if backpointer_storage == "device":
                back_choice_t[n, :] = take_prev1
            else:
                back_choice_np[n, :] = take_prev1.detach().cpu().numpy()

        best_final_state = int(torch.argmax(dp_prev).item())
        best_final_metric = float(dp_prev[best_final_state].item())

        if backpointer_storage == "device":
            back_choice = back_choice_t.detach().cpu().numpy()
        else:
            assert back_choice_np is not None
            back_choice = back_choice_np

        detected_bits = np.zeros(n_symbols, dtype=np.int8)
        state_path = np.zeros(n_symbols, dtype=np.int64)

        s = best_final_state
        oldest_mask_state_np = 1 << (state_len - 1)
        for n in range(n_symbols - 1, -1, -1):
            state_path[n] = s
            detected_bits[n] = np.int8(s & 1)
            if back_choice[n, s]:
                s = (s >> 1) | oldest_mask_state_np
            else:
                s = s >> 1

    df_data = {
        "bit_index": np.arange(n_symbols, dtype=np.int64),
        "detected_bit": detected_bits,
        "n_rx": counts,
        "state_path": state_path,
    }

    if bits_arr is not None:
        df_data["true_bit"] = bits_arr
        df_data["correct"] = detected_bits == bits_arr

    df = pd.DataFrame(df_data)

    metrics = None
    if bits_arr is not None:
        metrics = binary_detection_metrics(bits_arr, detected_bits)

    result = {
        "detected_bits": detected_bits,
        "final_log_metric": best_final_metric,
        "state_path": state_path,
        "K": int(K),
        "B": int(B),
        "L": int(leglib.L),
        "state_len": int(state_len),
        "n_states": int(n_states),
        "n_branches": int(n_branches),
        "lambda_tap": lambda_tap_np,
        "lambda_env_count": float(lambda_env_count),
        "full_ml": True,
        "torch_parallel": True,
        "torch_device": str(dev),
        "torch_dtype": "float32",
        "branch_chunk": None if branch_chunk is None else int(branch_chunk),
        "target_temp_bytes": int(target_temp_bytes),
        "preload_observations_to_device": bool(preload_observations_to_device),
        "backpointer_storage": backpointer_storage,
        "precompute_branch_bits": bool(precompute_branch_bits),
        "strict_intensity_check": bool(strict_intensity_check),
    }

    return df, metrics, result


# Backward-compatible explicit name.
full_ml_legendre_viterbi_detect_torch = full_ml_legendre_viterbi_detect


# ---------------------------------------------------------------------
# Minimal usage sketch
# ---------------------------------------------------------------------
#
# leglib = build_library_from_arrays(coeffs, tap_bin_prob)
#
# df, metrics, result = full_ml_legendre_viterbi_detect(
#     received_symbol_hits=received_symbol_hits,
#     bits=bits,
#     leglib=leglib,
#     n_emit_bit=N_tx,
#     Ts=Ts,
#     p_one=0.5,
#     axis=(0.0, 0.0, 1.0),
#     memory_taps=K,
#     lambda_env_count=0.0,
#     device="cuda",
#     require_cuda=True,
#     branch_chunk=None,               # automatic GPU-memory-based chunking
#     target_temp_bytes=4*1024*1024*1024,  # 4 GiB temporary workspace target
#     preload_observations_to_device=True,
#     backpointer_storage="device",    # use "cpu" if GPU memory is tight
#     precompute_branch_bits=True,      # faster; exact metric preserved
#     strict_intensity_check=True,      # safest; set False only after validation
# )
#
# print(metrics)