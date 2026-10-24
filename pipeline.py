"""Shared pipeline pieces: detector runner, channel-memory helpers, figure styles.

`run_one_detector_scenario` runs all five detectors on one set of received
symbols. The paper figures (`save_figure_*`) are drawn by make_figures.py from
the summary CSVs written by experiments.py.

Detectors:
  fixed_threshold          count-only threshold          (fixed_threshold.py)
  viterbi                  count-only Poisson Viterbi    (viterbi.py)
  legendre                 memoryless Legendre LRT       (legendre.py)
  legendre_viterbi         hybrid Legendre-Viterbi       (legendre_viterbi.py)
  legendre_viterbi_full    full Legendre-Viterbi         (legendre_viterbi_full.py)
"""

from __future__ import annotations

from pathlib import Path
import math
import os
import time

import numpy as np
import pandas as pd

from fixed_threshold import fixed_threshold_detect
from legendre import expected_isi_total_rate, legendre_detect
from legendre_viterbi import legendre_viterbi_detect
from legendre_viterbi_full import FullLegendreLibrary, full_ml_legendre_viterbi_detect
from main import DEFAULT_AXIS
from viterbi import viterbi_detect


PROJECT_ROOT_ENV = "LEGENDRE_MSD_ROOT"
PROJECT_ROOT = Path(os.environ.get(PROJECT_ROOT_ENV, Path(__file__).resolve().parent)).expanduser().resolve()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def memory_taps_from_ts(memory: float, ts: float) -> int:
    memory_taps = int(round(float(memory) / float(ts)))
    if not np.isclose(memory_taps * float(ts), float(memory)):
        raise ValueError(f"memory={memory} must be an integer multiple of ts={ts}.")
    return memory_taps


def get_first_existing_attr(obj, names: tuple[str, ...], default: float) -> float:
    """
    Read a channel parameter from main.parse_args() even if the project uses
    slightly different attribute names.
    """
    for name in names:
        if hasattr(obj, name):
            return float(getattr(obj, name))
    return float(default)


def compute_memory_for_ts_from_erfc(
    *,
    ts: float,
    distance: float,
    radius: float,
    diffusion_coef: float,
    target_probability: float = 0.7,
    max_taps: int = 100_000,
) -> tuple[float, int, float]:
    """
    Return the smallest positive memory = k * TS satisfying:

        erfc((distance - radius) / sqrt(4 * D * memory)) > target_probability

    Returns:
        memory, memory_taps, achieved_probability

    The strict '>' condition is used exactly as requested.
    """
    ts = float(ts)
    distance = float(distance)
    radius = float(radius)
    diffusion_coef = float(diffusion_coef)
    target_probability = float(target_probability)

    if ts <= 0.0:
        raise ValueError("ts must be positive.")
    if diffusion_coef <= 0.0:
        raise ValueError("diffusion_coef must be positive.")
    if not (0.0 < target_probability < 1.0):
        raise ValueError("target_probability must be between 0 and 1.")

    separation = max(distance - radius, 0.0)

    if separation <= 0.0:
        return ts, 1, 1.0

    for memory_taps in range(1, int(max_taps) + 1):
        memory = memory_taps * ts
        probability = math.erfc(
            separation / math.sqrt(4.0 * diffusion_coef * memory)
        )
        if probability > target_probability:
            return float(memory), int(memory_taps), float(probability)

    raise RuntimeError(
        "Could not find a memory multiple satisfying the erfc condition. "
        f"Increase max_taps. Last tested memory={max_taps * ts}."
    )


def compute_memory_for_args_ts(
    args,
    ts: float,
    target_probability: float,
) -> tuple[float, int, float]:
    """
    Compute memory from main.parse_args() channel parameters.

    Defaults match the notebook parameters if args does not expose the
    corresponding fields:
        distance = 12.5
        radius = 5.0
        diffusion_coef = 79.4
    """
    distance = get_first_existing_attr(
        args,
        ("distance", "p_distance", "P_DISTANCE", "tx_distance"),
        12.5,
    )
    radius = get_first_existing_attr(
        args,
        ("radius", "p_radius", "P_RADIUS", "receiver_radius"),
        5.0,
    )
    diffusion_coef = get_first_existing_attr(
        args,
        ("diffusion_coef", "diffusion_coefficient", "p_diffusion_coef", "P_DIFFUSION_COEF", "D"),
        79.4,
    )

    return compute_memory_for_ts_from_erfc(
        ts=ts,
        distance=distance,
        radius=radius,
        diffusion_coef=diffusion_coef,
        target_probability=target_probability,
    )


def poisson_equal_prior_count_threshold(lambda_h0: float, lambda_h1: float) -> float:
    """Threshold tau for deciding H1 when count-only Poisson LLR exceeds zero."""
    lambda_h0 = max(float(lambda_h0), 1e-15)
    lambda_h1 = max(float(lambda_h1), 1e-15)
    if np.isclose(lambda_h0, lambda_h1):
        return math.inf
    return float((lambda_h1 - lambda_h0) / math.log(lambda_h1 / lambda_h0))


def operating_points(df: pd.DataFrame) -> pd.DataFrame:
    """Return only default detector operating points, for plots that are not ROC-like."""
    if "point_kind" not in df.columns:
        return df.copy()
    return df[df["point_kind"].fillna("operating_point") == "operating_point"].copy()


def decision_boundary_points(df: pd.DataFrame) -> pd.DataFrame:
    """Return threshold-sweep points used for Figure 2 decision boundaries."""
    if "point_kind" not in df.columns:
        return df.iloc[0:0]
    return df[df["point_kind"] == "decision_boundary"].copy()


def threshold_candidates_from_scores(
    scores: np.ndarray,
    *,
    include_threshold: float | None = None,
    max_points: int = 400,
) -> np.ndarray:
    """
    Candidate thresholds for detectors with scalar scores and decision rule
    score > threshold -> 1.
    """
    scores = np.asarray(scores, dtype=float)
    finite_scores = np.unique(scores[np.isfinite(scores)])

    if finite_scores.size == 0:
        candidates = np.asarray([], dtype=float)
    elif finite_scores.size == 1:
        value = float(finite_scores[0])
        candidates = np.asarray([
            np.nextafter(value, -np.inf),
            value,
            np.nextafter(value, np.inf),
        ])
    else:
        mids = 0.5 * (finite_scores[:-1] + finite_scores[1:])
        candidates = np.concatenate([
            [np.nextafter(float(finite_scores[0]), -np.inf)],
            mids,
            [np.nextafter(float(finite_scores[-1]), np.inf)],
        ])

    if candidates.size > max_points:
        # Evenly spaced in rank: keeps the curve shape, drops redundant points.
        keep = np.unique(np.linspace(0, candidates.size - 1, max_points).round().astype(int))
        candidates = candidates[keep]

    if include_threshold is not None and np.isfinite(include_threshold):
        candidates = np.concatenate([candidates, [float(include_threshold)]])

    return np.unique(candidates)


def threshold_sweep_rows(
    *,
    detector: str,
    bits: np.ndarray,
    scores: np.ndarray,
    threshold_variable: str,
    default_threshold: float | None = None,
    lambda_h0: float = np.nan,
    lambda_h1: float = np.nan,
) -> list[dict]:
    """Build P_FA/P_miss operating-curve rows for a thresholded detector."""
    bits = np.asarray(bits, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)

    rows = []
    for threshold in threshold_candidates_from_scores(
        scores,
        include_threshold=default_threshold,
    ):
        detected = (scores > threshold).astype(np.int8)
        errors = int(np.sum(detected != bits))
        rows.append({
            "detector": detector,
            "BER": float(errors / max(len(bits), 1)),
            "errors": errors,
            "P_miss": float(np.mean(detected[bits == 1] == 0)) if np.any(bits == 1) else np.nan,
            "P_FA": float(np.mean(detected[bits == 0] == 1)) if np.any(bits == 0) else np.nan,
            "runtime_seconds": np.nan,
            "runtime_seconds_per_1000_bits": np.nan,
            "viterbi_memory_taps": np.nan,
            "viterbi_state_count": np.nan,
            "fixed_threshold": float(threshold) if detector == "fixed_threshold" else np.nan,
            "lambda_h0": lambda_h0,
            "lambda_h1": lambda_h1,
            "point_kind": "decision_boundary",
            "threshold_variable": threshold_variable,
            "threshold_value": float(threshold),
        })

    return rows


def canonical_detector_metrics(metrics: dict) -> dict:
    """Normalize metric keys across detector implementations."""
    if metrics is None:
        return {
            "BER": np.nan,
            "errors": np.nan,
            "P_miss": np.nan,
            "P_FA": np.nan,
        }
    return {
        "BER": metrics["BER"],
        "errors": metrics.get("errors", metrics.get("n_errors", np.nan)),
        "P_miss": metrics.get("P_miss", metrics.get("P_0_given_1", np.nan)),
        "P_FA": metrics.get("P_FA", metrics.get("P_1_given_0", np.nan)),
    }


def make_full_ml_legendre_library(leglib: dict, memory_taps: int | None = None) -> FullLegendreLibrary:
    """
    Convert the existing project Legendre library into the full-ML format.

    Existing libraries store P(hit in tap) and P(time-bin | hit in tap)
    separately. The strong detector expects their product:
    P(hit in tap and relative-time bin).
    """
    coeffs = np.asarray(leglib["coeffs"], dtype=np.float64)
    tap_bin_prob = (
        np.asarray(leglib["tap_prob"], dtype=np.float64)[:, None]
        * np.asarray(leglib["rel_mass"], dtype=np.float64)
    )
    if memory_taps is not None:
        coeffs = coeffs[: int(memory_taps)]
        tap_bin_prob = tap_bin_prob[: int(memory_taps)]
    return FullLegendreLibrary(
        coeffs=coeffs,
        tap_bin_prob=tap_bin_prob,
        eps_floor=float(leglib.get("eps_floor", 1e-12)),
    )


def detector_label(detector: str) -> str:
    labels = {
        "legendre": "Legendre",
        "viterbi": "Count Viterbi",
        "legendre_viterbi": "Hybrid Viterbi",
        "legendre_viterbi_full": "Legendre Viterbi",
        "fixed_threshold": "Fixed Threshold",
    }
    return labels.get(detector, detector)

def detector_style(detector: str) -> dict:
    """Muted, publication-style line/marker colors for each detector."""
    styles = {
        "legendre": {
            "color": "#1F4E79",  # deep blue
            "marker": "o",
            "linestyle": ":",
        },
        "viterbi": {
            "color": "#595959",  # charcoal gray
            "marker": "s",
            "linestyle": "--",
        },
        "legendre_viterbi": {
            "color": "#2B6E46",  # muted green
            "marker": "D",
            "linestyle": ":",
        },
        "legendre_viterbi_full": {
            "color": "#6E4B8B",  # muted violet
            "marker": "P",
            "linestyle": "-.",
        },
        "fixed_threshold": {
            "color": "#7F3B2E",  # muted brick
            "marker": "^",
            "linestyle": "--",
        },
    }
    return styles.get(detector, {"color": "#333333", "marker": "o", "linestyle": "--"})


def apply_publication_style(ax, *, grid_axis: str = "both") -> None:
    """Make axes, ticks, labels, and numeric tick labels publication-ready."""
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=15,
        width=1.7,
        length=7,
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=1.3,
        length=4,
        direction="in",
        top=True,
        right=True,
    )

    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(17)
    ax.title.set_size(17)
    ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=0.8)


FIGURE_2_N_TX = 100

DETECTOR_ORDER = [
    "fixed_threshold",
    "viterbi",
    "legendre",
    "legendre_viterbi",
    "legendre_viterbi_full",
]

RUNTIME_DETECTOR_ORDER = [
    "fixed_threshold",
    "legendre",
    "legendre_viterbi",
    "legendre_viterbi_full",
    "viterbi",
]


def maybe_make_out_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def float_tag(x: float) -> str:
    return f"{float(x):.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def configure_args(args, ts: float, memory: float, n_bits: int):
    args.ts = float(ts)
    args.memory = float(memory)
    args.n_bits = int(n_bits)
    return args


def choose_viterbi_memory_taps(
    physical_memory_taps: int,
    max_viterbi_memory_taps: int | None,
) -> int:
    if max_viterbi_memory_taps is None:
        return int(physical_memory_taps)
    return int(min(physical_memory_taps, max_viterbi_memory_taps))


ALL_DETECTORS = (
    "legendre",
    "viterbi",
    "legendre_viterbi",
    "legendre_viterbi_full",
    "fixed_threshold",
)


def _operating_point_row(detector: str, metrics: dict, runtime: float, n_bits: int, **extra) -> dict:
    row = {
        "detector": detector,
        "BER": metrics["BER"],
        "errors": metrics["errors"],
        "P_miss": metrics["P_miss"],
        "P_FA": metrics["P_FA"],
        "runtime_seconds": runtime,
        "runtime_seconds_per_1000_bits": runtime / max(n_bits, 1) * 1000.0,
        "viterbi_memory_taps": np.nan,
        "viterbi_state_count": np.nan,
        "fixed_threshold": np.nan,
        "lambda_h0": np.nan,
        "lambda_h1": np.nan,
        "point_kind": "operating_point",
        "threshold_variable": np.nan,
        "threshold_value": np.nan,
    }
    row.update(extra)
    return row


def run_one_detector_scenario(
    *,
    args,
    bits: np.ndarray,
    received_symbol_hits: list[dict],
    leglib: dict,
    n_tx: int,
    physical_memory_taps: int,
    max_viterbi_memory_taps: int | None,
    max_strong_viterbi_memory_taps: int | None,
    save_symbol_csvs: bool,
    add_threshold_curves: bool,
    out_dir: Path,
    scenario_tag: str,
) -> list[dict]:
    """Run all detector operating points for one scenario.

    Optional attributes on args (set by experiments.py):
        detectors     : iterable of detector keys to run (default: ALL_DETECTORS)
        detector_axis : Tx direction assumed by the Legendre detectors (default: true axis)
        full_device   : torch device for the full Legendre-Viterbi detector (default: "auto")
    """
    rows = []
    n_bits = len(bits)
    detectors = set(getattr(args, "detectors", None) or ALL_DETECTORS)
    unknown = detectors - set(ALL_DETECTORS)
    if unknown:
        raise ValueError(f"Unknown detectors: {sorted(unknown)}")
    axis = np.asarray(getattr(args, "detector_axis", DEFAULT_AXIS), dtype=np.float64)
    full_device = str(getattr(args, "full_device", "auto"))
    if full_device == "auto":
        try:
            import torch
            full_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ModuleNotFoundError:
            full_device = "cpu"

    viterbi_memory_taps = choose_viterbi_memory_taps(
        physical_memory_taps=physical_memory_taps,
        max_viterbi_memory_taps=max_viterbi_memory_taps,
    )
    viterbi_state_count = 1 if viterbi_memory_taps <= 1 else 1 << (viterbi_memory_taps - 1)

    def save_symbols(df: pd.DataFrame, name: str) -> None:
        if save_symbol_csvs:
            df.to_csv(out_dir / f"{scenario_tag}_{name}_detection_NTx{n_tx}.csv", index=False)

    legendre_df = None
    if "legendre" in detectors:
        legendre_llr_threshold = math.log((1.0 - float(args.p_one)) / max(float(args.p_one), 1e-15))
        t0 = time.perf_counter()
        legendre_df, legendre_metrics = legendre_detect(
            received_symbol_hits=received_symbol_hits,
            bits=bits,
            leglib=leglib,
            n_emit_bit=n_tx,
            axis=axis,
            p_one=args.p_one,
            lambda_env=args.legendre_lambda_env,
            llr_threshold=legendre_llr_threshold,
        )
        rows.append(_operating_point_row(
            "legendre", legendre_metrics, time.perf_counter() - t0, n_bits,
            threshold_variable="expected_isi_llr",
            threshold_value=legendre_llr_threshold,
        ))
        save_symbols(legendre_df, "legendre")

    if "viterbi" in detectors:
        t0 = time.perf_counter()
        viterbi_df, viterbi_metrics, _ = viterbi_detect(
            received_symbol_hits=received_symbol_hits,
            bits=bits,
            tap_prob=leglib["tap_prob"],
            n_emit_bit=n_tx,
            p_one=args.p_one,
            lambda_env=args.viterbi_lambda_env,
            memory_taps=viterbi_memory_taps,
            max_states=viterbi_state_count,
        )
        rows.append(_operating_point_row(
            "viterbi", viterbi_metrics, time.perf_counter() - t0, n_bits,
            viterbi_memory_taps=viterbi_memory_taps,
            viterbi_state_count=viterbi_state_count,
        ))
        save_symbols(viterbi_df, "viterbi")

    if "legendre_viterbi" in detectors:
        t0 = time.perf_counter()
        hybrid_df, hybrid_metrics, hybrid_result = legendre_viterbi_detect(
            received_symbol_hits=received_symbol_hits,
            bits=bits,
            leglib=leglib,
            n_emit_bit=n_tx,
            tap_prob=leglib["tap_prob"],
            p_one=args.p_one,
            axis=axis,
            lambda_env_count=args.viterbi_lambda_env,
            lambda_env_shape=args.legendre_lambda_env,
            memory_taps=viterbi_memory_taps,
            expected_prev_p1=args.p_one,
            max_states=viterbi_state_count,
        )
        rows.append(_operating_point_row(
            "legendre_viterbi", hybrid_metrics, time.perf_counter() - t0, n_bits,
            viterbi_memory_taps=viterbi_memory_taps,
            viterbi_state_count=hybrid_result["n_states"],
        ))
        save_symbols(hybrid_df, "legendre_viterbi")

    if "legendre_viterbi_full" in detectors:
        strong_viterbi_memory_taps = choose_viterbi_memory_taps(
            physical_memory_taps=physical_memory_taps,
            max_viterbi_memory_taps=max_strong_viterbi_memory_taps,
        )
        strong_viterbi_state_count = (
            1 if strong_viterbi_memory_taps <= 1 else 1 << (strong_viterbi_memory_taps - 1)
        )
        strong_leglib = make_full_ml_legendre_library(
            leglib,
            memory_taps=strong_viterbi_memory_taps,
        )
        t0 = time.perf_counter()
        strong_df, strong_metrics_raw, strong_result = full_ml_legendre_viterbi_detect(
            received_symbol_hits=received_symbol_hits,
            bits=bits,
            leglib=strong_leglib,
            n_emit_bit=n_tx,
            Ts=args.ts,
            p_one=args.p_one,
            axis=axis,
            memory_taps=strong_viterbi_memory_taps,
            lambda_env_count=args.viterbi_lambda_env,
            max_states=strong_viterbi_state_count,
            device=full_device,
            require_cuda=full_device.startswith("cuda"),
            target_temp_bytes=4 * 1024 * 1024 * 1024,
            branch_chunk=None,
            preload_observations_to_device=True,
            backpointer_storage="device",
            precompute_branch_bits=True,
            strict_intensity_check=True,
        )
        rows.append(_operating_point_row(
            "legendre_viterbi_full",
            canonical_detector_metrics(strong_metrics_raw),
            time.perf_counter() - t0,
            n_bits,
            viterbi_memory_taps=strong_viterbi_memory_taps,
            viterbi_state_count=strong_result["n_states"],
        ))
        save_symbols(strong_df, "legendre_viterbi_full")

    fixed_df = None
    lambda_h0 = expected_isi_total_rate(
        leglib=leglib,
        n_emit_bit=n_tx,
        current_bit_hypothesis=0,
        expected_prev_p1=args.p_one,
        lambda_env=args.legendre_lambda_env,
    )
    lambda_h1 = expected_isi_total_rate(
        leglib=leglib,
        n_emit_bit=n_tx,
        current_bit_hypothesis=1,
        expected_prev_p1=args.p_one,
        lambda_env=args.legendre_lambda_env,
    )
    fixed_threshold = poisson_equal_prior_count_threshold(lambda_h0, lambda_h1)
    if "fixed_threshold" in detectors:
        t0 = time.perf_counter()
        fixed_df, fixed_metrics, _ = fixed_threshold_detect(
            received_symbol_hits=received_symbol_hits,
            bits=bits,
            threshold=fixed_threshold,
        )
        rows.append(_operating_point_row(
            "fixed_threshold", fixed_metrics, time.perf_counter() - t0, n_bits,
            fixed_threshold=fixed_threshold,
            lambda_h0=lambda_h0,
            lambda_h1=lambda_h1,
            threshold_variable="n_rx",
            threshold_value=fixed_threshold,
        ))
        save_symbols(fixed_df, "fixed_threshold")

    if add_threshold_curves:
        if legendre_df is not None:
            rows.extend(threshold_sweep_rows(
                detector="legendre",
                bits=bits,
                scores=legendre_df["expected_isi_llr"].to_numpy(dtype=float),
                threshold_variable="expected_isi_llr",
                default_threshold=legendre_llr_threshold,
            ))
        if fixed_df is not None:
            rows.extend(threshold_sweep_rows(
                detector="fixed_threshold",
                bits=bits,
                scores=fixed_df["n_rx"].to_numpy(dtype=float),
                threshold_variable="n_rx",
                default_threshold=fixed_threshold,
                lambda_h0=lambda_h0,
                lambda_h1=lambda_h1,
            ))

    return rows


def save_figure_1_ber_vs_ntx(comparison: pd.DataFrame, out_file: str | Path) -> None:
    import matplotlib.pyplot as plt

    comparison = operating_points(comparison)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for detector in DETECTOR_ORDER:
        df = comparison[comparison["detector"] == detector].sort_values("N_Tx")
        style = detector_style(detector)
        ax.plot(
            df["N_Tx"],
            df["BER"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=2.8,
            markersize=8,
            markeredgewidth=1.2,
            label=detector_label(detector),
        )

    ax.set_xlabel(r"$N_{Tx}$")
    ax.set_ylabel("BER")
    ax.set_yscale("log")
    ax.set_ylim(top=1.0)
    apply_publication_style(ax)
    ax.xaxis.label.set_size(20)
    ax.legend(
        loc="lower left",
        frameon=True,
        fontsize=12,
        handlelength=2.2,
        borderpad=0.5,
        labelspacing=0.4,
    )

    fig.tight_layout()
    fig.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_figure_2_false_alarm_vs_missed_detection(
    comparison: pd.DataFrame,
    out_file: str | Path,
    n_tx: int = FIGURE_2_N_TX,
) -> None:
    """
    Plot false-alarm probability versus missed-detection probability
    at one fixed N_Tx.

    At a fixed N_Tx, Viterbi detectors contribute one operating point, while
    thresholded detectors can contribute a decision-boundary curve.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    df_ntx = comparison[comparison["N_Tx"] == int(n_tx)].copy()
    df_points = operating_points(df_ntx)
    df_boundaries = decision_boundary_points(df_ntx)

    if df_ntx.empty:
        available = sorted(comparison["N_Tx"].dropna().unique().astype(int).tolist())
        raise ValueError(
            f"No rows found for N_Tx={n_tx}. "
            f"Available N_Tx values are: {available}"
        )

    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    for detector in DETECTOR_ORDER:
        df_det = df_points[df_points["detector"] == detector]

        if df_det.empty:
            continue

        row = df_det.iloc[0]
        style = detector_style(detector)
        df_curve = df_boundaries[df_boundaries["detector"] == detector]

        if not df_curve.empty:
            df_curve = (
                df_curve
                .dropna(subset=["P_FA", "P_miss"])
                .drop_duplicates(subset=["P_FA", "P_miss"])
                .sort_values(["P_FA", "P_miss"])
            )
            ax.plot(
                df_curve["P_FA"],
                df_curve["P_miss"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.8,
                alpha=0.95,
                label=detector_label(detector),
            )

        if df_curve.empty or detector not in {"legendre", "fixed_threshold"}:
            ax.scatter(
                row["P_FA"],
                row["P_miss"],
                s=120,
                color=style["color"],
                marker=style["marker"],
                linewidths=1.2,
                label="_nolegend_" if not df_curve.empty else detector_label(detector),
            )

    ax.set_xlabel(r"$P_{1|0} = P(\hat{\mathit{x}}_n = 1 \mid \mathit{x}_n = 0)$")
    ax.set_ylabel(r"$P_{0|1} = P(\hat{\mathit{x}}_n = 0 \mid \mathit{x}_n = 1)$")
    apply_publication_style(ax)

    # Keep the axes interpretable even when all points are near zero.
    max_axis = max(
        float(df_ntx["P_FA"].max(skipna=True)),
        float(df_ntx["P_miss"].max(skipna=True)),
        1e-3,
    )
    ax.set_xlim(left=0.0, right=min(1.0, max_axis * 1.25 + 1e-3))
    ax.set_ylim(bottom=0.0, top=min(1.0, max_axis * 1.25 + 1e-3))
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, position: "" if np.isclose(value, 0.0) else f"{value:.1f}")
    )

    ax.legend(frameon=True, fontsize=14, handlelength=2.2, borderpad=0.7)

    fig.tight_layout()
    fig.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_figure_3_ber_vs_ts(ts_sweep_df: pd.DataFrame, out_file: str | Path) -> None:
    import matplotlib.pyplot as plt

    ts_sweep_df = operating_points(ts_sweep_df)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    n_tx_values = sorted(ts_sweep_df["N_Tx"].dropna().unique())

    for detector in DETECTOR_ORDER:
        style = detector_style(detector)
        legend_label = detector_label(detector)
        for n_tx in n_tx_values:
            df = ts_sweep_df[
                (ts_sweep_df["detector"] == detector)
                & (ts_sweep_df["N_Tx"] == n_tx)
            ].sort_values("ts")

            if df.empty:
                continue

            ax.plot(
                df["ts"],
                df["BER"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2.8,
                markersize=8,
                markeredgewidth=1.2,
                label=legend_label,
            )
            legend_label = "_nolegend_"

    ax.set_xlabel(r"Symbol duration $T_s$ [s]")
    ax.set_ylabel("BER")
    ax.set_yscale("log")
    ax.set_ylim(top=1.0)
    apply_publication_style(ax)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=17,
        width=2.0,
        length=8.5,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=1.5,
        length=5.0,
    )
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.legend(
        loc="lower left",
        ncol=1,
        frameon=True,
        fontsize=15,
        handlelength=2.2,
        borderpad=0.5,
        labelspacing=0.4,
    )

    fig.tight_layout()
    fig.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_figure_4_runtime(comparison: pd.DataFrame, out_file: str | Path) -> None:
    import matplotlib.pyplot as plt

    comparison = operating_points(comparison)
    runtime = (
        comparison
        .groupby("detector", as_index=False)
        .agg(
            mean_runtime_seconds_per_1000_bits=("runtime_seconds_per_1000_bits", "mean"),
            std_runtime_seconds_per_1000_bits=("runtime_seconds_per_1000_bits", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
    )
    order = RUNTIME_DETECTOR_ORDER
    runtime["detector"] = pd.Categorical(runtime["detector"], categories=order, ordered=True)
    runtime = runtime.sort_values("detector")
    colors = [detector_style(str(detector))["color"] for detector in runtime["detector"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(
        [detector_label(str(d)) for d in runtime["detector"]],
        runtime["mean_runtime_seconds_per_1000_bits"],
        yerr=runtime["std_runtime_seconds_per_1000_bits"].fillna(0.0),
        capsize=5,
        color=colors,
        edgecolor="#2A2A2A",
        linewidth=1.2,
    )

    ax.set_ylabel("Runtime [s] per 1000 bits")
    ax.set_title("Detector runtime comparison")
    apply_publication_style(ax, grid_axis="y")

    fig.tight_layout()
    fig.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_plot_csvs(baseline_df: pd.DataFrame, ts_sweep_df: pd.DataFrame, plots_dir: str | Path) -> dict[str, Path]:
    """Save the exact tabular data used by each figure next to the plots."""
    plots_dir = maybe_make_out_dir(plots_dir)

    # Defensive copies because this function may add missing metadata columns
    # when reading older summary CSV files.
    baseline_df = baseline_df.copy()
    ts_sweep_df = ts_sweep_df.copy()
    baseline_operating_df = operating_points(baseline_df)
    ts_sweep_operating_df = operating_points(ts_sweep_df)

    figure_1_cols = ["N_Tx", "detector", "BER", "P_miss", "P_FA", "detector_memory_taps", "channel_memory_taps"]
    figure_2_cols = [
        "N_Tx",
        "detector",
        "point_kind",
        "threshold_variable",
        "threshold_value",
        "P_FA",
        "P_miss",
    ]
    figure_3_cols = [
        "ts",
        "N_Tx",
        "detector",
        "BER",
        "P_miss",
        "P_FA",
        "detector_memory_taps",
        "channel_memory_taps",
        "lambda_tail_modelled",
    ]

    for col in figure_1_cols:
        if col not in baseline_operating_df.columns:
            baseline_operating_df[col] = np.nan

    for col in figure_2_cols:
        if col not in baseline_df.columns:
            baseline_df[col] = np.nan

    for col in figure_3_cols:
        if col not in ts_sweep_operating_df.columns:
            ts_sweep_operating_df[col] = np.nan

    figure_1_df = (
        baseline_operating_df[figure_1_cols]
        .sort_values(["N_Tx", "detector"])
    )

    figure_2_df = (
        baseline_df[baseline_df["N_Tx"] == FIGURE_2_N_TX]
        [figure_2_cols]
        .sort_values(["detector", "point_kind", "threshold_value"])
    )

    figure_3_df = (
        ts_sweep_operating_df[figure_3_cols]
        .sort_values(["ts", "N_Tx", "detector"])
    )

    figure_4_df = (
        baseline_operating_df
        .groupby("detector", as_index=False)
        .agg(
            mean_runtime_seconds_per_1000_bits=("runtime_seconds_per_1000_bits", "mean"),
            std_runtime_seconds_per_1000_bits=("runtime_seconds_per_1000_bits", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values("detector")
    )

    out_files = {
        "figure_1": plots_dir / "figure_1_ber_vs_NTx.csv",
        "figure_2": plots_dir / "figure_2_pfa_vs_pmiss_NTx100.csv",
        "figure_3": plots_dir / "figure_3_ber_vs_TS.csv",
        "figure_4": plots_dir / "figure_4_runtime_comparison.csv",
    }

    figure_1_df.to_csv(out_files["figure_1"], index=False)
    figure_2_df.to_csv(out_files["figure_2"], index=False)
    figure_3_df.to_csv(out_files["figure_3"], index=False)
    figure_4_df.to_csv(out_files["figure_4"], index=False)

    return out_files
