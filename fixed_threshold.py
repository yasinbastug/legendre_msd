"""Count-only fixed-threshold detection.

This module replaces count-only Poisson Viterbi sequence detection with a
symbol-by-symbol fixed-threshold detector.

Core decision rule:
    detected_bit[n] = 1 if count[n] > threshold else 0

The module supports either:
    1. counts directly, or
    2. received_symbol_hits, where count[n] = len(received_symbol_hits[n]["t_rel"])

No Viterbi state recursion is used.
"""

import math
import numpy as np
import pandas as pd


def binary_detection_metrics(y_true, y_hat):
    """
    Compute BER, miss probability, and false-alarm probability.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_hat = np.asarray(y_hat, dtype=np.int8)

    if len(y_true) != len(y_hat):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)}, len(y_hat)={len(y_hat)}."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics for an empty sequence.")

    errors = int(np.sum(y_hat != y_true))
    ber = float(errors / len(y_true))

    p_miss = (
        float(np.mean(y_hat[y_true == 1] == 0))
        if np.any(y_true == 1)
        else np.nan
    )

    p_fa = (
        float(np.mean(y_hat[y_true == 0] == 1))
        if np.any(y_true == 0)
        else np.nan
    )

    return {
        "BER": ber,
        "errors": errors,
        "P_miss": p_miss,
        "P_FA": p_fa,
        "n_bits": int(len(y_true)),
        "n_ones_true": int(np.sum(y_true == 1)),
        "n_zeros_true": int(np.sum(y_true == 0)),
        "n_ones_detected": int(np.sum(y_hat == 1)),
        "n_zeros_detected": int(np.sum(y_hat == 0)),
    }


def counts_from_received_symbol_hits(received_symbol_hits):
    """
    Convert received_symbol_hits into a count vector.

    Expected input format:
        received_symbol_hits[n]["t_rel"] contains the arrivals in symbol n.
    """
    return np.asarray(
        [len(obs["t_rel"]) for obs in received_symbol_hits],
        dtype=np.int64,
    )


def run_count_fixed_threshold(
    counts,
    threshold,
    tie_break_bit=0,
):
    """
    Run symbol-by-symbol fixed-threshold detection.

    Decision rule:
        count > threshold  -> 1
        count < threshold  -> 0
        count == threshold -> tie_break_bit

    Parameters
    ----------
    counts : array-like of int
        Received molecule counts per symbol.
    threshold : float
        Fixed count threshold.
    tie_break_bit : {0, 1}, default 0
        Decision when count equals threshold exactly.
        For a non-integer threshold, this has no practical effect.

    Returns
    -------
    result : dict
        Contains detected_bits, threshold, counts, and detector metadata.
    """
    counts = np.asarray(counts, dtype=np.int64)

    if counts.ndim != 1:
        raise ValueError("counts must be a one-dimensional array.")

    if len(counts) == 0:
        raise ValueError("counts must not be empty.")

    threshold = float(threshold)
    tie_break_bit = int(tie_break_bit)

    if tie_break_bit not in {0, 1}:
        raise ValueError("tie_break_bit must be 0 or 1.")

    detected = np.zeros(len(counts), dtype=np.int8)
    detected[counts > threshold] = 1

    if tie_break_bit == 1:
        detected[counts == threshold] = 1

    return {
        "detected_bits": detected,
        "threshold": threshold,
        "tie_break_bit": tie_break_bit,
        "counts": counts,
        "detector": "count_fixed_threshold",
    }


def fixed_threshold_detect_from_counts(
    counts,
    bits,
    threshold,
    tie_break_bit=0,
):
    """
    Run fixed-threshold detection using a precomputed count vector.

    Parameters
    ----------
    counts : array-like of int
        Received molecule counts per symbol.
    bits : array-like of int
        True transmitted bits for performance evaluation.
    threshold : float
        Fixed count threshold.
    tie_break_bit : {0, 1}, default 0
        Decision when count equals threshold exactly.

    Returns
    -------
    df : pandas.DataFrame
        Per-symbol detection table.
    metrics : dict
        BER, miss probability, false-alarm probability, and counts.
    result : dict
        Raw detector output.
    """
    bits = np.asarray(bits, dtype=np.int8)
    counts = np.asarray(counts, dtype=np.int64)

    if len(bits) != len(counts):
        raise ValueError(
            f"Length mismatch: len(bits)={len(bits)}, len(counts)={len(counts)}."
        )

    result = run_count_fixed_threshold(
        counts=counts,
        threshold=threshold,
        tie_break_bit=tie_break_bit,
    )

    detected = result["detected_bits"].astype(np.int8)
    metrics = binary_detection_metrics(bits, detected)

    df = pd.DataFrame({
        "bit_index": np.arange(len(bits), dtype=np.int64),
        "true_bit": bits,
        "detected_bit": detected,
        "correct": detected == bits,
        "n_rx": counts,
        "threshold": float(threshold),
        "tie_break_bit": int(tie_break_bit),
    })

    return df, metrics, result


def fixed_threshold_detect(
    received_symbol_hits,
    bits,
    threshold,
    tie_break_bit=0,
):
    """
    Run fixed-threshold detection using received_symbol_hits.

    This wrapper matches the style of the older Viterbi wrapper, but it does
    not use tap_prob, n_emit_bit, p_one, lambda_env, or any Viterbi states.
    """
    counts = counts_from_received_symbol_hits(received_symbol_hits)

    return fixed_threshold_detect_from_counts(
        counts=counts,
        bits=bits,
        threshold=threshold,
        tie_break_bit=tie_break_bit,
    )


def find_ber_optimal_threshold_from_counts(
    counts,
    bits,
    tie_break_bit=0,
):
    """
    Find the count threshold that minimizes BER on the provided labeled data.

    This is useful for calibration/debugging. If the same data are used for
    both calibration and testing, the resulting BER is optimistic.

    Decision rule tested:
        detected = 1 if count > threshold else 0

    Candidate thresholds are placed below the minimum count, between unique
    observed counts, and above the maximum count.
    """
    counts = np.asarray(counts, dtype=np.int64)
    bits = np.asarray(bits, dtype=np.int8)

    if len(counts) != len(bits):
        raise ValueError(
            f"Length mismatch: len(counts)={len(counts)}, len(bits)={len(bits)}."
        )

    if len(counts) == 0:
        raise ValueError("counts must not be empty.")

    unique_counts = np.sort(np.unique(counts.astype(np.float64)))

    thresholds = [unique_counts[0] - 1.0]

    for a, b in zip(unique_counts[:-1], unique_counts[1:]):
        thresholds.append(0.5 * (a + b))

    thresholds.append(unique_counts[-1] + 1.0)

    rows = []

    for tau in thresholds:
        df_tau, metrics_tau, _ = fixed_threshold_detect_from_counts(
            counts=counts,
            bits=bits,
            threshold=tau,
            tie_break_bit=tie_break_bit,
        )

        rows.append({
            "threshold": float(tau),
            "BER": float(metrics_tau["BER"]),
            "errors": int(metrics_tau["errors"]),
            "P_miss": float(metrics_tau["P_miss"]),
            "P_FA": float(metrics_tau["P_FA"]),
            "n_detected_ones": int(metrics_tau["n_ones_detected"]),
            "n_detected_zeros": int(metrics_tau["n_zeros_detected"]),
        })

    scan_df = pd.DataFrame(rows)
    scan_df["balanced_error_gap"] = np.abs(scan_df["P_miss"] - scan_df["P_FA"])

    scan_df = scan_df.sort_values(
        by=["BER", "balanced_error_gap", "threshold"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    best = scan_df.iloc[0].to_dict()

    return float(best["threshold"]), best, scan_df


def poisson_count_llr(
    counts,
    lambda_h0,
    lambda_h1,
):
    """
    Compute count-only Poisson log-likelihood ratio per symbol.

    LLR(k) = log P(k | H1) - log P(k | H0)
           = k log(lambda_h1 / lambda_h0) - (lambda_h1 - lambda_h0)

    This helper is optional. It is useful when you want a fixed LLR threshold
    derived from a two-rate Poisson model instead of a direct count threshold.
    """
    counts = np.asarray(counts, dtype=np.int64)

    lambda_h0 = max(float(lambda_h0), 1e-15)
    lambda_h1 = max(float(lambda_h1), 1e-15)

    return (
        counts.astype(np.float64) * math.log(lambda_h1 / lambda_h0)
        - (lambda_h1 - lambda_h0)
    )


def fixed_llr_threshold_detect_from_counts(
    counts,
    bits,
    lambda_h0,
    lambda_h1,
    llr_threshold=0.0,
):
    """
    Run fixed-threshold detection on the count-only Poisson LLR.

    Decision rule:
        LLR(count) > llr_threshold -> 1
        otherwise                  -> 0
    """
    bits = np.asarray(bits, dtype=np.int8)
    counts = np.asarray(counts, dtype=np.int64)

    if len(bits) != len(counts):
        raise ValueError(
            f"Length mismatch: len(bits)={len(bits)}, len(counts)={len(counts)}."
        )

    llr = poisson_count_llr(
        counts=counts,
        lambda_h0=lambda_h0,
        lambda_h1=lambda_h1,
    )

    detected = (llr > float(llr_threshold)).astype(np.int8)
    metrics = binary_detection_metrics(bits, detected)

    df = pd.DataFrame({
        "bit_index": np.arange(len(bits), dtype=np.int64),
        "true_bit": bits,
        "detected_bit": detected,
        "correct": detected == bits,
        "n_rx": counts,
        "count_llr": llr,
        "llr_threshold": float(llr_threshold),
        "lambda_h0": float(lambda_h0),
        "lambda_h1": float(lambda_h1),
    })

    result = {
        "detected_bits": detected,
        "count_llr": llr,
        "llr_threshold": float(llr_threshold),
        "lambda_h0": float(lambda_h0),
        "lambda_h1": float(lambda_h1),
        "detector": "count_fixed_llr_threshold",
    }

    return df, metrics, result
