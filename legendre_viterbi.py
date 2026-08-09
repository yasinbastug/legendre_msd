"""Hybrid Legendre + count-Viterbi detection."""

import math

import numpy as np
import pandas as pd

from legendre import expected_isi_llr_known_axis
from viterbi import binary_detection_metrics, log_poisson_pmf_vec


def legendre_viterbi_detect(
    received_symbol_hits,
    bits,
    leglib,
    n_emit_bit,
    tap_prob,
    p_one=0.5,
    axis=(0.0, 0.0, 1.0),
    lambda_env_count=0.0,
    lambda_env_shape=1.0,
    memory_taps=None,
    use_bit_prior=True,
    max_states=2_000_000,
    expected_prev_p1=0.5,
):
    """Hybrid detector: count-Poisson Viterbi + Legendre shape LLR bonus."""
    bits = np.asarray(bits, dtype=np.int8)
    counts = np.asarray([len(obs["t_rel"]) for obs in received_symbol_hits], dtype=np.int64)
    tap_prob = np.asarray(tap_prob, dtype=np.float64)
    if memory_taps is not None:
        tap_prob = tap_prob[: int(memory_taps)]

    n_bits = len(counts)
    if len(bits) != n_bits:
        raise ValueError(f"Length mismatch: len(bits)={len(bits)}, len(counts)={n_bits}.")

    K = len(tap_prob)
    if K < 1:
        raise ValueError("tap_prob must contain at least tap 0.")

    p_one = float(p_one)
    p_one_clip = min(max(p_one, 1e-15), 1.0 - 1e-15)
    log_p1 = math.log(p_one_clip)
    log_p0 = math.log(1.0 - p_one_clip)
    if not use_bit_prior:
        log_p1 = 0.0
        log_p0 = 0.0

    # Shape-only LLR per symbol:
    #   llr_shape[n] = log p(shape_n | H1) - log p(shape_n | H0)
    shape_llr = np.zeros(n_bits, dtype=np.float64)
    for n, obs in enumerate(received_symbol_hits):
        llr_parts = expected_isi_llr_known_axis(
            leglib=leglib,
            t_rel=obs["t_rel"],
            xyz_obs=obs["xyz"],
            n_emit_bit=n_emit_bit,
            axis=axis,
            expected_prev_p1=expected_prev_p1,
            lambda_env=lambda_env_shape,
            use_count_likelihood=False,
            use_shape_likelihood=True,
        )
        shape_llr[n] = float(llr_parts["llr"])

    state_len = K - 1
    if state_len == 0:
        lam0 = float(lambda_env_count)
        lam1 = float(lambda_env_count) + float(n_emit_bit) * tap_prob[0]
        ll0 = np.array(
            [log_p0 + log_poisson_pmf_vec(k, lam0) for k in counts],
            dtype=np.float64,
        )
        ll1 = np.array(
            [log_p1 + log_poisson_pmf_vec(k, lam1) + shape_llr[n] for n, k in enumerate(counts)],
            dtype=np.float64,
        )
        detected = (ll1 > ll0).astype(np.int8)
        result = {
            "detected_bits": detected,
            "final_log_metric": float(np.sum(np.maximum(ll0, ll1))),
            "state_path": np.zeros(n_bits, dtype=np.int64),
            "shape_llr": shape_llr,
            "K": int(K),
            "state_len": 0,
            "n_states": 1,
        }
    else:
        n_states = 1 << state_len
        if n_states > int(max_states):
            raise MemoryError(
                f"Hybrid Legendre-Viterbi would require {n_states:,} states for K={K} taps. "
                "Reduce --viterbi-memory-taps or increase max_states intentionally."
            )

        states = np.arange(n_states, dtype=np.int64)
        state_isi_rate = np.zeros(n_states, dtype=np.float64)
        for i in range(state_len):
            bit_i = ((states >> i) & 1).astype(np.float64)
            state_isi_rate += bit_i * float(n_emit_bit) * tap_prob[i + 1]

        new_states = states
        current_bit_from_new_state = (new_states & 1).astype(np.int8)
        prev0 = new_states >> 1
        oldest_mask = 1 << (state_len - 1)
        prev1 = prev0 | oldest_mask

        current_rate = (
            current_bit_from_new_state.astype(np.float64)
            * float(n_emit_bit)
            * tap_prob[0]
        )
        log_prior_current = np.where(
            current_bit_from_new_state == 1,
            log_p1,
            log_p0,
        ).astype(np.float64)

        dp_prev = np.full(n_states, -np.inf, dtype=np.float64)
        dp_prev[0] = 0.0
        back_choice = np.zeros((n_bits, n_states), dtype=np.uint8)

        for n in range(n_bits):
            k_obs = int(counts[n])
            shape_bonus = np.where(current_bit_from_new_state == 1, shape_llr[n], 0.0)

            lam_prev0 = float(lambda_env_count) + state_isi_rate[prev0] + current_rate
            lam_prev1 = float(lambda_env_count) + state_isi_rate[prev1] + current_rate

            score0 = (
                dp_prev[prev0]
                + log_prior_current
                + log_poisson_pmf_vec(k_obs, lam_prev0)
                + shape_bonus
            )
            score1 = (
                dp_prev[prev1]
                + log_prior_current
                + log_poisson_pmf_vec(k_obs, lam_prev1)
                + shape_bonus
            )

            take_prev1 = score1 > score0
            dp_prev = np.where(take_prev1, score1, score0)
            back_choice[n, :] = take_prev1.astype(np.uint8)

        best_final_state = int(np.argmax(dp_prev))
        best_final_metric = float(dp_prev[best_final_state])
        detected_bits = np.zeros(n_bits, dtype=np.int8)
        state_path = np.zeros(n_bits, dtype=np.int64)
        s = best_final_state
        for n in range(n_bits - 1, -1, -1):
            state_path[n] = s
            detected_bits[n] = np.int8(s & 1)
            choice = int(back_choice[n, s])
            s = (s >> 1) | (choice << (state_len - 1))

        result = {
            "detected_bits": detected_bits,
            "final_log_metric": best_final_metric,
            "state_path": state_path,
            "shape_llr": shape_llr,
            "K": int(K),
            "state_len": int(state_len),
            "n_states": int(n_states),
        }

    detected = result["detected_bits"].astype(np.int8)
    df = pd.DataFrame({
        "bit_index": np.arange(len(bits), dtype=np.int64),
        "true_bit": bits,
        "detected_bit": detected,
        "correct": detected == bits,
        "n_rx": counts,
        "shape_llr": shape_llr,
        "state_path": result["state_path"],
    })
    metrics = binary_detection_metrics(bits, detected)
    return df, metrics, result
