"""Count-only Poisson Viterbi detection."""

import math
import numpy as np
import pandas as pd

def log_poisson_pmf_vec(k, lam):
    """
    Vectorized log Poisson(k; lam).
    """
    k = int(k)
    lam = np.asarray(lam, dtype=np.float64)
    lam = np.maximum(lam, 1e-15)

    return k * np.log(lam) - lam - math.lgamma(k + 1)

def binary_detection_metrics(y_true, y_hat):
    """
    BER, miss probability, false-alarm probability.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_hat = np.asarray(y_hat, dtype=np.int8)

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

def count_channel_lambda_from_bits(
    bit_sequence,
    tap_prob,
    n_emit_bit,
    lambda_env=0.0,
):
    """
    Computes lambda_n for each symbol from a known bit sequence.
    """
    bit_sequence = np.asarray(bit_sequence, dtype=np.int8)
    tap_prob = np.asarray(tap_prob, dtype=np.float64)

    n_bits = len(bit_sequence)
    K = len(tap_prob)

    lam = np.full(n_bits, float(lambda_env), dtype=np.float64)

    for n in range(n_bits):
        s = 0.0
        for tap in range(K):
            k = n - tap
            if k < 0:
                continue
            s += float(bit_sequence[k]) * tap_prob[tap]

        lam[n] += float(n_emit_bit) * s

    return lam

def run_count_poisson_viterbi(
    counts,
    tap_prob,
    n_emit_bit,
    p_one=0.5,
    lambda_env=0.0,
    use_bit_prior=True,
    max_states=2_000_000,
):
    """
    Count-only Poisson Viterbi detector.

    State convention:
        state bit i stores b_{n-1-i}

        i = 0       -> previous bit b_{n-1}
        i = 1       -> b_{n-2}
        ...
        i = K-2     -> b_{n-(K-1)}

    At time n, branch bit x = b_n.

    Transition:
        new_state = ((old_state << 1) | x) masked to K-1 bits

    The implementation uses the inverse transition for vectorization.
    """
    counts = np.asarray(counts, dtype=np.int64)
    tap_prob = np.asarray(tap_prob, dtype=np.float64)

    n_bits = len(counts)
    K = len(tap_prob)

    if K < 1:
        raise ValueError("tap_prob must contain at least tap 0.")

    state_len = K - 1

    p_one = float(p_one)
    p_one_clip = min(max(p_one, 1e-15), 1.0 - 1e-15)

    log_p1 = math.log(p_one_clip)
    log_p0 = math.log(1.0 - p_one_clip)

    if not use_bit_prior:
        log_p1 = 0.0
        log_p0 = 0.0

    # --------------------------------------------------------
    # Special memoryless case
    # --------------------------------------------------------
    if state_len == 0:
        lam0 = float(lambda_env)
        lam1 = float(lambda_env) + float(n_emit_bit) * tap_prob[0]

        ll0 = np.array([
            log_p0 + log_poisson_pmf_vec(k, lam0)
            for k in counts
        ], dtype=np.float64)

        ll1 = np.array([
            log_p1 + log_poisson_pmf_vec(k, lam1)
            for k in counts
        ], dtype=np.float64)

        detected = (ll1 > ll0).astype(np.int8)

        return {
            "detected_bits": detected,
            "final_log_metric": float(np.sum(np.maximum(ll0, ll1))),
            "state_path": np.zeros(n_bits, dtype=np.int64),
            "K": int(K),
            "state_len": 0,
            "n_states": 1,
        }

    # --------------------------------------------------------
    # Full ISI Viterbi case
    # --------------------------------------------------------
    n_states = 1 << state_len

    if n_states > int(max_states):
        raise MemoryError(
            f"Count Viterbi would require {n_states:,} states "
            f"for K={K} taps. Reduce VITERBI_MEMORY_TAPS or increase "
            "COUNT_VITERBI_MAX_STATES intentionally."
        )

    states = np.arange(n_states, dtype=np.int64)

    # ISI rate contribution from the previous-bit state:
    #
    #   N_EMIT_BIT * sum_{i=0}^{state_len-1}
    #       state_bit_i * tap_prob[i+1]
    #
    # because state bit i corresponds to tap i+1.
    state_isi_rate = np.zeros(n_states, dtype=np.float64)

    for i in range(state_len):
        bit_i = ((states >> i) & 1).astype(np.float64)
        state_isi_rate += bit_i * float(n_emit_bit) * tap_prob[i + 1]

    # Inverse transition arrays.
    #
    # For a given new_state ns:
    #   current bit x = ns & 1
    #   possible previous states:
    #       prev0 = ns >> 1
    #       prev1 = (ns >> 1) with dropped oldest bit = 1
    #
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

    # DP initialization:
    # before bit 0, all past bits are zero.
    dp_prev = np.full(n_states, -np.inf, dtype=np.float64)
    dp_prev[0] = 0.0

    # back_choice[n, ns] = 0 means previous state was prev0[ns]
    # back_choice[n, ns] = 1 means previous state was prev1[ns]
    back_choice = np.zeros((n_bits, n_states), dtype=np.uint8)

    for n in range(n_bits):
        k_obs = int(counts[n])

        lam_prev0 = (
            float(lambda_env)
            + state_isi_rate[prev0]
            + current_rate
        )

        lam_prev1 = (
            float(lambda_env)
            + state_isi_rate[prev1]
            + current_rate
        )

        score0 = (
            dp_prev[prev0]
            + log_prior_current
            + log_poisson_pmf_vec(k_obs, lam_prev0)
        )

        score1 = (
            dp_prev[prev1]
            + log_prior_current
            + log_poisson_pmf_vec(k_obs, lam_prev1)
        )

        take_prev1 = score1 > score0

        dp_next = np.where(take_prev1, score1, score0)
        back_choice[n, :] = take_prev1.astype(np.uint8)

        dp_prev = dp_next

    best_final_state = int(np.argmax(dp_prev))
    best_final_metric = float(dp_prev[best_final_state])

    # Traceback.
    detected_bits = np.zeros(n_bits, dtype=np.int8)
    state_path = np.zeros(n_bits, dtype=np.int64)

    s = best_final_state

    for n in range(n_bits - 1, -1, -1):
        state_path[n] = s
        detected_bits[n] = np.int8(s & 1)

        choice = int(back_choice[n, s])
        s = (s >> 1) | (choice << (state_len - 1))

    return {
        "detected_bits": detected_bits,
        "final_log_metric": best_final_metric,
        "state_path": state_path,
        "K": int(K),
        "state_len": int(state_len),
        "n_states": int(n_states),
    }

def viterbi_detect(
    received_symbol_hits,
    bits,
    tap_prob,
    n_emit_bit,
    p_one=0.5,
    lambda_env=0.0,
    memory_taps=None,
    use_bit_prior=True,
    max_states=2_000_000,
):
    """Run count-only Poisson Viterbi on the same received symbols."""
    bits = np.asarray(bits, dtype=np.int8)
    tap_prob = np.asarray(tap_prob, dtype=np.float64)
    if memory_taps is not None:
        tap_prob = tap_prob[: int(memory_taps)]
    counts = np.asarray([len(obs["t_rel"]) for obs in received_symbol_hits], dtype=np.int64)
    result = run_count_poisson_viterbi(
        counts=counts, tap_prob=tap_prob, n_emit_bit=n_emit_bit, p_one=p_one,
        lambda_env=lambda_env, use_bit_prior=use_bit_prior, max_states=max_states,
    )
    detected = result["detected_bits"].astype(np.int8)
    df = pd.DataFrame({
        "bit_index": np.arange(len(bits), dtype=np.int64),
        "true_bit": bits,
        "detected_bit": detected,
        "correct": detected == bits,
        "n_rx": counts,
        "state_path": result["state_path"],
    })
    return df, binary_detection_metrics(bits, detected), result
