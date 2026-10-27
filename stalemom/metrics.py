import numpy as np


def compute_run_metrics(result: dict, cfg: dict) -> dict:
    """Compute summary statistics for a completed simulation run.

    Follows SPEC §6.2 and §6.3:
    - A_T: gap averaged over t = 1..T, then averaged over runs (with SE).
    - final_gap: gap at t = T (with SE).
    - steps_to_90: median steps to reach gap <= 0.1 (inf if not reached).
    - frac_reached_90: fraction of runs that reached gap <= 0.1.
    - mean_trunc_rate, mean_kl_pq, diverge_rate.
    - Both non-diverged (clean) and pessimistic (diverged gap = 1.0) versions for A_T and final_gap.
    """
    gap = result["gap"]  # shape (R, T+1)
    diverged = result["diverged"]  # shape (R,)
    R, T_plus_1 = gap.shape
    T = T_plus_1 - 1

    # Pessimistic gap: where gap is NaN or run diverged, set gap = 1.0
    gap_pessimistic = np.copy(gap)
    for r_idx in range(R):
        if diverged[r_idx]:
            d_step = result["diverged_step"][r_idx]
            if d_step >= 0:
                gap_pessimistic[r_idx, d_step:] = 1.0
            else:
                gap_pessimistic[r_idx, :] = 1.0
    gap_pessimistic = np.nan_to_num(gap_pessimistic, nan=1.0)

    # Valid (non-diverged) mask
    valid = ~diverged
    n_valid = int(np.sum(valid))

    # A_T for each run (over t = 1..T)
    # 1. Non-diverged only
    if n_valid > 0:
        a_t_valid = np.mean(gap[valid, 1:], axis=1)
        A_T_mean = float(np.mean(a_t_valid))
        A_T_se = float(np.std(a_t_valid, ddof=1) / np.sqrt(n_valid)) if n_valid > 1 else 0.0

        final_valid = gap[valid, -1]
        final_gap_mean = float(np.mean(final_valid))
        final_gap_se = float(np.std(final_valid, ddof=1) / np.sqrt(n_valid)) if n_valid > 1 else 0.0
    else:
        A_T_mean = float("nan")
        A_T_se = float("nan")
        final_gap_mean = float("nan")
        final_gap_se = float("nan")

    # 2. Pessimistic version
    a_t_pess = np.mean(gap_pessimistic[:, 1:], axis=1)
    A_T_pess_mean = float(np.mean(a_t_pess))
    A_T_pess_se = float(np.std(a_t_pess, ddof=1) / np.sqrt(R)) if R > 1 else 0.0

    final_pess = gap_pessimistic[:, -1]
    final_gap_pess_mean = float(np.mean(final_pess))
    final_gap_pess_se = float(np.std(final_pess, ddof=1) / np.sqrt(R)) if R > 1 else 0.0

    # Steps to 90% gap reduction (gap <= 0.1)
    steps_to_90_list = []
    for r_idx in range(R):
        # search over t = 0..T
        idx = np.where(gap_pessimistic[r_idx] <= 0.1)[0]
        if len(idx) > 0:
            steps_to_90_list.append(float(idx[0]))
        else:
            steps_to_90_list.append(float("inf"))

    steps_arr = np.array(steps_to_90_list)
    frac_reached_90 = float(np.mean(steps_arr < float("inf")))
    median_steps = float(np.median(steps_arr))

    mean_trunc_rate = float(np.mean(result["trunc_rate"]))
    mean_kl_pq = float(np.mean(result["kl_pq"]))
    diverge_rate = float(np.mean(diverged))

    metrics = {
        "A_T": A_T_mean,
        "A_T_se": A_T_se,
        "A_T_pessimistic": A_T_pess_mean,
        "A_T_pessimistic_se": A_T_pess_se,
        "final_gap": final_gap_mean,
        "final_gap_se": final_gap_se,
        "final_gap_pessimistic": final_gap_pess_mean,
        "final_gap_pessimistic_se": final_gap_pess_se,
        "steps_to_90": median_steps,
        "frac_reached_90": frac_reached_90,
        "mean_trunc_rate": mean_trunc_rate,
        "mean_kl_pq": mean_kl_pq,
        "diverge_rate": diverge_rate,
        "skipped": False,
        "skip_reason": "",
    }
    return metrics
