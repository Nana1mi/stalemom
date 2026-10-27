import numpy as np
from stalemom.bandit import (
    log_softmax,
    project,
    optimal_value,
    exact_npg_dir,
    vanilla_grad,
)
from stalemom.estimators import (
    sample_counts,
    reward_term,
    tis_bias,
    direction,
    truncation_rate,
)
from stalemom.optim import (
    bias_correction_den,
    ema_step,
    history_age,
)


def simulate(cfg: dict) -> dict:
    """Run a vectorized simulation for a single expanded configuration.

    Parameters
    ----------
    cfg : dict
        Configuration containing 'problem', 'run', 'estimator', 'k', 'eta', 'omb', 'c', 'mismatch'.

    Returns
    -------
    dict
        Simulation results including gap, J, diagnostics, etc.
    """
    prob_cfg = cfg["problem"]
    run_cfg = cfg["run"]

    K = prob_cfg["K"]
    n_good = prob_cfg["n_good"]
    lam = float(prob_cfg["lam"])
    init_type = prob_cfg.get("init", "random")
    init_seed = prob_cfg.get("init_seed", 0)
    init_scale = float(prob_cfg.get("init_scale", 1.0))

    estimator = cfg["estimator"]
    k = int(cfg.get("k", 0))
    eta = float(cfg["eta"])
    omb = float(cfg["omb"])
    c = cfg.get("c", None)
    if c is not None:
        c = float(c)
    mismatch = cfg.get("mismatch", "lag")
    gradient_type = cfg.get("gradient_type", "npg")

    R = int(run_cfg["R"])
    if estimator in ("exact", "tis_mean"):
        R = 1

    seed = int(run_cfg.get("seed", 12345))
    T = int(run_cfg["T"])
    B = int(run_cfg.get("B", 32))
    diag_every = int(run_cfg.get("diag_every", 8))
    return_traj = bool(run_cfg.get("return_traj", False))

    rng = np.random.default_rng(seed)

    # Reward vector: r = [1]*n_good + [0]*(K - n_good)
    r = np.zeros(K, dtype=np.float64)
    r[:n_good] = 1.0

    # Initial theta_0
    if init_type == "uniform":
        theta_0_single = np.zeros(K, dtype=np.float64)
    elif init_type == "random":
        init_rng = np.random.default_rng(init_seed)
        z = init_rng.normal(size=K)
        theta_0_single = init_scale * project(z)
    else:
        raise ValueError(f"Unknown init type: {init_type}")

    theta_0 = np.tile(theta_0_single, (R, 1))

    # Reference distribution
    logp_ref = log_softmax(theta_0_single)
    J_star = optimal_value(r, logp_ref, lam)

    # Pre-allocate arrays
    theta_all = np.zeros((R, T + 1, K), dtype=np.float64)
    theta_all[:, 0] = theta_0

    # Cache logp and p for each step to avoid re-computing log_softmax
    logp_cache = np.zeros((R, T + 1, K), dtype=np.float64)
    p_cache = np.zeros((R, T + 1, K), dtype=np.float64)

    lp_0 = log_softmax(theta_0)
    p_0 = np.exp(lp_0)
    logp_cache[:, 0] = lp_0
    p_cache[:, 0] = p_0

    J = np.zeros((R, T + 1), dtype=np.float64)
    gap = np.zeros((R, T + 1), dtype=np.float64)

    J_0 = np.sum(p_0 * r, axis=-1) - lam * np.sum(p_0 * (lp_0 - logp_ref), axis=-1)
    J[:, 0] = J_0
    gap[:, 0] = 1.0

    kl_pq = np.zeros((R, T), dtype=np.float64)
    trunc_rate = np.zeros((R, T), dtype=np.float64)

    if estimator in ("exact", "onpolicy"):
        lag = np.zeros(T, dtype=np.int64)
    else:
        lag = np.array([t - max(t - k, 0) for t in range(T)], dtype=np.int64)

    diag_t = np.arange(0, T, diag_every, dtype=np.int64)
    n_diag = len(diag_t)

    bias_norm = np.zeros((R, n_diag), dtype=np.float64)
    inner_update = np.zeros((R, n_diag), dtype=np.float64)
    inner_track = np.zeros((R, n_diag), dtype=np.float64)
    inner_bias_hist = np.zeros((R, n_diag), dtype=np.float64)
    inner_noise = np.zeros((R, n_diag), dtype=np.float64)
    norm_track = np.zeros((R, n_diag), dtype=np.float64)
    norm_bias_hist = np.zeros((R, n_diag), dtype=np.float64)
    norm_noise = np.zeros((R, n_diag), dtype=np.float64)
    H = np.zeros(n_diag, dtype=np.float64)

    diverged = np.zeros(R, dtype=bool)
    diverged_step = np.full(R, -1, dtype=np.int64)

    m = np.zeros((R, K), dtype=np.float64)
    E_h = np.zeros((R, K), dtype=np.float64)
    E_b = np.zeros((R, K), dtype=np.float64)
    E_xi = np.zeros((R, K), dtype=np.float64)

    if return_traj:
        g_all = np.zeros((R, T, K), dtype=np.float64)
        m_hat_all = np.zeros((R, T, K), dtype=np.float64)

    diag_idx = 0

    for t in range(T):
        theta_t = theta_all[:, t]
        logp_t = logp_cache[:, t]
        p_t = p_cache[:, t]

        # Check for divergence
        if not np.all(np.isfinite(theta_t)):
            bad = ~np.all(np.isfinite(theta_t), axis=-1)
            newly_diverged = bad & (~diverged)
            if np.any(newly_diverged):
                diverged[newly_diverged] = True
                diverged_step[newly_diverged] = t

        # Behavior policy q_t
        if estimator in ("exact", "onpolicy"):
            q_t = p_t
            logq_t = logp_t
            kl_pq[:, t] = 0.0
        else:
            q_idx = max(t - k, 0)
            if q_idx == t:
                q_t = p_t
                logq_t = logp_t
                kl_pq[:, t] = 0.0
            else:
                q_t = p_cache[:, q_idx]
                logq_t = logp_cache[:, q_idx]
                kl_pq[:, t] = np.sum(p_t * (logp_t - logq_t), axis=-1)

        # Update direction: h(theta)
        v_h = r - lam * (logp_t - logp_ref)
        if gradient_type == "vanilla":
            h_t = p_t * v_h - p_t * np.sum(p_t * v_h, axis=-1, keepdims=True)
        else:
            h_t = project(v_h)

        if estimator in ("tis", "tis_mean"):
            delta_r = r * (np.minimum(1.0, c * q_t / p_t) - 1.0)
            if gradient_type == "vanilla":
                b_t = p_t * delta_r - p_t * np.sum(p_t * delta_r, axis=-1, keepdims=True)
            else:
                b_t = project(delta_r)
        else:
            b_t = np.zeros_like(h_t)

        if estimator == "exact":
            g_t = h_t.copy()
            xi_t = np.zeros_like(h_t)
            trunc_rate[:, t] = 0.0
        elif estimator == "tis_mean":
            g_t = h_t + b_t
            xi_t = np.zeros_like(h_t)
            trunc_rate[:, t] = 0.0
        elif estimator == "onpolicy":
            counts = sample_counts(rng, p_t, B)
            rho_hat = reward_term("onpolicy", counts, p_t, q_t, r, c, B)
            v_t = rho_hat - lam * (logp_t - logp_ref)
            if gradient_type == "vanilla":
                g_t = p_t * v_t - p_t * np.sum(p_t * v_t, axis=-1, keepdims=True)
            else:
                g_t = project(v_t)
            xi_t = g_t - h_t - b_t
            trunc_rate[:, t] = 0.0
        else:
            counts = sample_counts(rng, q_t, B)
            rho_hat = reward_term(estimator, counts, p_t, q_t, r, c, B)
            v_t = rho_hat - lam * (logp_t - logp_ref)
            if gradient_type == "vanilla":
                g_t = p_t * v_t - p_t * np.sum(p_t * v_t, axis=-1, keepdims=True)
            else:
                g_t = project(v_t)
            xi_t = g_t - h_t - b_t
            if estimator == "tis":
                trunc_rate[:, t] = truncation_rate(counts, p_t, q_t, c, B)
            else:
                trunc_rate[:, t] = 0.0

        # EMA updates
        m = ema_step(m, g_t, omb)
        E_h = ema_step(E_h, h_t, omb)
        E_b = ema_step(E_b, b_t, omb)
        E_xi = ema_step(E_xi, xi_t, omb)

        den_t = bias_correction_den(omb, t)
        m_hat_t = m / den_t
        E_h_hat = E_h / den_t
        E_b_hat = E_b / den_t
        E_xi_hat = E_xi / den_t

        track_t = E_h_hat - h_t
        bias_hist_t = E_b_hat
        noise_t = E_xi_hat

        # Diagnostics
        if diag_idx < n_diag and t == diag_t[diag_idx]:
            grad_t = vanilla_grad(theta_t, r, logp_ref, lam)
            bias_norm[:, diag_idx] = np.linalg.norm(b_t, axis=-1)
            inner_update[:, diag_idx] = np.sum(grad_t * m_hat_t, axis=-1)
            inner_track[:, diag_idx] = np.sum(grad_t * track_t, axis=-1)
            inner_bias_hist[:, diag_idx] = np.sum(grad_t * bias_hist_t, axis=-1)
            inner_noise[:, diag_idx] = np.sum(grad_t * noise_t, axis=-1)
            norm_track[:, diag_idx] = np.linalg.norm(track_t, axis=-1)
            norm_bias_hist[:, diag_idx] = np.linalg.norm(bias_hist_t, axis=-1)
            norm_noise[:, diag_idx] = np.linalg.norm(noise_t, axis=-1)
            H[diag_idx] = history_age(omb, t)
            diag_idx += 1

        if return_traj:
            g_all[:, t] = g_t
            m_hat_all[:, t] = m_hat_t

        # Step theta
        theta_next = theta_t + eta * m_hat_t
        if np.any(diverged):
            theta_next[diverged] = theta_t[diverged]

        theta_all[:, t + 1] = theta_next

        # Compute next distribution and objective
        lp_next = log_softmax(theta_next)
        p_next = np.exp(lp_next)
        logp_cache[:, t + 1] = lp_next
        p_cache[:, t + 1] = p_next

        J_next = np.sum(p_next * r, axis=-1) - lam * np.sum(p_next * (lp_next - logp_ref), axis=-1)
        gap_next = (J_star - J_next) / (J_star - J_0)

        if not np.all(np.isfinite(J_next)) or not np.all(np.isfinite(gap_next)):
            bad_next = ~np.isfinite(J_next) | ~np.isfinite(gap_next)
            newly_div_next = bad_next & (~diverged)
            if np.any(newly_div_next):
                diverged[newly_div_next] = True
                diverged_step[newly_div_next] = t + 1

        if np.any(diverged):
            gap_next[diverged] = np.nan

        J[:, t + 1] = J_next
        gap[:, t + 1] = gap_next

    out = {
        "gap": gap,
        "J": J,
        "kl_pq": kl_pq,
        "trunc_rate": trunc_rate,
        "lag": lag,
        "diag_t": diag_t,
        "bias_norm": bias_norm,
        "inner_update": inner_update,
        "inner_track": inner_track,
        "inner_bias_hist": inner_bias_hist,
        "inner_noise": inner_noise,
        "norm_track": norm_track,
        "norm_bias_hist": norm_bias_hist,
        "norm_noise": norm_noise,
        "H": H,
        "diverged": diverged,
        "diverged_step": diverged_step,
        "J_star": J_star,
        "r": r,
        "logp_ref": logp_ref,
    }

    if return_traj:
        out["theta"] = theta_all
        out["g"] = g_all
        out["m_hat"] = m_hat_all

    return out
