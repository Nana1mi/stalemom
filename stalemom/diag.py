"""SPEC E5: Momentum & gradient alignment diagnostics and estimators (§3 & §5).

Reuses existing stalemom.bandit, stalemom.estimators, stalemom.sim without modifying them.
Provides:
  - sample_grads_at(theta, q, problem, estimator, c, B, K, rng) -> np.ndarray
  - nested_stats(G, m, R) -> dict
  - compute_c3(stats_t, stats_s) -> float
  - exact_bandit_stats(theta, q, m, problem, estimator, c, B) -> dict
  - CountSketch
"""
import hashlib
import numpy as np

from stalemom.bandit import log_softmax, project
from stalemom.estimators import sample_counts, reward_term, tis_bias, direction


def resolve_problem(problem: dict, dim: int | None = None) -> tuple[int, np.ndarray, np.ndarray, float]:
    """Extract (K_arms, r, logp_ref, lam) from a problem specification dict."""
    K_arms = int(problem.get("K", dim if dim is not None else 10))
    lam = float(problem.get("lam", 1.0))

    if "r" in problem and problem["r"] is not None:
        r = np.asarray(problem["r"], dtype=np.float64)
    else:
        n_good = int(problem.get("n_good", 2))
        r = np.zeros(K_arms, dtype=np.float64)
        r[:n_good] = 1.0

    if "logp_ref" in problem and problem["logp_ref"] is not None:
        logp_ref = np.asarray(problem["logp_ref"], dtype=np.float64)
    else:
        init_type = problem.get("init", "random")
        init_seed = int(problem.get("init_seed", 0))
        init_scale = float(problem.get("init_scale", 1.0))
        if init_type == "uniform":
            theta_0 = np.zeros(K_arms, dtype=np.float64)
        elif init_type == "random":
            init_rng = np.random.default_rng(init_seed)
            z = init_rng.normal(size=K_arms)
            theta_0 = init_scale * project(z)
        else:
            raise ValueError(f"Unknown init type: {init_type}")
        logp_ref = log_softmax(theta_0)

    return K_arms, r, logp_ref, lam


def sample_grads_at(theta, q, problem, estimator, c, B, K, rng) -> np.ndarray:
    """在固定的 theta 和行为分布 q 下，独立抽取 K 个随机方向 g，形状为 (K, dim)。"""
    theta = np.asarray(theta, dtype=np.float64)
    dim = theta.shape[-1]
    _, r, logp_ref, lam = resolve_problem(problem, dim=dim)

    logp = log_softmax(theta)
    p = np.exp(logp)

    if estimator in ("onpolicy", "exact") or q is None:
        q_eff = p
    else:
        q_eff = np.asarray(q, dtype=np.float64)

    if estimator == "exact":
        v_h = r - lam * (logp - logp_ref)
        h = project(v_h)
        return np.tile(h, (K, 1))
    elif estimator == "tis_mean":
        v_h = r - lam * (logp - logp_ref)
        h = project(v_h)
        b = tis_bias(p, q_eff, r, c)
        return np.tile(h + b, (K, 1))

    sample_dist = p if estimator == "onpolicy" else q_eff
    probs_tiled = np.tile(sample_dist, (K, 1))
    p_tiled = np.tile(p, (K, 1))
    q_tiled = np.tile(q_eff, (K, 1))

    counts = sample_counts(rng, probs_tiled, int(B))
    rho_hat = reward_term(estimator, counts, p_tiled, q_tiled, r, c, int(B))
    G = direction(rho_hat, logp, logp_ref, lam)
    return G


def exact_bandit_stats(theta, q, m, problem, estimator: str, c: float | None, B: int) -> dict:
    """Compute closed-form exact expectations, variances, SNR, and cosine similarities on bandit."""
    theta = np.asarray(theta, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    dim = theta.shape[-1]
    K_arms, r, logp_ref, lam = resolve_problem(problem, dim=dim)

    logp = log_softmax(theta)
    p = np.exp(logp)
    if estimator in ("onpolicy", "exact") or q is None:
        q_eff = p
    else:
        q_eff = np.asarray(q, dtype=np.float64)

    # Exact true NPG direction h = Pi[r - lam * (logp - logp_ref)]
    h = project(r - lam * (logp - logp_ref))

    # Estimator bias b
    if estimator in ("tis", "tis_mean"):
        b = tis_bias(p, q_eff, r, c)
    else:
        b = np.zeros_like(h)

    mu = h + b  # Exact E[g | theta, q]

    H_exact_h = float(np.sum(h ** 2))
    H_exact_mean = float(np.sum(mu ** 2))

    # Exact closed-form variance E||g - E[g]||^2
    if estimator in ("exact", "tis_mean"):
        var_exact = 0.0
    else:
        if estimator == "onpolicy":
            pi_sample = p
            u = np.where(p > 0, r / p, 0.0)
        elif estimator in ("is", "tis_debiased"):
            pi_sample = q_eff
            u = np.where(q_eff > 0, r / q_eff, 0.0)
        elif estimator == "tis":
            pi_sample = q_eff
            ratio = np.where(q_eff > 0, p / q_eff, 0.0)
            w = np.where(p > 0, np.minimum(ratio, float(c)) / p, 0.0)
            u = w * r
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        mean_rho = pi_sample * u
        proj_mean_rho_sq = float(np.sum(project(mean_rho) ** 2))
        second_moment = (1.0 - 1.0 / K_arms) * float(np.sum(pi_sample * (u ** 2)))
        var_exact = max(0.0, (second_moment - proj_mean_rho_sq) / float(B))

    N_exact = H_exact_mean + var_exact
    SNR_exact = float(H_exact_mean / var_exact) if var_exact > 0 else float("nan")
    SNR_exact_h = float(H_exact_h / var_exact) if var_exact > 0 else float("nan")

    norm_m = float(np.linalg.norm(m))
    norm_h = float(np.sqrt(H_exact_h))
    norm_mu = float(np.sqrt(H_exact_mean))
    norm_b = float(np.linalg.norm(b))

    inner_m_h_exact = float(np.dot(m, h))
    inner_m_mean_exact = float(np.dot(m, mu))
    inner_m_b_exact = float(np.dot(m, b))

    if norm_m > 0 and norm_h > 0:
        C2_exact_h = float(inner_m_h_exact / (norm_m * norm_h))
    else:
        C2_exact_h = float("nan")

    if norm_m > 0 and norm_mu > 0:
        C2_exact_mean = float(inner_m_mean_exact / (norm_m * norm_mu))
    else:
        C2_exact_mean = float("nan")

    if norm_m > 0 and norm_b > 1e-15:
        cos_m_b = float(inner_m_b_exact / (norm_m * norm_b))
    else:
        cos_m_b = float("nan")

    if norm_h > 0:
        norm_b_over_h = float(norm_b / norm_h)
    else:
        norm_b_over_h = float("nan")

    return {
        "h": h,
        "b": b,
        "mu": mu,
        "H_exact_h": H_exact_h,
        "H_exact_mean": H_exact_mean,
        "var_exact": var_exact,
        "N_exact": N_exact,
        "SNR_exact": SNR_exact,
        "SNR_exact_h": SNR_exact_h,
        "inner_m_h_exact": inner_m_h_exact,
        "inner_m_mean_exact": inner_m_mean_exact,
        "C2_exact_h": C2_exact_h,
        "C2_exact_mean": C2_exact_mean,
        "cos_m_b": cos_m_b,
        "norm_b_over_h": norm_b_over_h,
    }


def _compute_formulas_from_set_accumulators(
    n_set: np.ndarray,
    d_set: np.ndarray,
    inv_sqrt_n_set: np.ndarray,
    c1_set: np.ndarray,
    norm_T_sq: np.ndarray,
    A_sum: np.ndarray,
    B_sum: np.ndarray,
    S_A: int,
    S_B: int,
    R: int,
    norm_m: float,
) -> dict:
    """Internal helper to compute §5.3 point estimates from set-level accumulators."""
    S = S_A + S_B
    total_batches = S * R
    N_A = S_A * R
    N_B = S_B * R

    N_bar = float(np.sum(n_set) / total_batches)
    A_bar = A_sum / float(N_A)
    B_bar = B_sum / float(N_B)
    H = float(np.dot(A_bar, B_bar))

    if R >= 2:
        M_s = (norm_T_sq - n_set) / float(R * (R - 1))
        M = float(np.mean(M_s))
        var_rollout = N_bar - M
        var_prompt = M - H
    else:
        M = float("nan")
        var_rollout = float("nan")
        var_prompt = float("nan")

    var_total = N_bar - H
    SNR_hat = float(H / var_total) if var_total > 0 else float("nan")

    d_bar = float(np.sum(d_set) / total_batches)

    if norm_m > 0:
        C1 = float(np.sum(c1_set) / total_batches)
        C2_hat = float(d_bar / (norm_m * np.sqrt(H))) if H > 0 else float("nan")
        C1_pred = float(d_bar / (norm_m * np.sqrt(N_bar))) if N_bar > 0 else float("nan")
        mean_inv_sqrt_n = float(np.sum(inv_sqrt_n_set) / total_batches)
        C1_pred_legacy = float((d_bar / norm_m) * mean_inv_sqrt_n)
    else:
        C1 = float("nan")
        C2_hat = float("nan")
        C1_pred = float("nan")
        C1_pred_legacy = float("nan")

    return {
        "N_bar": N_bar,
        "H_hat": H,
        "M": M,
        "var_total": var_total,
        "var_rollout": var_rollout,
        "var_prompt": var_prompt,
        "SNR_hat": SNR_hat,
        "inner_m_h": d_bar,
        "C1": C1,
        "C2_hat": C2_hat,
        "C1_pred": C1_pred,
        "C1_pred_legacy": C1_pred_legacy,
        "A_bar": A_bar,
        "B_bar": B_bar,
    }


def nested_stats(G: np.ndarray, m: np.ndarray, R: int, compute_jk: bool = True) -> dict:
    """G: (S*R, dim) or (S, R, dim)，按 prompt-set 分组；bandit 中没有 prompt，S 组只代表独立重复。
    m: 动量向量。返回 §5 中定义的全部估计量（及 jackknife SE）。
    """
    G = np.asarray(G, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    R = int(R)

    if G.ndim == 2:
        total_batches, dim = G.shape
        if total_batches % R != 0:
            raise ValueError(f"G.shape[0]={total_batches} must be divisible by R={R}")
        S = total_batches // R
        G_3d = G.reshape(S, R, dim)
    elif G.ndim == 3:
        S, R_in, dim = G.shape
        if R_in != R:
            raise ValueError(f"G.shape[1]={R_in} does not match R={R}")
        G_3d = G
    else:
        raise ValueError(f"G must be 2D or 3D, got shape {G.shape}")

    if S < 2:
        raise ValueError(f"Number of sets S={S} must be >= 2 to split into halves A and B")

    half = S // 2
    S_A = half
    S_B = S - half

    norm_m = float(np.linalg.norm(m))

    # Online accumulators per set (§5.2)
    n_batch = np.sum(G_3d ** 2, axis=-1)  # (S, R)
    d_batch = np.sum(G_3d * m[None, None, :], axis=-1)  # (S, R)
    sqrt_n = np.sqrt(n_batch)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sqrt_n = np.where(sqrt_n > 0, 1.0 / sqrt_n, 0.0)
        if norm_m > 0:
            c1_batch = np.where(sqrt_n > 0, d_batch / (norm_m * sqrt_n), 0.0)
        else:
            c1_batch = np.full_like(d_batch, np.nan)

    n_set = np.sum(n_batch, axis=1)  # (S,)
    d_set = np.sum(d_batch, axis=1)  # (S,)
    inv_sqrt_n_set = np.sum(inv_sqrt_n, axis=1)  # (S,)
    c1_set = np.sum(c1_batch, axis=1)  # (S,)

    T_s = np.sum(G_3d, axis=1)  # (S, dim)
    norm_T_sq = np.sum(T_s ** 2, axis=-1)  # (S,)

    A_sum = np.sum(T_s[:half], axis=0)
    B_sum = np.sum(T_s[half:], axis=0)

    est = _compute_formulas_from_set_accumulators(
        n_set=n_set,
        d_set=d_set,
        inv_sqrt_n_set=inv_sqrt_n_set,
        c1_set=c1_set,
        norm_T_sq=norm_T_sq,
        A_sum=A_sum,
        B_sum=B_sum,
        S_A=S_A,
        S_B=S_B,
        R=R,
        norm_m=norm_m,
    )

    scalar_keys = [
        "N_bar",
        "H_hat",
        "M",
        "var_total",
        "var_rollout",
        "var_prompt",
        "SNR_hat",
        "inner_m_h",
        "C1",
        "C2_hat",
        "C1_pred",
        "C1_pred_legacy",
    ]

    # Jackknife over prompt-sets s = 0..S-1 (§5.4)
    if compute_jk and S_A >= 2 and S_B >= 2:
        jk_replicates = {k: [] for k in scalar_keys}
        for s_leave in range(S):
            mask = np.ones(S, dtype=bool)
            mask[s_leave] = False
            if s_leave < half:
                A_sum_jk = A_sum - T_s[s_leave]
                B_sum_jk = B_sum
                S_A_jk = S_A - 1
                S_B_jk = S_B
            else:
                A_sum_jk = A_sum
                B_sum_jk = B_sum - T_s[s_leave]
                S_A_jk = S_A
                S_B_jk = S_B - 1

            rep = _compute_formulas_from_set_accumulators(
                n_set=n_set[mask],
                d_set=d_set[mask],
                inv_sqrt_n_set=inv_sqrt_n_set[mask],
                c1_set=c1_set[mask],
                norm_T_sq=norm_T_sq[mask],
                A_sum=A_sum_jk,
                B_sum=B_sum_jk,
                S_A=S_A_jk,
                S_B=S_B_jk,
                R=R,
                norm_m=norm_m,
            )
            for k in scalar_keys:
                jk_replicates[k].append(rep[k])

        for k in scalar_keys:
            vals = np.asarray(jk_replicates[k], dtype=np.float64)
            valid = vals[np.isfinite(vals)]
            n_val = len(valid)
            if n_val >= 2:
                jk_mean = np.mean(valid)
                jk_se = float(np.sqrt(((n_val - 1.0) / n_val) * np.sum((valid - jk_mean) ** 2)))
            else:
                jk_se = float("nan")
            est[f"{k}_jk_se"] = jk_se
    else:
        for k in scalar_keys:
            est[f"{k}_jk_se"] = float("nan")

    return est


def compute_c3(stats_t: dict, stats_s: dict) -> float:
    """Compute cross-checkpoint C3(t, s) per §5.3:
    C3(t, s) = 0.5 * (<A_bar_t, B_bar_s> + <B_bar_t, A_bar_s>) / sqrt(H_t * H_s)
    """
    H_t = stats_t["H_hat"]
    H_s = stats_s["H_hat"]
    if H_t <= 0 or H_s <= 0 or not np.isfinite(H_t) or not np.isfinite(H_s):
        return float("nan")
    A_t, B_t = stats_t["A_bar"], stats_t["B_bar"]
    A_s, B_s = stats_s["A_bar"], stats_s["B_bar"]
    cross = 0.5 * (float(np.dot(A_t, B_s)) + float(np.dot(B_t, A_s)))
    return float(cross / np.sqrt(H_t * H_s))


class CountSketch:
    """Linear CountSketch (§5.5) for cross-checkpoint inner product estimation.

    Uses deterministic hash buckets and +/-1 signs seeded by parameter name so that
    sketches across different checkpoints are strictly aligned.
    """

    def __init__(self, D: int = 2 ** 22):
        self.D = int(D)
        self._cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _name_to_seed(name: str) -> int:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    def _get_hash_and_sign(self, name: str, numel: int) -> tuple[np.ndarray, np.ndarray]:
        key = (name, numel)
        if key not in self._cache:
            seed = self._name_to_seed(name)
            rng = np.random.default_rng(seed)
            buckets = rng.integers(0, self.D, size=numel, dtype=np.int64)
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=numel)
            self._cache[key] = (buckets, signs)
        return self._cache[key]

    def sketch_tensor(self, name: str, arr: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Sketch a single array/tensor with parameter name `name`."""
        flat = np.asarray(arr, dtype=np.float64).ravel()
        buckets, signs = self._get_hash_and_sign(name, flat.size)
        contrib = np.bincount(buckets, weights=signs * flat, minlength=self.D).astype(np.float64)
        if out is None:
            return contrib
        out += contrib
        return out

    def sketch_dict(self, named_tensors: dict[str, np.ndarray]) -> np.ndarray:
        """Sketch a dictionary of {param_name: array} into a single D-dimensional vector."""
        out = np.zeros(self.D, dtype=np.float64)
        for name, arr in named_tensors.items():
            self.sketch_tensor(name, arr, out=out)
        return out

    def sketch_vector(self, vec: np.ndarray, name: str = "default") -> np.ndarray:
        """Sketch a 1D vector."""
        return self.sketch_tensor(name, vec)

    @staticmethod
    def inner(sketch_u: np.ndarray, sketch_v: np.ndarray) -> float:
        """Unbiased inner product estimate <sketch(u), sketch(v)>."""
        return float(np.dot(sketch_u, sketch_v))


def _mean_and_se(arr: np.ndarray) -> tuple[float, float, int]:
    arr = np.asarray(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    n = len(valid)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(valid))
    se = float(np.std(valid, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, se, n


def run_d1_condition(task: dict) -> list[dict]:
    """Run D1 trajectory simulation and frozen-checkpoint gradient diagnostics for one (estimator, k, B, beta)."""
    from stalemom.sim import simulate

    estimator = task["estimator"]
    k = int(task["k"])
    B = int(task["B"])
    beta = float(task["beta"])
    eta = float(task["eta"])
    R_traj = int(task["R"])
    K_samples = int(task["K_samples"])
    R_rep = int(task["R_rep"])
    c = float(task["c"])
    seed = int(task["seed"])
    freeze_steps = task["freeze_steps"]

    T = max(freeze_steps) + 4
    omb = 1.0 - beta
    eff_c = c if estimator in ("tis", "tis_debiased", "tis_mean") else None

    problem = {
        "K": 10,
        "n_good": 2,
        "lam": 1.0,
        "init": "random",
        "init_seed": 0,
        "init_scale": 1.0,
        "ref": "init",
    }

    cfg = {
        "problem": problem,
        "run": {
            "R": R_traj,
            "seed": seed,
            "T": T,
            "B": B,
            "diag_every": T + 1,
            "return_traj": True,
        },
        "estimator": estimator,
        "k": k,
        "eta": eta,
        "omb": omb,
        "c": eff_c,
        "mismatch": "lag",
    }

    res = simulate(cfg)
    theta_all = res["theta"]  # (R_traj, T+1, 10)
    m_hat_all = res["m_hat"]  # (R_traj, T, 10)

    # Dedicated deterministic RNG for D1 frozen sampling
    cond_key = f"{estimator}_{k}_{B}_{beta:.4f}"
    diag_seed = seed + 100000 + (CountSketch._name_to_seed(cond_key) % 900000)
    rng = np.random.default_rng(int(diag_seed))

    rows = []
    c3_rows = []
    # Store per-trajectory (ex, st) across freeze_steps to also compute cross-step C3(t1, t2)
    step_traj_cache: dict[int, list[tuple[dict, dict]]] = {t: [] for t in freeze_steps}

    for t in freeze_steps:
        metric_keys = [
            "C2_exact_h",
            "C2_exact_mean",
            "cos_m_b",
            "norm_b_over_h",
            "SNR_exact",
            "SNR_exact_h",
            "H_exact_mean",
            "H_exact_h",
            "N_exact",
            "var_exact",
            "C1",
            "C2_hat",
            "C1_pred",
            "C1_pred_legacy",
            "SNR_hat",
            "H_hat",
            "N_bar",
            "var_total",
            "var_rollout",
            "var_prompt",
            "inner_m_h",
            "inner_m_mean_exact",
            "H_hat_err",
            "C2_hat_err",
            "C1_pred_err",
        ]
        per_traj = {key: [] for key in metric_keys}

        for r_idx in range(R_traj):
            theta_t = theta_all[r_idx, t]
            if estimator in ("onpolicy", "exact") or k == 0:
                q_t = np.exp(log_softmax(theta_t))
            else:
                q_idx = max(t - k, 0)
                q_t = np.exp(log_softmax(theta_all[r_idx, q_idx]))

            m_prev = m_hat_all[r_idx, t - 1]

            ex = exact_bandit_stats(theta_t, q_t, m_prev, problem, estimator, eff_c, B)
            G = sample_grads_at(theta_t, q_t, problem, estimator, eff_c, B, K_samples, rng)
            st = nested_stats(G, m_prev, R=R_rep, compute_jk=False)
            step_traj_cache[t].append((ex, st))

            per_traj["C2_exact_h"].append(ex["C2_exact_h"])
            per_traj["C2_exact_mean"].append(ex["C2_exact_mean"])
            per_traj["cos_m_b"].append(ex["cos_m_b"])
            per_traj["norm_b_over_h"].append(ex["norm_b_over_h"])
            per_traj["SNR_exact"].append(ex["SNR_exact"])
            per_traj["SNR_exact_h"].append(ex["SNR_exact_h"])
            per_traj["H_exact_mean"].append(ex["H_exact_mean"])
            per_traj["H_exact_h"].append(ex["H_exact_h"])
            per_traj["N_exact"].append(ex["N_exact"])
            per_traj["var_exact"].append(ex["var_exact"])
            per_traj["inner_m_mean_exact"].append(ex["inner_m_mean_exact"])

            per_traj["C1"].append(st["C1"])
            per_traj["C2_hat"].append(st["C2_hat"])
            per_traj["C1_pred"].append(st["C1_pred"])
            per_traj["C1_pred_legacy"].append(st["C1_pred_legacy"])
            per_traj["SNR_hat"].append(st["SNR_hat"])
            per_traj["H_hat"].append(st["H_hat"])
            per_traj["N_bar"].append(st["N_bar"])
            per_traj["var_total"].append(st["var_total"])
            per_traj["var_rollout"].append(st["var_rollout"])
            per_traj["var_prompt"].append(st["var_prompt"])
            per_traj["inner_m_h"].append(st["inner_m_h"])

            per_traj["H_hat_err"].append(st["H_hat"] - ex["H_exact_mean"])
            per_traj["C2_hat_err"].append(st["C2_hat"] - ex["C2_exact_mean"])
            per_traj["C1_pred_err"].append(st["C1_pred"] - st["C1"])

        row = {
            "estimator": estimator,
            "k": k,
            "B": B,
            "beta": beta,
            "eta": eta,
            "t": t,
            "R": R_traj,
            "K_samples": K_samples,
        }
        for key in metric_keys:
            mean_val, se_val, n_val = _mean_and_se(per_traj[key])
            row[f"{key}_mean"] = mean_val
            row[f"{key}_se"] = se_val
            if key == "C2_hat":
                row["C2_hat_valid_n"] = n_val
            elif key == "cos_m_b":
                row["cos_m_b_valid_n"] = n_val

        rows.append(row)

    # Cross-checkpoint C3(t1, t2) for all pairs t1 < t2
    for idx1 in range(len(freeze_steps)):
        for idx2 in range(idx1 + 1, len(freeze_steps)):
            t1, t2 = freeze_steps[idx1], freeze_steps[idx2]
            c3_ex_h_list = []
            c3_ex_m_list = []
            c3_hat_list = []
            for r_idx in range(R_traj):
                ex1, st1 = step_traj_cache[t1][r_idx]
                ex2, st2 = step_traj_cache[t2][r_idx]
                nh1 = np.sqrt(ex1["H_exact_h"])
                nh2 = np.sqrt(ex2["H_exact_h"])
                if nh1 > 0 and nh2 > 0:
                    c3_ex_h_list.append(float(np.dot(ex1["h"], ex2["h"]) / (nh1 * nh2)))
                else:
                    c3_ex_h_list.append(float("nan"))

                nm1 = np.sqrt(ex1["H_exact_mean"])
                nm2 = np.sqrt(ex2["H_exact_mean"])
                if nm1 > 0 and nm2 > 0:
                    c3_ex_m_list.append(float(np.dot(ex1["mu"], ex2["mu"]) / (nm1 * nm2)))
                else:
                    c3_ex_m_list.append(float("nan"))

                c3_hat_list.append(compute_c3(st1, st2))

            m_ex_h, se_ex_h, _ = _mean_and_se(c3_ex_h_list)
            m_ex_m, se_ex_m, _ = _mean_and_se(c3_ex_m_list)
            m_hat, se_hat, n_hat = _mean_and_se(c3_hat_list)
            c3_rows.append({
                "estimator": estimator,
                "k": k,
                "B": B,
                "beta": beta,
                "eta": eta,
                "t1": t1,
                "t2": t2,
                "C3_exact_h_mean": m_ex_h,
                "C3_exact_h_se": se_ex_h,
                "C3_exact_mean_mean": m_ex_m,
                "C3_exact_mean_se": se_ex_m,
                "C3_hat_mean": m_hat,
                "C3_hat_se": se_hat,
                "C3_hat_valid_n": n_hat,
            })

    return rows, c3_rows


def main():
    import argparse
    import csv
    import datetime
    import itertools
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor
    from stalemom.run_tuned_momentum import write_env_file

    parser = argparse.ArgumentParser(description="Run SPEC E5 Phase D1: Bandit gradient alignment diagnostics.")
    parser.add_argument("--out", type=str, default="results/e5/d1")
    parser.add_argument("--d2_best", type=str, default="results/e5/d2/best_eta_summary.csv")
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--K_samples", type=int, default=64)
    parser.add_argument("--R_rep", type=int, default=4)
    parser.add_argument("--c", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_env_file(args.out)
    log_path = os.path.join(args.out, "run_log.txt")

    # Load tuned eta from D2 if available, otherwise fallback to SPEC §3.3 defaults
    best_eta_map = {}
    if os.path.exists(args.d2_best):
        with open(args.d2_best, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["estimator"], int(r["k"]), int(r["B"]), float(r["beta"]))
                best_eta_map[key] = float(r["best_eta"])

    estimators = ["onpolicy", "is", "tis"]
    ks = [0, 16]
    Bs = [4, 32]
    betas = [0.0, 0.6, 0.9, 0.99]
    freeze_steps = [16, 64, 256]

    tasks = []
    for est, k, B, beta in itertools.product(estimators, ks, Bs, betas):
        eta = best_eta_map.get((est, k, B, beta), 0.1 if B == 32 else 0.025)
        tasks.append({
            "estimator": est,
            "k": k,
            "B": B,
            "beta": beta,
            "eta": eta,
            "R": args.R,
            "K_samples": args.K_samples,
            "R_rep": args.R_rep,
            "c": args.c,
            "seed": args.seed,
            "freeze_steps": freeze_steps,
        })

    t0 = time.time()
    msg = f"[D1] Starting {len(tasks)} conditions x {len(freeze_steps)} freeze steps (R={args.R}, K={args.K_samples})..."
    print(msg)
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(msg + "\n")

    all_rows = []
    all_c3_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for cond_rows, cond_c3_rows in executor.map(run_d1_condition, tasks):
            all_rows.extend(cond_rows)
            all_c3_rows.extend(cond_c3_rows)

    all_rows.sort(key=lambda r: (r["estimator"], r["k"], r["B"], r["beta"], r["t"]))
    all_c3_rows.sort(key=lambda r: (r["estimator"], r["k"], r["B"], r["beta"], r["t1"], r["t2"]))

    csv_path = os.path.join(args.out, "summary.csv")
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    c3_csv_path = os.path.join(args.out, "c3_summary.csv")
    c3_fieldnames = list(all_c3_rows[0].keys())
    with open(c3_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=c3_fieldnames)
        writer.writeheader()
        writer.writerows(all_c3_rows)

    elapsed = time.time() - t0
    done_msg = f"[D1] Completed {len(all_rows)} rows in {elapsed:.2f}s. Saved to {csv_path}"
    print(done_msg)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(done_msg + "\n")

    with open(os.path.join(args.out, "DONE"), "w", encoding="utf-8") as df:
        df.write(f"DONE at {datetime.datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()

