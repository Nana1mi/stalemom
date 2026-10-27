"""E0 正确性测试：包含 SPEC_E0-E2.md §5 E0 的全部 11 项检查。"""
import numpy as np
import pytest

from stalemom import bandit, estimators, optim, sim


@pytest.fixture
def rng():
    return np.random.default_rng(20261028)


# 1. 最优值：在 θ* = log p_ref + r/λ（再投影）处，objective 等于 optimal_value，误差 < 1e-12；任意 θ 处 objective ≤ J*。
def test_e0_1_optimal_value(rng):
    K = 8
    lam = 0.8
    r = np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=np.float64)
    logp_ref = bandit.log_softmax(rng.normal(size=K))
    J_star = bandit.optimal_value(r, logp_ref, lam)

    theta_star = bandit.project(logp_ref + r / lam)[None, :]
    J_opt = bandit.objective(theta_star, r, logp_ref, lam)[0]
    assert abs(J_opt - J_star) < 1e-12

    # 随机生成大量 theta，验证 objective(theta) <= J*
    thetas = rng.normal(scale=5.0, size=(1000, K))
    Js = bandit.objective(thetas, r, logp_ref, lam)
    assert np.all(Js <= J_star + 1e-12)


# 2. 精确方向：h(θ*) ≈ 0；vanilla_grad 与对 objective 做中心差分的结果一致（相对误差 < 1e-6）。
def test_e0_2_exact_direction_and_grad(rng):
    K = 6
    lam = 1.2
    r = np.array([1, 0, 1, 0, 0, 0], dtype=np.float64)
    logp_ref = bandit.log_softmax(rng.normal(size=K))
    theta_star = bandit.project(logp_ref + r / lam)[None, :]
    h_star = bandit.exact_npg_dir(theta_star, r, logp_ref, lam)
    np.testing.assert_allclose(h_star, 0.0, atol=1e-12)

    # 中心差分验证 vanilla_grad
    theta = rng.normal(scale=1.5, size=(1, K))
    vg = bandit.vanilla_grad(theta, r, logp_ref, lam)[0]
    eps = 1e-7
    fd = np.zeros(K, dtype=np.float64)
    for i in range(K):
        e = np.zeros((1, K), dtype=np.float64)
        e[0, i] = eps
        j_plus = bandit.objective(theta + e, r, logp_ref, lam)[0]
        j_minus = bandit.objective(theta - e, r, logp_ref, lam)[0]
        fd[i] = (j_plus - j_minus) / (2 * eps)
    np.testing.assert_allclose(vg, fd, rtol=1e-6, atol=1e-8)


# 3. IS 无偏：固定 p、q（随机生成，全支持），对所有可能的单样本动作枚举求期望：E[ĝ^{is}] = h，atol=1e-10、rtol=1e-8。
def test_e0_3_is_unbiased(rng):
    K = 5
    lam = 0.9
    r = np.array([1, 0, 1, 0, 0], dtype=np.float64)
    theta = rng.normal(scale=1.0, size=(1, K))
    q_theta = theta + rng.normal(scale=0.8, size=(1, K))
    p = bandit.softmax(theta)
    q = bandit.softmax(q_theta)
    logp_ref = bandit.log_softmax(rng.normal(size=K))
    h = bandit.exact_npg_dir(theta, r, logp_ref, lam)

    # 单样本动作枚举
    E_g = np.zeros((1, K), dtype=np.float64)
    for a in range(K):
        cnt = np.zeros((1, K), dtype=np.int64)
        cnt[0, a] = 1
        rho_hat = estimators.reward_term("is", cnt, p, q, r, None, 1)
        g_hat = estimators.direction(rho_hat, np.log(p), logp_ref, lam)
        E_g += q[0, a] * g_hat
    np.testing.assert_allclose(E_g, h, atol=1e-10, rtol=1e-8)


# 4. TIS 偏差闭式：同上枚举，E[ĝ^{tis}] − h = tis_bias，同样容差。
def test_e0_4_tis_bias_closed_form(rng):
    K = 6
    lam = 1.0
    c = 1.3
    r = np.array([1, 1, 0, 0, 0, 0], dtype=np.float64)
    theta = rng.normal(scale=1.2, size=(1, K))
    q_theta = theta + rng.normal(scale=1.0, size=(1, K))
    p = bandit.softmax(theta)
    q = bandit.softmax(q_theta)
    logp_ref = bandit.log_softmax(rng.normal(size=K))
    h = bandit.exact_npg_dir(theta, r, logp_ref, lam)
    b = estimators.tis_bias(p, q, r, c)

    E_g = np.zeros((1, K), dtype=np.float64)
    for a in range(K):
        cnt = np.zeros((1, K), dtype=np.int64)
        cnt[0, a] = 1
        rho_hat = estimators.reward_term("tis", cnt, p, q, r, c, 1)
        g_hat = estimators.direction(rho_hat, np.log(p), logp_ref, lam)
        E_g += q[0, a] * g_hat
    np.testing.assert_allclose(E_g - h, b, atol=1e-10, rtol=1e-8)


# 5. 蒙特卡洛一致性：用 10⁶ 次 B=1 采样估计 E[ĝ^{tis}]，与 h + b 的差在 5 倍标准误以内。
def test_e0_5_monte_carlo_consistency():
    rng = np.random.default_rng(42)
    K = 4
    lam = 1.0
    c = 1.2
    r = np.array([1, 1, 0, 0], dtype=np.float64)
    theta = np.array([[1.0, 0.5, -0.5, -1.0]])
    q_theta = np.array([[0.0, 0.0, 0.0, 0.0]])
    p = bandit.softmax(theta)
    q = bandit.softmax(q_theta)
    logp_ref = np.zeros(K)

    h = bandit.exact_npg_dir(theta, r, logp_ref, lam)
    b = estimators.tis_bias(p, q, r, c)
    target = (h + b)[0]

    N = 1_000_000
    Q_batch = np.tile(q, (N, 1))
    P_batch = np.tile(p, (N, 1))
    counts = estimators.sample_counts(rng, Q_batch, 1)
    rho_hat = estimators.reward_term("tis", counts, P_batch, Q_batch, r, c, 1)
    dirs = estimators.direction(rho_hat, np.log(P_batch), logp_ref, lam)

    mean = dirs.mean(axis=0)
    se = dirs.std(axis=0, ddof=1) / np.sqrt(N)
    assert np.all(np.abs(mean - target) <= 5 * se)


# 6. 同步退化：q = p 且 c ≥ 1 时，is 与 tis 的估计逐元素相等；b = 0。
def test_e0_6_sync_degeneracy(rng):
    K = 5
    c = 1.5
    r = np.array([1, 0, 1, 0, 0], dtype=np.float64)
    theta = rng.normal(size=(10, K))
    p = bandit.softmax(theta)
    counts = estimators.sample_counts(rng, p, 16)
    rho_is = estimators.reward_term("is", counts, p, p, r, c, 16)
    rho_tis = estimators.reward_term("tis", counts, p, p, r, c, 16)
    np.testing.assert_allclose(rho_is, rho_tis, atol=1e-12)

    b = estimators.tis_bias(p, p, r, c)
    np.testing.assert_allclose(b, 0.0, atol=1e-14)


# 7. 动量退化：omb = 1 时 m̂_t = ĝ_t；omb = 1e-6、t ≤ 1000、ĝ 为 O(1) 量级时，m̂_t 与 ĝ_0..ĝ_t 的等权平均的绝对误差 < 1e-4。
def test_e0_7_momentum_degeneracy(rng):
    T = 1000
    gs = rng.normal(size=(T, 3, 4))

    # omb = 1
    m = np.zeros((3, 4))
    m_hat_1 = []
    for t in range(T):
        m = optim.ema_step(m, gs[t], 1.0)
        den = optim.bias_correction_den(1.0, t)
        m_hat_1.append(m / den)
    np.testing.assert_allclose(np.array(m_hat_1), gs, atol=1e-14)

    # omb = 1e-6
    m = np.zeros((3, 4))
    m_hat_inf = []
    for t in range(T):
        m = optim.ema_step(m, gs[t], 1e-6)
        den = optim.bias_correction_den(1e-6, t)
        m_hat_inf.append(m / den)
    m_hat_inf = np.array(m_hat_inf)
    running_avg = np.cumsum(gs, axis=0) / np.arange(1, T + 1)[:, None, None]
    np.testing.assert_allclose(m_hat_inf, running_avg, atol=1e-4)


# 8. 偏差修正权重：Σ_j α_{t,j} = 1（误差 < 1e-12）；history_age 在 omb = 1 时为 0。
def test_e0_8_bias_correction_weights():
    assert optim.history_age(1.0, 10) == 0.0
    for omb in [1e-6, 0.01, 0.1, 0.5]:
        for t in [0, 5, 50, 300]:
            den = optim.bias_correction_den(omb, t)
            j = np.arange(t + 1)
            alpha = omb * np.exp(j * np.log1p(-omb)) / den
            assert abs(alpha.sum() - 1.0) < 1e-12


# 9. 分解恒等式：m̂_t = Ê^h + Ê^b + Ê^ξ，逐元素误差 < 1e-10。
def test_e0_9_decomposition_identity():
    cfg = {
        "problem": {"K": 8, "n_good": 2, "lam": 1.0, "init": "random", "init_seed": 1, "init_scale": 1.0, "ref": "init"},
        "run": {"R": 4, "seed": 42, "T": 40, "B": 16, "diag_every": 4, "return_traj": True},
        "estimator": "tis",
        "k": 4,
        "eta": 0.05,
        "omb": 0.1,
        "c": 1.5,
        "mismatch": "lag",
    }
    out = sim.simulate(cfg)
    th, r, lpr, lam = out["theta"], out["r"], out["logp_ref"], 1.0
    for i, t in enumerate(out["diag_t"]):
        grad = bandit.vanilla_grad(th[:, t], r, lpr, lam)
        h = bandit.exact_npg_dir(th[:, t], r, lpr, lam)
        inner_m = np.sum(grad * out["m_hat"][:, t], axis=-1)
        inner_sum = (
            np.sum(grad * h, axis=-1)
            + out["inner_track"][:, i]
            + out["inner_bias_hist"][:, i]
            + out["inner_noise"][:, i]
        )
        np.testing.assert_allclose(inner_m, inner_sum, atol=1e-10)
        np.testing.assert_allclose(out["inner_update"][:, i], inner_m, atol=1e-10)


# 10. 滞后记录：k = 3 时，t = 0..2 的 lag_t = t，t ≥ 3 的 lag_t = 3；q_t 确实等于 softmax(θ_{t−3})。
def test_e0_10_lag_recording():
    cfg = {
        "problem": {"K": 6, "n_good": 2, "lam": 1.0, "init": "random", "init_seed": 5, "init_scale": 1.0, "ref": "init"},
        "run": {"R": 2, "seed": 99, "T": 15, "B": 8, "diag_every": 5, "return_traj": True},
        "estimator": "is",
        "k": 3,
        "eta": 0.02,
        "omb": 1.0,
        "c": None,
        "mismatch": "lag",
    }
    out = sim.simulate(cfg)
    expected_lag = [0, 1, 2] + [3] * 12
    np.testing.assert_array_equal(out["lag"], expected_lag)
    th = out["theta"]
    for t in range(15):
        p_t = bandit.softmax(th[:, t])
        q_t = bandit.softmax(th[:, max(t - 3, 0)])
        kl = np.sum(p_t * (np.log(p_t) - np.log(q_t)), axis=-1)
        np.testing.assert_allclose(out["kl_pq"][:, t], kl, atol=1e-10)


# 11. 投影不变：整个训练过程中 sum(θ_t) 保持为 sum(θ_0)（误差 < 1e-9）。
def test_e0_11_projection_invariance():
    for est in ["is", "tis", "tis_debiased", "onpolicy"]:
        cfg = {
            "problem": {"K": 10, "n_good": 3, "lam": 1.0, "init": "random", "init_seed": 7, "init_scale": 2.0, "ref": "init"},
            "run": {"R": 3, "seed": 101, "T": 50, "B": 16, "diag_every": 10, "return_traj": True},
            "estimator": est,
            "k": 5,
            "eta": 0.05,
            "omb": 0.1,
            "c": 1.5,
            "mismatch": "lag",
        }
        out = sim.simulate(cfg)
        th = out["theta"]
        init_sum = th[:, 0].sum(-1, keepdims=True)
        all_sums = th.sum(-1)
        np.testing.assert_allclose(all_sums, np.broadcast_to(init_sum, all_sums.shape), atol=1e-9)
