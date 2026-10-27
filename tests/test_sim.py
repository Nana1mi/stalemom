import numpy as np
import pytest

from stalemom import sim
from conftest import (ref_log_softmax, ref_softmax, ref_objective, ref_optimal_value,
                      ref_vanilla_grad, ref_npg_dir, ref_tis_bias, ref_kl)


def make_cfg(estimator, k=0, eta=0.05, omb=1.0, c=2.0, R=4, T=64, B=8, init="random",
             seed=0, K=10, n_good=2, lam=1.0, diag_every=4, return_traj=True):
    return {
        "problem": {"K": K, "n_good": n_good, "lam": lam, "init": init, "init_seed": 3,
                    "init_scale": 1.0, "ref": "init"},
        "run": {"R": R, "seed": seed, "T": T, "B": B, "diag_every": diag_every,
                "return_traj": return_traj},
        "estimator": estimator, "k": k, "eta": eta, "omb": omb, "c": c, "mismatch": "lag",
    }


def _ref_momentum(gs, omb):
    m = np.zeros_like(gs[:, 0])
    out = np.zeros_like(gs)
    for t in range(gs.shape[1]):
        m = (1 - omb) * m + omb * gs[:, t]
        den = -np.expm1((t + 1) * np.log1p(-omb)) if omb < 1 else 1.0
        out[:, t] = m / den
    return out


# ---------- 输出结构 ----------

def test_output_keys_and_shapes():
    cfg = make_cfg("tis", k=4, R=3, T=40, diag_every=8)
    out = sim.simulate(cfg)
    R, T, K = 3, 40, 10
    assert out["gap"].shape == (R, T + 1) and out["J"].shape == (R, T + 1)
    assert out["kl_pq"].shape == (R, T) and out["trunc_rate"].shape == (R, T)
    assert out["lag"].shape == (T,)
    np.testing.assert_array_equal(out["diag_t"], np.arange(0, T, 8))
    n = len(out["diag_t"])
    for key in ["bias_norm", "inner_update", "inner_track", "inner_bias_hist", "inner_noise",
                "norm_track", "norm_bias_hist", "norm_noise"]:
        assert out[key].shape == (R, n), key
    assert out["H"].shape == (n,)
    assert out["diverged"].shape == (R,) and out["diverged_step"].shape == (R,)
    assert out["theta"].shape == (R, T + 1, K)
    assert out["g"].shape == (R, T, K) and out["m_hat"].shape == (R, T, K)


def test_reward_reference_and_initial_state():
    out = sim.simulate(make_cfg("is", k=2))
    np.testing.assert_array_equal(out["r"], [1, 1] + [0] * 8)
    theta0 = out["theta"][:, 0]
    assert np.allclose(theta0, theta0[0:1])                       # 所有运行同一初值
    np.testing.assert_allclose(theta0.sum(-1), 0.0, atol=1e-12)  # random 初值已投影
    np.testing.assert_allclose(out["logp_ref"], ref_log_softmax(theta0[0]), atol=1e-12)
    assert np.isclose(out["J_star"], ref_optimal_value(out["r"], out["logp_ref"], 1.0), atol=1e-12)


def test_uniform_init():
    out = sim.simulate(make_cfg("is", k=2, init="uniform"))
    np.testing.assert_allclose(out["theta"][:, 0], 0.0, atol=1e-15)


# ---------- 轨迹自洽性 ----------

@pytest.mark.parametrize("estimator,k,omb", [("is", 3, 0.2), ("tis", 5, 0.05), ("onpolicy", 0, 1.0),
                                             ("tis_debiased", 4, 0.5)])
def test_J_gap_and_updates_are_consistent(estimator, k, omb):
    eta = 0.05
    out = sim.simulate(make_cfg(estimator, k=k, eta=eta, omb=omb, c=1.3))
    th, r, lpr = out["theta"], out["r"], out["logp_ref"]
    J = ref_objective(th, r, lpr, 1.0)
    np.testing.assert_allclose(out["J"], J, atol=1e-10)
    gap = (out["J_star"] - J) / (out["J_star"] - J[:, :1])
    np.testing.assert_allclose(out["gap"], gap, atol=1e-10)
    np.testing.assert_allclose(out["gap"][:, 0], 1.0, atol=1e-12)
    # θ_{t+1} = θ_t + η m̂_t
    np.testing.assert_allclose(th[:, 1:] - th[:, :-1], eta * out["m_hat"], atol=1e-11)
    # m̂ 由 g 按带偏差修正的 EMA 得到
    np.testing.assert_allclose(out["m_hat"], _ref_momentum(out["g"], omb), atol=1e-11)
    # sum(θ) 不变
    np.testing.assert_allclose(th.sum(-1) - th[:, :1].sum(-1), 0.0, atol=1e-9)


@pytest.mark.parametrize("k", [0, 1, 3, 7])
def test_lag_and_behavior_policy(k):
    T = 30
    out = sim.simulate(make_cfg("is", k=k, T=T))
    np.testing.assert_array_equal(out["lag"], [t - max(t - k, 0) for t in range(T)])
    th = out["theta"]
    for t in range(T):
        p = ref_softmax(th[:, t])
        q = ref_softmax(th[:, max(t - k, 0)])
        np.testing.assert_allclose(out["kl_pq"][:, t], ref_kl(p, q), atol=1e-10)


def test_onpolicy_and_exact_have_zero_mismatch():
    for est in ["onpolicy", "exact"]:
        out = sim.simulate(make_cfg(est, k=5))
        np.testing.assert_allclose(out["kl_pq"], 0.0, atol=1e-14)
        np.testing.assert_allclose(out["trunc_rate"], 0.0, atol=1e-14)
        np.testing.assert_array_equal(out["lag"], 0)


def test_deterministic_modes_force_single_run():
    for est in ["exact", "tis_mean"]:
        out = sim.simulate(make_cfg(est, k=3, R=8))
        assert out["gap"].shape[0] == 1


def test_reproducible_with_same_seed():
    a = sim.simulate(make_cfg("tis", k=3, seed=11))
    b = sim.simulate(make_cfg("tis", k=3, seed=11))
    np.testing.assert_array_equal(a["theta"], b["theta"])
    c = sim.simulate(make_cfg("tis", k=3, seed=12))
    assert not np.allclose(a["theta"], c["theta"])


# ---------- 确定性轨迹：与独立参考实现逐步对比 ----------

def _ref_trajectory(estimator, k, eta, omb, c, T, theta0, r, lpr, lam):
    th = [theta0.copy()]
    m = np.zeros_like(theta0)
    for t in range(T):
        cur = th[-1]
        h = ref_npg_dir(cur, r, lpr, lam)
        if estimator == "exact":
            g = h
        else:  # tis_mean
            p = ref_softmax(cur)
            q = ref_softmax(th[max(t - k, 0)])
            g = h + ref_tis_bias(p, q, r, c)
        m = (1 - omb) * m + omb * g
        den = -np.expm1((t + 1) * np.log1p(-omb)) if omb < 1 else 1.0
        th.append(cur + eta * m / den)
    return np.stack(th, axis=1)


def test_exact_trajectory_matches_reference():
    eta, omb, T = 0.05, 0.05 / 0.25, 80
    out = sim.simulate(make_cfg("exact", eta=eta, omb=omb, T=T))
    ref = _ref_trajectory("exact", 0, eta, omb, None, T, out["theta"][:, 0], out["r"], out["logp_ref"], 1.0)
    np.testing.assert_allclose(out["theta"], ref, atol=1e-10)
    assert out["gap"][0, -1] < out["gap"][0, 0]      # 确实在优化


def test_tis_mean_lag_trajectory_matches_reference():
    eta, omb, k, c, T = 0.3, 0.1, 6, 1.2, 80
    out = sim.simulate(make_cfg("tis_mean", k=k, eta=eta, omb=omb, c=c, T=T))
    th0, r, lpr = out["theta"][:, 0], out["r"], out["logp_ref"]
    ref = _ref_trajectory("tis_mean", k, eta, omb, c, T, th0, r, lpr, 1.0)
    np.testing.assert_allclose(out["theta"], ref, atol=1e-10)
    # 保证截断在这条轨迹上确实发生过，否则本测试没有意义
    biases = [np.abs(ref_tis_bias(ref_softmax(ref[:, t]), ref_softmax(ref[:, max(t - k, 0)]), r, c)).max()
              for t in range(T)]
    assert max(biases) > 1e-3


# ---------- 诊断量 ----------

@pytest.mark.parametrize("estimator,omb", [("is", 0.2), ("tis", 0.1), ("tis_debiased", 0.5)])
def test_diagnostics_decomposition(estimator, omb):
    k, c, eta = 4, 1.3, 0.1
    out = sim.simulate(make_cfg(estimator, k=k, eta=eta, omb=omb, c=c, T=48, diag_every=4))
    th, r, lpr = out["theta"], out["r"], out["logp_ref"]
    for i, t in enumerate(out["diag_t"]):
        grad = ref_vanilla_grad(th[:, t], r, lpr, 1.0)
        h = ref_npg_dir(th[:, t], r, lpr, 1.0)
        np.testing.assert_allclose(out["inner_update"][:, i], (grad * out["m_hat"][:, t]).sum(-1), atol=1e-10)
        total = (grad * h).sum(-1) + out["inner_track"][:, i] + out["inner_bias_hist"][:, i] + out["inner_noise"][:, i]
        np.testing.assert_allclose(out["inner_update"][:, i], total, atol=1e-9)
        if estimator == "tis":
            b = ref_tis_bias(ref_softmax(th[:, t]), ref_softmax(th[:, max(t - k, 0)]), r, c)
            np.testing.assert_allclose(out["bias_norm"][:, i], np.linalg.norm(b, axis=-1), atol=1e-10)
        else:
            np.testing.assert_allclose(out["bias_norm"][:, i], 0.0, atol=1e-14)
            np.testing.assert_allclose(out["inner_bias_hist"][:, i], 0.0, atol=1e-14)


def test_history_age_output():
    out = sim.simulate(make_cfg("is", k=2, omb=0.1, T=40, diag_every=10))
    for i, t in enumerate(out["diag_t"]):
        j = np.arange(t + 1)
        w = 0.1 * 0.9 ** j / (-np.expm1((t + 1) * np.log1p(-0.1)))
        assert np.isclose(out["H"][i], (j * w).sum(), rtol=1e-10, atol=1e-12)
