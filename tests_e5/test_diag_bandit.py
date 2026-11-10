"""SPEC E5 §3.5: Verification tests for bandit momentum/gradient alignment estimators.

Tests:
1. H_hat is an unbiased estimator of ||E[g]||^2: at R=200, |mean(H_hat) - H_exact| < 3 * SE.
2. C2_hat vs C2_exact_mean difference < 3 * SE.
3. C1_pred vs measured C1 relative error < 10% (or absolute error < 0.02 when SNR is small).
4. Pure noise control (h = 0): both C1 and C2_hat are near 0, |mean| < 3 * SE.
"""
import numpy as np
import pytest

from stalemom.bandit import log_softmax, project
from stalemom.diag import (
    sample_grads_at,
    exact_bandit_stats,
    nested_stats,
    compute_c3,
)


def _make_test_state():
    problem = {
        "K": 10,
        "n_good": 2,
        "lam": 1.0,
        "init": "random",
        "init_seed": 0,
        "init_scale": 1.0,
        "ref": "init",
    }
    rng = np.random.default_rng(42)
    theta = project(rng.normal(scale=0.6, size=10))
    theta_old = project(theta - 0.25 * rng.normal(scale=0.5, size=10))
    q = np.exp(log_softmax(theta_old))
    # Momentum vector positively aligned with true direction
    ex = exact_bandit_stats(theta, q, np.ones(10), problem, "tis", 1.2, 32)
    m = project(0.8 * ex["mu"] + 0.2 * rng.normal(size=10))
    return theta, q, m, problem


@pytest.mark.parametrize("estimator,B", [("onpolicy", 32), ("is", 32), ("tis", 32), ("tis", 4)])
def test_1_h_hat_unbiased_for_norm_expected_grad_sq(estimator, B):
    """§3.5.1: H_hat is an unbiased estimator of ||E[g]||^2: at R=200, |mean(H_hat) - exact| < 3 * SE."""
    theta, q, m, problem = _make_test_state()
    c = 1.2 if estimator == "tis" else None
    ex = exact_bandit_stats(theta, q, m, problem, estimator, c, B)
    h_exact = ex["H_exact_mean"]

    R_trials = 200
    K = 64
    R_rep = 4
    rng = np.random.default_rng(20261110)

    h_hats = np.empty(R_trials, dtype=np.float64)
    n_bars = np.empty(R_trials, dtype=np.float64)
    inner_hats = np.empty(R_trials, dtype=np.float64)

    for i in range(R_trials):
        G = sample_grads_at(theta, q, problem, estimator, c, B, K, rng)
        st = nested_stats(G, m, R=R_rep, compute_jk=False)
        h_hats[i] = st["H_hat"]
        n_bars[i] = st["N_bar"]
        inner_hats[i] = st["inner_m_h"]

    mean_h = float(np.mean(h_hats))
    se_h = float(np.std(h_hats, ddof=1) / np.sqrt(R_trials))
    assert abs(mean_h - h_exact) < 3.0 * se_h, (
        f"H_hat biased: mean={mean_h:.6f}, exact={h_exact:.6f}, 3*SE={3*se_h:.6f}"
    )

    # Also check N_bar and inner_m_h unbiasedness
    mean_n = float(np.mean(n_bars))
    se_n = float(np.std(n_bars, ddof=1) / np.sqrt(R_trials))
    assert abs(mean_n - ex["N_exact"]) < 3.0 * se_n

    mean_in = float(np.mean(inner_hats))
    se_in = float(np.std(inner_hats, ddof=1) / np.sqrt(R_trials))
    assert abs(mean_in - ex["inner_m_mean_exact"]) < 3.0 * se_in


@pytest.mark.parametrize("estimator", ["onpolicy", "is", "tis"])
def test_2_c2_hat_matches_c2_exact_mean(estimator):
    """§3.5.2: |C2_hat - C2_exact_mean| < 3 * SE."""
    theta, q, m, problem = _make_test_state()
    c = 1.2 if estimator == "tis" else None
    B = 32
    ex = exact_bandit_stats(theta, q, m, problem, estimator, c, B)
    c2_exact = ex["C2_exact_mean"]

    R_trials = 200
    K = 64
    R_rep = 4
    rng = np.random.default_rng(102030)

    c2_hats = []
    for _ in range(R_trials):
        G = sample_grads_at(theta, q, problem, estimator, c, B, K, rng)
        st = nested_stats(G, m, R=R_rep, compute_jk=False)
        if np.isfinite(st["C2_hat"]):
            c2_hats.append(st["C2_hat"])

    c2_arr = np.asarray(c2_hats, dtype=np.float64)
    mean_c2 = float(np.mean(c2_arr))
    se_c2 = float(np.std(c2_arr, ddof=1) / np.sqrt(len(c2_arr)))
    assert abs(mean_c2 - c2_exact) < 3.0 * se_c2, (
        f"C2_hat mismatch: mean={mean_c2:.6f}, exact={c2_exact:.6f}, 3*SE={3*se_c2:.6f}"
    )


@pytest.mark.parametrize("estimator,B", [("onpolicy", 32), ("is", 32), ("tis", 32)])
def test_3_c1_pred_matches_measured_c1(estimator, B):
    """§3.5.3: C1_pred vs measured C1 relative error < 10% (or abs error < 0.02 when SNR is small)."""
    theta, q, m, problem = _make_test_state()
    c = 1.2 if estimator == "tis" else None
    ex = exact_bandit_stats(theta, q, m, problem, estimator, c, B)

    R_trials = 200
    K = 64
    R_rep = 4
    rng = np.random.default_rng(556677)

    c1_vals = np.empty(R_trials, dtype=np.float64)
    c1_preds = np.empty(R_trials, dtype=np.float64)

    for i in range(R_trials):
        G = sample_grads_at(theta, q, problem, estimator, c, B, K, rng)
        st = nested_stats(G, m, R=R_rep, compute_jk=False)
        c1_vals[i] = st["C1"]
        c1_preds[i] = st["C1_pred"]

    mean_c1 = float(np.mean(c1_vals))
    mean_pred = float(np.mean(c1_preds))
    abs_err = abs(mean_pred - mean_c1)
    rel_err = abs_err / max(abs(mean_c1), 1e-12)

    assert rel_err < 0.10 or abs_err < 0.02, (
        f"C1_pred vs C1 error too large: C1={mean_c1:.5f}, C1_pred={mean_pred:.5f}, "
        f"rel_err={rel_err:.4f}, abs_err={abs_err:.5f}, SNR={ex['SNR_exact']:.4f}"
    )


def test_4_pure_noise_control_h_equals_zero():
    """§3.5.4: Pure noise control (h = 0): both C1 and C2_hat are near 0, |mean| < 3 * SE."""
    problem = {
        "K": 10,
        "n_good": 2,
        "lam": 1.0,
        "init": "random",
        "init_seed": 0,
        "init_scale": 1.0,
        "ref": "init",
    }
    # At the optimal policy theta* = project(logp_ref + r / lam),
    # the exact NPG direction is h(theta*) = Pi[r - lam*(logp* - logp_ref)] = 0,
    # while single-batch gradients have non-zero variance!
    _, r, logp_ref, lam = (
        10,
        np.array([1.0, 1.0] + [0.0] * 8),
        log_softmax(project(np.random.default_rng(0).normal(size=10))),
        1.0,
    )
    theta_star = project(logp_ref + r / lam)
    p_star = np.exp(log_softmax(theta_star))

    rng = np.random.default_rng(999)
    m = project(rng.normal(size=10))

    ex = exact_bandit_stats(theta_star, p_star, m, problem, "onpolicy", None, 4)
    assert ex["H_exact_h"] < 1e-28
    assert ex["H_exact_mean"] < 1e-28
    assert ex["var_exact"] > 0.1  # Genuine stochastic noise with zero mean

    R_trials = 200
    K = 64
    R_rep = 4

    c1_vals = []
    c2_vals = []
    h_vals = []

    for _ in range(R_trials):
        m_r = project(rng.normal(size=10))
        G = sample_grads_at(theta_star, p_star, problem, "onpolicy", None, 4, K, rng)
        st = nested_stats(G, m_r, R=R_rep, compute_jk=False)
        c1_vals.append(st["C1"])
        h_vals.append(st["H_hat"])
        if np.isfinite(st["C2_hat"]):
            c2_vals.append(st["C2_hat"])

    c1_arr = np.asarray(c1_vals, dtype=np.float64)
    c2_arr = np.asarray(c2_vals, dtype=np.float64)
    h_arr = np.asarray(h_vals, dtype=np.float64)

    mean_c1 = float(np.mean(c1_arr))
    se_c1 = float(np.std(c1_arr, ddof=1) / np.sqrt(len(c1_arr)))
    assert abs(mean_c1) < 3.0 * se_c1, f"Pure noise C1 not near 0: {mean_c1} vs 3*SE={3*se_c1}"

    assert len(c2_arr) >= 50, f"Expected ~half of pure noise trials to have H_hat > 0, got {len(c2_arr)}"
    mean_c2 = float(np.mean(c2_arr))
    se_c2 = float(np.std(c2_arr, ddof=1) / np.sqrt(len(c2_arr)))
    assert abs(mean_c2) < 3.0 * se_c2, f"Pure noise C2_hat not near 0: {mean_c2} vs 3*SE={3*se_c2}"

    mean_h = float(np.mean(h_arr))
    se_h = float(np.std(h_arr, ddof=1) / np.sqrt(len(h_arr)))
    assert abs(mean_h) < 3.0 * se_h


def test_jackknife_and_c3():
    """Verify jackknife SEs and cross-checkpoint C3 estimator."""
    theta_1, q_1, m_1, problem = _make_test_state()
    theta_2 = project(theta_1 * 0.5)
    q_2 = np.exp(log_softmax(theta_2))

    rng = np.random.default_rng(77)
    G1 = sample_grads_at(theta_1, q_1, problem, "onpolicy", None, 32, 64, rng)
    G2 = sample_grads_at(theta_2, q_2, problem, "onpolicy", None, 32, 64, rng)

    st1 = nested_stats(G1, m_1, R=4, compute_jk=True)
    st2 = nested_stats(G2, m_1, R=4, compute_jk=True)

    assert st1["C1_jk_se"] > 0 and np.isfinite(st1["C1_jk_se"])
    assert st1["C2_hat_jk_se"] > 0 and np.isfinite(st1["C2_hat_jk_se"])
    assert st1["H_hat_jk_se"] > 0 and np.isfinite(st1["H_hat_jk_se"])

    c3_self = compute_c3(st1, st1)
    assert np.isclose(c3_self, 1.0, atol=1e-12)

    c3_cross = compute_c3(st1, st2)
    ex1 = exact_bandit_stats(theta_1, q_1, m_1, problem, "onpolicy", None, 32)
    ex2 = exact_bandit_stats(theta_2, q_2, m_1, problem, "onpolicy", None, 32)
    c3_exact = float(np.dot(ex1["mu"], ex2["mu"]) / (np.linalg.norm(ex1["mu"]) * np.linalg.norm(ex2["mu"])))
    assert abs(c3_cross - c3_exact) < 0.05


def test_c1_pred_in_minus_one_to_one_bounds():
    """Verify RMS C1_pred = mean(d) / (||m|| * sqrt(mean(n))) is strictly in [-1, 1] and C1_pred_legacy is preserved."""
    theta, q, m, problem = _make_test_state()
    rng = np.random.default_rng(2026)

    for estimator in ["onpolicy", "is", "tis"]:
        c = 1.2 if estimator == "tis" else None
        for B in [1, 4, 32]:
            for _ in range(25):
                m_rand = project(rng.normal(size=10))
                G = sample_grads_at(theta, q, problem, estimator, c, B, 64, rng)
                st = nested_stats(G, m_rand, R=4, compute_jk=False)
                assert -1.0 - 1e-12 <= st["C1_pred"] <= 1.0 + 1e-12, (
                    f"C1_pred={st['C1_pred']} out of [-1, 1] bounds"
                )
                assert "C1_pred_legacy" in st and np.isfinite(st["C1_pred_legacy"])

    # Extreme case: collinear gradients with varying norms where C1_pred_legacy > 1 but C1_pred <= 1
    m_unit = np.zeros(10)
    m_unit[0] = 1.0
    m_unit[1] = -1.0
    m_unit = m_unit / np.linalg.norm(m_unit)
    scales = np.array([0.01] * 32 + [10.0] * 32)[:, None]
    G_collinear = scales * m_unit[None, :]
    st_ext = nested_stats(G_collinear, m_unit, R=4, compute_jk=False)
    assert -1.0 <= st_ext["C1_pred"] <= 1.0
    assert st_ext["C1_pred_legacy"] > 1.0

