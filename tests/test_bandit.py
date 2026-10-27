import numpy as np
import pytest

from stalemom import bandit
from conftest import (ref_log_softmax, ref_softmax, ref_objective, ref_optimal_value,
                      ref_vanilla_grad, ref_npg_dir)


def test_softmax_and_log_softmax(rng):
    theta = rng.normal(scale=3.0, size=(5, 9))
    p = bandit.softmax(theta)
    lp = bandit.log_softmax(theta)
    assert p.shape == (5, 9) and lp.shape == (5, 9)
    assert p.dtype == np.float64
    np.testing.assert_allclose(p.sum(-1), 1.0, atol=1e-14)
    np.testing.assert_allclose(lp, ref_log_softmax(theta), atol=1e-12)
    np.testing.assert_allclose(p, np.exp(lp), atol=1e-14)


def test_log_softmax_is_numerically_stable():
    theta = np.array([[1000.0, 999.0, -1000.0], [-800.0, -801.0, -799.0]])
    lp = bandit.log_softmax(theta)
    assert np.all(np.isfinite(lp))
    np.testing.assert_allclose(lp, ref_log_softmax(theta), atol=1e-12)


def test_project(rng):
    v = rng.normal(size=(4, 6))
    pv = bandit.project(v)
    np.testing.assert_allclose(pv.mean(-1), 0.0, atol=1e-14)
    np.testing.assert_allclose(bandit.project(pv), pv, atol=1e-14)   # 幂等
    np.testing.assert_allclose(pv, v - v.mean(-1, keepdims=True), atol=1e-14)


def test_objective_matches_definition(problem, rng):
    theta = rng.normal(scale=2.0, size=(6, problem["K"]))
    J = bandit.objective(theta, problem["r"], problem["logp_ref"], problem["lam"])
    assert J.shape == (6,)
    np.testing.assert_allclose(J, ref_objective(theta, problem["r"], problem["logp_ref"], problem["lam"]),
                               atol=1e-12)


def test_optimal_value_and_maximizer(problem, rng):
    r, lpr, lam = problem["r"], problem["logp_ref"], problem["lam"]
    Jstar = bandit.optimal_value(r, lpr, lam)
    assert np.isclose(Jstar, ref_optimal_value(r, lpr, lam), atol=1e-12)
    theta_star = (lpr + r / lam)[None, :]
    theta_star = theta_star - theta_star.mean()
    assert np.isclose(bandit.objective(theta_star, r, lpr, lam)[0], Jstar, atol=1e-12)
    # 任意点不超过最优值
    theta = rng.normal(scale=3.0, size=(200, problem["K"]))
    assert np.all(bandit.objective(theta, r, lpr, lam) <= Jstar + 1e-12)


def test_exact_npg_dir_definition_and_fixed_point(problem, rng):
    r, lpr, lam = problem["r"], problem["logp_ref"], problem["lam"]
    theta = rng.normal(scale=2.0, size=(5, problem["K"]))
    h = bandit.exact_npg_dir(theta, r, lpr, lam)
    np.testing.assert_allclose(h, ref_npg_dir(theta, r, lpr, lam), atol=1e-12)
    np.testing.assert_allclose(h.sum(-1), 0.0, atol=1e-12)
    theta_star = (lpr + r / lam)[None, :]
    np.testing.assert_allclose(bandit.exact_npg_dir(theta_star, r, lpr, lam), 0.0, atol=1e-12)


def test_npg_equals_fisher_pseudoinverse_times_gradient(problem):
    """独立途径：NPG 方向 = F⁺ ∇J（伪逆只在测试里用，实现里禁止用）。"""
    r, lpr, lam, theta = problem["r"], problem["logp_ref"], problem["lam"], problem["theta"]
    p = ref_softmax(theta)[0]
    F = np.diag(p) - np.outer(p, p)
    grad = ref_vanilla_grad(theta, r, lpr, lam)[0]
    expected = np.linalg.pinv(F, rcond=1e-12) @ grad
    got = bandit.exact_npg_dir(theta, r, lpr, lam)[0]
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_vanilla_grad_matches_finite_differences(problem):
    r, lpr, lam, theta = problem["r"], problem["logp_ref"], problem["lam"], problem["theta"]
    g = bandit.vanilla_grad(theta, r, lpr, lam)[0]
    np.testing.assert_allclose(g, ref_vanilla_grad(theta, r, lpr, lam)[0], atol=1e-12)
    eps = 1e-6
    fd = np.zeros(problem["K"])
    for i in range(problem["K"]):
        e = np.zeros_like(theta)
        e[0, i] = eps
        fd[i] = (ref_objective(theta + e, r, lpr, lam)[0] - ref_objective(theta - e, r, lpr, lam)[0]) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-9)
