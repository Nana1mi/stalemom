import numpy as np
import pytest

from stalemom import bandit, estimators
from conftest import ref_npg_dir, ref_tis_bias, onehot_counts


def _expect_over_one_sample(mode, prob_sample, p, q, r, c, K):
    """对单样本（B=1）的所有可能动作精确枚举，返回奖励项估计的期望。"""
    out = np.zeros((1, K))
    for a, cnt in enumerate(onehot_counts(K)):
        out += prob_sample[0, a] * estimators.reward_term(mode, cnt, p, q, r, c, 1)
    return out


# ---------- 采样 ----------

def test_sample_counts_basic(rng):
    q = np.tile(np.array([[0.5, 0.2, 0.2, 0.1]]), (3000, 1))
    counts = estimators.sample_counts(rng, q, 40)
    assert counts.shape == (3000, 4)
    assert np.issubdtype(counts.dtype, np.integer)
    assert np.all(counts >= 0)
    assert np.all(counts.sum(-1) == 40)
    freq = counts / 40.0
    se = np.sqrt(q[0] * (1 - q[0]) / 40.0 / 3000)
    assert np.all(np.abs(freq.mean(0) - q[0]) < 5 * se)


def test_sample_counts_rows_are_independent_distributions(rng):
    q = np.array([[0.98, 0.01, 0.01], [0.01, 0.01, 0.98]])
    counts = estimators.sample_counts(rng, q, 1000)
    assert counts[0, 0] > 900 and counts[1, 2] > 900


# ---------- 估计器公式 ----------

def test_reward_term_formulas(problem, rng):
    K, r, p, q = problem["K"], problem["r"], problem["p"], problem["q"]
    B, c = 6, 1.3
    counts = estimators.sample_counts(rng, np.tile(q, (5, 1)), B)
    pp, qq = np.tile(p, (5, 1)), np.tile(q, (5, 1))
    np.testing.assert_allclose(estimators.reward_term("is", counts, pp, qq, r, c, B),
                               counts / B * r / qq, atol=1e-12)
    np.testing.assert_allclose(estimators.reward_term("onpolicy", counts, pp, qq, r, c, B),
                               counts / B * r / pp, atol=1e-12)
    np.testing.assert_allclose(estimators.reward_term("tis", counts, pp, qq, r, c, B),
                               counts / B * np.minimum(pp / qq, c) * r / pp, atol=1e-12)
    np.testing.assert_allclose(estimators.reward_term("tis_debiased", counts, pp, qq, r, c, B),
                               counts / B * np.minimum(pp / qq, c) * r / pp
                               - r * (np.minimum(1.0, c * qq / pp) - 1.0), atol=1e-12)


def test_reward_term_is_mean_over_samples(problem):
    """B>1 时估计量必须是单样本估计的平均（线性、无批内归一化）。"""
    K, r, p, q = problem["K"], problem["r"], problem["p"], problem["q"]
    actions = [0, 2, 2, 5, 3]
    counts = np.zeros((1, K), dtype=np.int64)
    for a in actions:
        counts[0, a] += 1
    ones = onehot_counts(K)
    for mode in ["is", "tis", "onpolicy", "tis_debiased"]:
        single = np.mean([estimators.reward_term(mode, ones[a], p, q, r, 1.4, 1) for a in actions], axis=0)
        np.testing.assert_allclose(estimators.reward_term(mode, counts, p, q, r, 1.4, len(actions)),
                                   single, atol=1e-12)


def test_direction_formula(problem, rng):
    K, lpr, lam = problem["K"], problem["logp_ref"], problem["lam"]
    rho_hat = rng.normal(size=(3, K))
    logp = np.log(np.tile(problem["p"], (3, 1)))
    d = estimators.direction(rho_hat, logp, lpr, lam)
    v = rho_hat - lam * (logp - lpr)
    np.testing.assert_allclose(d, v - v.mean(-1, keepdims=True), atol=1e-12)


# ---------- 无偏性与偏差（精确枚举） ----------

def test_is_estimator_is_unbiased_by_enumeration(problem):
    K, r, p, q, lpr, lam = (problem[k] for k in ["K", "r", "p", "q", "logp_ref", "lam"])
    E_rho = _expect_over_one_sample("is", q, p, q, r, None, K)
    np.testing.assert_allclose(E_rho, r[None, :], atol=1e-10, rtol=1e-8)
    E_dir = estimators.direction(E_rho, np.log(p), lpr, lam)   # direction 对 ρ̂ 线性
    h = bandit.exact_npg_dir(problem["theta"], r, lpr, lam)
    np.testing.assert_allclose(E_dir, h, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(E_dir, ref_npg_dir(problem["theta"], r, lpr, lam), atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("c", [1.05, 1.5, 2.0, 5.0])
def test_tis_bias_closed_form_by_enumeration(problem, c):
    K, r, p, q, lpr, lam = (problem[k] for k in ["K", "r", "p", "q", "logp_ref", "lam"])
    E_rho = _expect_over_one_sample("tis", q, p, q, r, c, K)
    E_dir = estimators.direction(E_rho, np.log(p), lpr, lam)
    h = ref_npg_dir(problem["theta"], r, lpr, lam)
    b = estimators.tis_bias(p, q, r, c)
    np.testing.assert_allclose(E_dir - h, b, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(b, ref_tis_bias(p, q, r, c), atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(estimators.expected_reward_term("tis", p, q, r, c), E_rho, atol=1e-10, rtol=1e-8)


def test_this_problem_actually_truncates(problem):
    """保证上面的偏差测试不是空测试：至少有一个好动作 p/q > 1.5。"""
    ratio = problem["p"] / problem["q"]
    assert np.any((ratio > 1.5) & (problem["r"][None, :] > 0))


def test_tis_debiased_is_unbiased_by_enumeration(problem):
    K, r, p, q = problem["K"], problem["r"], problem["p"], problem["q"]
    E_rho = _expect_over_one_sample("tis_debiased", q, p, q, r, 1.5, K)
    np.testing.assert_allclose(E_rho, r[None, :], atol=1e-10, rtol=1e-8)


def test_onpolicy_is_unbiased_when_sampling_from_p(problem):
    K, r, p, q = problem["K"], problem["r"], problem["p"], problem["q"]
    E_rho = _expect_over_one_sample("onpolicy", p, p, q, r, None, K)
    np.testing.assert_allclose(E_rho, r[None, :], atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("mode", ["is", "onpolicy", "tis_debiased"])
def test_expected_reward_term_returns_r(problem, mode):
    got = estimators.expected_reward_term(mode, problem["p"], problem["q"], problem["r"], 1.5)
    np.testing.assert_allclose(np.broadcast_to(got, (1, problem["K"])), problem["r"][None, :], atol=1e-12)


# ---------- 蒙特卡洛一致性 ----------

def test_tis_monte_carlo_matches_h_plus_b(problem):
    K, r, p, q, lpr, lam = (problem[k] for k in ["K", "r", "p", "q", "logp_ref", "lam"])
    c, R = 1.5, 400_000
    rng = np.random.default_rng(7)
    P, Q = np.tile(p, (R, 1)), np.tile(q, (R, 1))
    counts = estimators.sample_counts(rng, Q, 1)
    dirs = estimators.direction(estimators.reward_term("tis", counts, P, Q, r, c, 1), np.log(P), lpr, lam)
    mean, se = dirs.mean(0), dirs.std(0, ddof=1) / np.sqrt(R)
    target = (ref_npg_dir(problem["theta"], r, lpr, lam) + ref_tis_bias(p, q, r, c))[0]
    assert np.all(np.abs(mean - target) <= 5 * se + 1e-12), (mean, target, se)


# ---------- 同步退化与截断率 ----------

@pytest.mark.parametrize("c", [1.0, 1.5, 3.0])
def test_sync_degeneracy(problem, rng, c):
    K, r, p = problem["K"], problem["r"], problem["p"]
    P = np.tile(p, (4, 1))
    counts = estimators.sample_counts(rng, P, 5)
    np.testing.assert_allclose(estimators.reward_term("tis", counts, P, P, r, c, 5),
                               estimators.reward_term("is", counts, P, P, r, c, 5), atol=1e-12)
    np.testing.assert_allclose(estimators.tis_bias(P, P, r, c), 0.0, atol=1e-14)


def test_truncation_rate():
    p = np.array([[0.4, 0.3, 0.2, 0.1]])
    q = np.array([[0.1, 0.3, 0.4, 0.2]])      # p/q = 4, 1, 0.5, 0.5
    counts = np.array([[3, 2, 4, 1]])          # B = 10
    np.testing.assert_allclose(estimators.truncation_rate(counts, p, q, 2.0, 10), [0.3], atol=1e-14)
    np.testing.assert_allclose(estimators.truncation_rate(counts, p, q, 0.9, 10), [0.5], atol=1e-14)
    # p/q 恰好等于 c 不算截断
    np.testing.assert_allclose(estimators.truncation_rate(counts, p, q, 4.0, 10), [0.0], atol=1e-14)
