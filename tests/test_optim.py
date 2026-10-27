from fractions import Fraction

import numpy as np
import pytest

from stalemom import optim


def test_omb_from_rho():
    assert optim.omb_from_rho(0.01, 0.0) == 1.0
    assert np.isclose(optim.omb_from_rho(0.01, 0.25), 0.04, rtol=0, atol=1e-15)
    assert np.isclose(optim.omb_from_rho(0.01, 0.01), 1.0, atol=1e-15)
    with pytest.raises(ValueError):
        optim.omb_from_rho(0.1, 0.05)


@pytest.mark.parametrize("omb", [1e-6, 1e-3, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("t", [0, 1, 10, 300])
def test_bias_correction_den_is_exact(omb, t):
    exact = Fraction(1) - (Fraction(1) - Fraction(omb)) ** (t + 1)
    got = optim.bias_correction_den(omb, t)
    assert abs(got - float(exact)) <= 1e-13 * float(exact)


def test_ema_step_formula(rng):
    m, g = rng.normal(size=(3, 5)), rng.normal(size=(3, 5))
    np.testing.assert_allclose(optim.ema_step(m, g, 0.3), 0.7 * m + 0.3 * g, atol=1e-15)


def _weights(omb, t):
    j = np.arange(t + 1)
    return omb * (1 - omb) ** j / optim.bias_correction_den(omb, t)


@pytest.mark.parametrize("omb", [1e-6, 0.01, 0.1, 1.0])
@pytest.mark.parametrize("t", [0, 5, 200])
def test_history_weights_and_age(omb, t):
    w = _weights(omb, t)
    assert abs(w.sum() - 1.0) < 1e-12
    expected_age = float((np.arange(t + 1) * w).sum())
    assert np.isclose(optim.history_age(omb, t), expected_age, rtol=1e-10, atol=1e-12)


def test_history_age_limits():
    assert optim.history_age(1.0, 50) == 0.0
    assert np.isclose(optim.history_age(0.1, 3000), 0.9 / 0.1, rtol=1e-8)   # 稳态 (1-omb)/omb
    assert np.isclose(optim.history_age(1e-6, 99), 49.5, rtol=1e-4)        # 近似等权：(t)/2


def _momentum_sequence(gs, omb):
    m = np.zeros_like(gs[0])
    out = []
    for t, g in enumerate(gs):
        m = optim.ema_step(m, g, omb)
        out.append(m / optim.bias_correction_den(omb, t))
    return np.array(out)


def test_momentum_degenerates_without_memory(rng):
    gs = rng.normal(size=(40, 2, 4))
    np.testing.assert_allclose(_momentum_sequence(gs, 1.0), gs, atol=1e-15)


def test_long_memory_is_running_average(rng):
    gs = rng.normal(size=(1000, 2, 4)) + 1.0
    mh = _momentum_sequence(gs, 1e-6)
    running = np.cumsum(gs, axis=0) / np.arange(1, 1001)[:, None, None]
    # g 为 O(1) 量级时，1e-6 衰减的 EMA 与等权平均的真实差距约 1e-5，用绝对误差判断
    np.testing.assert_allclose(mh, running, rtol=0, atol=1e-4)


def test_momentum_equals_weighted_history(rng):
    gs = rng.normal(size=(30, 1, 3))
    omb = 0.2
    mh = _momentum_sequence(gs, omb)
    for t in [0, 7, 29]:
        w = _weights(omb, t)[::-1]           # w[j] 作用于 g_{t-j}，翻转后与 g_0..g_t 对齐
        np.testing.assert_allclose(mh[t], np.tensordot(w, gs[: t + 1], axes=1), atol=1e-13)
