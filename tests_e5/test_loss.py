"""Unit tests for LLM GRPO-TIS loss, weights, rewards, and online diagnostics (SPEC E5 §6, §12)."""

import numpy as np
import pytest

from llm.reward import (
    extract_boxed_answer,
    extract_gsm8k_ground_truth,
    compute_exact_match_reward,
    compute_grpo_advantages,
)
from llm.train import (
    compute_token_weights_numpy,
    compute_grpo_tis_loss_and_grad_numpy,
    StalenessBuffer,
    compute_online_c1,
)
from llm.diag import OnlineLLMDiagAccumulator
from stalemom.diag import nested_stats


def test_k0_forces_weights_strictly_one():
    """SPEC E5 §6.2 & §12: when k=0, w_tok must be strictly 1.0 even if log_pi_theta != log_pi_behav."""
    rng = np.random.default_rng(101)
    log_p = rng.normal(-2.0, 1.0, size=(8, 16))
    log_q = log_p + rng.normal(0.0, 0.5, size=(8, 16))

    for c_val in [1.2, 1.5, float("inf")]:
        w, stats = compute_token_weights_numpy(log_p, log_q, k=0, c=c_val)
        np.testing.assert_allclose(w, 1.0, rtol=0.0, atol=0.0)
        assert stats["trunc_rate"] == 0.0
        assert stats["mean_w"] == 1.0
        assert stats["approx_kl"] == 0.0


def test_c_inf_unclipped_exact_ratio():
    """SPEC E5 §6.2 & §12: when k>0 and c=inf, w_tok must equal the exact ratio pi_theta / pi_behav."""
    rng = np.random.default_rng(102)
    log_p = rng.normal(-2.0, 0.8, size=(8, 16))
    log_q = rng.normal(-2.0, 0.8, size=(8, 16))
    expected_ratio = np.exp(log_p - log_q)

    w, stats = compute_token_weights_numpy(log_p, log_q, k=4, c=float("inf"))
    np.testing.assert_allclose(w, expected_ratio, rtol=1e-12, atol=1e-12)
    assert stats["trunc_rate"] == 0.0
    assert np.isclose(stats["mean_w"], np.mean(expected_ratio))


def test_c_finite_truncates_ratio():
    """SPEC E5 §6.2: when k>0 and c=1.2, w_tok = min(ratio, 1.2) and trunc_rate is recorded."""
    log_q = np.zeros((2, 4))
    # Ratios: [0.5, 1.0, 1.5, 2.0]
    ratios = np.array([[0.5, 1.0, 1.5, 2.0], [0.8, 1.2, 1.25, 3.0]])
    log_p = np.log(ratios)

    w, stats = compute_token_weights_numpy(log_p, log_q, k=2, c=1.2)
    np.testing.assert_allclose(w, np.minimum(ratios, 1.2), rtol=1e-12, atol=1e-12)
    # 4 out of 8 tokens exceed 1.2 (1.5, 2.0, 1.25, 3.0)
    assert np.isclose(stats["trunc_rate"], 4.0 / 8.0)


def test_zero_advantage_group_produces_zero_gradient():
    """SPEC E5 §6.2 & §12: std=0 groups must have advantage == 0 and exact zero gradient."""
    # 3 prompt groups of size 4:
    # Group 0: all 0s (std=0)
    # Group 1: all 1s (std=0)
    # Group 2: mixed [1, 0, 1, 0] (std>0)
    rewards = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    adv, zero_adv_frac = compute_grpo_advantages(rewards, group_size=4)
    assert np.isclose(zero_adv_frac, 2.0 / 3.0)
    np.testing.assert_array_equal(adv[0], np.zeros(4))
    np.testing.assert_array_equal(adv[1], np.zeros(4))
    assert not np.allclose(adv[2], 0.0)

    # Now verify gradient for the zero-advantage groups is strictly 0
    rng = np.random.default_rng(103)
    log_p = rng.normal(-1.5, 0.5, size=(12, 10))
    log_q = rng.normal(-1.5, 0.5, size=(12, 10))
    loss, grad_log_p, w, _ = compute_grpo_tis_loss_and_grad_numpy(
        log_p, log_q, adv.reshape(-1), k=4, c=1.2
    )
    # Rows 0..7 belong to Group 0 and Group 1 (zero advantage)
    np.testing.assert_array_equal(grad_log_p[:8], np.zeros((8, 10)))
    # Rows 8..11 belong to Group 2 (non-zero advantage)
    assert np.all(np.abs(grad_log_p[8:]) > 0.0)

    # If ALL groups have std=0, total loss and entire gradient tensor must be strictly 0
    all_zero_adv, frac_all = compute_grpo_advantages(np.ones((2, 4)), group_size=4)
    assert frac_all == 1.0
    loss_z, grad_z, _, _ = compute_grpo_tis_loss_and_grad_numpy(
        log_p[:8], log_q[:8], all_zero_adv.reshape(-1), k=4, c=1.2
    )
    assert loss_z == 0.0
    np.testing.assert_array_equal(grad_z, np.zeros_like(grad_z))


def test_stop_gradient_on_w_tok_matches_finite_difference():
    """Verify that dL/d(log_pi_theta) treats w_tok as stop-gradient constant."""
    rng = np.random.default_rng(104)
    log_p = rng.normal(-1.0, 0.4, size=(4, 6))
    log_q = rng.normal(-1.0, 0.4, size=(4, 6))
    adv = np.array([1.0, -1.0, 0.5, -0.5])

    loss, grad_log_p, w, _ = compute_grpo_tis_loss_and_grad_numpy(
        log_p, log_q, adv, k=4, c=1.2
    )
    # Holding w fixed (stop-gradient), check finite difference w.r.t. log_p[0, 0]
    eps = 1e-6
    log_p_plus = log_p.copy()
    log_p_plus[0, 0] += eps
    n_tok = log_p.size
    loss_plus = -np.sum(w * adv[:, None] * log_p_plus) / n_tok
    fd_grad = (loss_plus - loss) / eps
    assert np.isclose(grad_log_p[0, 0], fd_grad, rtol=1e-6, atol=1e-8)


def test_staleness_buffer_ring_semantics():
    """SPEC E5 §6.3: t < k uses theta_0; t >= k uses theta_{t-k}."""
    buf = StalenessBuffer(k=3)
    for t in range(6):
        buf.push(t, {"w": np.array([float(t)], dtype=np.float32)})
        behav = buf.get_behavior_weights(t)
        expected_step = 0.0 if t < 3 else float(t - 3)
        assert behav["w"][0] == expected_step


def test_boxed_reward_parsing():
    """SPEC E5 §6.1: exact match on \\boxed{} numeric answer."""
    assert extract_boxed_answer("Step 1... Therefore \\boxed{42}.") == "42"
    assert extract_boxed_answer("Result: \\boxed{1,234.00}") == "1234"
    assert extract_boxed_answer("No boxed answer here 42") is None
    assert compute_exact_match_reward("The answer is \\boxed{72}", "Reasoning...\n#### 72") == 1.0
    assert compute_exact_match_reward("The answer is 72", "#### 72") == 0.0


def test_online_llm_diag_accumulator_matches_nested_stats():
    """Verify OnlineLLMDiagAccumulator matches nested_stats on a multi-tensor dictionary."""
    rng = np.random.default_rng(105)
    S, R = 8, 4
    d1, d2 = 50, 70
    G1 = rng.normal(0.5, 1.0, size=(S * R, d1))
    G2 = rng.normal(-0.3, 1.0, size=(S * R, d2))
    G_full = np.concatenate([G1, G2], axis=1)

    m1 = rng.normal(0.2, 1.0, size=d1)
    m2 = rng.normal(-0.1, 1.0, size=d2)
    m_full = np.concatenate([m1, m2])
    norm_m = float(np.linalg.norm(m_full))

    acc = OnlineLLMDiagAccumulator(S=S, R=R, norm_m=norm_m, sketch_dim=2**15)
    idx = 0
    for s in range(S):
        acc.begin_set(s)
        for _ in range(R):
            acc.add_batch_grad(
                {"layer1.weight": G1[idx], "layer2.weight": G2[idx]},
                {"layer1.weight": m1, "layer2.weight": m2},
            )
            idx += 1
        acc.end_set()

    res_online = acc.finalize()
    res_batch = nested_stats(G_full, m_full, R=R, compute_jk=True)

    for k in ["N_bar", "H_hat", "M", "var_total", "var_rollout", "var_prompt", "SNR_hat", "inner_m_h", "C1", "C2_hat", "C1_pred"]:
        assert np.isclose(res_online[k], res_batch[k], rtol=1e-10, atol=1e-10)
    assert res_online["sketch_rel_err"] < 0.05
