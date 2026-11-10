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


def test_staleness_buffer_missing_snapshot_raises():
    """Item 4: StalenessBuffer must raise KeyError directly if target snapshot is missing."""
    buf = StalenessBuffer(k=4)
    # Empty buffer -> raises KeyError
    with pytest.raises(KeyError, match="StalenessBuffer is empty"):
        buf.get_behavior_weights(0)

    # Push steps 1..5 without step 0 -> t=2 needs step 0, which is missing
    for t in range(1, 6):
        buf.push(t, {"w": t})
    with pytest.raises(KeyError, match="Target behavior snapshot step=0 not found"):
        buf.get_behavior_weights(2)

    # Push step 10 -> evicts earlier steps from maxlen=5; requesting step 5 (target=1) raises KeyError
    for t in range(6, 11):
        buf.push(t, {"w": t})
    with pytest.raises(KeyError, match="Target behavior snapshot step=1 not found"):
        buf.get_behavior_weights(5)


def test_microbatch_1_vs_4_gradient_equivalence():
    """Item 3: 1 micro-batch and 4 micro-batches using global N_tok must produce identical gradients (error < 1e-5)."""
    import torch
    from llm.train import accumulate_grpo_tis_microbatches

    torch.manual_seed(2026)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TinyCausalLM(torch.nn.Module):
        def __init__(self, vocab_size: int = 32, hidden: int = 16):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden)
            self.proj = torch.nn.Linear(hidden, vocab_size, bias=True)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            h = torch.tanh(self.embed(input_ids))
            logits = self.proj(h)

            class Out:
                pass

            out = Out()
            out.logits = logits
            return out

    model_1 = TinyCausalLM().to(dev, dtype=torch.float32)
    model_4 = TinyCausalLM().to(dev, dtype=torch.float32)
    model_4.load_state_dict(model_1.state_dict())
    behav_model = TinyCausalLM().to(dev, dtype=torch.float32)

    B, T = 8, 12
    prompt_len = 4
    R_len = T - 1
    input_ids = torch.randint(0, 32, (B, T), device=dev)
    attention_mask = torch.ones((B, T), dtype=torch.long, device=dev)

    # Give each sequence a different valid response length so micro-batch token counts differ!
    response_mask = torch.zeros((B, R_len), dtype=torch.float32, device=dev)
    lengths = [2, 7, 4, 8, 3, 6, 5, 8]
    for i, L in enumerate(lengths):
        response_mask[i, prompt_len - 1 : prompt_len - 1 + L] = 1.0

    advantages = torch.tensor([1.2, -0.8, 0.5, -0.9, 1.1, -1.1, 0.4, -0.4], dtype=torch.float32, device=dev)

    stats_1 = accumulate_grpo_tis_microbatches(
        model=model_1,
        behav_model=behav_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        advantages=advantages,
        k=4,
        c=1.2,
        micro_batch_size=8,  # 1 micro-batch
        use_bf16=False,
    )
    stats_4 = accumulate_grpo_tis_microbatches(
        model=model_4,
        behav_model=behav_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        advantages=advantages,
        k=4,
        c=1.2,
        micro_batch_size=2,  # 4 micro-batches with unequal token counts
        use_bf16=False,
    )

    assert abs(stats_1["loss"] - stats_4["loss"]) < 1e-6
    assert abs(stats_1["trunc_rate"] - stats_4["trunc_rate"]) < 1e-6
    assert abs(stats_1["mean_w"] - stats_4["mean_w"]) < 1e-6

    for (n1, p1), (n4, p4) in zip(model_1.named_parameters(), model_4.named_parameters()):
        assert n1 == n4
        max_diff = float(torch.max(torch.abs(p1.grad - p4.grad)).item())
        assert max_diff < 1e-5, f"Param {n1} max grad diff {max_diff} >= 1e-5"


def test_boxed_reward_parsing():
    """SPEC E5 §6.1: exact match on \\boxed{} numeric answer."""
    assert extract_boxed_answer("Step 1... Therefore \\boxed{42}.") == "42"
    assert extract_boxed_answer("Result: \\boxed{1,234.00}") == "1234"
    assert extract_boxed_answer("No boxed answer here 42") is None
    assert compute_exact_match_reward("The answer is \\boxed{72}", "Reasoning...\n#### 72") == 1.0
    assert compute_exact_match_reward("The answer is 72", "#### 72") == 0.0


def test_online_llm_diag_accumulator_matches_nested_stats():
    """Verify OnlineLLMDiagAccumulator (GPU fp32 vectors, float64 scalars) matches nested_stats on a multi-tensor dict."""
    rng = np.random.default_rng(105)
    S, R = 8, 4
    d1, d2 = 50, 70
    G1 = rng.normal(0.5, 1.0, size=(S * R, d1)).astype(np.float32)
    G2 = rng.normal(-0.3, 1.0, size=(S * R, d2)).astype(np.float32)
    G_full = np.concatenate([G1, G2], axis=1).astype(np.float64)

    m1 = rng.normal(0.2, 1.0, size=d1).astype(np.float32)
    m2 = rng.normal(-0.1, 1.0, size=d2).astype(np.float32)
    m_full = np.concatenate([m1, m2]).astype(np.float64)
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

    for k in ["N_bar", "H_hat", "M", "var_total", "var_rollout", "var_prompt", "SNR_hat", "inner_m_h", "C1", "C2_hat", "C1_pred", "C1_pred_legacy"]:
        assert np.isclose(res_online[k], res_batch[k], rtol=1e-5, atol=1e-5), f"Mismatch on {k}: {res_online[k]} vs {res_batch[k]}"
    assert res_online["sketch_rel_err"] < 0.05


def test_all_l0_to_l4_configs_have_micro_batch_size_4():
    """Verify every L0–L4 YAML config file specifies micro_batch_size: 4."""
    from pathlib import Path
    import yaml

    cfg_root = Path("llm/configs")
    yaml_files = sorted(cfg_root.rglob("*.yaml"))
    assert len(yaml_files) >= 6
    for yf in yaml_files:
        with open(yf, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("micro_batch_size") == 4, f"{yf} missing micro_batch_size: 4"


def test_chunked_compute_response_logprobs_matches_naive_log_softmax():
    """Verify chunked logsumexp + gather matches full fp32 log_softmax in forward values and backward grads."""
    import torch
    from llm.train import compute_response_logprobs

    torch.manual_seed(3001)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TinyLM(torch.nn.Module):
        def __init__(self, vocab_size: int = 128, hidden: int = 32):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden)
            self.proj = torch.nn.Linear(hidden, vocab_size, bias=True)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            h = torch.tanh(self.embed(input_ids))
            logits = self.proj(h)

            class Out:
                pass

            out = Out()
            out.logits = logits
            return out

    m_chunk = TinyLM().to(dev, dtype=torch.float32)
    m_naive = TinyLM().to(dev, dtype=torch.float32)
    m_naive.load_state_dict(m_chunk.state_dict())

    B, L = 4, 15
    input_ids = torch.randint(0, 128, (B, L), device=dev)
    attention_mask = torch.ones((B, L), dtype=torch.long, device=dev)
    weights = torch.randn((B, L - 1), dtype=torch.float32, device=dev)

    # 1. Chunked logsumexp + gather with small chunk_size=4
    lp_chunk = compute_response_logprobs(
        m_chunk, input_ids, attention_mask, use_bf16=False, chunk_size=4
    )
    loss_chunk = (lp_chunk * weights).sum()
    loss_chunk.backward()

    # 2. Naive full (B, L-1, V) fp32 log_softmax + gather
    logits_naive = m_naive(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    log_probs_all = torch.log_softmax(logits_naive.to(dtype=torch.float32), dim=-1)
    lp_naive = torch.gather(log_probs_all, dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    loss_naive = (lp_naive * weights).sum()
    loss_naive.backward()

    torch.testing.assert_close(lp_chunk, lp_naive, rtol=1e-6, atol=1e-6)
    for (n1, p1), (n2, p2) in zip(m_chunk.named_parameters(), m_naive.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-6, atol=1e-6)


def test_step_and_compute_update_norm_and_gpu_online_c1_no_clone_or_cpu_copy():
    """Verify GPU-direct online_C1 and per-parameter step_and_compute_update_norm match full-clone reference."""
    import torch
    from llm.train import (
        build_optimizer,
        compute_grad_norm_and_online_c1_gpu,
        compute_online_c1,
        extract_momentum_buffers,
        step_and_compute_update_norm,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for opt_name in ["sgd", "sgd_mom", "adamw"]:
        torch.manual_seed(4001)
        m_new = torch.nn.Sequential(
            torch.nn.Linear(16, 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, 8),
        ).to(dev, dtype=torch.float32)
        m_ref = torch.nn.Sequential(
            torch.nn.Linear(16, 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, 8),
        ).to(dev, dtype=torch.float32)
        m_ref.load_state_dict(m_new.state_dict())

        opt_new = build_optimizer(m_new, opt_name=opt_name, lr=0.01, momentum=0.9)
        opt_ref = build_optimizer(m_ref, opt_name=opt_name, lr=0.01, momentum=0.9)

        for step_idx in range(3):
            x = torch.randn((6, 16), device=dev, dtype=torch.float32)
            opt_new.zero_grad(set_to_none=True)
            opt_ref.zero_grad(set_to_none=True)

            loss_new = (m_new(x) ** 2).mean()
            loss_ref = (m_ref(x) ** 2).mean()
            loss_new.backward()
            loss_ref.backward()

            # Reference: CPU buffer extraction + full prev_params clone
            m_prev_ref, _, _ = extract_momentum_buffers(opt_ref, m_ref, opt_name)
            prev_params_ref = {n: p.detach().clone() for n, p in m_ref.named_parameters()}
            grad_dict_ref = {n: p.grad.detach() for n, p in m_ref.named_parameters()}
            c1_ref = compute_online_c1(m_prev_ref, grad_dict_ref)
            opt_ref.step()
            upd_ref_sq = sum(
                float(torch.sum((p.detach() - prev_params_ref[n]).to(torch.float64) ** 2).item())
                for n, p in m_ref.named_parameters()
            )
            upd_ref = float(np.sqrt(upd_ref_sq))

            # New: GPU-direct online_C1 + memory-free step_and_compute_update_norm
            grad_norm_new, c1_new = compute_grad_norm_and_online_c1_gpu(opt_new, m_new, opt_name)
            upd_new = step_and_compute_update_norm(opt_new)

            assert np.isclose(upd_new, upd_ref, rtol=1e-12, atol=1e-12)
            if np.isnan(c1_ref):
                assert np.isnan(c1_new)
            else:
                assert np.isclose(c1_new, c1_ref, rtol=1e-6, atol=1e-6)
            for p_n, p_r in zip(m_new.parameters(), m_ref.parameters()):
                torch.testing.assert_close(p_n, p_r, rtol=0.0, atol=0.0)


def test_save_checkpoint_omits_behav_state_dict_when_k0_and_includes_when_k_gt_0(tmp_path):
    """Verify k=0 checkpoint omits behav_state_dict while k>0 checkpoint saves behav_state_dict."""
    import torch
    from llm.train import build_optimizer, save_checkpoint

    model = torch.nn.Linear(8, 4)
    opt = build_optimizer(model, "sgd_mom", lr=0.01, momentum=0.9)
    behav_sd = {k: v.clone() for k, v in model.state_dict().items()}

    p_k0 = save_checkpoint(
        ckpt_dir=tmp_path / "k0",
        step=0,
        model=model,
        behav_state_dict=behav_sd,
        optimizer=opt,
        opt_name="sgd_mom",
        config={"k": 0},
        min_free_gb=0.01,
        k=0,
    )
    payload_k0 = torch.load(p_k0, map_location="cpu", weights_only=False)
    assert "behav_state_dict" not in payload_k0
    assert "model_state_dict" in payload_k0

    p_k2 = save_checkpoint(
        ckpt_dir=tmp_path / "k2",
        step=2,
        model=model,
        behav_state_dict=behav_sd,
        optimizer=opt,
        opt_name="sgd_mom",
        config={"k": 2},
        min_free_gb=0.01,
        k=2,
    )
    payload_k2 = torch.load(p_k2, map_location="cpu", weights_only=False)
    assert "behav_state_dict" in payload_k2


def test_vllm_sleep_wake_and_engine_seed_and_dup_group_frac(monkeypatch):
    """Verify dup_group_frac calculation, vLLM sleep(level=1)/wake_up(), and engine-level seed without per-request seed."""
    import sys
    import types
    import torch
    from llm.data import GSM8KItem
    from llm.train import RolloutBackend, compute_dup_group_frac

    # 1. Test compute_dup_group_frac directly
    # Group 0: all 4 identical -> duplicate group
    # Group 1: 3 identical, 1 different -> not all identical
    resps = [
        [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],
        [4, 5], [4, 5], [4, 5], [4, 6],
    ]
    assert np.isclose(compute_dup_group_frac(resps, group_size=4), 0.5)

    # 2. Mock vllm module to test RolloutBackend vllm lifecycle without a Linux vLLM server
    events = []
    captured_llm_kwargs = {}
    captured_sp_kwargs = []

    class DummySamplingParams:
        def __init__(self, **kwargs):
            captured_sp_kwargs.append(kwargs)
            self.kwargs = kwargs

    class DummyComp:
        def __init__(self, token_ids, text):
            self.token_ids = token_ids
            self.text = text
            self.logprobs = [{tid: types.SimpleNamespace(logprob=-0.5)} for tid in token_ids]

    class DummyReqOut:
        def __init__(self, prompt_token_ids, outputs):
            self.prompt_token_ids = prompt_token_ids
            self.outputs = outputs

    class DummyLLM:
        def __init__(self, **kwargs):
            captured_llm_kwargs.update(kwargs)
            events.append(("init", kwargs.get("seed")))

        def sleep(self, level: int = 1):
            events.append(("sleep", level))

        def wake_up(self):
            events.append(("wake_up", None))

        def apply_model(self, fn):
            dummy_model = types.SimpleNamespace(
                load_weights=lambda w: events.append(("load_weights_apply_model", len(w)))
            )
            return fn(dummy_model)

        def generate(self, prompts, sp, use_tqdm=False):
            events.append(("generate", len(prompts)))
            outs = []
            for idx, _ in enumerate(prompts):
                if idx == 0:
                    # All 4 responses identical in group 0
                    comps = [DummyComp([10, 11], "\\boxed{3}") for _ in range(sp.kwargs["n"])]
                else:
                    # Mixed responses in group 1
                    comps = [
                        DummyComp([10, 11 + g], "\\boxed{3}" if g % 2 == 0 else "\\boxed{4}")
                        for g in range(sp.kwargs["n"])
                    ]
                outs.append(DummyReqOut([1, 2, 3], comps))
            return outs

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = DummyLLM
    fake_vllm.SamplingParams = DummySamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    dummy_tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=2)
    rb = RolloutBackend(
        backend="vllm",
        model_name="dummy/model",
        tokenizer=dummy_tok,
        device="cpu",
        vllm_cfg={"enable_sleep_mode": True, "gpu_memory_utilization": 0.3},
        use_bf16=False,
        seed=1234,
    )

    # Engine-level seed must be passed to LLM(...)
    assert captured_llm_kwargs["seed"] == 1234

    # Sleep -> sync_weights should auto wake_up and use apply_model fallback
    rb.sleep(level=1)
    assert rb._is_sleeping is True
    rb.sync_weights({"w": torch.ones(4)})
    assert rb._is_sleeping is False

    # Sleep -> generate_group_rollouts should auto wake_up and NOT pass seed to SamplingParams
    rb.sleep(level=1)
    items = [
        GSM8KItem(idx=0, question="q0", ground_truth="3", prompt_text="p0"),
        GSM8KItem(idx=1, question="q1", ground_truth="3", prompt_text="p1"),
    ]
    batch = rb.generate_group_rollouts(
        items=items,
        group_size=4,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=16,
        seed=9999,
    )
    rb.sleep(level=1)

    assert "seed" not in captured_sp_kwargs[-1]
    assert np.isclose(batch["dup_group_frac"], 0.5)
    assert ("sleep", 1) in events
    assert ("wake_up", None) in events
    assert ("load_weights_apply_model", 1) in events


def test_run_l0_pipeline_steps_and_snr_recommendation(tmp_path):
    """Verify run_l0 helpers (recommend_l2_k_from_snr, estimate_l1_to_l4_gpu_hours, step1_record_environment)."""
    from llm.run_l0 import (
        estimate_l1_to_l4_gpu_hours,
        recommend_l2_k_from_snr,
        step1_record_environment,
    )

    pinned = step1_record_environment(tmp_path)
    assert "torch" in pinned
    assert (tmp_path / "pinned_versions.txt").exists()

    rec_high = recommend_l2_k_from_snr(snr_hat=0.5, target_cv=0.25, r_rep=4)
    assert rec_high["recommended_K"] >= 64
    assert rec_high["recommended_S"] % 2 == 0

    rec_low = recommend_l2_k_from_snr(snr_hat=0.05, target_cv=0.25, r_rep=4)
    assert rec_low["recommended_K"] > rec_high["recommended_K"]

    est = estimate_l1_to_l4_gpu_hours(mean_step_sec=2.0, recommended_k_l2=64)
    assert est["total_L1_to_L4_core_gpu_hours"] > 0.0
    assert np.isclose(
        est["total_L1_to_L4_core_gpu_hours"],
        est["L1_gpu_hours"] + est["L2_gpu_hours"] + est["L3_gpu_hours"] + est["L4_core_gpu_hours"],
    )


