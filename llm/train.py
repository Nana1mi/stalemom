"""Minimal RLVR training loop with staleness ring buffer, token-level TIS, and online C1 logging (SPEC E5 §6)."""

from collections import deque
import copy
import csv
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import numpy as np
import yaml

from llm.reward import compute_exact_match_reward, compute_grpo_advantages


def compute_token_weights_numpy(
    log_pi_theta: np.ndarray,
    log_pi_behav: np.ndarray,
    k: int,
    c: float = float("inf"),
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute token-level IS / TIS weights w_tok and diagnostics (SPEC E5 §6.2, §6.3)."""
    log_p = np.asarray(log_pi_theta, dtype=np.float64)
    log_q = np.asarray(log_pi_behav, dtype=np.float64)
    if mask is None:
        valid = np.ones_like(log_p, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)

    n_tok = int(np.sum(valid))
    if n_tok == 0:
        return np.ones_like(log_p, dtype=np.float64), {
            "trunc_rate": 0.0,
            "mean_w": 1.0,
            "approx_kl": 0.0,
        }

    if k == 0:
        w = np.ones_like(log_p, dtype=np.float64)
        return w, {
            "trunc_rate": 0.0,
            "mean_w": 1.0,
            "approx_kl": 0.0,
        }

    log_ratio = log_p - log_q
    ratio = np.exp(np.clip(log_ratio, -50.0, 50.0))
    if c is None or np.isinf(c):
        w = ratio
        trunc_mask = np.zeros_like(valid, dtype=bool)
    else:
        w = np.minimum(ratio, float(c))
        trunc_mask = (ratio > float(c)) & valid

    w = np.where(valid, w, 0.0)
    trunc_rate = float(np.sum(trunc_mask) / n_tok)
    mean_w = float(np.sum(w[valid]) / n_tok)
    approx_kl = float(np.sum(log_ratio[valid]) / n_tok)
    return w, {
        "trunc_rate": trunc_rate,
        "mean_w": mean_w,
        "approx_kl": approx_kl,
    }


def compute_grpo_tis_loss_and_grad_numpy(
    log_pi_theta: np.ndarray,
    log_pi_behav: np.ndarray,
    advantages: np.ndarray,
    k: int,
    c: float = float("inf"),
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    """Compute GRPO token-level TIS loss and exact gradient w.r.t. log_pi_theta (SPEC E5 §6.2)."""
    log_p = np.asarray(log_pi_theta, dtype=np.float64)
    log_q = np.asarray(log_pi_behav, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)
    if adv.ndim == 1 and log_p.ndim == 2:
        adv = adv[:, None]

    if mask is None:
        valid = np.ones_like(log_p, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)

    n_tok = int(np.sum(valid))
    if n_tok == 0:
        return 0.0, np.zeros_like(log_p), np.ones_like(log_p), {
            "trunc_rate": 0.0,
            "mean_w": 1.0,
            "approx_kl": 0.0,
        }

    w, stats = compute_token_weights_numpy(log_p, log_q, k=k, c=c, mask=valid)
    adv_tok = np.broadcast_to(adv, log_p.shape)
    active_terms = np.where(valid, w * adv_tok * log_p, 0.0)
    loss = float(-np.sum(active_terms) / n_tok)
    grad_log_p = np.where(valid, -(w * adv_tok) / n_tok, 0.0)
    return loss, grad_log_p, w, stats


def compute_grpo_tis_loss_torch(
    log_pi_theta,
    log_pi_behav,
    advantages,
    k: int,
    c: float = float("inf"),
    mask=None,
    global_n_tok: int | float | None = None,
):
    """PyTorch implementation of SPEC E5 §6.2 loss with stop-gradient on w_tok and global N_tok normalization."""
    import torch

    if mask is None:
        valid = torch.ones_like(log_pi_theta, dtype=torch.bool)
    else:
        valid = mask.to(dtype=torch.bool)

    local_n_tok = int(valid.sum().item())
    denom = float(global_n_tok) if global_n_tok is not None else float(max(local_n_tok, 1))
    denom = max(denom, 1.0)

    with torch.no_grad():
        if k == 0:
            w = torch.ones_like(log_pi_theta)
            trunc_count = 0.0
            sum_w = float(local_n_tok)
            sum_kl = 0.0
        else:
            log_ratio = log_pi_theta.detach() - log_pi_behav.detach()
            ratio = torch.exp(torch.clamp(log_ratio, -50.0, 50.0))
            if c is None or np.isinf(c):
                w = ratio
                trunc_count = 0.0
            else:
                w = torch.clamp(ratio, max=float(c))
                trunc_count = float(((ratio > float(c)) & valid).sum().item())
            sum_w = float(w[valid].sum().item()) if local_n_tok > 0 else 0.0
            sum_kl = float(log_ratio[valid].sum().item()) if local_n_tok > 0 else 0.0

    adv = advantages
    if adv.ndim == 1 and log_pi_theta.ndim == 2:
        adv = adv.unsqueeze(-1)

    valid_f = valid.to(dtype=log_pi_theta.dtype)
    loss = -(w.detach() * adv.detach() * log_pi_theta * valid_f).sum() / denom
    stats = {
        "trunc_rate": trunc_count / max(local_n_tok, 1) if k > 0 else 0.0,
        "mean_w": sum_w / max(local_n_tok, 1) if local_n_tok > 0 else 1.0,
        "approx_kl": sum_kl / max(local_n_tok, 1) if k > 0 and local_n_tok > 0 else 0.0,
        "trunc_count": trunc_count,
        "sum_w": sum_w,
        "sum_kl": sum_kl,
        "local_n_tok": local_n_tok,
    }
    return loss, w, stats


def compute_response_logprobs(
    model,
    input_ids,
    attention_mask,
     use_bf16: bool = True,
):
    """Compute per-token log probabilities log pi(y_t | y_{<t}) for positions 1..L-1 in float32."""
    import torch

    dev_type = "cuda" if input_ids.is_cuda else "cpu"
    amp_enabled = bool(use_bf16 and (dev_type == "cuda" or torch.cpu.amp is not None))
    with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp_enabled):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]  # (B, L-1, V)

    log_probs_all = torch.log_softmax(logits.to(dtype=torch.float32), dim=-1)
    target_ids = input_ids[:, 1:].unsqueeze(-1)  # (B, L-1, 1)
    token_logprobs = torch.gather(log_probs_all, dim=-1, index=target_ids).squeeze(-1)  # (B, L-1)
    return token_logprobs


def accumulate_grpo_tis_microbatches(
    model,
    behav_model,
    input_ids,
    attention_mask,
    response_mask,
    advantages,
    k: int,
    c: float = float("inf"),
    micro_batch_size: int | None = None,
    use_bf16: bool = True,
) -> dict[str, float]:
    """Run forward/backward over micro-batches using the global N_tok across the entire batch (Item 3).

    Guarantees that 1 micro-batch vs M micro-batches yields the exact same parameter gradients.
    """
    import torch

    total_seqs = int(input_ids.shape[0])
    mb_size = total_seqs if (micro_batch_size is None or micro_batch_size <= 0) else int(micro_batch_size)

    # Global total response tokens across the entire batch
    global_n_tok = int(response_mask.sum().item())
    if global_n_tok == 0:
        return {
            "loss": 0.0,
            "trunc_rate": 0.0,
            "mean_w": 1.0,
            "approx_kl": 0.0,
            "global_n_tok": 0,
        }

    total_loss = 0.0
    total_trunc = 0.0
    total_sum_w = 0.0
    total_sum_kl = 0.0

    for start in range(0, total_seqs, mb_size):
        end = min(start + mb_size, total_seqs)
        mb_ids = input_ids[start:end]
        mb_attn = attention_mask[start:end]
        mb_resp_mask = response_mask[start:end]
        mb_adv = advantages[start:end]

        if int(mb_resp_mask.sum().item()) == 0:
            continue

        log_pi_theta = compute_response_logprobs(model, mb_ids, mb_attn, use_bf16=use_bf16)
        if k == 0 or behav_model is None:
            log_pi_behav = log_pi_theta.detach()
        else:
            with torch.no_grad():
                log_pi_behav = compute_response_logprobs(behav_model, mb_ids, mb_attn, use_bf16=use_bf16)

        loss_mb, _, st_mb = compute_grpo_tis_loss_torch(
            log_pi_theta=log_pi_theta,
            log_pi_behav=log_pi_behav,
            advantages=mb_adv,
            k=k,
            c=c,
            mask=mb_resp_mask,
            global_n_tok=global_n_tok,
        )
        loss_mb.backward()

        total_loss += float(loss_mb.detach().item())
        total_trunc += st_mb["trunc_count"]
        total_sum_w += st_mb["sum_w"]
        total_sum_kl += st_mb["sum_kl"]

    return {
        "loss": total_loss,
        "trunc_rate": total_trunc / global_n_tok if k > 0 else 0.0,
        "mean_w": total_sum_w / global_n_tok if k > 0 else 1.0,
        "approx_kl": total_sum_kl / global_n_tok if k > 0 else 0.0,
        "global_n_tok": global_n_tok,
    }


class StalenessBuffer:
    """CPU fp32 ring buffer storing the most recent k+1 parameter snapshots (SPEC E5 §6.3).

    - At step t < k: returns the initial snapshot theta_0.
    - At step t >= k: returns theta_{t-k}.
    - Raises KeyError if the target snapshot step is not found in the buffer.
    """

    def __init__(self, k: int):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.k = int(k)
        self._snapshots: deque[tuple[int, dict[str, Any]]] = deque(maxlen=self.k + 1)
        self._theta_0: dict[str, Any] | None = None

    def push(self, step: int, state_dict: dict[str, Any]):
        """Store a deep copy of fp32 CPU weights at `step`."""
        snap = {}
        for name, val in state_dict.items():
            if hasattr(val, "detach"):
                import torch
                snap[name] = val.detach().to(device="cpu", dtype=torch.float32).clone()
            else:
                snap[name] = np.array(val, dtype=np.float32, copy=True)
        if int(step) == 0:
            self._theta_0 = copy.deepcopy(snap)
        self._snapshots.append((int(step), snap))

    def get_behavior_weights(self, current_step: int) -> dict[str, Any]:
        """Return theta_{t-k} (or theta_0 when current_step < k).

        Raises KeyError if the required target snapshot is missing from the buffer.
        """
        if current_step < 0:
            raise ValueError(f"current_step must be >= 0, got {current_step}")
        if not self._snapshots:
            raise KeyError("StalenessBuffer is empty; call push(0, init_weights) first")

        if self.k == 0:
            target_step = int(current_step)
        elif current_step < self.k:
            target_step = 0
            if self._theta_0 is not None:
                return self._theta_0
        else:
            target_step = int(current_step - self.k)

        for step_idx, snap in self._snapshots:
            if step_idx == target_step:
                return snap

        available = [s for s, _ in self._snapshots]
        raise KeyError(
            f"Target behavior snapshot step={target_step} not found in StalenessBuffer "
            f"(current_step={current_step}, k={self.k}, available={available})"
        )


def compute_online_c1(
    m_dict: dict[str, Any],
    g_dict: dict[str, Any],
) -> float:
    """Compute online_C1 = cos(m_{t-1}, g_t) across all parameters in float64 (SPEC E5 §6.5)."""
    import torch

    dot = 0.0
    norm_m_sq = 0.0
    norm_g_sq = 0.0
    for name, g in g_dict.items():
        if isinstance(g, torch.Tensor):
            g64 = g.detach().to(dtype=torch.float64)
            norm_g_sq += float(torch.sum(g64 * g64).item())
            if name in m_dict and m_dict[name] is not None:
                m_val = m_dict[name]
                m64 = (
                    m_val.detach().to(device=g64.device, dtype=torch.float64)
                    if isinstance(m_val, torch.Tensor)
                    else torch.as_tensor(m_val, dtype=torch.float64, device=g64.device)
                )
                norm_m_sq += float(torch.sum(m64 * m64).item())
                dot += float(torch.sum(m64 * g64).item())
        else:
            g64 = np.asarray(g, dtype=np.float64)
            norm_g_sq += float(np.sum(g64 * g64))
            if name in m_dict and m_dict[name] is not None:
                m64 = np.asarray(m_dict[name], dtype=np.float64)
                norm_m_sq += float(np.sum(m64 * m64))
                dot += float(np.sum(m64 * g64))
    if norm_m_sq <= 0.0 or norm_g_sq <= 0.0:
        return float("nan")
    return float(dot / (np.sqrt(norm_m_sq) * np.sqrt(norm_g_sq)))


def check_disk_free_gb(path: str | Path, min_free_gb: float = 10.0) -> float:
    """Verify disk free space >= min_free_gb (SPEC E5 §6.4)."""
    p = Path(path).resolve()
    while not p.exists() and p.parent != p:
        p = p.parent
    usage = shutil.disk_usage(p)
    free_gb = usage.free / (1024.0**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Disk free space ({free_gb:.2f} GB) is below the {min_free_gb} GB safety threshold (§6.4). Stopping."
        )
    return free_gb


def extract_momentum_buffers(optimizer, model, opt_name: str) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Extract m_{t-1} (and exp_avg_sq if AdamW) from PyTorch optimizer state (§6.4).

    Returns:
        m_dict: {param_name: fp32 CPU tensor} representing m_{t-1}
        sq_dict: {param_name: fp32 CPU tensor} representing exp_avg_sq (if AdamW, else empty)
        norm_m: float64 L2 norm of m_{t-1}
    """
    import torch

    m_dict = {}
    sq_dict = {}
    norm_m_sq = 0.0
    opt_key = opt_name.lower()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        state = optimizer.state.get(p, {})
        buf = None
        if opt_key in ("sgd_mom", "sgd"):
            buf = state.get("momentum_buffer", None)
        elif opt_key in ("adamw", "rmsprop_via_adamw"):
            buf = state.get("exp_avg", None)
            sq = state.get("exp_avg_sq", None)
            if sq is not None:
                sq_dict[name] = sq.detach().to(device="cpu", dtype=torch.float32).clone()

        if buf is not None:
            buf_cpu = buf.detach().to(device="cpu", dtype=torch.float32).clone()
            m_dict[name] = buf_cpu
            b64 = buf.detach().to(dtype=torch.float64)
            norm_m_sq += float(torch.sum(b64 * b64).item())

    return m_dict, sq_dict, float(np.sqrt(norm_m_sq))


class RolloutBackend:
    """Switchable rollout backend supporting 'hf' (local debug) and 'vllm' (server) (§6.2)."""

    def __init__(
        self,
        backend: str,
        model_name: str,
        tokenizer,
        device: str | Any,
        vllm_cfg: dict | None = None,
        use_bf16: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM

        self.backend = backend.lower()
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.use_bf16 = bool(use_bf16)

        if self.backend == "hf":
            dtype = torch.bfloat16 if (self.use_bf16 and self.device.type == "cuda") else torch.float32
            self.hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
            self.hf_model.eval()
            self.hf_model.requires_grad_(False)
            self.vllm_engine = None
        elif self.backend == "vllm":
            from vllm import LLM
            vcfg = vllm_cfg or {}
            self.vllm_engine = LLM(
                model=model_name,
                dtype="bfloat16",
                enable_sleep_mode=bool(vcfg.get("enable_sleep_mode", True)),
                gpu_memory_utilization=float(vcfg.get("gpu_memory_utilization", 0.30)),
                trust_remote_code=True,
            )
            self.hf_model = None
        else:
            raise ValueError(f"Unsupported rollout backend: {backend}")

    def sync_weights(self, fp32_state_dict: dict[str, Any]):
        """Synchronize behavior weights theta_{t-k} (converted to bf16) into the rollout engine."""
        import torch

        if self.backend == "hf":
            target_dtype = torch.bfloat16 if (self.use_bf16 and self.device.type == "cuda") else torch.float32
            cast_sd = {
                k: v.to(device=self.device, dtype=target_dtype)
                for k, v in fp32_state_dict.items()
            }
            self.hf_model.load_state_dict(cast_sd, strict=True)
            self.hf_model.eval()
        elif self.backend == "vllm":
            model_runner = (
                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner
            )
            weights = [
                (k, v.to(dtype=torch.bfloat16))
                for k, v in fp32_state_dict.items()
            ]
            model_runner.model.load_weights(weights)

    def generate_group_rollouts(
        self,
        items: list,
        group_size: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate `group_size` responses per prompt and pack padded tensors for training."""
        import torch

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id or 0)
        )
        eos_id = self.tokenizer.eos_token_id

        all_seq_ids: list[list[int]] = []
        all_prompt_lens: list[int] = []
        all_resp_lens: list[int] = []
        all_rewards: list[float] = []
        all_rollout_logprobs: list[np.ndarray] = []

        if self.backend == "hf":
            if seed is not None:
                torch.manual_seed(int(seed))
            with torch.no_grad():
                for item in items:
                    enc = self.tokenizer(item.prompt_text, return_tensors="pt", add_special_tokens=False)
                    p_ids = enc["input_ids"].to(self.device)
                    p_len = int(p_ids.shape[1])
                    p_ids_rep = p_ids.repeat(group_size, 1)
                    attn_rep = torch.ones_like(p_ids_rep)

                    gen_out = self.hf_model.generate(
                        input_ids=p_ids_rep,
                        attention_mask=attn_rep,
                        max_new_tokens=max_new_tokens,
                        do_sample=(temperature > 0.0),
                        temperature=max(temperature, 1e-5) if temperature > 0.0 else 1.0,
                        top_p=top_p,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                    )
                    for g_idx in range(group_size):
                        seq = gen_out[g_idx].tolist()
                        # Trim trailing pad tokens after EOS
                        resp_part = seq[p_len:]
                        if eos_id is not None and eos_id in resp_part:
                            first_eos = resp_part.index(eos_id)
                            resp_part = resp_part[: first_eos + 1]
                        if len(resp_part) == 0:
                            resp_part = [eos_id if eos_id is not None else 0]
                        full_seq = seq[:p_len] + resp_part
                        text = self.tokenizer.decode(resp_part, skip_special_tokens=True)
                        rew = compute_exact_match_reward(text, item.ground_truth)

                        all_seq_ids.append(full_seq)
                        all_prompt_lens.append(p_len)
                        all_resp_lens.append(len(resp_part))
                        all_rewards.append(rew)

        elif self.backend == "vllm":
            from vllm import SamplingParams

            sp = SamplingParams(
                n=group_size,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                seed=seed,
                logprobs=1,
            )
            prompts = [it.prompt_text for it in items]
            outputs = self.vllm_engine.generate(prompts, sp, use_tqdm=False)
            for item, out in zip(items, outputs):
                p_ids = list(out.prompt_token_ids)
                p_len = len(p_ids)
                for comp in out.outputs:
                    r_ids = list(comp.token_ids)
                    if len(r_ids) == 0:
                        r_ids = [eos_id if eos_id is not None else 0]
                    full_seq = p_ids + r_ids
                    rew = compute_exact_match_reward(comp.text, item.ground_truth)
                    all_seq_ids.append(full_seq)
                    all_prompt_lens.append(p_len)
                    all_resp_lens.append(len(r_ids))
                    all_rewards.append(rew)
                    if comp.logprobs is not None:
                        lp_list = [
                            float(step_lp[tok_id].logprob)
                            for tok_id, step_lp in zip(r_ids, comp.logprobs)
                        ]
                        all_rollout_logprobs.append(np.array(lp_list, dtype=np.float32))

        # Pad sequences into tensors (B_seq, max_len)
        max_len = max(len(s) for s in all_seq_ids)
        bsz = len(all_seq_ids)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=self.device)
        response_mask = torch.zeros((bsz, max_len - 1), dtype=torch.bool, device=self.device)

        for i, (seq, p_len, r_len) in enumerate(zip(all_seq_ids, all_prompt_lens, all_resp_lens)):
            slen = len(seq)
            input_ids[i, :slen] = torch.tensor(seq, dtype=torch.long, device=self.device)
            attention_mask[i, :slen] = 1
            # In token_logprobs of shape (B, max_len - 1), index j corresponds to predicting input_ids[i, j+1]
            # Response tokens are at input_ids indices [p_len, p_len + r_len - 1],
            # which correspond to logprob indices [p_len - 1, p_len + r_len - 2].
            start_lp = max(p_len - 1, 0)
            end_lp = min(p_len - 1 + r_len, max_len - 1)
            response_mask[i, start_lp:end_lp] = True

        rewards_np = np.array(all_rewards, dtype=np.float64)
        adv_np, zero_adv_frac = compute_grpo_advantages(rewards_np, group_size=group_size)
        advantages_t = torch.tensor(adv_np, dtype=torch.float32, device=self.device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "advantages": advantages_t,
            "rewards": rewards_np,
            "zero_adv_frac": zero_adv_frac,
            "resp_len_mean": float(np.mean(all_resp_lens)) if all_resp_lens else 0.0,
            "vllm_rollout_logprobs": all_rollout_logprobs,
            "prompt_lens": all_prompt_lens,
            "resp_lens": all_resp_lens,
        }

    def verify_logprob_sync(
        self,
        behav_model,
        rollout_batch: dict[str, Any],
    ) -> tuple[float, float]:
        """Compare token logprobs between rollout engine and GPU behavior model (§6.2, §6.3)."""
        import torch

        with torch.no_grad():
            input_ids = rollout_batch["input_ids"][:4]
            attention_mask = rollout_batch["attention_mask"][:4]
            response_mask = rollout_batch["response_mask"][:4]

            behav_lp = compute_response_logprobs(
                behav_model, input_ids, attention_mask, use_bf16=self.use_bf16
            )

            if self.backend == "hf":
                # Compare rollout hf_model (bf16 weights) forward logprobs against behav_model (fp32 weights + bf16 autocast)
                rollout_lp = compute_response_logprobs(
                    self.hf_model, input_ids, attention_mask, use_bf16=self.use_bf16
                )
                diff = torch.abs(behav_lp[response_mask] - rollout_lp[response_mask])
                if diff.numel() == 0:
                    return 0.0, 0.0
                return float(diff.max().item()), float(diff.mean().item())

            elif self.backend == "vllm" and rollout_batch.get("vllm_rollout_logprobs"):
                diffs = []
                for i in range(min(4, len(rollout_batch["vllm_rollout_logprobs"]))):
                    v_lp = rollout_batch["vllm_rollout_logprobs"][i]
                    p_len = rollout_batch["prompt_lens"][i]
                    r_len = rollout_batch["resp_lens"][i]
                    b_lp = behav_lp[i, p_len - 1 : p_len - 1 + r_len].detach().cpu().numpy()
                    n = min(len(v_lp), len(b_lp))
                    if n > 0:
                        diffs.append(np.abs(v_lp[:n] - b_lp[:n]))
                if not diffs:
                    return 0.0, 0.0
                cat = np.concatenate(diffs)
                return float(np.max(cat)), float(np.mean(cat))

        return 0.0, 0.0


def write_llm_env_file(out_dir: str | Path):
    """Write pip freeze, nvidia-smi, and git commit hash to env.txt (§2.7)."""
    out_path = Path(out_dir) / "env.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Timestamp: {datetime.datetime.now().isoformat()}"]
    for cmd, title in [
        (["git", "rev-parse", "HEAD"], "Git Commit"),
        (["nvidia-smi"], "NVIDIA-SMI"),
        ([sys.executable, "-m", "pip", "freeze"], "Pip Freeze"),
    ]:
        lines.append(f"\n--- {title} ---")
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=15)
            lines.append(out.strip())
        except Exception as e:
            lines.append(f"Unavailable: {e}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model,
    behav_state_dict: dict[str, Any],
    optimizer,
    opt_name: str,
    config: dict,
    min_free_gb: float = 10.0,
) -> Path:
    """Save step checkpoint containing fp32 weights, behavior weights, and m_{t-1} (§6.4)."""
    import torch

    check_disk_free_gb(ckpt_dir, min_free_gb=min_free_gb)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ckpt_step_{step:04d}.pt"

    fp32_weights = {
        k: v.detach().to(device="cpu", dtype=torch.float32).clone()
        for k, v in model.state_dict().items()
    }
    behav_cpu = {
        k: v.detach().to(device="cpu", dtype=torch.float32).clone()
        for k, v in behav_state_dict.items()
    }
    m_dict, sq_dict, norm_m = extract_momentum_buffers(optimizer, model, opt_name)

    payload = {
        "step": int(step),
        "optimizer_name": opt_name,
        "model_state_dict": fp32_weights,
        "behav_state_dict": behav_cpu,
        "momentum_buffer": m_dict,
        "exp_avg_sq": sq_dict,
        "norm_m": norm_m,
        "config": config,
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def build_optimizer(model, opt_name: str, lr: float, momentum: float = 0.9, dampening: float = 0.0, betas=(0.9, 0.999), weight_decay: float = 0.0):
    """Build SGD, SGD+mom, or AdamW optimizer (§6.2, §8, §11)."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    key = opt_name.lower()
    if key == "sgd":
        return torch.optim.SGD(params, lr=float(lr), momentum=0.0, dampening=0.0, weight_decay=float(weight_decay))
    elif key == "sgd_mom":
        return torch.optim.SGD(params, lr=float(lr), momentum=float(momentum), dampening=float(dampening), weight_decay=float(weight_decay))
    elif key in ("adamw", "rmsprop_via_adamw"):
        b1, b2 = float(betas[0]), float(betas[1])
        return torch.optim.AdamW(params, lr=float(lr), betas=(b1, b2), weight_decay=float(weight_decay))
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def main(config: dict | str | Path) -> dict[str, Any]:
    """Main RLVR training loop implementing SPEC E5 §6.1–§6.5."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm.data import load_gsm8k_pools
    from llm.eval import evaluate_gsm8k_hf, evaluate_gsm8k_vllm

    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = copy.deepcopy(config)

    out_dir = Path(cfg.get("output_dir", cfg.get("out_dir", "results/e5/l0")))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_llm_env_file(out_dir)

    seed = int(cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    use_bf16 = bool(cfg.get("compute_dtype", "bf16") == "bf16" and device.type == "cuda")

    model_name = cfg["model"]
    backend_name = cfg.get("rollout_backend", "hf" if "vllm" not in cfg else "vllm")
    k = int(cfg.get("k", 0))
    c_raw = cfg.get("c", float("inf"))
    c = float("inf") if (c_raw is None or str(c_raw).lower() in ("inf", "none")) else float(c_raw)
    steps = int(cfg.get("steps", cfg.get("smoke_checks", {}).get("steps", 10)))
    eval_every = int(cfg.get("eval_every", 50))
    save_ckpts_at = set(int(x) for x in cfg.get("save_checkpoints_at", [0, 50, 100]))
    min_free_gb = float(cfg.get("min_free_disk_gb", 10.0))

    opt_name = cfg.get("optimizer", cfg.get("smoke_checks", {}).get("optimizer", "sgd_mom"))
    lr = float(cfg.get("lr", cfg.get("smoke_checks", {}).get("lr", 0.01)))
    momentum = float(cfg.get("momentum", cfg.get("smoke_checks", {}).get("momentum", 0.9)))
    dampening = float(cfg.get("dampening", cfg.get("smoke_checks", {}).get("dampening", 0.0)))
    betas = cfg.get("betas", [0.9, 0.999])
    micro_batch_size = cfg.get("micro_batch_size", None)

    rollout_cfg = cfg.get("rollout", {})
    prompts_per_step = int(rollout_cfg.get("prompts_per_step", cfg.get("prompts_per_step", 32)))
    group_size = int(rollout_cfg.get("group_size", cfg.get("group_size", 4)))
    temperature = float(rollout_cfg.get("temperature", cfg.get("temperature", 1.0)))
    top_p = float(rollout_cfg.get("top_p", cfg.get("top_p", 1.0)))
    max_new_tokens = int(rollout_cfg.get("max_new_tokens", cfg.get("max_new_tokens", 512)))

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset", cfg.get("dataset", "openai/gsm8k"))
    train_pool_size = int(data_cfg.get("train_pool_size", cfg.get("train_pool_size", 1024)))
    eval_pool_size = int(data_cfg.get("eval_pool_size", cfg.get("eval_pool_size", 500)))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_pool, eval_pool = load_gsm8k_pools(
        tokenizer=tokenizer,
        train_pool_size=train_pool_size,
        eval_pool_size=eval_pool_size,
        seed=seed,
        dataset_name=dataset_name,
    )

    # 1. Master training model in fp32 (§6.2)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.train()
    if bool(cfg.get("gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 2. Behavior model replica on GPU in fp32 (§6.3)
    behav_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    behav_model.eval()
    behav_model.requires_grad_(False)

    # 3. Switchable rollout backend ('hf' or 'vllm')
    rollout_engine = RolloutBackend(
        backend=backend_name,
        model_name=model_name,
        tokenizer=tokenizer,
        device=device,
        vllm_cfg=cfg.get("vllm", {}),
        use_bf16=use_bf16,
    )

    optimizer = build_optimizer(
        model, opt_name=opt_name, lr=lr, momentum=momentum, dampening=dampening, betas=betas
    )
    staleness_buf = StalenessBuffer(k=k)

    log_file = out_dir / "run_log.txt"
    csv_file = out_dir / "train_log.csv"
    ckpt_dir = out_dir / "checkpoints"

    rng = np.random.default_rng(seed)
    step_rows = []
    saved_ckpt_paths = {}

    fieldnames = [
        "step",
        "loss",
        "reward_mean",
        "zero_adv_frac",
        "resp_len_mean",
        "grad_norm",
        "update_norm",
        "online_C1",
        "trunc_rate",
        "mean_w",
        "approx_kl",
        "sync_logprob_max_diff",
        "sync_logprob_mean_diff",
        "eval_acc",
        "step_time_gen",
        "step_time_train",
        "step_time_sync",
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(steps):
            # Push current fp32 snapshot theta_t into staleness ring buffer
            staleness_buf.push(step, model.state_dict())
            behav_sd = staleness_buf.get_behavior_weights(step)

            # Save checkpoint at step t BEFORE step t's gradient update so m_{t-1} is exact (§6.4)
            if step in save_ckpts_at:
                cp = save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    step=step,
                    model=model,
                    behav_state_dict=behav_sd,
                    optimizer=optimizer,
                    opt_name=opt_name,
                    config=cfg,
                    min_free_gb=min_free_gb,
                )
                saved_ckpt_paths[step] = str(cp)

            # Sync behavior weights theta_{t-k} into GPU behavior model and rollout engine (§6.3)
            t_sync_0 = time.perf_counter()
            behav_model.load_state_dict(
                {name: val.to(device=device, dtype=torch.float32) for name, val in behav_sd.items()},
                strict=True,
            )
            rollout_engine.sync_weights(behav_sd)
            step_time_sync = time.perf_counter() - t_sync_0

            # Generate rollouts
            t_gen_0 = time.perf_counter()
            batch_idx = rng.choice(len(train_pool), size=min(prompts_per_step, len(train_pool)), replace=False)
            batch_items = [train_pool[int(i)] for i in batch_idx]
            rollout_batch = rollout_engine.generate_group_rollouts(
                items=batch_items,
                group_size=group_size,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=seed + step,
            )
            step_time_gen = time.perf_counter() - t_gen_0

            # Weight sync logprob consistency verification (§6.2)
            sync_max_diff, sync_mean_diff = rollout_engine.verify_logprob_sync(behav_model, rollout_batch)

            # Extract m_{t-1} before optimizer step for online_C1 (§6.4, §6.5)
            m_prev_dict, _, _ = extract_momentum_buffers(optimizer, model, opt_name)
            prev_params = {
                name: p.detach().clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }

            # Forward & backward with global N_tok normalization
            t_train_0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            train_stats = accumulate_grpo_tis_microbatches(
                model=model,
                behav_model=behav_model,
                input_ids=rollout_batch["input_ids"],
                attention_mask=rollout_batch["attention_mask"],
                response_mask=rollout_batch["response_mask"],
                advantages=rollout_batch["advantages"],
                k=k,
                c=c,
                micro_batch_size=micro_batch_size,
                use_bf16=use_bf16,
            )

            grad_dict = {}
            grad_norm_sq = 0.0
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grad_dict[name] = p.grad.detach()
                    g64 = p.grad.detach().to(dtype=torch.float64)
                    grad_norm_sq += float(torch.sum(g64 * g64).item())
            grad_norm = float(np.sqrt(grad_norm_sq))
            online_c1 = compute_online_c1(m_prev_dict, grad_dict)

            optimizer.step()

            update_norm_sq = 0.0
            for name, p in model.named_parameters():
                if p.requires_grad:
                    diff64 = (p.detach() - prev_params[name]).to(dtype=torch.float64)
                    update_norm_sq += float(torch.sum(diff64 * diff64).item())
            update_norm = float(np.sqrt(update_norm_sq))
            step_time_train = time.perf_counter() - t_train_0

            # Periodic evaluation (§6.5)
            eval_acc = float("nan")
            if eval_every > 0 and (step % eval_every == 0 or step == steps - 1):
                if backend_name == "vllm":
                    rollout_engine.sync_weights(model.state_dict())
                    ev = evaluate_gsm8k_vllm(rollout_engine.vllm_engine, eval_pool, max_new_tokens=max_new_tokens)
                else:
                    ev = evaluate_gsm8k_hf(model, tokenizer, eval_pool, max_new_tokens=max_new_tokens, device=device)
                eval_acc = ev["eval_acc"]

            row = {
                "step": step,
                "loss": train_stats["loss"],
                "reward_mean": float(np.mean(rollout_batch["rewards"])),
                "zero_adv_frac": rollout_batch["zero_adv_frac"],
                "resp_len_mean": rollout_batch["resp_len_mean"],
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "online_C1": online_c1,
                "trunc_rate": train_stats["trunc_rate"],
                "mean_w": train_stats["mean_w"],
                "approx_kl": train_stats["approx_kl"],
                "sync_logprob_max_diff": sync_max_diff,
                "sync_logprob_mean_diff": sync_mean_diff,
                "eval_acc": eval_acc,
                "step_time_gen": step_time_gen,
                "step_time_train": step_time_train,
                "step_time_sync": step_time_sync,
            }
            step_rows.append(row)
            writer.writerow(row)
            cf.flush()

            log_line = (
                f"[step {step:03d}/{steps:03d}] loss={row['loss']:.4f} reward={row['reward_mean']:.3f} "
                f"zero_adv={row['zero_adv_frac']:.2f} grad_norm={grad_norm:.4f} C1={online_c1:.4f} "
                f"trunc={row['trunc_rate']:.4f} sync_max_diff={sync_max_diff:.2e} eval_acc={eval_acc:.3f}"
            )
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(log_line + "\n")

    # Write run summary CSV
    summary_path = out_dir / "summary.csv"
    diverged = any(not np.isfinite(r["loss"]) or not np.isfinite(r["grad_norm"]) for r in step_rows)
    summary_row = {
        "model": model_name,
        "optimizer": opt_name,
        "lr": lr,
        "momentum": momentum if opt_name == "sgd_mom" else 0.0,
        "effective_lr": lr / (1.0 - momentum) if opt_name == "sgd_mom" and momentum < 1.0 else lr,
        "k": k,
        "c": c,
        "seed": seed,
        "steps": steps,
        "final_reward_mean": step_rows[-1]["reward_mean"] if step_rows else float("nan"),
        "final_eval_acc": step_rows[-1]["eval_acc"] if step_rows else float("nan"),
        "diverged": diverged,
        "mean_step_time_gen": float(np.mean([r["step_time_gen"] for r in step_rows])) if step_rows else 0.0,
        "mean_step_time_train": float(np.mean([r["step_time_train"] for r in step_rows])) if step_rows else 0.0,
        "mean_step_time_sync": float(np.mean([r["step_time_sync"] for r in step_rows])) if step_rows else 0.0,
        "max_sync_logprob_diff": float(np.max([r["sync_logprob_max_diff"] for r in step_rows])) if step_rows else 0.0,
    }
    with open(summary_path, "w", newline="", encoding="utf-8") as sf:
        sw = csv.DictWriter(sf, fieldnames=list(summary_row.keys()))
        sw.writeheader()
        sw.writerow(summary_row)

    (out_dir / "DONE").write_text(f"DONE at {datetime.datetime.now().isoformat()}\n", encoding="utf-8")
    return {
        "step_rows": step_rows,
        "summary": summary_row,
        "checkpoints": saved_ckpt_paths,
        "output_dir": str(out_dir),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SPEC E5 LLM RLVR training.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    main(args.config)
