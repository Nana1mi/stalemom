"""Minimal RLVR training loop with staleness ring buffer, token-level TIS, and online C1 logging (SPEC E5 §6)."""

from collections import deque
import copy
import shutil
from pathlib import Path
from typing import Any
import numpy as np

from llm.reward import compute_grpo_advantages


def compute_token_weights_numpy(
    log_pi_theta: np.ndarray,
    log_pi_behav: np.ndarray,
    k: int,
    c: float = float("inf"),
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute token-level IS / TIS weights w_tok and diagnostics (SPEC E5 §6.2, §6.3).

    Rules:
    - k == 0: strictly force w_tok == 1.0 everywhere.
    - k > 0 and c == inf: full unclipped IS ratio exp(log_pi_theta - log_pi_behav).
    - k > 0 and finite c: w_tok = min(exp(log_pi_theta - log_pi_behav), c).
    """
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
    """Compute GRPO token-level TIS loss and exact gradient w.r.t. log_pi_theta (SPEC E5 §6.2).

    Loss:
        L = - (1 / N_tok) * sum_{tokens} sg(w_tok) * A * log_pi_theta(y_tok | .)

    Because stop-gradient sg(.) is applied to w_tok, the gradient w.r.t. log_pi_theta is:
        dL / d(log_pi_theta) = - (1 / N_tok) * w_tok * A  (on valid tokens, 0 on masked tokens).
    """
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
    # Broadcast advantages across sequence tokens
    adv_tok = np.broadcast_to(adv, log_p.shape)
    active_terms = np.where(valid, w * adv_tok * log_p, 0.0)
    loss = float(-np.sum(active_terms) / n_tok)

    # Exact gradient w.r.t. log_pi_theta with stop-gradient on w
    grad_log_p = np.where(valid, -(w * adv_tok) / n_tok, 0.0)
    return loss, grad_log_p, w, stats


def compute_grpo_tis_loss_torch(
    log_pi_theta,
    log_pi_behav,
    advantages,
    k: int,
    c: float = float("inf"),
    mask=None,
):
    """PyTorch implementation of SPEC E5 §6.2 loss with stop-gradient on w_tok."""
    import torch

    if mask is None:
        valid = torch.ones_like(log_pi_theta, dtype=torch.bool)
    else:
        valid = mask.to(dtype=torch.bool)

    n_tok = valid.sum().clamp(min=1)
    with torch.no_grad():
        if k == 0:
            w = torch.ones_like(log_pi_theta)
            trunc_rate = 0.0
            mean_w = 1.0
            approx_kl = 0.0
        else:
            log_ratio = log_pi_theta.detach() - log_pi_behav.detach()
            ratio = torch.exp(torch.clamp(log_ratio, -50.0, 50.0))
            if c is None or np.isinf(c):
                w = ratio
                trunc_rate = 0.0
            else:
                w = torch.clamp(ratio, max=float(c))
                trunc_rate = float(((ratio > float(c)) & valid).sum().item() / n_tok.item())
            mean_w = float(w[valid].mean().item())
            approx_kl = float(log_ratio[valid].mean().item())

    adv = advantages
    if adv.ndim == 1 and log_pi_theta.ndim == 2:
        adv = adv.unsqueeze(-1)

    valid_f = valid.to(dtype=log_pi_theta.dtype)
    loss = -(w.detach() * adv.detach() * log_pi_theta * valid_f).sum() / n_tok
    stats = {
        "trunc_rate": trunc_rate,
        "mean_w": mean_w,
        "approx_kl": approx_kl,
    }
    return loss, w, stats


class StalenessBuffer:
    """CPU fp32 ring buffer storing the most recent k+1 parameter snapshots (SPEC E5 §6.3).

    - At step t < k: returns the initial snapshot theta_0.
    - At step t >= k: returns theta_{t-k}.
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
                snap[name] = val.detach().to(device="cpu", dtype=val.dtype).clone()
            else:
                snap[name] = np.array(val, dtype=np.float32, copy=True)
        if self._theta_0 is None:
            self._theta_0 = copy.deepcopy(snap)
        self._snapshots.append((int(step), snap))

    def get_behavior_weights(self, current_step: int) -> dict[str, Any]:
        """Return theta_{t-k} (or theta_0 when current_step < k)."""
        if not self._snapshots:
            raise RuntimeError("StalenessBuffer is empty; call push(0, init_weights) first")
        if self.k == 0:
            return self._snapshots[-1][1]
        if current_step < self.k:
            return self._theta_0
        target_step = current_step - self.k
        for step_idx, snap in self._snapshots:
            if step_idx == target_step:
                return snap
        return self._snapshots[0][1]


def compute_online_c1(
    m_dict: dict[str, np.ndarray],
    g_dict: dict[str, np.ndarray],
) -> float:
    """Compute online_C1 = cos(m_{t-1}, g_t) across all parameters (SPEC E5 §6.5)."""
    dot = 0.0
    norm_m_sq = 0.0
    norm_g_sq = 0.0
    for name, g in g_dict.items():
        g64 = np.asarray(g, dtype=np.float64)
        norm_g_sq += float(np.sum(g64 * g64))
        if name in m_dict:
            m64 = np.asarray(m_dict[name], dtype=np.float64)
            norm_m_sq += float(np.sum(m64 * m64))
            dot += float(np.sum(m64 * g64))
    if norm_m_sq <= 0.0 or norm_g_sq <= 0.0:
        return float("nan")
    return float(dot / (np.sqrt(norm_m_sq) * np.sqrt(norm_g_sq)))


def check_disk_free_gb(path: str | Path, min_free_gb: float = 10.0) -> float:
    """Verify disk free space >= min_free_gb (SPEC E5 §6.4)."""
    usage = shutil.disk_usage(Path(path).resolve())
    free_gb = usage.free / (1024.0**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Disk free space ({free_gb:.2f} GB) is below the 10 GB safety threshold (§6.4). Stopping."
        )
    return free_gb
