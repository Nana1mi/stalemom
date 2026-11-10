"""Reward extraction and GRPO group advantage normalization for GSM8K (SPEC E5 §6.1, §6.2)."""

import re
from decimal import Decimal, InvalidOperation
import numpy as np


_BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_GSM8K_GT_PATTERN = re.compile(r"####\s*([^\n\r]+)")
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def normalize_numeric_str(text: str | None) -> str | None:
    """Normalize a numeric string into a canonical representation for exact match."""
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    if not cleaned:
        return None
    m = _NUMBER_PATTERN.search(cleaned)
    if not m:
        return None
    num_str = m.group(0).replace(",", "")
    if "/" in num_str:
        parts = num_str.split("/")
        if len(parts) == 2:
            try:
                num = Decimal(parts[0])
                den = Decimal(parts[1])
                if den == 0:
                    return None
                val = num / den
                return str(val.normalize())
            except InvalidOperation:
                return None
    try:
        d = Decimal(num_str)
        # Canonicalize integer vs decimal (e.g. 42.0 -> 42)
        if d == d.to_integral_value():
            return str(int(d))
        return str(d.normalize())
    except InvalidOperation:
        return None


def extract_boxed_answer(response: str) -> str | None:
    """Extract the last \\boxed{...} content from model response and normalize its number.

    Per SPEC E5 §6.1: prompt instructs the model to put the final answer inside \\boxed{}.
    If parsing fails, returns None (which scores 0).
    """
    matches = _BOXED_PATTERN.findall(response)
    if not matches:
        return None
    return normalize_numeric_str(matches[-1])


def extract_gsm8k_ground_truth(answer_field: str) -> str | None:
    """Extract the canonical ground-truth number after '####' in GSM8K dataset."""
    m = _GSM8K_GT_PATTERN.search(answer_field)
    if m:
        return normalize_numeric_str(m.group(1))
    return normalize_numeric_str(answer_field)


def compute_exact_match_reward(response: str, ground_truth: str) -> float:
    """Return 1.0 if extracted \\boxed{} answer matches ground truth, else 0.0."""
    pred = extract_boxed_answer(response)
    gt = normalize_numeric_str(ground_truth) if "####" not in ground_truth else extract_gsm8k_ground_truth(ground_truth)
    if pred is None or gt is None:
        return 0.0
    return 1.0 if pred == gt else 0.0


def compute_grpo_advantages(
    rewards: np.ndarray,
    group_size: int = 4,
    eps: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Compute GRPO group-normalized advantages (SPEC E5 §6.2).

    Args:
        rewards: 1D array of shape (num_prompts * group_size,) or 2D (num_prompts, group_size).
        group_size: Number of responses per prompt (default 4).
        eps: Numerical stabilizer in (r - mean) / (std + eps).

    Returns:
        advantages: Same shape as input `rewards`, float32/float64.
            Groups with std == 0 have advantages strictly equal to 0.0.
        zero_adv_frac: Fraction of prompt groups where std == 0.
    """
    orig_shape = rewards.shape
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    means = np.mean(r, axis=1, keepdims=True)
    stds = np.std(r, axis=1, keepdims=True)
    zero_mask = (stds.squeeze(-1) == 0.0)
    zero_adv_frac = float(np.mean(zero_mask)) if r.shape[0] > 0 else 0.0

    adv = (r - means) / (stds + eps)
    # Explicitly enforce exact 0.0 when group std == 0
    adv[zero_mask, :] = 0.0
    return adv.reshape(orig_shape), zero_adv_frac
