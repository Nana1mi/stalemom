import numpy as np
from stalemom.bandit import project


def sample_counts(rng: np.random.Generator, probs: np.ndarray, B: int) -> np.ndarray:
    """Sample action counts from categorical distributions.

    probs: (R, K) or (K,) probabilities, rows sum to 1.
    B: number of samples per row.
    Returns: (R, K) integer counts, each row summing to B.
    """
    probs = np.asarray(probs, dtype=np.float64)
    is_1d = (probs.ndim == 1)
    if is_1d:
        probs = probs[None, :]
    # Normalize probabilities to avoid slight floating point issues in multinomial
    probs = np.maximum(probs, 0.0)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    counts = rng.multinomial(B, probs)
    return counts


def reward_term(mode: str, counts, p, q, r, c, B) -> np.ndarray:
    """Compute the reward term estimate rho_hat according to Table 2.3.

    mode: 'exact' | 'onpolicy' | 'is' | 'tis' | 'tis_mean' | 'tis_debiased'
    counts: (R, K) integer counts
    p: current policy probabilities (R, K)
    q: behavior policy probabilities (R, K)
    r: reward vector (K,) or (R, K)
    c: truncation threshold (float or None)
    B: batch size (int)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    B = float(B)

    if mode == "exact":
        return np.broadcast_to(r, p.shape).copy()
    elif mode == "onpolicy":
        counts = np.asarray(counts, dtype=np.float64)
        return (counts / B) * r / p
    elif mode == "is":
        counts = np.asarray(counts, dtype=np.float64)
        return (counts / B) * r / q
    elif mode == "tis":
        counts = np.asarray(counts, dtype=np.float64)
        return (counts / B) * np.minimum(p / q, c) * r / p
    elif mode == "tis_mean":
        return r * np.minimum(1.0, c * q / p)
    elif mode == "tis_debiased":
        counts = np.asarray(counts, dtype=np.float64)
        tis_est = (counts / B) * np.minimum(p / q, c) * r / p
        bias = r * (np.minimum(1.0, c * q / p) - 1.0)
        return tis_est - bias
    else:
        raise ValueError(f"Unknown estimator mode: {mode}")


def expected_reward_term(mode: str, p, q, r, c) -> np.ndarray:
    """Exact expectation of the reward term: E[rho_hat].

    For 'is', 'onpolicy', 'tis_debiased', 'exact', returns r.
    For 'tis', 'tis_mean', returns r * min(1, c * q / p).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    if mode in ("tis", "tis_mean"):
        return r * np.minimum(1.0, c * q / p)
    else:
        return np.broadcast_to(r, p.shape)


def tis_bias(p, q, r, c) -> np.ndarray:
    """Closed-form truncation bias: b = Pi [ r * (min(1, c * q / p) - 1) ]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    v = r * (np.minimum(1.0, c * q / p) - 1.0)
    return project(v)


def direction(reward_est, logp, logp_ref, lam) -> np.ndarray:
    """Stochastic / exact direction: g_hat = Pi [ rho_hat - lam * (log p - log p_ref) ]."""
    reward_est = np.asarray(reward_est, dtype=np.float64)
    logp = np.asarray(logp, dtype=np.float64)
    logp_ref = np.asarray(logp_ref, dtype=np.float64)
    v = reward_est - lam * (logp - logp_ref)
    return project(v)


def truncation_rate(counts, p, q, c, B: int) -> np.ndarray:
    """Fraction of truncated samples per run. Truncated when p / q > c (strictly)."""
    if c is None:
        return np.zeros(counts.shape[:-1], dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    mask = (p / q) > c
    return np.sum(np.where(mask, counts, 0.0), axis=-1) / float(B)
