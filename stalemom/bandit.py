import numpy as np


def log_softmax(theta: np.ndarray) -> np.ndarray:
    """Compute stable log-softmax along the last dimension."""
    theta = np.asarray(theta, dtype=np.float64)
    m = np.max(theta, axis=-1, keepdims=True)
    z = theta - m
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def softmax(theta: np.ndarray) -> np.ndarray:
    """Compute softmax along the last dimension."""
    return np.exp(log_softmax(theta))


def project(v: np.ndarray) -> np.ndarray:
    """Project onto simplex tangent space: Pi v = v - mean(v, axis=-1, keepdims=True)."""
    v = np.asarray(v, dtype=np.float64)
    return v - np.mean(v, axis=-1, keepdims=True)


def objective(theta: np.ndarray, r: np.ndarray, logp_ref: np.ndarray, lam: float) -> np.ndarray:
    """Compute objective J(theta) = sum_a p_theta(a) r_a - lam * KL(p_theta || p_ref).
    
    Returns array of shape (R,) matching batch dimensions of theta.
    """
    theta = np.asarray(theta, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    logp_ref = np.asarray(logp_ref, dtype=np.float64)
    
    lp = log_softmax(theta)
    p = np.exp(lp)
    # E_p[r] - lam * E_p[log p - log p_ref]
    val = np.sum(p * r, axis=-1) - lam * np.sum(p * (lp - logp_ref), axis=-1)
    return val


def optimal_value(r: np.ndarray, logp_ref: np.ndarray, lam: float) -> float:
    """Closed-form optimal value: J* = lam * log sum_a p_ref(a) * exp(r_a / lam)."""
    r = np.asarray(r, dtype=np.float64)
    logp_ref = np.asarray(logp_ref, dtype=np.float64)
    
    a = logp_ref + r / lam
    m = np.max(a)
    return float(lam * (m + np.log(np.sum(np.exp(a - m)))))


def exact_npg_dir(theta: np.ndarray, r: np.ndarray, logp_ref: np.ndarray, lam: float) -> np.ndarray:
    """Exact natural policy gradient direction: h(theta) = Pi [ r - lam * (log p_theta - log p_ref) ]."""
    theta = np.asarray(theta, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    logp_ref = np.asarray(logp_ref, dtype=np.float64)
    
    lp = log_softmax(theta)
    v = r - lam * (lp - logp_ref)
    return project(v)


def vanilla_grad(theta: np.ndarray, r: np.ndarray, logp_ref: np.ndarray, lam: float) -> np.ndarray:
    """Standard policy gradient: grad J(theta) = F(theta) [ r - lam * (log p_theta - log p_ref) ].
    
    F(theta) v = p * v - p * (p^T v).
    """
    theta = np.asarray(theta, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    logp_ref = np.asarray(logp_ref, dtype=np.float64)
    
    lp = log_softmax(theta)
    p = np.exp(lp)
    v = r - lam * (lp - logp_ref)
    return p * v - p * np.sum(p * v, axis=-1, keepdims=True)
