import numpy as np


def omb_from_rho(eta: float, rho: float) -> float:
    """Compute omb = 1 - beta from learning rate eta and rho = eta / (1 - beta).

    rho = 0.0 denotes no momentum (omb = 1.0).
    Requires rho >= eta, otherwise raises ValueError.
    """
    if rho == 0.0:
        return 1.0
    if rho < eta:
        raise ValueError(f"rho ({rho}) must be >= eta ({eta})")
    return float(eta / rho)


def bias_correction_den(omb: float, t: int) -> float:
    """Compute bias correction denominator: den_t = 1 - (1 - omb)^(t+1).

    Must be computed using -np.expm1((t+1) * np.log1p(-omb)) for numerical stability.
    When omb >= 1.0, returns 1.0.
    """
    if omb >= 1.0:
        return 1.0
    return float(-np.expm1((t + 1) * np.log1p(-omb)))


def ema_step(m: np.ndarray, g: np.ndarray, omb: float) -> np.ndarray:
    """Single EMA step: m_t = (1 - omb) * m_{t-1} + omb * g_t."""
    return (1.0 - omb) * m + omb * g


def history_age(omb: float, t: int) -> float:
    """Compute weighted history age H_t = sum_{j=0}^t j * alpha_{t,j},
    where alpha_{t,j} = omb * (1 - omb)^j / den_t.
    """
    if omb >= 1.0:
        return 0.0
    den = bias_correction_den(omb, t)
    j = np.arange(t + 1, dtype=np.float64)
    # alpha_{t,j} = omb * exp(j * log1p(-omb)) / den
    w = omb * np.exp(j * np.log1p(-omb)) / den
    return float(np.sum(j * w))
