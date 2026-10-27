"""独立参考公式（不依赖被测实现），供各测试文件共用。

这些函数故意用与 SPEC 不同的写法实现（例如直接用定义、伪逆、枚举），
用来交叉验证 stalemom 包。不要修改本目录下的任何文件。
"""
import numpy as np
import pytest


# ---------- 参考公式 ----------

def ref_log_softmax(theta):
    theta = np.asarray(theta, dtype=np.float64)
    m = theta.max(axis=-1, keepdims=True)
    z = theta - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def ref_softmax(theta):
    return np.exp(ref_log_softmax(theta))


def ref_objective(theta, r, logp_ref, lam):
    lp = ref_log_softmax(theta)
    p = np.exp(lp)
    return (p * r).sum(-1) - lam * (p * (lp - logp_ref)).sum(-1)


def ref_optimal_value(r, logp_ref, lam):
    a = logp_ref + r / lam
    m = a.max()
    return lam * (m + np.log(np.exp(a - m).sum()))


def ref_vanilla_grad(theta, r, logp_ref, lam):
    lp = ref_log_softmax(theta)
    p = np.exp(lp)
    v = r - lam * (lp - logp_ref)
    return p * v - p * (p * v).sum(-1, keepdims=True)


def ref_npg_dir(theta, r, logp_ref, lam):
    lp = ref_log_softmax(theta)
    v = r - lam * (lp - logp_ref)
    return v - v.mean(-1, keepdims=True)


def ref_tis_bias(p, q, r, c):
    v = r * (np.minimum(1.0, c * q / p) - 1.0)
    return v - v.mean(-1, keepdims=True)


def ref_kl(p, q):
    return (p * (np.log(p) - np.log(q))).sum(-1)


# ---------- 公共夹具 ----------

@pytest.fixture
def rng():
    return np.random.default_rng(20261027)


@pytest.fixture
def problem(rng):
    """一个小的随机问题实例：K=7，3 个好动作，单行 theta 与一个不同的行为分布 q。"""
    K = 7
    r = np.array([1, 0, 1, 0, 0, 1, 0], dtype=np.float64)
    lam = 0.7
    logp_ref = ref_log_softmax(rng.normal(size=K))
    theta = rng.normal(scale=1.5, size=(1, K))
    q_theta = theta + rng.normal(scale=1.0, size=(1, K))
    p = ref_softmax(theta)
    q = ref_softmax(q_theta)
    return dict(K=K, r=r, lam=lam, logp_ref=logp_ref, theta=theta, p=p, q=q)


def onehot_counts(K, R=1):
    """返回 K 个 (R,K) 的单样本计数矩阵，第 a 个只在动作 a 上为 1。"""
    eye = np.eye(K, dtype=np.int64)
    return [np.repeat(eye[a:a + 1], R, axis=0) for a in range(K)]
