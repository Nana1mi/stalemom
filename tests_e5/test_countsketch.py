"""SPEC E5 §5.5 & §12: Unit tests for CountSketch linearity and unbiased inner product estimation."""
import numpy as np

from stalemom.diag import CountSketch


def test_countsketch_linearity():
    """Verify CountSketch is strictly linear: sketch(a*u + b*v) == a*sketch(u) + b*sketch(v)."""
    cs = CountSketch(D=2 ** 14)
    rng = np.random.default_rng(123)
    u = {
        "layer1.weight": rng.normal(size=(64, 128)),
        "layer2.bias": rng.normal(size=(128,)),
    }
    v = {
        "layer1.weight": rng.normal(size=(64, 128)),
        "layer2.bias": rng.normal(size=(128,)),
    }
    alpha, beta = 2.5, -1.3
    comb = {k: alpha * u[k] + beta * v[k] for k in u}

    sk_u = cs.sketch_dict(u)
    sk_v = cs.sketch_dict(v)
    sk_comb = cs.sketch_dict(comb)

    np.testing.assert_allclose(sk_comb, alpha * sk_u + beta * sk_v, rtol=1e-12, atol=1e-12)


def test_countsketch_inner_product_unbiased_and_accurate():
    """Verify inner product estimate is unbiased across sketch seeds and accurate within 5% (§5.5)."""
    rng = np.random.default_rng(456)
    u = rng.normal(size=50000)
    v = 0.6 * u + 0.8 * rng.normal(size=50000)
    exact_inner = float(np.dot(u, v))

    # 1. Single sketch with D = 2^18 has relative error well below 5%
    cs = CountSketch(D=2 ** 18)
    sk_u = cs.sketch_vector(u, name="param_block")
    sk_v = cs.sketch_vector(v, name="param_block")
    est_inner = cs.inner(sk_u, sk_v)
    rel_err = abs(est_inner - exact_inner) / abs(exact_inner)
    assert rel_err < 0.05, f"Relative error {rel_err:.4f} >= 5%"

    # 2. Unbiasedness across different parameter names (independent hash functions)
    estimates = []
    cs_small = CountSketch(D=2 ** 12)
    for idx in range(100):
        name = f"tensor_{idx}"
        su = cs_small.sketch_vector(u, name=name)
        sv = cs_small.sketch_vector(v, name=name)
        estimates.append(cs_small.inner(su, sv))

    arr = np.asarray(estimates, dtype=np.float64)
    mean_est = float(np.mean(arr))
    se_est = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    assert abs(mean_est - exact_inner) < 3.0 * se_est


def test_countsketch_cross_instance_determinism():
    """Verify two separate CountSketch instances produce identical sketches for the same parameter name."""
    rng = np.random.default_rng(789)
    x = rng.normal(size=(32, 32))
    cs1 = CountSketch(D=2 ** 15)
    cs2 = CountSketch(D=2 ** 15)
    np.testing.assert_array_equal(
        cs1.sketch_tensor("model.embed.weight", x),
        cs2.sketch_tensor("model.embed.weight", x),
    )
