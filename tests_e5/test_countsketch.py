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


def test_llm_torch_countsketch_gpu_fp32_no_cache():
    """Item 5: Verify GPU torch CountSketch uses parameter-name seed, no cache, index_add_, fp32 sketch, float64 inner."""
    import torch
    from llm.diag import CountSketch as TorchCountSketch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cs1 = TorchCountSketch(D=2 ** 14, device=dev)
    cs2 = TorchCountSketch(D=2 ** 14, device=dev)

    # Must not cache bucket or sign tensors
    assert not hasattr(cs1, "_cache")

    g = torch.Generator(device=dev)
    g.manual_seed(42)
    u = {
        "layer1.weight": torch.randn((32, 64), generator=g, device=dev, dtype=torch.float32),
        "layer2.bias": torch.randn((64,), generator=g, device=dev, dtype=torch.float32),
    }
    v = {
        "layer1.weight": torch.randn((32, 64), generator=g, device=dev, dtype=torch.float32),
        "layer2.bias": torch.randn((64,), generator=g, device=dev, dtype=torch.float32),
    }

    sk_u1 = cs1.sketch_dict(u)
    sk_u2 = cs2.sketch_dict(u)
    assert sk_u1.dtype == torch.float32
    assert sk_u1.device.type == dev.type
    torch.testing.assert_close(sk_u1, sk_u2, rtol=1e-6, atol=1e-6)

    # Linearity in fp32
    alpha, beta = 1.75, -0.85
    comb = {k: alpha * u[k] + beta * v[k] for k in u}
    sk_v = cs1.sketch_dict(v)
    sk_comb = cs1.sketch_dict(comb)
    torch.testing.assert_close(sk_comb, alpha * sk_u1 + beta * sk_v, rtol=1e-5, atol=1e-5)

    # Inner product returns Python float (computed in float64)
    inner_val = cs1.inner(sk_u1, sk_v)
    assert isinstance(inner_val, float)


def test_llm_torch_countsketch_consistent_with_numpy_on_small_vectors():
    """Item 5: Verify new GPU torch CountSketch and numpy CountSketch produce consistent results on small vectors."""
    import torch
    from llm.diag import CountSketch as TorchCountSketch, OnlineLLMDiagAccumulator

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(2026)
    u_np = rng.normal(size=4096).astype(np.float32)
    v_np = (0.7 * u_np + 0.5 * rng.normal(size=4096)).astype(np.float32)
    exact_inner = float(np.dot(u_np.astype(np.float64), v_np.astype(np.float64)))

    cs_np = CountSketch(D=2 ** 16)
    cs_th = TorchCountSketch(D=2 ** 16, device=dev)

    est_np = cs_np.inner(
        cs_np.sketch_tensor("small_vec", u_np),
        cs_np.sketch_tensor("small_vec", v_np),
    )
    est_th = cs_th.inner(
        cs_th.sketch_tensor("small_vec", torch.from_numpy(u_np).to(dev)),
        cs_th.sketch_tensor("small_vec", torch.from_numpy(v_np).to(dev)),
    )

    # Both match exact inner product within 2% and agree with each other within 2%
    assert abs(est_np - exact_inner) / abs(exact_inner) < 0.02
    assert abs(est_th - exact_inner) / abs(exact_inner) < 0.02
    assert abs(est_th - est_np) / abs(exact_inner) < 0.02

    # Also verify across 50 small vectors that mean difference between torch and numpy is < 1.5 SE
    diffs = []
    cs_np_s = CountSketch(D=2 ** 13)
    cs_th_s = TorchCountSketch(D=2 ** 13, device=dev)
    for i in range(50):
        name = f"small_param_{i}"
        in_np = cs_np_s.inner(
            cs_np_s.sketch_tensor(name, u_np),
            cs_np_s.sketch_tensor(name, v_np),
        )
        in_th = cs_th_s.inner(
            cs_th_s.sketch_tensor(name, torch.from_numpy(u_np).to(dev)),
            cs_th_s.sketch_tensor(name, torch.from_numpy(v_np).to(dev)),
        )
        diffs.append(in_th - in_np)
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    se_diff = float(np.std(diffs_arr, ddof=1) / np.sqrt(len(diffs_arr)))
    assert abs(float(np.mean(diffs_arr))) < 2.5 * se_diff

