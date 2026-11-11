"""SPEC E5 §7 — L0 Smoke Test, Weight Sync Check, Mini-L2 SNR Check, and GPU-Hour Estimator.

Executes the 5-step L0 protocol in strict order, stopping immediately and reporting if any step fails:
  1. Record environment and pin vLLM & torch versions.
  2. Standalone vLLM weight synchronization test under random weight perturbation (require mean logprob diff < 0.02).
  3. Run 10 steps of SGD+mom for k in {0, 2}, recording peak VRAM, per-stage step times, dup_group_frac, zero_adv_frac.
  4. Run mini-L2 (S=4, R=2), reporting SNR_hat, CountSketch relative error, and required K for L2.
  5. Re-estimate GPU hours required for L1–L4 from measured throughput, write report, and stop.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from llm.data import load_gsm8k_pools
from llm.train import (
    RolloutBackend,
    write_llm_env_file,
)


def recommend_l2_k_from_snr(snr_hat: float, target_cv: float = 0.25, r_rep: int = 4) -> dict[str, Any]:
    """Compute recommended total batches K = S * R for L2 given measured SNR_hat (§5.4, §7).

    Under half-split estimation H_hat = <A_bar, B_bar> with K = S * R total batches (K/2 per half),
    Var(H_hat) / H^2 ≈ 4 / (K * SNR) + 4 / (K^2 / 4 * SNR^2) (or conservative 2 / (K * SNR^2) + 4 / (K * SNR)).
    We report both the theoretical minimum K for CV(H_hat) <= target_cv and the practical S x R design.
    """
    if not np.isfinite(snr_hat) or snr_hat <= 0.0:
        return {
            "snr_hat": float(snr_hat) if np.isfinite(snr_hat) else float("nan"),
            "recommended_K": 128,
            "recommended_S": 32,
            "recommended_R": r_rep,
            "note": "SNR_hat <= 0 or NaN; recommending conservative K=128 (S=32, R=4).",
        }

    # Conservative variance bound for split-half inner product:
    # Var(H_hat)/H^2 <= (4 / (K * snr_hat)) + (4 / (K * snr_hat**2))
    raw_k = (4.0 / snr_hat + 4.0 / (snr_hat * snr_hat)) / (target_cv * target_cv)
    # Round up to an even number of prompt sets S (minimum S=16 -> K=64 per SPEC §9)
    min_s = max(16, int(math.ceil(raw_k / float(r_rep))))
    if min_s % 2 != 0:
        min_s += 1
    rec_k = min_s * r_rep
    return {
        "snr_hat": float(snr_hat),
        "raw_K_for_target_cv": float(raw_k),
        "recommended_K": int(rec_k),
        "recommended_S": int(min_s),
        "recommended_R": int(r_rep),
        "note": f"Target CV={target_cv:.2f} achieved with S={min_s}, R={r_rep} (K={rec_k}).",
    }


def step1_record_environment(out_dir: Path) -> dict[str, str]:
    """Step 1: Record environment and pin torch / vLLM / transformers versions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_llm_env_file(out_dir)

    import transformers

    try:
        import vllm
        vllm_ver = getattr(vllm, "__version__", "unknown")
    except Exception as e:
        vllm_ver = f"not_installed ({e})"

    cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    cuda_ver = str(torch.version.cuda) if cuda_available else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "none"
    vram_total_gb = (
        float(torch.cuda.get_device_properties(0).total_memory / (1024.0**3))
        if cuda_available
        else 0.0
    )

    pinned = {
        "torch": torch.__version__,
        "vllm": vllm_ver,
        "transformers": transformers.__version__,
        "cuda": cuda_ver,
        "gpu_name": gpu_name,
        "vram_total_gb": f"{vram_total_gb:.2f}",
        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else "unavailable",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() if shutil.which("git") else "unavailable",
    }
    pin_lines = [f"{k}=={v}" for k, v in pinned.items()]
    (out_dir / "pinned_versions.txt").write_text("\n".join(pin_lines) + "\n", encoding="utf-8")
    return pinned


def step2_test_weight_sync(
    model_name: str,
    backend_name: str,
    device: torch.device,
    vllm_cfg: dict,
    use_bf16: bool = True,
    seed: int = 42,
    dataset_name: str = "openai/gsm8k",
    max_new_tokens: int = 64,
    tol_mean_diff: float = 0.02,
) -> dict[str, Any]:
    """Step 2: Standalone weight sync test after random parameter perturbation (mean logprob diff < 0.02)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_pool, _ = load_gsm8k_pools(
        tokenizer=tokenizer,
        train_pool_size=16,
        eval_pool_size=8,
        seed=seed,
        dataset_name=dataset_name,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    hf_model.eval()
    hf_model.requires_grad_(False)

    # Perturb weights randomly so we genuinely test that sync_weights updates the rollout engine
    g = torch.Generator(device=device)
    g.manual_seed(seed + 999)
    with torch.no_grad():
        for p in hf_model.parameters():
            noise = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype) * 0.01
            p.add_(noise)

    rollout_engine = RolloutBackend(
        backend=backend_name,
        model_name=model_name,
        tokenizer=tokenizer,
        device=device,
        vllm_cfg=vllm_cfg,
        use_bf16=use_bf16,
        seed=seed,
    )

    t0 = time.perf_counter()
    perturbed_sd = {
        k: v.detach().to(device="cpu", dtype=torch.float32).clone()
        for k, v in hf_model.state_dict().items()
    }
    rollout_engine.wake_up()
    rollout_engine.sync_weights(perturbed_sd)
    sync_sec = time.perf_counter() - t0

    batch = rollout_engine.generate_group_rollouts(
        items=train_pool[:4],
        group_size=4,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    rollout_engine.sleep(level=1)

    max_diff, mean_diff = rollout_engine.verify_logprob_sync(hf_model, batch)
    interface = rollout_engine.sync_interface
    del rollout_engine, hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    passed = bool(np.isfinite(mean_diff) and mean_diff < tol_mean_diff)
    return {
        "backend": backend_name,
        "sync_interface": interface,
        "sync_time_sec": float(sync_sec),
        "logprob_max_diff": float(max_diff),
        "logprob_mean_diff": float(mean_diff),
        "tol_mean_diff": float(tol_mean_diff),
        "dup_group_frac": float(batch["dup_group_frac"]),
        "passed": passed,
    }


def estimate_l1_to_l4_gpu_hours(
    mean_step_sec: float,
    recommended_k_l2: int,
    prompts_per_step: int = 32,
) -> dict[str, float]:
    """Step 5: Estimate GPU hours for L1, L2, L3, L4 Core, and L4 Extension from measured step time."""
    # L1: 2 optimizers x 3 LRs x 150 steps + 2 extra seed runs x 150 steps = 8 runs x 150 = 1200 steps
    l1_steps = 8 * 150
    # L2: 3 checkpoints x recommended_K batch evaluations (each batch has same cost as 1 step without opt step)
    l2_equiv_steps = 3 * recommended_k_l2
    # L3: (2 k_values x 2 c_values + 1 extra seed) = 5 runs x 150 steps + 5 runs x 2 ckpts x (S=8 * R=4 = 32)
    l3_steps = 5 * 150 + 5 * 2 * 32
    # L4 Core: 2 async conditions x 2 optimizers x 3 LRs = 12 runs + 4 extra seed runs = 16 runs x 150 = 2400 steps
    l4_core_steps = 16 * 150
    # L4 Ext (optional): 3 conditions x 2 optimizers x 3 LRs = 18 runs x 150 = 2700 steps
    l4_ext_steps = 18 * 150

    sec_to_hr = 1.0 / 3600.0
    l1_hr = l1_steps * mean_step_sec * sec_to_hr
    l2_hr = l2_equiv_steps * mean_step_sec * sec_to_hr
    l3_hr = l3_steps * mean_step_sec * sec_to_hr
    l4_core_hr = l4_core_steps * mean_step_sec * sec_to_hr
    l4_ext_hr = l4_ext_steps * mean_step_sec * sec_to_hr

    return {
        "mean_step_sec": float(mean_step_sec),
        "L1_gpu_hours": float(l1_hr),
        "L2_gpu_hours": float(l2_hr),
        "L3_gpu_hours": float(l3_hr),
        "L4_core_gpu_hours": float(l4_core_hr),
        "total_L1_to_L4_core_gpu_hours": float(l1_hr + l2_hr + l3_hr + l4_core_hr),
        "L4_ext_optional_gpu_hours": float(l4_ext_hr),
    }


def run_l0_pipeline(config_path: str = "llm/configs/l0/smoke.yaml", backend_override: str | None = None) -> dict[str, Any]:
    """Run only L0, isolating CUDA stages and preserving failures for review."""
    from llm.l0_pipeline import run_l0_pipeline as execute
    return execute(config_path, backend_override)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPEC E5 L0 5-step smoke test.")
    parser.add_argument("--config", type=str, default="llm/configs/l0/smoke.yaml")
    parser.add_argument("--backend", type=str, default=None, help="Override rollout backend (vllm or hf).")
    args = parser.parse_args()
    run_l0_pipeline(args.config, backend_override=args.backend)
