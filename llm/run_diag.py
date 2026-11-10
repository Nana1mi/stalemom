"""Driver for SPEC E5 L2/L3 frozen-checkpoint nested gradient diagnostics (SPEC E5 §5, §9, §10)."""

import copy
import csv
import datetime
from pathlib import Path
import time
from typing import Any
import numpy as np
import torch
import yaml

from llm.data import load_gsm8k_pools
from llm.diag import OnlineLLMDiagAccumulator
from llm.train import (
    RolloutBackend,
    accumulate_grpo_tis_microbatches,
    write_llm_env_file,
)
from stalemom.diag import compute_c3


def run_checkpoint_diagnostics(config: dict | str | Path) -> dict[str, Any]:
    """Load frozen checkpoints and momentum buffers, sample S x R nested batches, and output diagnostic CSVs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = copy.deepcopy(config)

    out_dir = Path(cfg.get("output_dir", cfg.get("out_dir", "results/e5/l2")))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_llm_env_file(out_dir)

    seed = int(cfg.get("seed", 42))
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
    micro_batch_size = cfg.get("micro_batch_size", None)

    nested_cfg = cfg.get("nested_design", {})
    S = int(nested_cfg.get("S", cfg.get("diag_S", 16)))
    R = int(nested_cfg.get("R", cfg.get("diag_R", 4)))
    prompts_per_set = int(nested_cfg.get("prompts_per_set", cfg.get("prompts_per_step", 32)))
    group_size = int(nested_cfg.get("group_size", cfg.get("group_size", 4)))
    temperature = float(nested_cfg.get("temperature", cfg.get("temperature", 1.0)))
    top_p = float(nested_cfg.get("top_p", cfg.get("top_p", 1.0)))
    max_new_tokens = int(nested_cfg.get("max_new_tokens", cfg.get("max_new_tokens", 512)))

    sketch_cfg = cfg.get("sketch", {})
    sketch_dim = int(sketch_cfg.get("D", cfg.get("sketch_dim", 2**22)))
    delete_after = bool(cfg.get("delete_checkpoint_after_diag", cfg.get("delete_checkpoints_after_diag", False)))

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset", cfg.get("dataset", "openai/gsm8k"))
    train_pool_size = int(data_cfg.get("train_pool_size", cfg.get("train_pool_size", 1024)))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_pool, _ = load_gsm8k_pools(
        tokenizer=tokenizer,
        train_pool_size=train_pool_size,
        eval_pool_size=32,
        seed=seed,
        dataset_name=dataset_name,
    )

    # Resolve checkpoint paths
    ckpt_paths: list[Path] = []
    if "checkpoint_paths" in cfg:
        ckpt_paths = [Path(p) for p in cfg["checkpoint_paths"]]
    else:
        ckpt_dir = Path(cfg.get("checkpoint_dir", out_dir / "checkpoints"))
        if not ckpt_dir.exists():
            ckpt_dir = Path(cfg.get("checkpoint_dir", out_dir.parent / "l1" / "checkpoints"))
        target_steps = cfg.get("checkpoints", cfg.get("save_checkpoints_at", [0, 50, 100]))
        for st in target_steps:
            candidate = ckpt_dir / f"ckpt_step_{int(st):04d}.pt"
            if candidate.exists():
                ckpt_paths.append(candidate)

    if not ckpt_paths:
        raise FileNotFoundError("No checkpoint files found for run_diag.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.train()
    if bool(cfg.get("gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    behav_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    behav_model.eval()
    behav_model.requires_grad_(False)

    rollout_engine = RolloutBackend(
        backend=backend_name,
        model_name=model_name,
        tokenizer=tokenizer,
        device=device,
        vllm_cfg=cfg.get("vllm", {}),
        use_bf16=use_bf16,
    )

    rng = np.random.default_rng(seed + 1000)
    # Pre-sample the S prompt sets so all checkpoints share the exact same S prompt sets
    prompt_sets = []
    for _ in range(S):
        idxs = rng.choice(len(train_pool), size=min(prompts_per_set, len(train_pool)), replace=False)
        prompt_sets.append([train_pool[int(i)] for i in idxs])

    diag_rows = []
    step_stats_for_c3: list[tuple[int, dict[str, Any]]] = []
    log_file = out_dir / "run_log.txt"

    for ckpt_path in ckpt_paths:
        t0 = time.perf_counter()
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        step = int(payload["step"])
        model_sd = payload["model_state_dict"]
        behav_sd = payload.get("behav_state_dict", model_sd)
        m_dict_cpu = payload.get("momentum_buffer", {})
        norm_m = float(payload.get("norm_m", 0.0))
        if step == 0:
            # Per SPEC E5 §9: at step 0, m = 0, skip C1 and C2
            norm_m = 0.0
            m_dict_gpu = {}
        else:
            m_dict_gpu = {
                name: tensor.to(device=device, dtype=torch.float32)
                for name, tensor in m_dict_cpu.items()
            }

        model.load_state_dict(
            {name: val.to(device=device, dtype=torch.float32) for name, val in model_sd.items()},
            strict=True,
        )
        behav_model.load_state_dict(
            {name: val.to(device=device, dtype=torch.float32) for name, val in behav_sd.items()},
            strict=True,
        )
        rollout_engine.sync_weights(behav_sd)

        acc = OnlineLLMDiagAccumulator(
            S=S,
            R=R,
            norm_m=norm_m,
            sketch_dim=sketch_dim,
            device=device,
        )

        rewards_all = []
        zero_adv_all = []
        trunc_all = []
        kl_all = []

        for s_idx in range(S):
            acc.begin_set(s_idx)
            set_items = prompt_sets[s_idx]
            for r_idx in range(R):
                rollout_seed = seed + step * 10000 + s_idx * 100 + r_idx
                rb = rollout_engine.generate_group_rollouts(
                    items=set_items,
                    group_size=group_size,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    seed=rollout_seed,
                )
                model.zero_grad(set_to_none=True)
                st_mb = accumulate_grpo_tis_microbatches(
                    model=model,
                    behav_model=behav_model,
                    input_ids=rb["input_ids"],
                    attention_mask=rb["attention_mask"],
                    response_mask=rb["response_mask"],
                    advantages=rb["advantages"],
                    k=k,
                    c=c,
                    micro_batch_size=micro_batch_size,
                    use_bf16=use_bf16,
                )
                grad_dict = {
                    name: (
                        p.grad.detach()
                        if p.grad is not None
                        else torch.zeros_like(p, dtype=torch.float32, device=device)
                    )
                    for name, p in model.named_parameters()
                    if p.requires_grad
                }
                acc.add_batch_grad(grad_dict, m_dict_gpu if norm_m > 0.0 else None)

                rewards_all.append(float(np.mean(rb["rewards"])))
                zero_adv_all.append(rb["zero_adv_frac"])
                trunc_all.append(st_mb["trunc_rate"])
                kl_all.append(st_mb["approx_kl"])
            acc.end_set()

        res = acc.finalize()
        diag_time = time.perf_counter() - t0

        # Save compact sketch for cross-checkpoint analysis (§5.5, §6.4)
        sketch_out_path = out_dir / f"sketch_step_{step:04d}.npz"
        np.savez_compressed(
            sketch_out_path,
            step=step,
            H_hat=res["H_hat"],
            A_bar=res["A_bar"],
            B_bar=res["B_bar"],
        )
        step_stats_for_c3.append((step, res))

        var_tot = res["var_total"]
        prompt_var_frac = float(res["var_prompt"] / var_tot) if var_tot > 0 else float("nan")
        rollout_var_frac = float(res["var_rollout"] / var_tot) if var_tot > 0 else float("nan")

        row = {
            "step": step,
            "S": S,
            "R": R,
            "prompts_per_set": prompts_per_set,
            "group_size": group_size,
            "k": k,
            "c": c,
            "norm_m": norm_m,
            "reward_mean": float(np.mean(rewards_all)),
            "zero_adv_frac": float(np.mean(zero_adv_all)),
            "trunc_rate": float(np.mean(trunc_all)),
            "approx_kl": float(np.mean(kl_all)),
            "N_bar": res["N_bar"],
            "N_bar_jk_se": res["N_bar_jk_se"],
            "H_hat": res["H_hat"],
            "H_hat_jk_se": res["H_hat_jk_se"],
            "H_sketch": res["H_sketch"],
            "sketch_rel_err": res["sketch_rel_err"],
            "M": res["M"],
            "M_jk_se": res["M_jk_se"],
            "var_total": res["var_total"],
            "var_total_jk_se": res["var_total_jk_se"],
            "var_rollout": res["var_rollout"],
            "var_rollout_jk_se": res["var_rollout_jk_se"],
            "var_prompt": res["var_prompt"],
            "var_prompt_jk_se": res["var_prompt_jk_se"],
            "prompt_var_frac": prompt_var_frac,
            "rollout_var_frac": rollout_var_frac,
            "SNR_hat": res["SNR_hat"],
            "SNR_hat_jk_se": res["SNR_hat_jk_se"],
            "C1": res["C1"],
            "C1_jk_se": res["C1_jk_se"],
            "C1_pred": res["C1_pred"],
            "C1_pred_jk_se": res["C1_pred_jk_se"],
            "C1_pred_legacy": res["C1_pred_legacy"],
            "C1_pred_legacy_jk_se": res["C1_pred_legacy_jk_se"],
            "C2_hat": res["C2_hat"],
            "C2_hat_jk_se": res["C2_hat_jk_se"],
            "diag_time_sec": diag_time,
        }
        diag_rows.append(row)

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(
                f"[diag step={step:04d}] H_hat={res['H_hat']:.4e} SNR={res['SNR_hat']:.4f} "
                f"C1={res['C1']:.4f} C1_pred={res['C1_pred']:.4f} C2_hat={res['C2_hat']:.4f} "
                f"sketch_rel_err={res['sketch_rel_err']:.4f} time={diag_time:.2f}s\n"
            )

        if delete_after and ckpt_path.exists():
            ckpt_path.unlink()

    # Write diag_summary.csv
    summary_csv = out_dir / "diag_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diag_rows)

    # Write c3_matrix.csv across all pairs of steps
    c3_rows = []
    for i in range(len(step_stats_for_c3)):
        for j in range(i, len(step_stats_for_c3)):
            s1, st1 = step_stats_for_c3[i]
            s2, st2 = step_stats_for_c3[j]
            c3_val = 1.0 if i == j else compute_c3(st1, st2)
            c3_rows.append({
                "step_t": s1,
                "step_s": s2,
                "C3_hat": c3_val,
                "H_hat_t": st1["H_hat"],
                "H_hat_s": st2["H_hat"],
            })

    c3_csv = out_dir / "c3_matrix.csv"
    with open(c3_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(c3_rows[0].keys()))
        writer.writeheader()
        writer.writerows(c3_rows)

    (out_dir / "DONE").write_text(f"DONE at {datetime.datetime.now().isoformat()}\n", encoding="utf-8")
    return {
        "diag_rows": diag_rows,
        "c3_rows": c3_rows,
        "diag_summary_csv": str(summary_csv),
        "c3_matrix_csv": str(c3_csv),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SPEC E5 L2/L3 frozen checkpoint gradient diagnostics.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    run_checkpoint_diagnostics(args.config)
