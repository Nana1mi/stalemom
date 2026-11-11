"""Rented-4090 L0 orchestration. Every CUDA stage runs in a fresh process."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import yaml

from llm.runtime import output_directory, checkpoint_directory, check_storage


class StageFailure(RuntimeError):
    def __init__(self, label, payload):
        self.payload = payload
        super().__init__(f"{label}: {payload.get('error_type', 'failure')}: {payload.get('message', '')}")


def _stop_own_child(process):
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def run_stage(kind: str, label: str, config: dict, out: Path, timeout: float | None = None) -> dict:
    check_storage()
    stage = out / "stages"
    stage.mkdir(parents=True, exist_ok=True)
    request, response = stage / f"{label}.config.json", stage / f"{label}.result.json"
    request.write_text(json.dumps(config, indent=2), encoding="utf-8")
    command = [sys.executable, "-u", "-m", "llm.l0_worker", kind, str(request.resolve()), str(response.resolve())]
    start = time.monotonic()
    with (stage / f"{label}.out").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        try:
            while process.poll() is None:
                check_storage()
                if timeout is not None and time.monotonic() - start > timeout:
                    raise TimeoutError(f"{label} exceeded {timeout} seconds")
                time.sleep(1)
        except BaseException:
            if process.poll() is None:
                _stop_own_child(process)
            raise
    payload = json.loads(response.read_text(encoding="utf-8")) if response.exists() else {
        "status": "failed", "error_type": "WorkerExit", "message": f"exit={process.returncode}; see {label}.out",
    }
    payload["elapsed_seconds"] = time.monotonic() - start
    payload["worker_exit_code"] = process.returncode
    response.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if process.returncode or payload.get("status") != "complete":
        raise StageFailure(label, payload)
    print(f"[L0] {label} completed; physical peak={payload.get('physical_peak_gib')} GiB", flush=True)
    return payload


def sync_failure_result(error: Exception, backend: str) -> dict:
    """Keep measured sync errors even if the worker exits during CUDA cleanup."""
    payload = error.payload if isinstance(error, StageFailure) else {}
    result = copy.deepcopy(payload.get("result", {}))
    result.update(passed=False, backend=backend, error=str(error))
    result["worker_exit_code"] = payload.get("worker_exit_code")
    return result


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def training_checks(rows: list[dict], k: int, max_gib: float, expected_steps: int) -> list[dict]:
    def check(name, target, value, passed):
        return {"check": name, "target": target, "value": value, "passed": bool(passed)}

    finite = all(math.isfinite(float(row[key])) for row in rows for key in ("loss", "grad_norm"))
    torch_peak = max((float(row["peak_vram_gb"]) for row in rows), default=float("nan"))
    physical = [float(row["nvidia_smi_peak_gb"]) for row in rows if row.get("nvidia_smi_peak_gb") not in (None, "")]
    physical_peak = max(physical, default=float("nan"))
    duplicate = max((float(row["dup_group_frac"]) for row in rows), default=float("nan"))
    checks = [check(f"k{k}:steps", expected_steps, len(rows), len(rows) == expected_steps),
              check(f"k{k}:finite_loss_and_grad", "all finite", finite, finite),
              check(f"k{k}:torch_peak_gib", f"< {max_gib}", torch_peak, torch_peak < max_gib),
              check(f"k{k}:physical_peak_gib_sampled", f"< {max_gib}", physical_peak, physical_peak < max_gib),
              check(f"k{k}:dup_group_frac", "< 0.05", duplicate, duplicate < 0.05)]
    if k == 0:
        values = [float(row["mean_w"]) for row in rows]
        checks.append(check("k0:mean_w", "exactly 1", values, all(value == 1.0 for value in values)))
    else:
        values = [float(row["trunc_rate"]) for row in rows if int(row["step"]) >= k]
        checks.append(check(f"k{k}:trunc_rate_after_warmup", "> 0 at every step >= k", values,
                            bool(values) and all(value > 0 for value in values)))
    return checks


def recommend_budget(snr: float, h: float) -> dict:
    detected = math.isfinite(h) and h > 0
    total = 64 if math.isfinite(snr) and snr >= 0.05 else 128 if math.isfinite(snr) and snr >= 0.01 else 256
    if not detected:
        total = 256
    return {"SNR_hat": snr, "H": h, "signal_detected": detected, "recommended_K": total,
            "note": "Signal unresolved at this sample size; L2 cost may increase substantially." if not detected else
                    "Low SNR; L2 cost increases substantially." if total == 256 else "CODEX_L0_4090.md SNR tier."}


def _estimates(summary: dict, out: Path) -> dict:
    logs = {k: read_csv(out / f"train_log_k{k}.csv") for k in (0, 2)}
    seconds = {k: sum(sum(float(row[f"step_time_{part}"]) for part in ("gen", "train", "sync")) for row in rows) / len(rows)
               for k, rows in logs.items()}
    diag = read_csv(out / "diag_summary.csv")
    diag_batch = sum(float(row["diag_time_sec"]) / (int(row["S"]) * int(row["R"])) for row in diag) / len(diag)
    budget = summary["step4_mini_l2"]["k_recommendation"]["recommended_K"]
    hours = {"L1": 8 * 150 * seconds[0] / 3600,
             "L2": 3 * budget * diag_batch / 3600,
             "L3": (5 * 150 * seconds[2] + 5 * 2 * 32 * diag_batch) / 3600,
             "L4_core": 16 * 150 * seconds[2] / 3600}
    price = float(summary["config"].get("gpu_price_yuan_per_hour", 2.68))
    return {"mean_train_step_seconds": {str(k): value for k, value in seconds.items()},
            "mean_diag_batch_seconds": diag_batch, "gpu_hours": hours,
            "gpu_price_yuan_per_hour": price, "cost_yuan": {key: value * price for key, value in hours.items()},
            "note": "Estimates only, based on the measured baseline runs. No L1-L4 jobs were started."}


def run_l0_pipeline(config_path: str, backend_override: str | None = None) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out = output_directory(config, "results/e5/l0").resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite scientific outputs from an earlier run.
    if (out / "l0_summary.json").exists():
        raise FileExistsError(f"L0 outputs already exist: {out}; choose a new STALEMOM_OUT")
    config["rollout_backend"] = backend_override or config.get("rollout_backend", "vllm")
    checkpoint_root = checkpoint_directory(config, out).resolve()
    summary = {"status": "running", "checks": [], "attempts": [], "config": copy.deepcopy(config),
               "output_dir": str(out), "checkpoint_root": str(checkpoint_root), "step_status": {str(i): "not_run" for i in range(1, 6)}}

    def persist():
        (out / "l0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def attempt(kind, label, cfg, timeout=None):
        record = {"stage": kind, "label": label, "micro_batch_size": cfg.get("micro_batch_size"),
                  "vllm": cfg.get("vllm", {}), "backend": cfg["rollout_backend"],
                  "hf_batch_prompts": cfg.get("rollout", {}).get("hf_batch_prompts", 1) if cfg["rollout_backend"] == "hf" else None,
                  "status": "running"}
        summary["attempts"].append(record)
        persist()
        try:
            payload = run_stage(kind, label, cfg, out, timeout)
            record.update(status="complete", physical_peak_gib=payload.get("physical_peak_gib"), elapsed_seconds=payload["elapsed_seconds"])
            return payload["result"]
        except StageFailure as error:
            record.update(status="failed", error=error.payload)
            raise
        finally:
            persist()

    try:
        summary["step_status"]["1"] = "running"
        setup_file = Path(os.environ.get("WORK", ".")) / "setup" / "environment_setup.json"
        if setup_file.exists():
            summary["environment_setup"] = json.loads(setup_file.read_text(encoding="utf-8"))
        summary["storage_before"] = check_storage()
        env_free = float(config.get("environment_min_free_gib", 70))
        if summary["storage_before"] and summary["storage_before"]["data_free_gib"] < env_free:
            raise RuntimeError(f"Environment acceptance failed: need >= {env_free} GiB free before L0")
        summary["df_before"] = subprocess.run(["df", "-h"], capture_output=True, text=True).stdout if os.name == "posix" else "Windows development environment"
        summary["step1_env"] = attempt("env", "environment", {**config, "output_dir": str(out), "_paths_resolved": True})
        summary["step_status"]["1"] = "passed"
        summary["step_status"]["2"] = "running"
        sync_deadline = time.monotonic() + 1800
        try:
            sync = attempt("sync", "sync_auto", config, max(1, sync_deadline - time.monotonic()))
            if not sync["passed"]:
                raise StageFailure("sync_auto", {"error_type": "LogprobMismatch", "message": str(sync), "result": sync})
        except (StageFailure, TimeoutError) as error:
            summary["initial_sync_error"] = str(error)
            retry = copy.deepcopy(config)
            retry.setdefault("vllm", {})["sync_method"] = "apply_model"
            try:
                if time.monotonic() >= sync_deadline:
                    raise TimeoutError("Weight-sync problem reached the 30-minute budget")
                sync = attempt("sync", "sync_public_api", retry, sync_deadline - time.monotonic())
                if sync.get("passed"):
                    config.setdefault("vllm", {})["sync_method"] = "apply_model"
            except (StageFailure, TimeoutError) as second:
                sync = sync_failure_result(second, config["rollout_backend"])
        (out / "weight_sync_check.json").write_text(json.dumps(sync, indent=2), encoding="utf-8")
        summary["step2_weight_sync"] = sync
        summary["checks"].append({"check": "vllm_weight_sync" if config["rollout_backend"] == "vllm" else "hf_weight_consistency",
                                  "target": "mean absolute logprob difference < 0.02",
                                  "value": sync.get("logprob_mean_diff"), "passed": bool(sync.get("passed"))})
        if not sync.get("passed"):
            if config["rollout_backend"] != "vllm":
                raise RuntimeError("HF weight consistency test failed")
            config["rollout_backend"] = "hf"
            summary["step_status"]["2"] = "vllm_failed_hf_fallback"
        else:
            summary["step_status"]["2"] = "passed"
        summary["backend_for_steps_3_4"] = config["rollout_backend"]
        summary["step_status"]["3"] = "running"
        smoke = config.get("smoke_checks", {})
        steps = int(smoke.get("steps", 10))
        summary["step3_runs"] = {}
        final_training_config = copy.deepcopy(config)
        for k in (0, 2):
            run = copy.deepcopy(final_training_config)
            run.update(k=k, c=float("inf") if k == 0 else 1.2, steps=steps, optimizer="sgd_mom", lr=0.01,
                       momentum=0.9, eval_every=0, save_checkpoints_at=[0, steps - 1], _paths_resolved=True)
            result = None
            for index in range(3):
                folder = f"train_k{k}_attempt{index}"
                run.update(output_dir=str(out / folder), _checkpoint_output_dir=str(checkpoint_root / folder))
                try:
                    result = attempt("train", folder, run)
                    break
                except StageFailure as error:
                    is_oom = error.payload.get("error_type") in ("OutOfMemoryError", "CUDAOutOfMemoryError") or "cuda out of memory" in error.payload.get("message", "").lower()
                    if not is_oom or index == 2:
                        raise
                    if index == 0:
                        run["micro_batch_size"] = 2
                    else:
                        run.setdefault("vllm", {})["gpu_memory_utilization"] = float(smoke.get("oom_gpu_memory_utilization", 0.15))
            assert result is not None
            shutil.copy2(Path(result["output_dir"]) / "train_log.csv", out / f"train_log_k{k}.csv")
            rows = read_csv(out / f"train_log_k{k}.csv")
            checks = training_checks(rows, k, float(smoke.get("max_vram_gb", 22)), steps)
            summary["checks"].extend(checks)
            summary["step3_runs"][f"k={k}"] = result
            summary["step3_runs"][f"k={k}"]["all_advantages_zero"] = all(float(row["zero_adv_frac"]) == 1 for row in rows)
            if not all(check["passed"] for check in checks):
                raise RuntimeError(f"k={k} training acceptance checks failed; see l0_summary.json")
            final_training_config["micro_batch_size"] = run.get("micro_batch_size", 4)
            final_training_config["vllm"] = copy.deepcopy(run.get("vllm", {}))
        summary["final_training_config"] = final_training_config
        baseline_rows = {k: read_csv(out / f"train_log_k{k}.csv") for k in (0, 2)}
        baseline_peak = max(float(row[key]) for rows in baseline_rows.values() for row in rows
                            for key in ("peak_vram_gb", "nvidia_smi_peak_gb"))
        if smoke.get("throughput_probe", False) and final_training_config.get("micro_batch_size", 4) == 4 and baseline_peak < 25:
            probe = copy.deepcopy(final_training_config)
            probe.update(k=0, c=float("inf"), steps=3, optimizer="sgd_mom", lr=0.01, momentum=0.9,
                         micro_batch_size=8, eval_every=0, save_checkpoints_at=[], _paths_resolved=True,
                         output_dir=str(out / "throughput_mb8"), _checkpoint_output_dir=str(checkpoint_root / "throughput_mb8"))
            result = attempt("train", "throughput_mb8", probe)
            shutil.copy2(Path(result["output_dir"]) / "train_log.csv", out / "train_log_mb8.csv")
            rows = read_csv(out / "train_log_mb8.csv")
            if len(rows) != 3 or not all(math.isfinite(float(row[key])) for row in rows for key in ("loss", "grad_norm")):
                raise RuntimeError("micro_batch_size=8 throughput probe failed")
            training_seconds = sum(float(row["step_time_train"]) for row in rows) / len(rows)
            baseline_seconds = sum(float(row["step_time_train"]) for row in baseline_rows[0]) / len(baseline_rows[0])
            summary["throughput_probe"] = {"micro_batch_size": 8, "steps": len(rows),
                                           "mean_training_seconds": training_seconds,
                                           "baseline_k0_mean_training_seconds": baseline_seconds,
                                           "training_speedup": baseline_seconds / training_seconds if training_seconds > 0 else None,
                                           "torch_peak_gib": max(float(row["peak_vram_gb"]) for row in rows),
                                           "physical_peak_gib_sampled": max(float(row["nvidia_smi_peak_gb"]) for row in rows),
                                           "adopted_for_later_stages": False}
        else:
            summary["throughput_probe"] = {"status": "not_run", "baseline_peak_gib": baseline_peak,
                                           "reason": "Requires enabled probe, baseline micro-batch 4 and peak below 25 GiB."}
        summary["step_status"]["3"] = "passed"
        summary["step_status"]["4"] = "running"
        mini = smoke.get("mini_l2", {})
        diag_rows = []
        for step in (0, steps - 1):
            diag = copy.deepcopy(final_training_config)
            diag.update(k=0, c=float("inf"), _paths_resolved=True,
                        output_dir=str(out / f"diag_step{step}"),
                        checkpoint_paths=[summary["step3_runs"]["k=0"]["checkpoints"][str(step)]],
                        delete_checkpoint_after_diag=False)
            rollout = config.get("rollout", {})
            diag["nested_design"] = {"S": mini.get("S", 4), "R": mini.get("R", 2),
                                     **{key: rollout.get(key, default) for key, default in (
                                         ("group_size", 4), ("temperature", 1.0), ("top_p", 1.0), ("max_new_tokens", 512))},
                                     "prompts_per_set": rollout.get("prompts_per_step", 32)}
            diag["sketch"] = {"D": mini.get("sketch_dim", 2**22)}
            result = attempt("diag", f"diag_step{step}", diag)
            row = result["diag_rows"][0]
            row["H"] = row["H_hat"]
            row["signal_detected"] = math.isfinite(row["H_hat"]) and row["H_hat"] > 0
            diag_rows.append(row)
            summary["step4_mini_l2"] = {"rows": list(diag_rows)}
            with (out / "diag_summary.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader(); writer.writerows(diag_rows)
            error = float(row["sketch_rel_err"])
            passed = math.isfinite(error) and error < 0.05
            summary["checks"].append({"check": f"diag_step{step}:sketch_rel_err", "target": "< 0.05", "value": error, "passed": passed})
            if not passed:
                raise RuntimeError(f"Diagnostic sketch check failed at step {step}")
        last = diag_rows[-1]
        summary["step4_mini_l2"] = {"rows": diag_rows, "k_recommendation": recommend_budget(last["SNR_hat"], last["H_hat"])}
        summary["step_status"]["4"] = "passed"
        summary["step_status"]["5"] = "running"
        summary["step5_estimates"] = _estimates(summary, out)
        summary["step_status"]["5"] = "passed"
        summary["status"] = "complete" if sync.get("passed") else "complete_with_hf_fallback"
    except BaseException as error:
        for step, status in summary["step_status"].items():
            if status == "running":
                summary["step_status"][step] = "failed"
        summary.update(status="stopped", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            summary["storage_after"] = check_storage()
        except Exception as storage_error:
            summary["storage_after"] = {"safety_stop": str(storage_error)}
        persist()
        from llm.l0_report import generate_report
        generate_report(out)
    return summary
