"""Regression coverage for the rented-GPU L0 execution and resource boundaries."""

import csv
import json
import math
from pathlib import Path
import types

import numpy as np
import pytest
import torch
import yaml

from llm.runtime import output_directory, checkpoint_directory, check_storage, tensor_inner64
from llm.l0_pipeline import training_checks, recommend_budget, StageFailure


@pytest.fixture(autouse=True)
def clean_runtime_environment(monkeypatch):
    for key in ("WORK", "STALEMOM_OUT", "STALEMOM_CKPT", "STALEMOM_MIN_FREE_GIB"):
        monkeypatch.delenv(key, raising=False)


def test_environment_paths_keep_separate_run_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("STALEMOM_OUT", str(tmp_path / "results"))
    monkeypatch.setenv("STALEMOM_CKPT", str(tmp_path / "checkpoints"))
    output = output_directory({"output_dir": "results/e5/l0"}, "results")
    assert output == tmp_path / "results/e5/l0"
    zero = checkpoint_directory({}, output / "train_k0")
    stale = checkpoint_directory({}, output / "train_k2")
    assert zero == tmp_path / "checkpoints/e5/l0/train_k0"
    assert stale != zero
    internal = {"output_dir": str(output / "train_k0"), "_paths_resolved": True}
    assert output_directory(internal, "results") == output / "train_k0"
    with pytest.raises(ValueError):
        output_directory({"output_dir": str(output)}, "results")


def test_default_paths_and_explicit_checkpoint_directory():
    out = output_directory({}, "results/e5/l0")
    assert checkpoint_directory({}, out) == Path("results/e5/l0/checkpoints")
    assert checkpoint_directory({"_checkpoint_output_dir": "custom"}, out) == Path("custom")


@pytest.mark.parametrize("data_free,system_free", [(9.99, 20), (20, 4.99)])
def test_storage_guard_stops_at_either_disk_threshold(tmp_path, monkeypatch, data_free, system_free):
    work = str(tmp_path / "work")
    monkeypatch.setenv("WORK", work)
    monkeypatch.setattr("llm.runtime.shutil.disk_usage", lambda path: types.SimpleNamespace(
        free=(data_free if str(path) == work else system_free) * 2**30))
    with pytest.raises(RuntimeError, match="Storage safety stop"):
        check_storage()


def test_chunked_float64_reduction_and_online_cosine():
    from llm.train import compute_grad_norm_and_online_c1_gpu

    left = torch.tensor([1e8, 3, -1e8, 5, 7], dtype=torch.float32)
    right = torch.tensor([1, 2, 1, -3, 4], dtype=torch.float32)
    assert tensor_inner64(left, right, chunk_elements=2).item() == np.dot(left.numpy().astype(np.float64), right.numpy().astype(np.float64))
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3., 4.]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer.state[model.weight]["momentum_buffer"] = torch.tensor([[0., 5.]])
    norm, cosine = compute_grad_norm_and_online_c1_gpu(optimizer, model, "sgd_mom")
    assert norm == pytest.approx(5)
    assert cosine == pytest.approx(0.8)


def test_environment_record_handles_cuda_available_with_no_visible_devices(tmp_path, monkeypatch):
    import llm.run_l0 as runner

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: pytest.fail("No visible CUDA device"))
    monkeypatch.setattr(runner, "write_llm_env_file", lambda out: (out / "env.txt").write_text("unit test fixture"))
    pinned = runner.step1_record_environment(tmp_path)
    assert pinned["gpu_name"] == "none" and pinned["cuda"] == "cpu"


def test_vllm_limits_reach_engine_and_cache_is_freed_before_wake(monkeypatch):
    import sys
    from llm.train import RolloutBackend

    events, options = [], {}
    class Engine:
        def __init__(self, **kwargs):
            options.update(kwargs)
        def sleep(self, level):
            events.append("sleep")
        def wake_up(self):
            events.append("wake")
        def apply_model(self, function):
            function(types.SimpleNamespace(load_weights=lambda weights: events.append("weights")))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=Engine))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    limits = dict(enable_sleep_mode=True, gpu_memory_utilization=0.25, enforce_eager=True,
                  max_model_len=2048, max_num_seqs=16, max_num_batched_tokens=512,
                  enable_chunked_prefill=True, sync_method="apply_model")
    backend = RolloutBackend("vllm", "test-fixture", types.SimpleNamespace(), "cpu", limits)
    backend.sleep(); backend.sync_weights({"weight": torch.ones(2)})
    assert events == ["sleep", "empty_cache", "wake", "weights"]
    assert options["enforce_eager"] is True and options["max_num_seqs"] == 16
    assert options["max_num_batched_tokens"] == 512
    assert backend.sync_interface == "LLM.apply_model"


def test_hf_batched_prompts_preserve_group_order_left_padding_and_response_masks():
    from llm.train import RolloutBackend
    from llm.data import GSM8KItem

    calls = []
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2
        def __call__(self, text, **_):
            ids = {"short": [1, 3], "long": [1, 5, 6, 7], "middle": [1, 8, 9]}[text]
            return {"input_ids": torch.tensor([ids])}
        def decode(self, ids, **_):
            return "\\boxed{3}" if ids[0] == 10 else "\\boxed{4}"
    class Model:
        def generate(self, input_ids, attention_mask, **_):
            calls.append((input_ids.clone(), attention_mask.clone()))
            response = torch.tensor([[10 + index % 2, 2, 0] for index in range(len(input_ids))])
            return torch.cat([input_ids, response], dim=1)
    def backend(batch_prompts):
        result = object.__new__(RolloutBackend)
        result.backend, result.device = "hf", torch.device("cpu")
        result.tokenizer, result.hf_model = Tokenizer(), Model()
        result.hf_batch_prompts = batch_prompts
        return result
    items = [GSM8KItem(index, text, "3", text) for index, text in enumerate(["short", "long", "middle"])]
    combined = backend(2).generate_group_rollouts(items, group_size=2, temperature=1, top_p=1, max_new_tokens=3, seed=42)
    assert len(calls) == 2
    assert calls[0][0].tolist() == [[0, 0, 1, 3], [0, 0, 1, 3], [1, 5, 6, 7], [1, 5, 6, 7]]
    assert calls[0][1].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    assert combined["prompt_lens"] == [2, 2, 4, 4, 3, 3]
    assert combined["resp_lens"] == [2] * 6
    np.testing.assert_array_equal(combined["rewards"], [1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(np.sign(combined["advantages"].numpy()), [1, -1, 1, -1, 1, -1])
    np.testing.assert_allclose(np.abs(combined["advantages"].numpy()), 1, rtol=1e-5)
    assert combined["response_mask"].sum().item() == 12
    calls.clear()
    separate = backend(1).generate_group_rollouts(items, group_size=2, temperature=1, top_p=1, max_new_tokens=3, seed=42)
    assert len(calls) == 3
    assert torch.equal(combined["input_ids"], separate["input_ids"])
    assert torch.equal(combined["response_mask"], separate["response_mask"])


def _training_rows(k, steps=10):
    return [dict(step=i, loss=0.1, grad_norm=1, peak_vram_gb=12,
                 nvidia_smi_peak_gb=13, dup_group_frac=0, mean_w=1,
                 trunc_rate=0.1 if k and i >= k else 0, reward_mean=0.2,
                 zero_adv_frac=0.5, step_time_gen=1, step_time_train=2, step_time_sync=0.1)
            for i in range(steps)]


def test_acceptance_rejects_missing_telemetry_nonfinite_grad_and_absent_truncation():
    rows = _training_rows(2)
    assert all(check["passed"] for check in training_checks(rows, 2, 22, 10))
    rows[5]["grad_norm"] = float("nan")
    rows[5]["trunc_rate"] = 0
    for row in rows:
        row["nvidia_smi_peak_gb"] = ""
    failed = {check["check"] for check in training_checks(rows, 2, 22, 10) if not check["passed"]}
    assert {"k2:finite_loss_and_grad", "k2:physical_peak_gib_sampled", "k2:trunc_rate_after_warmup"} <= failed
    rows = _training_rows(0)
    rows[2]["mean_w"] = 1 + 1e-12
    assert not next(check for check in training_checks(rows, 0, 22, 10) if check["check"] == "k0:mean_w")["passed"]


def test_sync_worker_exit_keeps_measured_logprob_errors_and_interface():
    from llm.l0_pipeline import sync_failure_result

    measured = {"passed": False, "logprob_mean_diff": 0.3, "logprob_max_diff": 2.0,
                "sync_interface": "LLM.apply_model", "tol_mean_diff": 0.02}
    failure = StageFailure("fixture", {"status": "complete", "result": measured, "worker_exit_code": -1})
    result = sync_failure_result(failure, "vllm")
    assert result["logprob_mean_diff"] == 0.3 and result["logprob_max_diff"] == 2.0
    assert result["sync_interface"] == "LLM.apply_model" and result["worker_exit_code"] == -1
    assert result["passed"] is False and "error" not in measured


@pytest.mark.parametrize("snr,h,expected", [(0.05, 1, 64), (0.01, 1, 128), (0.009, 1, 256), (math.nan, 0, 256), (1, -1, 256)])
def test_budget_tiers_and_unresolved_signal(snr, h, expected):
    result = recommend_budget(snr, h)
    assert result["recommended_K"] == expected
    assert result["signal_detected"] is (h > 0)


def test_stage_failure_stops_before_any_later_stage_and_writes_report(tmp_path, monkeypatch):
    import llm.l0_pipeline as pipeline

    out, seen = tmp_path / "output", []
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model": "test-fixture", "output_dir": str(out)}))
    def fail(kind, label, *_):
        seen.append(kind)
        raise StageFailure(label, {"error_type": "TestFixtureFailure", "message": "fixture"})
    monkeypatch.setattr(pipeline, "run_stage", fail)
    with pytest.raises(StageFailure):
        pipeline.run_l0_pipeline(str(config))
    assert seen == ["env"]
    assert json.loads((out / "l0_summary.json").read_text())["status"] == "stopped"
    assert json.loads((out / "l0_summary.json").read_text())["step_status"]["1"] == "failed"
    assert (out / "l0_report.md").exists()
    assert "| 1 | failed |" in (out / "l0_report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("initial_ooms,probe_enabled", [(1, False), (2, False), (0, True)])
def test_pipeline_preserves_oom_attempt_and_runs_only_l0(tmp_path, monkeypatch, initial_ooms, probe_enabled):
    import llm.l0_pipeline as pipeline

    out, calls = tmp_path / "output", []
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model": "test-fixture", "output_dir": str(out),
                                     "vllm": {"gpu_memory_utilization": 0.2}, "micro_batch_size": 4,
                                     "gpu_price_yuan_per_hour": 2.68,
                                     "smoke_checks": {"throughput_probe": probe_enabled}}))
    def worker(kind, label, cfg, *_):
        calls.append((kind, label, cfg.get("micro_batch_size"), cfg.get("vllm", {}).get("gpu_memory_utilization")))
        if kind == "env":
            result = {"torch": "fixture", "driver": "fixture"}
        elif kind == "sync":
            result = {"passed": True, "backend": "vllm", "sync_interface": "fixture", "logprob_mean_diff": 0.01}
        elif kind == "train":
            if label == "train_k0_attempt0" and initial_ooms >= 1 or label == "train_k0_attempt1" and initial_ooms >= 2:
                raise StageFailure(label, {"error_type": "OutOfMemoryError", "message": "fixture"})
            folder = Path(cfg["output_dir"]); folder.mkdir(parents=True)
            rows = _training_rows(cfg["k"], cfg.get("steps", 10))
            with (folder / "train_log.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            paths = {}
            for step in cfg.get("save_checkpoints_at", [0, 9]):
                path = folder / f"fixture_{step}.pt"; path.write_bytes(b"unit-test-fixture")
                paths[str(step)] = str(path)
            result = {"output_dir": str(folder), "checkpoints": paths, "summary": {"diverged": False}}
        else:
            step = 0 if label == "diag_step0" else 9
            result = {"diag_rows": [{"step": step, "S": 4, "R": 2, "SNR_hat": 0.05,
                                    "H_hat": 0.1, "N_bar": 1, "sketch_rel_err": 0.01,
                                    "zero_adv_frac": 0.5, "diag_time_sec": 1}]}
        return {"status": "complete", "result": result, "physical_peak_gib": 13, "elapsed_seconds": 1}
    monkeypatch.setattr(pipeline, "run_stage", worker)
    summary = pipeline.run_l0_pipeline(str(config))
    assert summary["status"] == "complete"
    assert [call[0] for call in calls] == ["env", "sync"] + ["train"] * (2 + initial_ooms + int(probe_enabled)) + ["diag", "diag"]
    if initial_ooms:
        assert calls[3][2] == 2
        assert summary["attempts"][2]["status"] == "failed"
        if initial_ooms == 2:
            assert calls[4][3] == 0.15
    else:
        assert calls[4][1:3] == ("throughput_mb8", 8)
        assert summary["throughput_probe"]["steps"] == 3
        assert summary["throughput_probe"]["adopted_for_later_stages"] is False
    assert [row["step"] for row in summary["step4_mini_l2"]["rows"]] == [0, 9]
    assert (out / "cost_estimates.csv").exists() and (out / "l0_report.md").exists()
    assert summary["step5_estimates"]["gpu_price_yuan_per_hour"] == 2.68
    with pytest.raises(FileExistsError):
        pipeline.run_l0_pipeline(str(config))
