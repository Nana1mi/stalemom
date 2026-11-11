"""One isolated L0 stage, so a completed stage releases its CUDA context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from llm.runtime import MemorySampler, check_storage


def execute(kind: str, config: dict) -> dict:
    if kind == "env":
        from llm.run_l0 import step1_record_environment
        return step1_record_environment(Path(config["output_dir"]))
    if kind == "sync":
        import torch
        from llm.run_l0 import step2_test_weight_sync

        return step2_test_weight_sync(
            model_name=config["model"], backend_name=config["rollout_backend"],
            device=torch.device(config.get("device", "cuda")),
            vllm_cfg=config.get("vllm", {}), seed=config.get("data", {}).get("seed", 42),
            dataset_name=config.get("data", {}).get("dataset", "openai/gsm8k"),
            max_new_tokens=64,
        )
    if kind == "train":
        from llm.train import main
        return main(config)
    if kind == "diag":
        from llm.run_diag import run_checkpoint_diagnostics
        return run_checkpoint_diagnostics(config)
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["env", "sync", "train", "diag"])
    parser.add_argument("config")
    parser.add_argument("result")
    args = parser.parse_args()
    target = Path(args.result)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    sampler = None
    try:
        check_storage()
        with MemorySampler(target.with_suffix(".memory.jsonl")) as sampler:
            result = execute(args.kind, config)
        payload = {"status": "complete", "result": result, "physical_peak_gib": sampler.peak}
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        payload = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "physical_peak_gib": sampler.peak if sampler else None}
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
