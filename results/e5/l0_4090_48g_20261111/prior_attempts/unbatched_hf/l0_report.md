# L0 报告

状态：`stopped`。本报告由 `llm.l0_report` 从保存的 JSON/CSV 生成。

## 1. 环境与磁盘

```json
{
  "torch": "2.6.0+cu124",
  "vllm": "0.8.5",
  "transformers": "4.51.3",
  "cuda": "12.4",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "vram_total_gb": "47.50",
  "driver": "550.90.07",
  "git_commit": "67932fe6fb982e4fc2d7d60b5ae4ceec7b13e216"
}
```

环境安装记录与版本选择：

```json
{
  "time": "2026-11-11T11:41:12.803262",
  "driver": "550.90.07",
  "df_before": "Filesystem     Type  Size  Used Avail Use% Mounted on\n/dev/vda1      ext4   97G   13G   84G  14% /\n",
  "disk_free_bytes_before": 89934114816,
  "version_selection_reason": "Driver 550.90.07 reports CUDA 12.4 support. Use vLLM 0.8.5 CUDA 12.4 wheel and its required torch 2.6.0 instead of the CUDA 12.8 stack used on S17.",
  "version_reference": "https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html",
  "network_note": "Direct HTTPS failed. Temporary loopback-only SSH reverse proxy used for public package/model downloads; no credentials copied to the server.",
  "installation_note": "Conda environment completed. Slow initial mirror proxy was replaced by a physical-interface-routed SSH loopback tunnel before large dependency downloads.",
  "python": "3.11.18 (main, Oct 26 2026, 18:19:35) [GCC 14.3.0]",
  "torch": "2.6.0+cu124",
  "torch_cuda": "12.4",
  "vllm": "0.8.5",
  "transformers": "4.51.3",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_total_bytes": 51002015744,
  "disk_free_bytes_after_environment": 79168962560,
  "cuda_matmul_validation": "passed",
  "cached_assets": {
    "model": "Qwen/Qwen3-0.6B",
    "model_type": "qwen3",
    "vocab_size": 151669,
    "max_prompt_tokens_in_fixed_pools": 231,
    "max_new_tokens": 512,
    "max_model_len": 4096,
    "dataset": {
      "repo": "openai/gsm8k",
      "config": "main",
      "splits": {
        "train": {
          "rows": 7473,
          "fingerprint": "c6f812ae33c9159d"
        },
        "test": {
          "rows": 1319,
          "fingerprint": "59ec1b7f9357c7a2"
        }
      }
    },
    "disk_free_bytes_before_l0": 79125778432
  },
  "model_cache_manifest": {
    "model": "Qwen/Qwen3-0.6B",
    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "snapshot": "/root/work/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
    "files": [
      {
        "file": "config.json",
        "bytes": 726,
        "digest": "f5c3703b78ae2a478ae15b247e9f855e0ce2107b",
        "verified": true
      },
      {
        "file": "generation_config.json",
        "bytes": 239,
        "digest": "20a8a9156fc8c3f25295ca067f61fdf120d517c5",
        "verified": true
      },
      {
        "file": "merges.txt",
        "bytes": 1671853,
        "digest": "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7",
        "verified": true
      },
      {
        "file": "model.safetensors",
        "bytes": 1503300328,
        "digest": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        "verified": true
      },
      {
        "file": "tokenizer.json",
        "bytes": 11422654,
        "digest": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "verified": true
      },
      {
        "file": "tokenizer_config.json",
        "bytes": 9732,
        "digest": "417d038a63fa3de29cfde265caedae14d1a58d92",
        "verified": true
      },
      {
        "file": "vocab.json",
        "bytes": 2776833,
        "digest": "4783fe10ac3adce15ac8f358ef5462739852c569",
        "verified": true
      }
    ],
    "completed_at": "2026-11-11T12:13:19.397409",
    "weight_source": "https://www.modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master/model.safetensors",
    "method": "Validated HTTP ranges; assembled SHA256 verified against official Hugging Face metadata. Final registration used Python 3.11."
  },
  "unit_tests": {
    "tests": "149",
    "failures": "0",
    "errors": "0",
    "skipped": "0",
    "time": "9.576"
  },
  "offline_assets_validation": "passed",
  "prelaunch_attempt": {
    "exit_code": 127,
    "reason": "A CRLF terminator on the temporary bootstrap final command prevented L0 from starting. Rewritten with LF; no training outputs existed.",
    "preserved_artifacts": "/root/work/results/e5/l0_prelaunch_crlf_20261111_121537"
  }
}
```

数据盘与空间检查：

```json
{
  "storage_before": {
    "work": "/root/work",
    "data_free_gib": 73.6914291381836,
    "system_free_gib": 73.6914291381836,
    "minimum_free_gib": 15.0
  },
  "storage_after": {
    "work": "/root/work",
    "data_free_gib": 70.89065933227539,
    "system_free_gib": 70.89065933227539,
    "minimum_free_gib": 15.0
  }
}
```

运行前 df 输出（实际挂载目录以此及 storage_before.work 为准）：

```text
Filesystem                                         Size  Used Avail Use% Mounted on
tmpfs                                              8.8G  1.4M  8.8G   1% /run
/dev/vda1                                           97G   24G   74G  24% /
tmpfs                                               44G     0   44G   0% /dev/shm
tmpfs                                              5.0M     0  5.0M   0% /run/lock
/dev/vda15                                         105M  6.3M   99M   7% /boot/efi
//nm2.storage.spacehpc.com/smb_user_3h7tevy1rst8n   30T   29T  1.1T  97% /mnt/storage
//nm2.pub.storage.spacehpc.com/pub_data             22T   21T  1.1T  96% /mnt/pub_storage
tmpfs                                              8.8G  8.0K  8.8G   1% /run/user/0

```

## 2. 合格检查表

| step | status |
| --- | --- |
| 1 | passed |
| 2 | vllm_failed_hf_fallback |
| 3 | failed |
| 4 | not_run |
| 5 | not_run |


| check | target | value | passed |
| --- | --- | --- | --- |
| vllm_weight_sync | mean absolute logprob difference < 0.02 |  | False |


权重同步检查：

```json
{
  "passed": false,
  "backend": "vllm",
  "error": "sync_public_api: failure: "
}
```

## 3. 显存与单步用时

未运行或未产出数据。


驱动显存为周期采样峰值，可能漏掉采样间瞬态；PyTorch 的睡眠池计数不能直接等同于物理占用。

micro-batch 测速（仅测试，后续配置由用户决定）：

```json
{
  "status": "not_run"
}
```

## 4. 训练指标

未运行或未产出数据。


## 5. 诊断

未运行或未产出数据。


K 建议与信号是否可测：

```json
{
  "status": "not_run"
}
```

## 6. 后续用时与费用估算

未运行或未产出数据。


估算依据：

```json
{
  "status": "not_run"
}
```

## 7. 问题与修改

正式入口现在使用独立子进程隔离各阶段，统一结果和 checkpoint 路径，限制 vLLM 并发并在唤醒前释放闲置缓存；统计归约采用分块 float64。代码差异见 code_changes.diff。

全部尝试及参数（失败尝试保留）：

```json
[
  {
    "stage": "env",
    "label": "environment",
    "micro_batch_size": 4,
    "vllm": {
      "enable_sleep_mode": true,
      "gpu_memory_utilization": 0.2,
      "enforce_eager": true,
      "max_model_len": 4096,
      "max_num_seqs": 16,
      "max_num_batched_tokens": 512,
      "enable_chunked_prefill": true
    },
    "backend": "vllm",
    "status": "complete",
    "physical_peak_gib": 0.0,
    "elapsed_seconds": 7.002434583000195
  },
  {
    "stage": "sync",
    "label": "sync_auto",
    "micro_batch_size": 4,
    "vllm": {
      "enable_sleep_mode": true,
      "gpu_memory_utilization": 0.2,
      "enforce_eager": true,
      "max_model_len": 4096,
      "max_num_seqs": 16,
      "max_num_batched_tokens": 512,
      "enable_chunked_prefill": true
    },
    "backend": "vllm",
    "status": "failed",
    "error": {
      "status": "complete",
      "result": {
        "backend": "vllm",
        "sync_interface": "driver_worker.model_runner.model.load_weights",
        "sync_time_sec": 4.469640933999926,
        "logprob_max_diff": 3.5473339557647705,
        "logprob_mean_diff": 0.5818749666213989,
        "tol_mean_diff": 0.02,
        "dup_group_frac": 0.0,
        "passed": false
      },
      "physical_peak_gib": 9.521484375,
      "elapsed_seconds": 31.008812989000035
    }
  },
  {
    "stage": "sync",
    "label": "sync_public_api",
    "micro_batch_size": 4,
    "vllm": {
      "enable_sleep_mode": true,
      "gpu_memory_utilization": 0.2,
      "enforce_eager": true,
      "max_model_len": 4096,
      "max_num_seqs": 16,
      "max_num_batched_tokens": 512,
      "enable_chunked_prefill": true,
      "sync_method": "apply_model"
    },
    "backend": "vllm",
    "status": "failed",
    "error": {
      "status": "complete",
      "result": {
        "backend": "vllm",
        "sync_interface": "LLM.apply_model",
        "sync_time_sec": 3.5848590270002205,
        "logprob_max_diff": 3.5473339557647705,
        "logprob_mean_diff": 0.5818749666213989,
        "tol_mean_diff": 0.02,
        "dup_group_frac": 0.0,
        "passed": false
      },
      "physical_peak_gib": 9.521484375,
      "elapsed_seconds": 29.008055280000008
    }
  },
  {
    "stage": "train",
    "label": "train_k0_attempt0",
    "micro_batch_size": 4,
    "vllm": {
      "enable_sleep_mode": true,
      "gpu_memory_utilization": 0.2,
      "enforce_eager": true,
      "max_model_len": 4096,
      "max_num_seqs": 16,
      "max_num_batched_tokens": 512,
      "enable_chunked_prefill": true
    },
    "backend": "hf"
  }
]
```

L0 后停止，未运行后续实验，也未关机或修改计费。
