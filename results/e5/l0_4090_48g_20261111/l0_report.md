# L0 报告

状态：`complete_with_hf_fallback`。本报告由 `llm.l0_report` 从保存的 JSON/CSV 生成。

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
    "disk_free_bytes_before_l0": 76117516288
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
    "tests": "151",
    "failures": "0",
    "errors": "0",
    "skipped": "0",
    "time": "9.820"
  },
  "offline_assets_validation": "passed",
  "prelaunch_attempt": {
    "exit_code": 127,
    "reason": "A CRLF terminator on the temporary bootstrap final command prevented L0 from starting. Rewritten with LF; no training outputs existed.",
    "preserved_artifacts": "/root/work/results/e5/l0_prelaunch_crlf_20261111_121537"
  },
  "prior_unbatched_hf_attempt": {
    "preserved_paths": {
      "/root/work/results/e5/l0": "/root/work/results/e5/l0_unbatched_hf_20261111_123808",
      "/root/work/checkpoints/e5/l0": "/root/work/checkpoints/e5/l0_unbatched_hf_20261111_123808",
      "/root/work/l0_results.tar.gz": "/root/work/l0_unbatched_hf_20261111_123808.tar.gz"
    },
    "completed_train_rows": {
      "0": 2,
      "2": 0
    },
    "hf_batch_prompts_before": 1,
    "hf_batch_prompts_after": 4,
    "sampling_note": "Prompt count, group size and max generation length unchanged. Batched sampling changes RNG consumption; new run restarts seed42."
  }
}
```

数据盘与空间检查：

```json
{
  "storage_before": {
    "work": "/root/work",
    "data_free_gib": 70.88995361328125,
    "system_free_gib": 70.88995361328125,
    "minimum_free_gib": 15.0
  },
  "storage_after": {
    "work": "/root/work",
    "data_free_gib": 49.586936950683594,
    "system_free_gib": 49.586936950683594,
    "minimum_free_gib": 15.0
  }
}
```

运行前 df 输出（实际挂载目录以此及 storage_before.work 为准）：

```text
Filesystem                                         Size  Used Avail Use% Mounted on
tmpfs                                              8.8G  1.4M  8.8G   1% /run
/dev/vda1                                           97G   26G   71G  27% /
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
| 3 | passed |
| 4 | passed |
| 5 | passed |


| check | target | value | passed |
| --- | --- | --- | --- |
| vllm_weight_sync | mean absolute logprob difference < 0.02 | 0.5818749666213989 | False |
| k0:steps | 10 | 10 | True |
| k0:finite_loss_and_grad | all finite | True | True |
| k0:torch_peak_gib | < 40.0 | 13.727941513061523 | True |
| k0:physical_peak_gib_sampled | < 40.0 | 20.359375 | True |
| k0:dup_group_frac | < 0.05 | 0.0 | True |
| k0:mean_w | exactly 1 | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | True |
| k2:steps | 10 | 10 | True |
| k2:finite_loss_and_grad | all finite | True | True |
| k2:torch_peak_gib | < 40.0 | 13.774137020111084 | True |
| k2:physical_peak_gib_sampled | < 40.0 | 20.39453125 | True |
| k2:dup_group_frac | < 0.05 | 0.0 | True |
| k2:trunc_rate_after_warmup | > 0 at every step >= k | [0.028144853875476494, 0.03771786492374728, 0.04154193903109536, 0.04381581295734428, 0.04573034043697881, 0.04808760024676126, 0.056633233651785006, 0.0636515126622386] | True |
| diag_step0:sketch_rel_err | < 0.05 | 0.0014672168387200328 | True |
| diag_step9:sketch_rel_err | < 0.05 | 0.0030400284476720194 | True |


权重同步检查：

```json
{
  "backend": "vllm",
  "sync_interface": "LLM.apply_model",
  "sync_time_sec": 3.797737288999997,
  "logprob_max_diff": 3.5473339557647705,
  "logprob_mean_diff": 0.5818749666213989,
  "tol_mean_diff": 0.02,
  "dup_group_frac": 0.0,
  "passed": false,
  "error": "sync_public_api: failure: ",
  "worker_exit_code": -6
}
```

## 3. 显存与单步用时

| k | steps | torch_peak_gib | physical_peak_gib_sampled | generation_seconds | training_seconds | sync_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 10 | 13.727941513061523 | 20.359375 | 143.3042353265999 | 10.988233816299907 | 1.5309910511999079 |
| 2 | 10 | 13.774137020111084 | 20.39453125 | 143.2394680947 | 13.39906506999996 | 1.9184359949000283 |


驱动显存为周期采样峰值，可能漏掉采样间瞬态；PyTorch 的睡眠池计数不能直接等同于物理占用。

micro-batch 测速（仅测试，后续配置由用户决定）：

```json
{
  "micro_batch_size": 8,
  "steps": 3,
  "mean_training_seconds": 11.128668890667237,
  "baseline_k0_mean_training_seconds": 10.988233816299907,
  "training_speedup": 0.987380784193777,
  "torch_peak_gib": 16.80008840560913,
  "physical_peak_gib_sampled": 18.658203125,
  "adopted_for_later_stages": false
}
```

## 4. 训练指标

| k | reward_mean | zero_adv_frac | dup_group_frac | trunc_rate | mean_w |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.66640625 | 0.55625 | 0.0 | 0.0 | 1.0 |
| 2 | 0.65234375 | 0.5375 | 0.0 | 0.03749071331321143 | 0.9809975933072517 |


## 5. 诊断

| step | SNR_hat | H | N_bar | sketch_rel_err | zero_adv_frac | signal_detected |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.01461438253389026 | 0.013172539867341222 | 0.9145133824786897 | 0.0014672168387200328 | 0.5390625 | True |
| 9 | 0.03260864345750934 | 0.019275597029219917 | 0.610394852092282 | 0.0030400284476720194 | 0.5859375 | True |


K 建议与信号是否可测：

```json
{
  "SNR_hat": 0.03260864345750934,
  "H": 0.019275597029219917,
  "signal_detected": true,
  "recommended_K": 128,
  "note": "CODEX_L0_4090.md SNR tier."
}
```

## 6. 后续用时与费用估算

| phase | gpu_hours | cost_yuan |
| --- | --- | --- |
| L1 | 51.94115339803324 | 139.2022911067291 |
| L2 | 17.405975582999993 | 46.64801456243998 |
| L3 | 47.537681560749995 | 127.40098658280999 |
| L4_core | 105.70464610639998 | 283.28845156515195 |


估算依据：

```json
{
  "mean_train_step_seconds": {
    "0": 155.8234601940997,
    "2": 158.5569691596
  },
  "mean_diag_batch_seconds": 163.18102109062494,
  "gpu_hours": {
    "L1": 51.94115339803324,
    "L2": 17.405975582999993,
    "L3": 47.537681560749995,
    "L4_core": 105.70464610639998
  },
  "gpu_price_yuan_per_hour": 2.68,
  "cost_yuan": {
    "L1": 139.2022911067291,
    "L2": 46.64801456243998,
    "L3": 127.40098658280999,
    "L4_core": 283.28845156515195
  },
  "note": "Estimates only, based on the measured baseline runs. No L1-L4 jobs were started."
}
```

## 7. 问题与修改

正式入口现在使用独立子进程隔离各阶段，统一结果和 checkpoint 路径，限制 vLLM 并发并在唤醒前释放闲置缓存；统计归约采用分块 float64。代码差异见 code_changes.diff。

HF 回退支持带左填充的 prompt 分批生成，具体批量数记录在 config.rollout.hf_batch_prompts 与各次尝试中。同步子进程退出异常时仍保留已经写出的误差和接口信息。

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
    "hf_batch_prompts": null,
    "status": "complete",
    "physical_peak_gib": 0.0,
    "elapsed_seconds": 7.002454448000208
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
    "hf_batch_prompts": null,
    "status": "failed",
    "error": {
      "status": "complete",
      "result": {
        "backend": "vllm",
        "sync_interface": "driver_worker.model_runner.model.load_weights",
        "sync_time_sec": 4.253141056000459,
        "logprob_max_diff": 3.5473339557647705,
        "logprob_mean_diff": 0.5818749666213989,
        "tol_mean_diff": 0.02,
        "dup_group_frac": 0.0,
        "passed": false
      },
      "physical_peak_gib": 9.521484375,
      "elapsed_seconds": 31.00865447200067,
      "worker_exit_code": -6
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
    "hf_batch_prompts": null,
    "status": "failed",
    "error": {
      "status": "complete",
      "result": {
        "backend": "vllm",
        "sync_interface": "LLM.apply_model",
        "sync_time_sec": 3.797737288999997,
        "logprob_max_diff": 3.5473339557647705,
        "logprob_mean_diff": 0.5818749666213989,
        "tol_mean_diff": 0.02,
        "dup_group_frac": 0.0,
        "passed": false
      },
      "physical_peak_gib": 9.521484375,
      "elapsed_seconds": 30.008612618999905,
      "worker_exit_code": -6
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
    "backend": "hf",
    "hf_batch_prompts": 4,
    "status": "complete",
    "physical_peak_gib": 20.359375,
    "elapsed_seconds": 1618.3632285200001
  },
  {
    "stage": "train",
    "label": "train_k2_attempt0",
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
    "backend": "hf",
    "hf_batch_prompts": 4,
    "status": "complete",
    "physical_peak_gib": 20.39453125,
    "elapsed_seconds": 1667.359228661
  },
  {
    "stage": "train",
    "label": "throughput_mb8",
    "micro_batch_size": 8,
    "vllm": {
      "enable_sleep_mode": true,
      "gpu_memory_utilization": 0.2,
      "enforce_eager": true,
      "max_model_len": 4096,
      "max_num_seqs": 16,
      "max_num_batched_tokens": 512,
      "enable_chunked_prefill": true
    },
    "backend": "hf",
    "hf_batch_prompts": 4,
    "status": "complete",
    "physical_peak_gib": 18.658203125,
    "elapsed_seconds": 468.1057373089998
  },
  {
    "stage": "diag",
    "label": "diag_step0",
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
    "backend": "hf",
    "hf_batch_prompts": 4,
    "status": "complete",
    "physical_peak_gib": 22.990234375,
    "elapsed_seconds": 1185.2441677380002
  },
  {
    "stage": "diag",
    "label": "diag_step9",
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
    "backend": "hf",
    "hf_batch_prompts": 4,
    "status": "complete",
    "physical_peak_gib": 29.208984375,
    "elapsed_seconds": 1450.3061710149996
  }
]
```

L0 后停止，未运行后续实验，也未关机或修改计费。

## 交付补充

旧的逐 prompt HF 尝试在用户同意批量优化后停止，完整原始结果保留于 `prior_attempts/unbatched_hf/`，路径迁移映射见 `prior_attempts/relocation_mapping.json`。正式训练数据未与旧尝试合并。

首步批量对比（序列因批量采样的 RNG 消耗不同，仅作初步性能比较）：

```json
{
  "source_unbatched": "/root/work/results/e5/l0_unbatched_hf_20261111_123808/train_k0_attempt0/train_log.csv",
  "source_batched": "/root/work/results/e5/l0/train_k0_attempt0/train_log.csv",
  "comparison": "First completed step, same settings except prompt batching. Sampled sequences differ due RNG consumption. Initial measurement only, not controlled benchmark.",
  "before_hf_batch_prompts": 1,
  "after_hf_batch_prompts": 4,
  "before_generation_seconds": 375.85353426300026,
  "after_generation_seconds": 141.03787660499984,
  "generation_speedup": 2.6649120315079684,
  "before_sum_stage_seconds": 389.22364341,
  "after_sum_stage_seconds": 154.19224851799936,
  "sum_stage_speedup": 2.5242750342574105,
  "torch_peak_gib": 11.372558116912842,
  "physical_peak_gib_sampled": 15.783203125,
  "price_yuan_per_hour": 2.68,
  "projection_note": "Crude total estimate based on1training step and39train-equivalent batches. Diagnostics overhead and token lengths vary.",
  "projected_cost_yuan": 4.476714948639248,
  "total_l0_39batch_equivalent_projection_hours": 1.6704160256116598
}
```

计时口径与估算局限：

```json
{
  "training": [
    {
      "k": 0,
      "steps": 10,
      "logged_phase_sum_seconds": 1558.234601940997,
      "worker_elapsed_seconds": 1618.3632285200001,
      "unallocated_worker_seconds": 60.12862657900314
    },
    {
      "k": 2,
      "steps": 10,
      "logged_phase_sum_seconds": 1585.5696915959998,
      "worker_elapsed_seconds": 1667.359228661,
      "unallocated_worker_seconds": 81.78953706500033
    }
  ],
  "cost_estimation_limitations": "Future training estimates use the logged generation+training+sync times. Model initialization, checkpoint IO, logprob verification and other overhead outside these phase timers are excluded. Diagnostic estimates use measured diag_time_sec. Future k4/k8 training is approximated by the measured k2 baseline; their CPU snapshot/IO overhead was not measured. The microbatch probe is a short raw timing comparison, not a controlled equal-token benchmark."
}
```

最终资源状态：

```json
{
  "time": "2026-11-11T14:30:11.846278",
  "l0_status": "complete_with_hf_fallback",
  "bootstrap_exit_code": 0,
  "gpu_compute_processes": [],
  "disk_free_gib": 49.52785110473633,
  "L1_L4_started": false,
  "server_shutdown": false,
  "billing_changed": false
}
```
