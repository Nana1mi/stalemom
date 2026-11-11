"""Generate the L0 report exclusively from the saved JSON and CSV artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def _table(rows, columns):
    if not rows:
        return "未运行或未产出数据。\n"
    def text(value):
        if isinstance(value, float):
            return f"{value:.6g}" if math.isfinite(value) else str(value)
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(text(row.get(column, "")) for column in columns) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def generate_report(out: Path):
    summary = json.loads((out / "l0_summary.json").read_text(encoding="utf-8"))
    metrics = []
    for k in (0, 2):
        rows = _csv(out / f"train_log_k{k}.csv")
        if not rows:
            continue
        def mean(key):
            return sum(float(row[key]) for row in rows) / len(rows)
        physical = [float(row["nvidia_smi_peak_gb"]) for row in rows if row.get("nvidia_smi_peak_gb") not in (None, "")]
        metrics.append({"k": k, "steps": len(rows), "torch_peak_gib": max(float(row["peak_vram_gb"]) for row in rows),
                        "physical_peak_gib_sampled": max(physical) if physical else "unavailable",
                        "generation_seconds": mean("step_time_gen"), "training_seconds": mean("step_time_train"),
                        "sync_seconds": mean("step_time_sync"), "reward_mean": mean("reward_mean"),
                        "zero_adv_frac": mean("zero_adv_frac"), "dup_group_frac": mean("dup_group_frac"),
                        "trunc_rate": mean("trunc_rate"), "mean_w": mean("mean_w")})
    _write(out / "l0_metrics.csv", metrics)
    _write(out / "step_status.csv", [{"step": step, "status": status} for step, status in summary.get("step_status", {}).items()])
    _write(out / "acceptance_checks.csv", summary.get("checks", []))
    estimates = summary.get("step5_estimates", {})
    costs = [{"phase": phase, "gpu_hours": hours, "cost_yuan": estimates["cost_yuan"][phase]}
             for phase, hours in estimates.get("gpu_hours", {}).items()]
    _write(out / "cost_estimates.csv", costs)
    env = summary.get("step1_env", {})
    lines = ["# L0 报告", "", f"状态：`{summary['status']}`。本报告由 `llm.l0_report` 从保存的 JSON/CSV 生成。", "",
             "## 1. 环境与磁盘", "", "```json", json.dumps(env, ensure_ascii=False, indent=2), "```", "",
             "环境安装记录与版本选择：", "", "```json", json.dumps(summary.get("environment_setup", {}), ensure_ascii=False, indent=2), "```", "",
             "数据盘与空间检查：", "", "```json", json.dumps({key: summary.get(key) for key in ("storage_before", "storage_after")}, ensure_ascii=False, indent=2), "```", "",
             "运行前 df 输出（实际挂载目录以此及 storage_before.work 为准）：", "", "```text", summary.get("df_before", "unavailable"), "```", "",
             "## 2. 合格检查表", "", _table(_csv(out / "step_status.csv"), ["step", "status"]), "",
             _table(_csv(out / "acceptance_checks.csv"), ["check", "target", "value", "passed"]), "",
             "权重同步检查：", "", "```json", json.dumps(summary.get("step2_weight_sync", {"status": "not_run"}), ensure_ascii=False, indent=2), "```", "",
             "## 3. 显存与单步用时", "", _table(_csv(out / "l0_metrics.csv"), ["k", "steps", "torch_peak_gib", "physical_peak_gib_sampled", "generation_seconds", "training_seconds", "sync_seconds"]), "",
             "驱动显存为周期采样峰值，可能漏掉采样间瞬态；PyTorch 的睡眠池计数不能直接等同于物理占用。", "",
             "micro-batch 测速（仅测试，后续配置由用户决定）：", "", "```json", json.dumps(summary.get("throughput_probe", {"status": "not_run"}), ensure_ascii=False, indent=2), "```", "",
             "## 4. 训练指标", "", _table(_csv(out / "l0_metrics.csv"), ["k", "reward_mean", "zero_adv_frac", "dup_group_frac", "trunc_rate", "mean_w"]), "",
             "## 5. 诊断", "", _table(_csv(out / "diag_summary.csv"), ["step", "SNR_hat", "H", "N_bar", "sketch_rel_err", "zero_adv_frac", "signal_detected"]), "",
             "K 建议与信号是否可测：", "", "```json", json.dumps(summary.get("step4_mini_l2", {}).get("k_recommendation", {"status": "not_run"}), ensure_ascii=False, indent=2), "```", "",
             "## 6. 后续用时与费用估算", "", _table(_csv(out / "cost_estimates.csv"), ["phase", "gpu_hours", "cost_yuan"]), "",
             "估算依据：", "", "```json", json.dumps(estimates or {"status": "not_run"}, ensure_ascii=False, indent=2), "```", "",
             "## 7. 问题与修改", "", "正式入口现在使用独立子进程隔离各阶段，统一结果和 checkpoint 路径，限制 vLLM 并发并在唤醒前释放闲置缓存；统计归约采用分块 float64。代码差异见 code_changes.diff。", "",
             "HF 回退支持带左填充的 prompt 分批生成，具体批量数记录在 config.rollout.hf_batch_prompts 与各次尝试中。同步子进程退出异常时仍保留已经写出的误差和接口信息。", "",
             "全部尝试及参数（失败尝试保留）：", "", "```json", json.dumps(summary.get("attempts", []), ensure_ascii=False, indent=2), "```", ""]
    if summary.get("error"):
        lines += ["停止原因：", "", "```text", summary["error"], "```", ""]
    for run in summary.get("step3_runs", {}).values():
        if run.get("all_advantages_zero"):
            lines += ["警告：至少一个训练 run 的全部 batch 优势均为零；资源链路通过不代表发生了有效学习。", ""]
            break
    lines += ["L0 后停止，未运行后续实验，也未关机或修改计费。", ""]
    (out / "l0_report.md").write_text("\n".join(lines), encoding="utf-8")
