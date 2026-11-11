# 4090 48GB L0 结果备份

状态：`complete_with_hf_fallback`。训练与诊断使用 HF 后端；vLLM 权重同步未通过。

测试：151 项通过。k=0、k=2 各完成 10 步。

全过程驱动采样显存峰值：29.209 GiB。HF 首步生成速度比优化前约快 2.66 倍（初步对比，采样序列不同）。

| checkpoint | SNR_hat | H | sketch_rel_err |
| --- | ---: | ---: | ---: |
| 0 | 0.0146144 | 0.0131725 | 0.00146722 |
| 9 | 0.0326086 | 0.0192756 | 0.00304003 |

按点估计建议 K=128。

| 后续阶段（未运行） | 估算 GPU 时 | 估算费用（元） |
| --- | ---: | ---: |
| L1 | 51.94 | 139.20 |
| L2 | 17.41 | 46.65 |
| L3 | 47.54 | 127.40 |
| L4_core | 105.70 | 283.29 |

合计约 222.59 GPU 时、¥596.54，基于本次 HF 速度；估算局限见报告。

- [完整报告](l0_report.md)
- [完整结果归档](l0_results.tar.gz)（含诊断 sketch、原始日志和旧尝试）
- [归档 SHA256](l0_results.sha256)
- [最终汇总](l0_summary.json)
- [代码差异](code_changes.diff)
- [交付校验](delivery_verification.json)

本目录的 CSV、JSON 和日志供直接查看；被通用忽略规则排除的诊断 sketch 保存在完整归档中。checkpoint 按原要求未下载、未上传，仍仅保留在服务器上。密码文件未上传。服务器未由本任务关闭。

本索引由脚本读取 final_facts.json、hf_batching_initial_comparison.json 和 pytest_results.xml 生成；实验数字均来自已保存的 CSV/JSON。
