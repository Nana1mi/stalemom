# SPEC E5：RLVR 中动量与梯度对齐的诊断实验

> 适用仓库：`Nana1mi/stalemom`｜执行者：Gemini｜编写日期：2026-11-10
> 前置文档：`SPEC_E0-E2.md`、`docs/theory_derivation.tex`

---

## 0. 背景与要回答的问题

*Do We Need Adam?*（ICML 2026，arXiv 2602.07729）观察到：在 RLVR 中，历史动量和当前 batch 梯度的余弦约为 −0.007，并把这一现象归因于策略非平稳。我们怀疑，单批梯度噪声大也足以让余弦接近 0，而这并不说明动量偏离了真正的优化方向。

本 SPEC 要回答三个问题：

1. **Q1**：单批余弦 C1 低，究竟是噪声造成的，还是真实梯度方向在漂移？
2. **Q2**：在同步设定下，动量是否真的有用？能否复现原论文的结论？
3. **Q3**：staleness（k）和 IS 截断（c）会怎样改变上面两个问题的答案？

**三个核心指标**（全文统一）：

| 记号 | 定义 | 含义 |
|---|---|---|
| C1 | cos(m_{t−1}, g_t) | 动量与单批随机梯度的余弦，即原论文的指标 |
| C2 | cos(m_{t−1}, h_t)，其中 h_t = E[g_t \| θ_t] | 动量与平均梯度的余弦 |
| C3 | cos(h_t, h_s) | 不同 checkpoint 的平均梯度之间的余弦，衡量真实漂移 |

还要估计信噪比 SNR = ‖h‖² / E‖g − h‖²，以及方差分解（prompt 异质性 vs rollout 随机性）。

---

## 1. 算力与租卡总表（先读）

| 阶段 | 内容 | 硬件 | 预计时长 | 是否租卡 |
|---|---|---|---|---|
| D1 | bandit 诊断与估计器验证 | 本地 CPU | 2–4 小时 CPU | **不需要** |
| D2 | bandit 调参后的动量对照 | 本地 CPU | 1–2 小时 CPU | **不需要** |
| L0 | LLM 冒烟测试与吞吐测量 | RTX 3090 24GB | 1–2 GPU 时 | **需要**，测完先关机 |
| L1 | 同步训练（SGD / SGD+mom），保存 checkpoint | RTX 3090 | 约 7 GPU 时 | **需要** |
| L2 | 冻结 checkpoint 做梯度诊断 | RTX 3090 | 约 3–4 GPU 时 | **需要** |
| L3 | 异步训练（k、c）加诊断 | RTX 3090 | 约 7–8 GPU 时 | **需要** |
| L4 | 异步下的优化器对比 | RTX 3090 | 核心约 14 GPU 时，可选扩展约 40 GPU 时 | **需要** |

- **核心部分合计（L0–L4 核心）**：约 35–50 GPU 时，按 ¥1.18/h 计约 **¥40–60**。
- **加上可选扩展（AdamW / RMSProp 全对比）**：约 80–110 GPU 时，约 **¥95–130**。
- 以上都是估算。**L0 结束后必须按实测吞吐重新估算，交给用户确认后再开 L1。**

**服务器配置要求**：
- GPU：1 张 RTX 3090 24GB 即可，不需要多卡。
- 内存：≥ 48GB。L3 的 k=8 需要在内存里存 9 份 fp32 权重快照，约 22GB。
- **磁盘：建议数据盘扩到 ≥ 100GB。** 默认的 50GB 不够：环境约 15GB，模型约 1.5GB，每个 checkpoint（fp32 权重 + 动量缓冲）约 4.8GB。如果不扩盘，就必须严格按 §6.4 的规则边跑边删。
- CUDA ≥ 12.1，Ubuntu 22.04。

**租卡时机**：D1、D2 在本地跑完并经用户审查后，再开卡做 L0。L0 跑完后关机，等用户确认预算再开 L1–L4。

---

## 2. 通用规则（违反任何一条都要停下来报告）

1. **不得修改 `tests/` 下的任何文件。** 新测试放在 `tests_e5/`。
2. 报告里的每个数字都必须能对应到某个 CSV 的某一行。表格要用脚本从 CSV 生成，并提交生成脚本。
3. 每个阶段结束后**停下来写报告，等用户确认再继续**，不得自行进入下一阶段。
4. 不得为了让结果"好看"而调参、挑种子或删除发散的 run。发散的 run 记为 NaN/inf，并保留在 CSV 中。
5. 不在服务器上保存任何个人凭据（GitHub token、API key）。代码由用户自己 push。
6. 长任务用 `nohup` 或 `tmux` 运行，日志写入 `results/e5/<phase>/run_log.txt`。任务完成后写一个 `DONE` 标记文件。**不要自己关机，不要动计费设置。**
7. 记录完整环境：`pip freeze`、`nvidia-smi`、git commit hash，写入每个阶段的 `env.txt`。
8. 结论措辞：没有直接实验证据的机理一律写"假设"，不要写"证明""确证"。

---

## 3. D1：bandit 上的诊断与估计器验证（本地 CPU）

### 3.1 目的

bandit 中的真实平均梯度 h_t 可以精确计算。利用这一点，先验证 §5 中的无偏估计方法是否正确，再把同一套方法搬到 LLM 上。

### 3.2 实现

新建 `stalemom/diag.py`，**复用现有的 `bandit`、`estimators`、`sim` 模块，不修改它们的行为**。

```python
def sample_grads_at(theta, q, problem, estimator, c, B, K, rng) -> np.ndarray:
    """在固定的 theta 和行为分布 q 下，独立抽取 K 个随机方向 g，形状为 (K, dim)。"""

def nested_stats(G, m, R) -> dict:
    """G: (S*R, dim)，按 prompt-set 分组；bandit 中没有 prompt，S 组只代表独立重复。
    m: 动量向量。返回 §5 中定义的全部估计量。"""
```

### 3.3 实验网格

- 问题实例：与 E2 相同（K=10，n_good=2，λ=1，init_seed=0）
- 估计器 ∈ {onpolicy, is, tis}，c=1.2
- k ∈ {0, 16}；B ∈ {4, 32}；β ∈ {0, 0.6, 0.9, 0.99}
- η：每个 (估计器, k, B, β) 取 D2 中调出的最优 η。先做 D2，或者直接用下表的初值：B=32 时 η=0.1，B=4 时 η=0.025
- 冻结点：先用 `simulate(..., return_traj=True)` 跑出轨迹，在 t ∈ {16, 64, 256} 取 θ_t、q_t 和 m̂_{t−1}
- 在每个冻结点上抽取 K=64 个 g，R=200 条轨迹（每条轨迹单独冻结和抽样）

### 3.4 输出

`results/e5/d1/summary.csv`，每行一个 (估计器, k, B, β, t)，需要包含：

- 精确值：C2_exact_h = cos(m, h)；C2_exact_mean = cos(m, h + b)，其中 b 是估计器的偏差（tis 时非零）；SNR_exact
- 估计值：C1、C2_hat、C1_pred、SNR_hat、H_hat 与 ‖E g‖² 的精确值
- 以上各量在 R 条轨迹上的均值 ± SE

### 3.5 验证标准（写进 `tests_e5/test_diag_bandit.py`）

1. H_hat 是 ‖E g‖² 的无偏估计：在 R=200 时，|mean(H_hat) − 精确值| < 3·SE。
2. C2_hat 与 C2_exact_mean 的差 < 3·SE。
3. C1_pred 与实测 C1 的相对误差 < 10%（SNR 很小时可以改用绝对误差 < 0.02）。
4. 构造一个"纯噪声"对照（h=0）：C1、C2_hat 都应在 0 附近，|·| < 3·SE。

### 3.6 预期（假设，待验证）

B=4、β=0.9 时，C1 很低（约 0.1），但 C2_exact 明显更高。这说明"C1 低"可以完全由噪声造成。

---

## 4. D2：bandit 上调好步长后的动量对照（本地 CPU）

- 新增 `stalemom/run_tuned_momentum.py`
- 网格：估计器 ∈ {onpolicy, is, tis, tis_debiased} × k ∈ {0, 16} × B ∈ {4, 32} × β ∈ {0, 0.6, 0.9, 0.99}
- 每个 β 在 η ∈ geomspace(0.01, 1.0, 11) 上取 A_T 最小的 η。发散的 run 记为 inf，不得跳过。R=200
- 输出：`results/e5/d2/tuned_momentum.csv`，包含每个 η 的 A_T ± SE（不只是最优值）、最优 η、final_gap
- 图：每个 (估计器, k, B) 画一张"β vs 调好后的 A_T"图
- 用户的初步结果供核对（R=100）：onpolicy、k=0、B=32 时，β=0、0.6、0.9、0.99 的最优 A_T 依次约为 0.0255、0.0237、0.0226、0.0466

**D1 和 D2 完成后，停下来交报告。**

---

## 5. 估计量定义（D1 与 L2 共用）

### 5.1 嵌套抽样设计

- 共 S 个 prompt-set，每个 set 包含 P 个 prompt（LLM 中 P=32）
- 每个 set 独立重复 R 次 rollout（LLM 中 R=4）
- 共 S·R 个 batch，每个 batch 的梯度记为 g_{s,i}，i = 1…R
- 把 set 分成两半：s < S/2 为 A 组，其余为 B 组。**按 set 分组，不按 batch 分组**，以保证两半互相独立

### 5.2 在线累积量（不保存单个梯度向量）

逐个处理 set，对每个 batch 记录标量：

- n_{s,i} = ‖g_{s,i}‖²
- d_{s,i} = ⟨m, g_{s,i}⟩

对每个 set，累加向量 T_s = Σ_i g_{s,i}，记录标量 ‖T_s‖²，再把 T_s 加进 A 或 B 的总和。

### 5.3 估计公式

记 N_A = N_B = (S/2)·R，Ā = A / N_A，B̄ = B / N_B。

| 量 | 估计式 | 说明 |
|---|---|---|
| E‖g‖² | N̄ = mean(n_{s,i}) | |
| ‖h‖² | H = ⟨Ā, B̄⟩ | 无偏，因为两半互相独立 |
| 每个 set 的 ‖E[g\|P_s]‖² | (‖T_s‖² − Σ_i n_{s,i}) / (R(R−1)) | 无偏，用 set 内两两交叉项 |
| M | 上式对 s 取平均 | |
| 总方差 | N̄ − H | |
| rollout 方差 | N̄ − M | |
| prompt 方差 | M − H | |
| SNR | H / (N̄ − H) | |
| ⟨m, h⟩ | mean(d_{s,i}) | 线性量，直接平均即无偏 |
| C1 | mean_{s,i}[ d_{s,i} / (‖m‖·√n_{s,i}) ] | 原论文指标 |
| C2_hat | mean(d) / (‖m‖·√H) | H ≤ 0 时记为 NaN，表示信号测不出 |
| C1_pred | (mean(d)/‖m‖)·mean(1/√n_{s,i}) | 由"噪声主导"假设预测的 C1 |
| C3(t, s) | ½(⟨Ā_t, B̄_s⟩ + ⟨B̄_t, Ā_s⟩) / √(H_t·H_s) | 跨 checkpoint |

### 5.4 标准误

对 prompt-set 做 jackknife（每次留出一个 set）。为此每个 set 需要额外保存：

- 标量 n、d
- T_s 的 CountSketch（见 §5.5）

### 5.5 跨 checkpoint 的存储：CountSketch

LLM 的完整梯度约 2.4GB，不能为每个 set 或每个 checkpoint 都存一份。改为存 CountSketch：

- 维度 D = 2²²（约 420 万维，fp32 约 16MB）
- 每个参数张量用固定种子生成 hash 桶号和 ±1 符号，种子 = hash(参数名)，保证跨 checkpoint 一致
- CountSketch 是线性的，所以对 T_s、A、B 做 sketch，等价于对每个梯度做 sketch 再求和
- **必须验证**：在同一个 checkpoint 内，用 sketch 算出的 ⟨Ā, B̄⟩ 与精确值的相对误差 < 5%。验证不通过就把 D 加倍

---

## 6. L0–L4：LLM 实验通用设定

### 6.1 模型与数据

- 模型：`Qwen/Qwen3-0.6B`，**全参数**训练（不用 LoRA），chat template 设 `enable_thinking=False`
- 训练数据：GSM8K train。固定种子，抽 1024 题作为 prompt 池
- 评测：GSM8K test 中固定的 500 题，贪心解码，max_new_tokens=512
- 奖励：解析答案中的最终数字，与标准答案精确匹配得 1，否则得 0。prompt 要求模型把答案写在 `\boxed{}` 里，解析失败计 0
- 生成：每个 prompt 4 条回答，temperature=1.0，top_p=1.0，max_new_tokens=512

### 6.2 训练循环（自写最小循环，不用 verl 或 TRL 的 trainer）

**推理引擎**：vLLM，与训练进程共用同一张卡。开启 `enable_sleep_mode`，`gpu_memory_utilization` 约 0.3，**版本要固定**。每步训练后把权重同步进 vLLM。

**权重同步必须验证**：同步后，对同一条序列，比较 vLLM 和训练模型算出的 token logprob，报告最大绝对差和平均绝对差，写入 L0 报告。

**精度**：
- **主权重必须是 fp32**，前向和反向用 bf16 autocast。原论文提到，bf16 主权重会把 SGD 的小更新舍入掉，所以这一条不能省
- 开启梯度检查点，micro-batch 自行设定

**Advantage**：GRPO 组内归一化，A = (r − mean) / (std + 1e-6)。std=0 的组优势全为 0，记录其比例 `zero_adv_frac`。

**损失**（每个 batch 只更新一次，不做 PPO 多 epoch）：

$$L = -\frac{1}{N_{tok}}\sum_{\text{tokens}} \mathrm{sg}\big(w_{tok}\big)\cdot A\cdot \log \pi_\theta(y_{tok}\mid\cdot)$$

- w_tok = min(π_θ / π_behav, c)，按 token 计算。c=∞ 表示不截断的完整 IS
- k=0 时强制 w ≡ 1
- 不加 KL 项（β_KL = 0）。这一点与 bandit 不同，要在报告中写明
- 不做梯度裁剪。如果某个 run 必须裁剪才能稳定，要停下来报告

### 6.3 staleness 的实现

- 在 CPU 内存中维护最近 k+1 份 **fp32** 权重快照组成的环形缓冲
- 第 t 步：
  1. 把 θ_{t−k} 转成 bf16，同步进 vLLM，生成 rollout
  2. 把 θ_{t−k} 的 fp32 快照载入 GPU 上一份**行为模型副本**，计算 log π_behav。这样行为概率和当前概率用的是同一套数值代码，k=0 时比值严格等于 1
  3. 在当前 θ_t 上计算损失并更新
- t < k 时，行为策略取 θ_0
- 每步记录：截断率（w 被截断的 token 比例）、mean(w)、近似 KL（mean(log π_θ − log π_behav)）、vLLM 与行为模型的 logprob 差

### 6.4 checkpoint 与磁盘规则

- 只在 step ∈ {0, 50, 100} 保存：fp32 权重和动量缓冲 m_{t−1}
  - SGD+mom：torch 的 `momentum_buffer`
  - AdamW：`exp_avg`，同时保存 `exp_avg_sq`，供后续分析用
- m_{t−1} 的定义：在第 t 步，即将与 g_t 结合的那个动量缓冲（已经包含 g_{t−1}，还没包含 g_t）
- 一个 run 的 L2 诊断做完后，立即删除它的 checkpoint，只保留 sketch、CSV 和图
- 磁盘剩余 < 10GB 时停止，并报告

### 6.5 训练过程中每步在线记录（所有 run 都要记）

| 字段 | 说明 |
|---|---|
| step、reward_mean、zero_adv_frac、resp_len_mean | |
| grad_norm、update_norm | |
| online_C1 = cos(m_{t−1}, g_t) | 与原论文同一个指标，开销很小 |
| trunc_rate、mean_w、approx_kl | k>0 时 |
| eval_acc | 每 50 步评测一次 |
| step_time_gen、step_time_train、step_time_sync | 用于估算算力 |

---

## 7. L0：冒烟测试（需要租卡，1–2 GPU 时）

1. 安装环境，固定版本，写 `env.txt`
2. 测试权重同步：比较 vLLM 和训练模型的 logprob（见 §6.2）
3. 用 SGD+mom 训练 10 步，验证：
   - 损失有限
   - k=0 时 w ≡ 1
   - k=2 时 trunc_rate > 0
   - 显存峰值 < 22GB
4. 测量吞吐：每步生成、训练、同步各用多少秒；L2 中每个 batch 的梯度诊断用多少秒
5. 做一次小规模 L2：S=4，R=2，验证 CountSketch 的精度（§5.5）
6. **按实测吞吐重新估算 L1–L4 的 GPU 时，写成表格交给用户。然后停下，等用户确认再继续。**

---

## 8. L1：同步训练（需要租卡，约 7 GPU 时）

- 设定：k=0，150 步，每步 32 个 prompt × 4 条回答
- 优化器与学习率网格：
  - SGD（无动量）：lr ∈ {0.03, 0.1, 0.3}
  - SGD+mom（torch 默认，momentum=0.9，dampening=0）：lr ∈ {0.003, 0.01, 0.03}
- 注意：torch 的动量是不做归一化的 heavy-ball，等效步长约为 lr/(1−β)，所以 0.01 大致对应 SGD 的 0.1。报告中要同时写出等效步长
- 种子：先每个配置 1 个种子，共 6 个 run。在最优 lr 上各补 1 个种子，共 8 个 run
- 两个最优配置（SGD 和 SGD+mom 各一个）保存 step 0、50、100 的 checkpoint，供 L2 使用
- 输出：`results/e5/l1/summary.csv`（每个 run 一行：最终 reward、最终 eval_acc、发散与否）和每步日志 CSV

---

## 9. L2：冻结 checkpoint 做梯度诊断（需要租卡，约 3–4 GPU 时）

- 对象：L1 中 SGD+mom 最优 run 的 step {0, 50, 100}。step 0 时 m=0，只算 SNR 和 H，跳过 C1、C2
- 嵌套设计：S=16 个 prompt-set × R=4 次 rollout = 64 个 batch，每个 batch 32 个 prompt × 4 条回答
- 梯度定义：与训练完全相同的损失（k=0 时 w ≡ 1），相同的 token 归约方式和精度
- 按 §5 计算全部估计量，用 jackknife 给出 SE
- C3：按 §5.5 用 CountSketch 计算 step 0、50、100 两两之间的值
- 输出：`results/e5/l2/diag_summary.csv`，`results/e5/l2/c3_matrix.csv`
- 报告中必须包含：
  - C1 与 C1_pred 的对比表
  - prompt 方差与 rollout 方差的占比
  - 零优势组比例

**如何解读**（写进报告，但不要预设结论）：

| 观测 | 更可能的解释 |
|---|---|
| C1 ≈ C1_pred，且 C2 明显 > 0 | C1 低主要是噪声造成的 |
| C2 ≈ 0，且 C3 低 | 平均方向确实在漂移，支持原论文的解释 |
| C2 ≈ 0，但 C3 高 | 平均方向稳定，但动量本身积累了噪声 |

**L1 和 L2 完成后，停下来交报告。**

---

## 10. L3：异步训练与诊断（需要租卡，约 7–8 GPU 时）

- 优化器：L1 中最优的 SGD+mom 配置
- 网格：k ∈ {4, 8} × c ∈ {1.2, ∞}，共 4 个 run，150 步，1 个种子。另加 k=8、c=1.2 的第 2 个种子
- 每个 run 在 step 50 和 100 做一次缩小版的 L2：S=8，R=4。这时梯度用带截断权重的训练损失，行为策略为 θ_{t−k}，可以从环形缓冲中取
- 额外记录：截断率和 approx_kl 随时间的变化
- 输出：`results/e5/l3/`

**预期（假设）**：k 越大，C2 越低（跟踪误差变大）；c=1.2 时 SNR 高于 c=∞（截断减小方差）。

---

## 11. L4：异步下的优化器对比（需要租卡）

### 11.1 核心（约 14 GPU 时）

- 优化器 ∈ {SGD, SGD+mom}，各用 L1 的 lr 网格
- 配置：(k=8, c=1.2) 和 (k=8, c=∞)
- 先每个 lr 跑 1 个种子，共 12 个 run；在最优 lr 上各补 1 个种子，共 16 个 run
- 与 L1 的同步结果合并，回答：同步时 momentum 是否有用？异步加截断后结论是否改变？

### 11.2 可选扩展（约 40 GPU 时，需用户单独批准）

- 加入 AdamW（β1=0.9，β2=0.999，lr ∈ {3e-7, 1e-6, 3e-6}）
- 加入 RMSProp（AdamW 设 β1=0，学习率网格相同）
- 覆盖 k ∈ {0, 8} × c ∈ {1.2, ∞}

### 11.3 输出

- `results/e5/l4/summary.csv`
- 主表：行为 (k, c)，列为各优化器在最优 lr 下的最终 eval_acc ± 种子间差异

---

## 12. 交付物清单

```
stalemom/diag.py                    # D1 估计器
stalemom/run_tuned_momentum.py      # D2
llm/                                # 新目录，LLM 部分
  train.py                          # §6 训练循环（含 staleness、TIS、在线记录）
  diag.py                           # §5 嵌套诊断与 CountSketch
  data.py, reward.py, eval.py
  configs/{l0,l1,l2,l3,l4}/*.yaml
tests_e5/                           # 新测试（tests/ 不得改动）
  test_diag_bandit.py               # §3.5 的四条验证
  test_countsketch.py               # sketch 是线性的，内积估计无偏
  test_loss.py                      # k=0 时 w≡1；c=∞ 时 w=比值；零优势组梯度为 0
results/e5/{d1,d2,l0,l1,l2,l3,l4}/  # CSV、图、run_log、env.txt
reports/e5_<phase>.md               # 每个阶段一份报告，所有数字由脚本从 CSV 生成
```

## 13. 执行顺序与停止点

```
D1 → D2 → 【停：用户审查】
→ 开卡 → L0 → 【停：用户确认预算，可先关机】
→ L1 → L2 → 【停：用户审查】
→ L3 → L4 核心 → 【停：用户审查】
→ （可选）L4 扩展
```
