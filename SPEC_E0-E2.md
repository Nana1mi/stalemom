# 实现规格说明：陈旧 Rollout × 截断 × 动量（E0–E2，bandit 部分）

> 给编码助手的说明：请严格按本文档实现。**所有公式、接口、默认值都是确定的，不要自行替换成"等价"或"更常见"的写法。** 遇到本文档没写清楚的地方，停下来列出问题，不要自己猜。

---

## 0. 背景（一段话）

研究异步强化学习中：rollout 由旧策略生成（陈旧），对重要性权重做截断会产生偏差；这种偏差与优化器的长记忆动量（EMA）如何相互作用，长动量何时有益、何时有害。本阶段只在一个**可以精确计算**的 KL 正则 softmax bandit 上做实验，不涉及神经网络和 GPU。

核心建模原则（必须始终满足）：**数据来自旧策略 q，梯度在当前参数 θ_t 处计算。** 不是"旧参数处算出的梯度晚到"。

---

## 1. 硬性约束

1. **纯 numpy，float64，CPU。** 不要用 PyTorch / JAX / 自动微分 / GPU。所有梯度都用下文的闭式公式。
2. **按随机种子维度向量化。** 状态张量形状为 `(R, K)`，R = 并行运行数（种子数），K = 动作数。不要写 Python 层面的逐种子循环。
3. **不得修改 `tests/` 目录下由用户提供的测试文件。** 你可以新增自己的测试文件，命名为 `tests/test_impl_*.py`。测试失败时修实现，不许放宽断言或改容差。
4. **不得丢弃发散或失败的运行。** 必须记录下来并在汇总里体现（见 §6.4）。
5. **不得悄悄改网格或默认值。** 如果某个配置非法（例如 β < 0），跳过并在日志里列出被跳过的配置。
6. 完整 IS 路径**不得做批内权重归一化**（不得除以权重之和）。
7. 每个实验输出目录里保存：完整配置副本、git commit hash、numpy 版本、随机种子、原始指标数组。

---

## 2. 数学定义（实现必须与此逐字一致）

### 2.1 问题

- 动作数 K；参数 θ ∈ R^K；策略 p_θ = softmax(θ)。
- 奖励向量 r ∈ {0,1}^K，已知、确定。
- 参考分布 p_ref，全支持。**默认 p_ref = p_{θ_0}（初始策略）。**
- KL 系数 λ > 0。
- 目标：

  J(θ) = Σ_a p_θ(a) r_a − λ · KL(p_θ ‖ p_ref)，其中 KL(p‖q) = Σ_a p(a) log(p(a)/q(a))

- 最优值（闭式）：J* = λ · log Σ_a p_ref(a) · exp(r_a / λ)
- 投影矩阵 Π = I − (1/K)·11ᵀ（对向量 v：Π v = v − mean(v)）。

实现要求：log p 用稳定的 log-softmax 计算；θ 每步更新后保持 sum(θ) 不变（下面的方向都已投影，自然保持）。

### 2.2 精确方向

- **精确自然策略梯度（NPG）方向**：

  h(θ) = Π [ r − λ (log p_θ − log p_ref) ]

- **普通梯度**（用于诊断量）：

  ∇J(θ) = F(θ) [ r − λ (log p_θ − log p_ref) ]，F(θ) = diag(p) − p pᵀ

不要用伪逆数值求 NPG，直接用上面的闭式。

### 2.3 采样与估计器

每一步 t：从行为策略 q_t 采样 B 个动作，得到每个动作的计数 `counts ∈ N^K`（`counts ~ Multinomial(B, q_t)`，每个运行独立）。

**奖励项**的估计（KL 项一律按精确值计算，不采样）：

| 模式 `estimator` | 奖励项估计 ρ̂（形状 K） | 是否采样 |
|---|---|---|
| `exact` | r | 否 |
| `onpolicy` | (counts/B) ⊙ r / p_t，其中 counts 从 **p_t** 采样（强制 k=0） | 是 |
| `is` | (counts/B) ⊙ r / q_t | 是 |
| `tis` | (counts/B) ⊙ min(p_t/q_t, c) ⊙ r / p_t | 是 |
| `tis_mean` | r ⊙ min(1, c·q_t/p_t) | 否（精确期望） |
| `tis_debiased` | tis 的估计 − (r ⊙ (min(1, c·q_t/p_t) − 1)) | 是 |

（⊙、/、min 都是逐元素运算。）

随机方向统一为：

  ĝ_t = Π [ ρ̂_t − λ (log p_t − log p_ref) ]

**截断偏差（闭式，用于 `tis_debiased` 和诊断）**：

  b_t = Π [ r ⊙ ( min(1, c·q_t/p_t) − 1 ) ]

性质（测试会检查）：E_{counts~q}[ĝ_t^{tis}] = h(θ_t) + b_t；E[ĝ_t^{is}] = h(θ_t)。

### 2.4 陈旧（行为策略）

- `mismatch = "lag"`（主设置）：q_t = softmax(θ_{max(t−k, 0)})。用环形缓冲区保存最近 k+1 步的 θ。记录实际滞后 `lag_t = t − max(t−k, 0)`。
- `mismatch = "fixed"`（补充对照，**优先级最低，E0–E1 完成后再做**）：q_t = softmax(θ_t + s·u)，u = Π z / ‖Π z‖，z ~ N(0, I_K)，每个运行一个固定的 u；标量 s 通过二分法校准，使整条轨迹上 KL(p_t‖q_t) 的时间平均值等于对应 `lag` 运行的时间平均值。

### 2.5 动量（带偏差修正的 EMA）

为避免 β 接近 1 时的数值抵消，**内部一律存储 `omb = 1 − β`，不要存 β。**

- m_{−1} = 0
- m_t = (1 − omb)·m_{t−1} + omb·ĝ_t
- 偏差修正分母：den_t = 1 − (1−omb)^{t+1}，必须用 `-np.expm1((t+1) * np.log1p(-omb))` 计算
- m̂_t = m_t / den_t（omb = 1 时 m̂_t = ĝ_t）
- 更新（最大化）：θ_{t+1} = θ_t + η · m̂_t

动量的两种指定方式：

- 按 ρ：`omb = η / ρ`（ρ = 0 表示无动量，omb = 1）。要求 ρ ≥ η，否则跳过该配置。
- 按固定 β：`beta_fixed ∈ {0.9, 1 − 1e-6}`，直接给出 omb = 0.1 或 1e-6。

**加权历史年龄**（诊断量）：权重 α_{t,j} = omb·(1−omb)^j / den_t，j = 0..t；H_t = Σ_j j·α_{t,j}。

### 2.6 动量误差分解（诊断量，必须实现）

令 h_t = h(θ_t)，ξ_t = ĝ_t − h_t − b_t（对 `is` 模式 b_t 取 0）。用与 m 完全相同的 EMA 递推和偏差修正，额外维护三条 EMA：

- E^h_t：输入 h_t
- E^b_t：输入 b_t
- E^ξ_t：输入 ξ_t

由线性性，m̂_t = Ê^h_t + Ê^b_t + Ê^ξ_t（测试会检查）。记录：

- 跟踪误差 `track = Ê^h_t − h_t`
- 历史偏差 `bias_hist = Ê^b_t`
- 平滑噪声 `noise = Ê^ξ_t`

---

## 3. 代码结构与接口

```
stalemom/
  __init__.py
  bandit.py       # 问题定义与精确量
  estimators.py   # 采样、估计器、闭式偏差
  optim.py        # EMA 动量、omb 计算、历史年龄
  sim.py          # 向量化主循环
  metrics.py      # 指标与汇总
  run.py          # CLI：读取配置、展开网格、保存结果
plots/
  plot_e0.py  plot_e1.py  plot_e2.py
configs/
  e0_repro.yaml  e1_grid.yaml  e2_paths.yaml
tests/            # 用户提供，禁止修改；你的测试放 tests/test_impl_*.py
results/          # 输出
README.md
```

以下函数签名必须**完全一致**（用户会用独立测试直接导入它们）。所有数组 float64；`(R,K)` 表示批量，K 维向量可广播。

```python
# bandit.py
def log_softmax(theta: np.ndarray) -> np.ndarray: ...            # (R,K) -> (R,K)
def softmax(theta: np.ndarray) -> np.ndarray: ...                # (R,K) -> (R,K)
def project(v: np.ndarray) -> np.ndarray: ...                    # Π v，沿最后一维
def objective(theta, r, logp_ref, lam) -> np.ndarray: ...        # -> (R,)
def optimal_value(r, logp_ref, lam) -> float: ...                # J*
def exact_npg_dir(theta, r, logp_ref, lam) -> np.ndarray: ...    # h(θ), (R,K)
def vanilla_grad(theta, r, logp_ref, lam) -> np.ndarray: ...     # ∇J(θ), (R,K)

# estimators.py
def sample_counts(rng, probs: np.ndarray, B: int) -> np.ndarray: ...   # (R,K) 整数计数，每行和为 B
def reward_term(mode: str, counts, p, q, r, c, B) -> np.ndarray: ...   # 表 2.3 的 ρ̂，(R,K)
def expected_reward_term(mode: str, p, q, r, c) -> np.ndarray: ...     # 'is'/'onpolicy' 返回 r；'tis' 返回 r*min(1,c*q/p)
def tis_bias(p, q, r, c) -> np.ndarray: ...                            # b，(R,K)
def direction(reward_est, logp, logp_ref, lam) -> np.ndarray: ...      # Π[ρ̂ − λ(log p − log p_ref)]
def truncation_rate(counts, p, q, c, B) -> np.ndarray: ...             # 被截断样本占比，(R,)

# optim.py
def omb_from_rho(eta: float, rho: float) -> float: ...   # rho=0 -> 1.0；rho<eta -> raise ValueError
def bias_correction_den(omb: float, t: int) -> float: ... # 1-(1-omb)^(t+1)，用 expm1/log1p
def ema_step(m, g, omb) -> np.ndarray: ...
def history_age(omb: float, t: int) -> float: ...        # H_t

# sim.py
def simulate(cfg: dict) -> dict: ...   # 返回 §6 中的全部数组
```

补充约定：

- `reward_term` 的 `mode` 取值为 `onpolicy`、`is`、`tis`、`tis_debiased`；`expected_reward_term` 对 `onpolicy`、`is`、`tis_debiased` 返回 r，对 `tis` 返回 r⊙min(1, c·q/p)。
- `r`、`logp_ref` 可以是形状 `(K,)`，与 `(R,K)` 的数组广播。
- `p/q > c` 时才算截断（等于 c 不算）。

### 3.1 `simulate` 的输入与输出（单个已展开配置）

输入：

```python
cfg = {
  "problem": {"K": 10, "n_good": 2, "lam": 1.0, "init": "random", "init_seed": 0, "init_scale": 1.0, "ref": "init"},
  "run": {"R": 20, "seed": 12345, "T": 512, "B": 32, "diag_every": 8, "return_traj": False},
  "estimator": "tis",     # exact | onpolicy | is | tis | tis_mean | tis_debiased
  "k": 8,                 # 滞后；exact / onpolicy 忽略（视为 0）
  "eta": 0.01,            # 实际学习率 η（不是 eta_lam）
  "omb": 0.04,            # 1 − β，已由 run.py 计算好
  "c": 2.0,               # 截断阈值；is / exact / onpolicy 可为 None
  "mismatch": "lag",      # lag | fixed
}
```

- 奖励：r = [1]*n_good + [0]*(K − n_good)。
- `init = "uniform"` 时 θ_0 = 0；`random` 时 θ_0 = init_scale·Π z，z 由 `init_seed` 生成，**同一配置下所有 R 个运行的 θ_0 相同**。
- `ref = "init"` 时 log p_ref = log_softmax(θ_0)。
- `exact`、`tis_mean` 强制 R = 1。

输出字典（在 §6.1 基础上明确形状）：

| 键 | 形状 | 含义 |
|---|---|---|
| `gap`、`J` | (R, T+1) | t = 0..T |
| `kl_pq`、`trunc_rate` | (R, T) | 第 t 次更新（t = 0..T−1）所用的 q_t；`exact`/`onpolicy` 时 kl 为 0 |
| `lag` | (T,) | lag_t = t − max(t−k, 0) |
| `diag_t` | (n_diag,) | 诊断步 = `np.arange(0, T, diag_every)` |
| `bias_norm`、`inner_update`、`inner_track`、`inner_bias_hist`、`inner_noise`、`norm_track`、`norm_bias_hist`、`norm_noise` | (R, n_diag) | 在第 t 次更新时、用更新前的 θ_t 和该步的 m̂_t 计算 |
| `H` | (n_diag,) | 加权历史年龄 |
| `diverged`、`diverged_step` | (R,) | |
| `J_star` | 标量 | |
| `r`、`logp_ref` | (K,) | |

`return_traj = True` 时额外返回：`theta` (R, T+1, K)、`g` (R, T, K)（即 ĝ_t）、`m_hat` (R, T, K)。

恒等式：m̂_t = h(θ_t) + track_t + bias_hist_t + noise_t，因此 `inner_update` = ⟨∇J(θ_t), h(θ_t)⟩ + `inner_track` + `inner_bias_hist` + `inner_noise`。

分解中的 b_t 定义为**该估计器自身的均值偏差** E[ĝ_t] − h(θ_t)：`tis`、`tis_mean` 取 §2.3 的闭式 b_t；`exact`、`onpolicy`、`is`、`tis_debiased` 取 0（它们都是无偏的）。`bias_norm` 记录的就是这个 b_t 的范数。`exact`、`tis_mean` 的 ξ_t ≡ 0。

---

## 4. 配置字段

```yaml
name: e1_grid
problem:
  K: 10
  n_good: 2              # 前 n_good 个动作奖励为 1，其余为 0
  lam: 1.0
  init: random           # random | uniform
  init_seed: 0           # 固定问题实例
  init_scale: 1.0        # random 时 θ0 = init_scale * Π z，z~N(0,I)
  ref: init              # 参考分布 = 初始策略
run:
  R: 20                  # 种子数（并行运行数）
  seed: 12345
  T: 512
  B: 32
  log_every: 1           # 标量指标每步记录；向量诊断每 diag_every 步记录
  diag_every: 8
sweep:                   # 笛卡尔积展开，非法组合跳过并记录
  estimator: [is, tis]
  k: [0, 2, 8, 16]
  eta_lam: [0.001, 0.003, 0.01, 0.03, 0.1]   # 实际 η = eta_lam / lam
  momentum:              # 二选一：rho_lam 或 beta_fixed
    - {rho_lam: 0.0}
    - {rho_lam: 0.05}
    - {rho_lam: 0.1}
    - {rho_lam: 0.25}
    - {beta_fixed: 0.9}
    - {beta_fixed: 0.999999}
  c: [1.2, 1.5, 2.0]     # 仅对 tis 系列有效；is 只跑一次
  mismatch: [lag]
```

确定性模式（`exact`、`tis_mean`）自动令 R = 1。

---

## 5. 实验清单（按此顺序做，前一步通过再做下一步）

### E0 正确性（必须全部通过才能进入 E1）

在 `tests/test_impl_e0.py` 中实现下列检查（用户还会另外提供一套独立测试）：

1. **最优值**：在 θ* = log p_ref + r/λ（再投影）处，objective 等于 optimal_value，误差 < 1e-12；任意 θ 处 objective ≤ J*。
2. **精确方向**：h(θ*) ≈ 0；vanilla_grad 与对 objective 做中心差分的结果一致（相对误差 < 1e-6）。
3. **IS 无偏**：固定 p、q（随机生成，全支持），对所有可能的单样本动作枚举求期望：E[ĝ^{is}] = h，atol=1e-10、rtol=1e-8。
4. **TIS 偏差闭式**：同上枚举，E[ĝ^{tis}] − h = tis_bias，同样容差。
5. **蒙特卡洛一致性**：用 10⁶ 次 B=1 采样估计 E[ĝ^{tis}]，与 h + b 的差在 5 倍标准误以内。
6. **同步退化**：q = p 且 c ≥ 1 时，is 与 tis 的估计逐元素相等；b = 0。
7. **动量退化**：omb = 1 时 m̂_t = ĝ_t；omb = 1e-6、t ≤ 1000、ĝ 为 O(1) 量级时，m̂_t 与 ĝ_0..ĝ_t 的等权平均的绝对误差 < 1e-4（不要用相对误差：等权平均接近 0 时相对误差没有意义）。
8. **偏差修正权重**：Σ_j α_{t,j} = 1（误差 < 1e-12）；history_age 在 omb = 1 时为 0。
9. **分解恒等式**：m̂_t = Ê^h + Ê^b + Ê^ξ，逐元素误差 < 1e-10。
10. **滞后记录**：k = 3 时，t = 0..2 的 lag_t = t，t ≥ 3 的 lag_t = 3；q_t 确实等于 softmax(θ_{t−3})。
11. **投影不变**：整个训练过程中 sum(θ_t) 保持为 sum(θ_0)（误差 < 1e-9）。

**同步复现（E0-R）**：复现 Wang et al. (2026) "When Does Momentum Help in RLVR?" §6.1 的趋势（不要求数值一致）：

- K = 10，3 个奖励为 1 的动作，λ = 1，B = 1，p_ref = 初始策略，`estimator = onpolicy`；
- 令 τ = ηT = 1，T ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}，η = 1/T；
- 三组：无动量（omb=1）、固定 β = 0.9、按比例 β = 1 − 4η（即 ρ = 0.25）；
- 每组至少 20000 次独立重复；均匀初始化和随机初始化各一套；
- 统计：J(θ_T^β) − J(θ_T^0) 的均值、标准差（**不同组使用独立样本**）、SNR、β 组胜出的概率。

应看到的趋势：标准差约按 √η 缩小；固定 β 的平均收益约按 η 缩小、胜率趋近 0.5；按比例 β 的平均收益趋近正常数、胜率趋近 1。**如果趋势不符，先报告，不要调参去"凑"。**

### E1 动量收益区域

按 `configs/e1_grid.yaml` 运行（§4 的网格）。种子数 R = 20。

### E2 精确去偏因果对照

对 E1 中选出的若干配置（由用户指定，先用默认：k ∈ {8, 16}，c = 主阈值，ηλ ∈ {0.01, 0.03}，动量取 ρλ ∈ {0, 0.25} 和 β = 1−1e-6），分别运行下列路径，R = 100：

`exact`、`is`、`tis`、`tis_mean`、`tis_debiased`

它们是**相互独立的训练分支**（同一初始化，不同估计器）。`mismatch: fixed` 的对照在其余部分完成后再做。

---

## 6. 输出

### 6.1 每次运行（每个配置）保存为一个 `.npz`

标量（形状见 §3.1）：
- `gap`：归一化目标差距 (J* − J(θ_t)) / (J* − J(θ_0))
- `J`：目标值
- `kl_pq`：KL(p_t ‖ q_t)
- `trunc_rate`：截断率（非 tis 模式为 0）
- `lag`：实际滞后

诊断（每 `diag_every` 步，形状 `(R, n_diag)`）：
- `bias_norm`：‖b_t‖
- `inner_update`：⟨∇J(θ_t), m̂_t⟩
- `inner_track`、`inner_bias_hist`、`inner_noise`：∇J(θ_t) 分别与三个分解项的内积
- `norm_track`、`norm_bias_hist`、`norm_noise`
- `H`：加权历史年龄（形状 `(n_diag,)`）

状态：
- `diverged`：bool `(R,)`；`diverged_step`：int `(R,)`，未发散为 −1

### 6.2 汇总表 `summary.csv`（每个配置一行）

配置的全部字段，加上：
- `A_T`：gap 在 t = 1..T 上的平均，再对 R 个运行取均值；同时给出标准误
- `final_gap` 均值与标准误
- `steps_to_90`：首次 gap ≤ 0.1 的步数的中位数；未达到的运行计为 ∞，并另列 `frac_reached_90`
- `mean_trunc_rate`、`mean_kl_pq`、`diverge_rate`
- `skipped`：该配置是否因非法被跳过及原因

### 6.3 数值发散

任何运行出现非有限值时：标记 `diverged`，记录步数，冻结该运行的 θ 不再更新；其后的 gap 记为 NaN。汇总中 A_T 和 final_gap 给出两个版本：(a) 仅未发散运行；(b) 发散运行在发散后 gap 按 1.0 计入（悲观版本）。两者都要报告。

### 6.4 日志

每个实验目录写一个 `run_log.txt`：配置数、跳过的配置及原因、发散的运行数、总耗时。

---

## 7. 图（plots/）

- `plot_e0.py`：E0-R 的四联图（均值收益 vs η、标准差 vs η、SNR vs η、胜率 vs η，双对数坐标，三组动量各一条线）。
- `plot_e1.py`：
  1. **截断率热图**：横轴 ηλ，纵轴 k，每个 c 一张；
  2. **收益区域**：对每个 (estimator, c, k)，画"每种动量设置在其最佳学习率下的 A_T"（每种动量的学习率搜索预算相同），误差棒为标准误；
  3. 选定配置下 gap 随 t 的曲线（中位数 + 四分位带）。
- `plot_e2.py`：
  1. 五条路径的 gap 随 t 曲线（同一张图，按动量设置分面）；
  2. 分解项内积 `inner_track / inner_bias_hist / inner_noise` 随 t 的曲线。

所有图从 `.npz` 和 `summary.csv` 重新生成，不在仿真代码里画图。

---

## 8. 运行方式与服务器

- 命令：`python -m stalemom.run --config configs/e1_grid.yaml --out results/e1/`
- 支持 `--dry-run`：只打印展开后的配置数和被跳过的配置，不运行。
- 支持用 `multiprocessing` 在配置维度并行（默认进程数 = CPU 核数）；每个配置内部已经按种子向量化。
- 在服务器上用 `tmux` 运行；运行结束后打印汇总并退出；**不要使用 GPU**。
- 先在本地用 `R=2, T=32` 的小配置跑通全流程，再上完整网格。

---

## 9. 交付顺序与验收

1. 实现 `bandit.py`、`estimators.py`、`optim.py`，通过 §5 E0 的第 1–11 项检查。**把测试输出贴给用户确认后再继续。**
2. 实现 `sim.py`、`run.py`，跑 E0-R，交 `plot_e0.py` 的图和文字说明（趋势是否与原论文一致）。
3. 跑 E1，交截断率热图、收益区域图、`summary.csv`。
4. 跑 E2，交两张图。

每一步结束时报告：做了什么、测试结果原文、任何异常或与本规格不一致的地方。**不要只给结论性的总结，要附上原始数字。**

---

## 10. 禁止事项清单

- 不用自动微分、不用伪逆求 NPG、不用 GPU
- 不修改 `tests/` 中用户提供的测试，不放宽容差
- 完整 IS 不做权重归一化
- 不丢弃发散运行，不只报告"成功"的运行
- 不悄悄改网格、默认值、指标定义
- 不存 β，只存 omb = 1 − β
- 不在趋势不符时调参去"凑"结果
