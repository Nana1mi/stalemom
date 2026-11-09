# stalemom

> **Asynchronous Reinforcement Learning: Interaction Between Stale Rollout Truncation Bias and Optimizer Momentum**  
> 研究异步强化学习中，陈旧 rollout 的重要性采样截断偏差（Truncation Bias）与优化器动量记忆（Momentum Memory）的动力学相互作用。

---

## 目录
- [一、核心研究问题与学术背景](#一核心研究问题与学术背景)
- [二、主要研究发现与理论洞见](#二主要研究发现与理论洞见)
- [三、代码库结构说明](#三代码库结构说明)
- [四、环境安装与单元测试](#四环境安装与单元测试)
- [五、核心实验与一键复现指南](#五核心实验与一键复现指南)
- [六、相关理论推导文档](#六相关理论推导文档)

---

## 一、核心研究问题与学术背景

在现代大规模强化学习（如 LLM Reasoning / RLVR 训练）系统中，为了最大化算力集群吞吐，架构广泛采用 **Actor-Learner 解耦的异步架构**。这不可避免地引入了策略陈旧性（Staleness，延迟 $k$ 步），通常依赖重要性采样截断（Truncated Importance Sampling, TIS / PPO/GRPO ratio clipping）来控制方差。

然而，截断权重必然带来**系统性负偏差**。近期学术界（例如针对 ICML 2026 Oral *Do We Need Adam?* 中“RL 中动量普遍有害、余弦相似度接近 0”的主张）对动量有效性展开了广泛讨论。本项目通过 KL 正则化 Softmax Bandit 与严格受控的动力学仿真，深入解答：
1. **在异步与截断存在时，动量究竟是有益还是有害？**
2. **截断偏差是否仅等价于缩减步长？**
3. **截断偏差何时从阻碍收敛的“纯代价”翻转为压制过冲的“阻尼收益”？**

---

## 二、主要研究发现与理论洞见

### 1. 符号翻转与自适应阻尼假说（Adaptive Damping Hypothesis）
在配对实验（$\Delta = \text{TIS} - \text{TIS}_{\text{debiased}}$）中，截断偏差表现出明确的动力学**符号翻转**：
- **安全收敛区（无过冲倾向，如 $\beta \le 0.6, \rho \le 0.25$）**：截断偏差系统性低估优势动作更新幅度，是拖慢收敛的**纯代价（Cost，$\Delta > 0, +11\sigma \sim +15\sigma$）**。
- **动量过冲反弹区（如 $\beta = 0.99, \rho = 10.0$）**：长动量累积严重陈旧的历史方向，使策略在极值点附近剧烈震荡反弹（例如 Exact NPG 在 $t=16 \to 32$ 期间 gap 从 $0.0086$ 反弹至 $0.2099$）。此时，截断偏差以负偏差形式提供了非线性自适应阻尼，彻底压制了反弹，使收敛质量显著反超无偏估计器（**$\Delta = -0.0224 \pm 0.0011, -20\sigma$ 强烈收益**）。
- **动量失谐的“掩盖效应”（Masking Effect）**：在无截断时，长动量必然导致崩溃；但引入截断后，损失被非线性阻尼奇迹般抹平。这解释了为何工业级系统中使用大动量未见崩溃——**截断机制在暗中替动量过冲提供了保护**。

### 2. 步长对照确证（非步长缩减等价性）
将 Exact NPG 在 $\beta=0.99$ 下沿 $\eta \in [0.01, 2.0]$ 进行精细扫描：
- Exact NPG 在调至最优步长（$\eta=0.5$）时的下界为 $A_T = 0.0176$。
- 而同一动量下，含有截断偏差的 $\text{TIS}_{\text{mean}}$ 在 $\eta=0.10 \sim 0.14$ 下可达到 **$0.0062 \sim 0.0081$**。
- 证明截断偏差并非等效于均匀调小学习率，而是具有状态依赖的动力学阻尼特性。

### 3. 临界零点 $\rho^*$ 移动规律
符号翻转的临界边界并非由单一参数 $\rho = \frac{\eta}{1-\beta}$ 孤立决定：
- **基准设定**：$\eta\lambda=0.10, k=16$ 时，双侧显著（$|t|>3$）零点落在 $\rho^* \approx 1.0$ 附近（约为动量理论安全阈值 $1/(4\lambda)=0.25$ 的 4 倍）。
- **梯度噪声影响**：当 Batch Size $B$ 从 128 降至 4 时，采样噪声剧增诱发剧烈游走，零点持续左移（$1.218 \to 1.026 \to 0.408 \to <0.10$）。高噪声下全区间均为负效应。
- **环境多峰与漂移影响**：好动作数 $n_{\text{good}}$ 从 2 增至 5 时，策略漂移加快，零点同样向左推移至 $<0.10$。

### 4. 全维度稳健性（Robustness）
符号翻转在 5 个随机初始化种子、截断阈值 $c \in [1.2, 2.0]$、不同批量 $B$，以及标准策略梯度（Vanilla PG，$\eta=1.0$）下均严格成立，排除偶然性。

---

## 三、代码库结构说明

```
E:\stalemom
│   pyproject.toml                  # pytest 配置与 pythonpath 声明
│   requirements.txt                # 运行依赖 (numpy, matplotlib, pyyaml, pytest)
│   .gitignore                      # 排除大型中间数组与 LaTeX 辅助文件
│   SPEC_E0-E2.md                   # 基础技术与算法规格说明书
│   README.md                       # 本说明文档
├───stalemom/                       # 核心轻量化纯 NumPy 仿真引擎 (float64)
│       bandit.py                   # KL 正则化 Softmax Bandit、环境定义与投影算子
│       estimators.py               # Exact, On-policy, IS, TIS, TIS_Mean, TIS_Debiased
│       optim.py                    # 动量 EMA 缓冲维护与偏差修正
│       sim.py                      # 核心主循环 (支持 NPG 与 Vanilla PG)
│       metrics.py                  # A_T, final_gap, steps_to_90, KL 漂移与截断率指标
│       run.py                      # 统一实验执行与配对差值计算入口
│       run_e0.py                   # E0-R 复现实验脚本
│       reproduce_table_e2.py       # §2.1 核心对比表直接复现脚本
│       run_stepsize_control.py     # 步长精细对照实验脚本 (扫至 eta=2.0)
│       run_zero_crossing.py        # 零点定位与双侧 |t|>3 显著性筛选脚本
│       run_noise_dimension.py      # 噪声 (B) 与难度 (n_good) 维度扫描脚本
│       run_robustness.py           # 5 维度稳健性压力测试脚本
├───plots/                          # 绘图脚本
│       plot_e0.py                  # E0-R 曲线绘制
│       plot_e1.py                  # E1 热力图与增益区域绘制
│       plot_e2.py                  # E2 配对效应图绘制
│       plot_stepsize_control.py    # 步长对照图
│       plot_overshoot_trajectories.py # 对数轴动量过冲反弹与截断压制轨迹图
│       plot_zero_crossing.py       # 零点定位曲线图
│       plot_noise_dimension.py     # 噪声与难度对零点移动的影响图
│       plot_robustness.py          # 稳健性多维度对比柱状图
├───results/                        # 实验汇总 CSV、配对数据与图表
│   ├───e0_repro/                   # E0-R 数据与图
│   ├───e1/                         # E1 扩展网格汇总
│   ├───e2/                         # E2 实验数据与 summary.csv
│   ├───stepsize_control/           # 步长实验数据与过冲反弹轨迹图
│   ├───zero_crossing/              # 零点定位数据与零点估算表
│   ├───noise_dimension/            # 噪声扫描数据与零点漂移表
│   └───robustness/                 # 稳健性检验汇总数据与图表
├───docs/                           # 文档与理论手册
│       plan.md                     # 项目初始研究规划
│       theory_derivation.tex       # 理论推导手册 LaTeX 源文件
│       theory_derivation.pdf       # 编译生成的完整理论推导手册
└───tests/                          # 独立测试套件 (98 项测试，严格未受改动)
        test_bandit.py
        test_estimators.py
        test_optim.py
        test_sim.py
        test_impl_e0.py
```

---

## 四、环境安装与单元测试

### 1. 安装依赖
本项目基于纯 CPU NumPy 实现（无 PyTorch/JAX 外部依赖），需 Python $\ge 3.10$：
```powershell
pip install -r requirements.txt
```

### 2. 运行测试套件
```powershell
python -m pytest
```
预期输出：`98 passed in <1.0s`。

---

## 五、核心实验与一键复现指南

全部核心实验均支持多进程并行加速，并输出标准 CSV 与出版级图表：

### 1. 打印 §2.1 核心对比表（精确对齐至 5 位小数）
```powershell
python -m stalemom.reproduce_table_e2
```

### 2. 动量过冲反弹与截断压制轨迹图（对数轴）
生成展示 $t=16 \to 32$ 期间 exact/IS 反弹以及 TIS 平滑压制的高清对比图：
```powershell
python -m plots.plot_overshoot_trajectories
```
产出文件：`results/stepsize_control/gap_trajectory_overshoot.png`。

### 3. 步长精细对照实验（$\eta \in [0.01, 2.0]$）
```powershell
python -m stalemom.run_stepsize_control --R 200
python -m plots.plot_stepsize_control
```
产出文件：`results/stepsize_control/stepsize_summary.csv` 与 `stepsize_control.png`。

### 4. 符号翻转零点定位与显著性筛选
```powershell
python -m stalemom.run_zero_crossing --R 200
python -m plots.plot_zero_crossing
```
产出文件：`results/zero_crossing/zero_crossing_points.csv` 与 `zero_crossing_curves.png`。

### 5. 噪声（$B$）与环境难度（$n_{\text{good}}$）维度扫描
```powershell
python -m stalemom.run_noise_dimension --R 200
python -m plots.plot_noise_dimension
```
产出文件：`results/noise_dimension/noise_and_difficulty_shift.png`。

### 6. 全维度稳健性压力测试
```powershell
python -m stalemom.run_robustness --R 200
python -m plots.plot_robustness
```
产出文件：`results/robustness/robustness_bars.png`。

---

## 六、相关理论推导文档

关于连续时间极限动力学、二阶常微分方程阻尼项推导、Fisher 信息矩阵与截断负偏差数学证明，详见手册：
- **LaTeX 源码**：[`docs/theory_derivation.tex`](file:///E:/stalemom/docs/theory_derivation.tex)
- **PDF 手册**：[`docs/theory_derivation.pdf`](file:///E:/stalemom/docs/theory_derivation.pdf)
