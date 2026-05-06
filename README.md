# 贪吃蛇 Q-Learning 强化学习项目

## 项目简介

本项目用**表格 Q-Learning** 训练一个能玩简化版贪吃蛇（Snake）游戏的强化学习智能体。环境为 6×6 的方形棋盘，智能体通过与环境反复交互，逐步学会避开墙壁/自身并引导蛇头朝向食物。

---

## 环境说明

| 属性 | 值 |
|---|---|
| 棋盘大小 | 6×6（默认，可设 8×8） |
| 状态空间 | 标准: 11 位二进制 → **2048** 个状态；增强: 15 位 → **32768** 个状态 |
| 动作空间 | 3 个相对动作：直行 / 左转 / 右转 |
| 奖励 | 吃到食物 **+10**，死亡 **-10**，普通每步 **-0.1** |

**标准状态编码（11 位）**：

```
[危险_直行, 危险_左转, 危险_右转,  方向_上, 方向_下, 方向_左, 方向_右,  食物_上, 食物_下, 食物_左, 食物_右]
```

**增强状态编码（15 位，`--enhanced-state`）**：在上述 11 位基础上追加 4 位：

```
[d2_直行→直行, d2_左转→直行, d2_右转→直行,  蛇身_长标志]
```
前三位为**两步前方碰壁**预判，最后一位表示蛇身是否超过 `grid_size`（长蛇阶段）。

---

## 算法说明

### 核心方法：Q-Learning（离策略 TD 控制）

Q 值更新公式：

```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

### 消融对比变种

| 算法 | 类型 | 说明 |
|---|---|---|
| Q-Learning | 离策略 | 本项目主方法 |
| SARSA | 在策略 | 使用实际执行的下一动作更新 |
| Expected SARSA | 在策略 | 用行为策略期望值替代采样动作 |
| Double Q-Learning | 离策略 | 双表减小 max 操作的过估计偏差 |

### 探索策略

| 策略 | 说明 |
|---|---|
| ε-greedy | 主方法，ε 从 1.0 按 0.995 衰减至 0.01 |
| Softmax | 按 Q 值的 Boltzmann 分布采样 |
| UCB | 对访问次数少的动作给乐观奖励加成 |

### 奖励塑形（可选）

- **simple shaping**：移近食物给小正奖励，移远给小负惩罚
- **potential-based shaping**：以到食物的 Manhattan 距离的负值为势函数，理论更严谨

### 新增功能

#### TD(λ) / 资格迹（`--lambda`）

使用累积资格迹（accumulating eligibility traces）进行 Q(λ) 更新：

```
E(s,a) ← E(s,a) + 1        # 当前 (s,a) 的 trace 加 1
Q      ← Q + α·δ·E          # 广播到全表
E      ← E · γλ             # 衰减所有 trace
```

- `λ=0` 等价于标准 1-step Q-learning（默认）
- `λ→1` 趋近于 Monte-Carlo 回报
- 可讨论 bias-variance tradeoff

#### 自适应学习率（`--adaptive-lr`）

对每个 (s,a) 单独计算学习率，满足 Robbins-Monro 条件：

```
α(s,a) = 1 / (1 + visit_count(s,a))
```

#### 乐观初始值（`--optimistic-initial-value`）

将 Q 表初始化为正值（如 1.0、5.0）以鼓励早期探索。消融实验中已加入 `oiv=1` 和 `oiv=5` 对比。

#### 基准线体系（三档对比）

| 基准 | 说明 |
|---|---|
| 随机基准 | 均匀随机选择 3 个动作 |
| **规则基准**（新增） | 无碰撞时朝食物方向行进，否则取安全随机动作 |
| Q-Learning 智能体 | 10000 回合训练后的贪心策略 |

三档对比更能体现 RL 的价值：随机 < 规则 < Q-Learning。

#### 8×8 棋盘泛化（`--grid-size 8`）

使用同一 11 位特征在更大棋盘上验证表格法的泛化能力，
暴露其局限性（状态覆盖率下降），为引入函数近似（DQN）提供动机。

#### 多种子 + 置信区间（`--num-seeds`）

消融脚本支持对每个变种运行多个独立种子，在 CSV 汇总中输出 mean ± std，报告中可绘制误差棒。

---

## 主要实验结果

### 核心结果（Q-Learning + ε-greedy，10000 回合训练后，贪心评估 300 回合）

| 指标 | 训练智能体（贪心） | 随机基准 |
|---|---|---|
| 平均得分 | **13.73** | 0.19 |
| 最高得分 | **27** | 3 |
| 平均总奖励 | **121.67** | -8.89 |
| 平均存活步数 | **70.0** | 8.7 |

> 训练后的 Q-Learning 智能体得分是随机基准的约 **73 倍**。

### 消融实验汇总（所有变种，10000 回合，贪心评估 300 回合）

| 变种 | 算法 | 探索策略 | 奖励塑形 | 平均分 | 最高分 |
|---|---|---|---|---|---|
| q_learning_epsilon（主） | Q-Learning | ε-greedy | 无 | 13.73 | 27 |
| q_learning_slow_decay（最优） | Q-Learning | ε-greedy（慢衰减） | 无 | **14.79** | **30** |
| q_learning_potential_shaping | Q-Learning | ε-greedy | potential | 13.80 | 27 |
| double_q_learning_epsilon | Double Q | ε-greedy | 无 | 13.98 | 25 |
| expected_sarsa_epsilon | Expected SARSA | ε-greedy | 无 | 13.45 | 30 |
| q_learning_ucb | Q-Learning | UCB | 无 | 13.42 | 26 |
| q_learning_softmax | Q-Learning | Softmax | 无 | 13.37 | 29 |
| sarsa_epsilon | SARSA | ε-greedy | 无 | 13.33 | 24 |
| q_learning_simple_shaping | Q-Learning | ε-greedy | simple | 12.99 | 27 |
| q_learning_low_initial_epsilon | Q-Learning | ε-greedy（低初始ε） | 无 | 12.79 | 28 |

**结论**：ε 慢衰减版本得分最高（均分 14.79，最高 30）；各算法整体差距不大，主要体现在探索充分性和方差上。

---

## 文件结构

```
snake/
├── snake_env.py              # 贪吃蛇游戏环境（状态编码、奖励、step 接口）
├── q_agent.py                # 强化学习智能体（Q-Learning / SARSA / Double Q 等）
├── train_part2.py            # 主训练脚本（含评估、日志、保存）
├── run_part2_ablation.py     # 一键运行所有消融实验
├── plot_part2_results.py     # 绘制学习曲线
├── sanity_check_part2.py     # 快速健全性检查
├── requirements.txt          # 依赖（numpy、matplotlib）
├── results/
│   ├── main_q_learning/      # 主实验结果（Q-table、训练历史、评估摘要、曲线图）
│   ├── part2_ablation/       # 消融实验结果汇总（ablation_summary.csv）
│   └── figures/              # 对比图（基准对比、学习曲线、ε衰减对比等）
└── Evaluation/               # 独立评估脚本（evaluate_q_learning.py 等）
```

每次训练产生的输出文件：

| 文件 | 内容 |
|---|---|
| `training_history.csv` | 每回合得分、奖励、存活步数、ε 值、TD 误差、`q_std` 等 |
| `summary.txt / summary.json` | 最终评估摘要（含随机/规则/训练三档基准对比） |
| `agent_config.json` | 超参数，用于复现 |
| `q_table.npy` | 学到的 Q 值表 |
| `state_visit_counts.npy` | 各状态访问次数 |

---

## 快速使用

```bash
# 安装依赖
pip install -r requirements.txt

# 健全性检查（含新功能测试）
python sanity_check_part2.py

# 主实验（10000 回合 Q-Learning）
python train_part2.py \
  --episodes 10000 \
  --algorithm q_learning \
  --exploration epsilon_greedy \
  --output-dir results/main_q_learning

# 增强状态 + TD(λ=0.5) + 自适应学习率
python train_part2.py \
  --episodes 10000 \
  --enhanced-state \
  --lambda 0.5 \
  --adaptive-lr \
  --output-dir results/q_learning_enhanced

# 8×8 棋盘泛化实验
python train_part2.py \
  --episodes 10000 \
  --grid-size 8 \
  --output-dir results/q_learning_8x8

# 一键消融实验（所有变种，含新增变种）
python run_part2_ablation.py --episodes 10000 --eval-episodes 300

# 多种子消融（每变种 3 个独立种子，输出 mean±std）
python run_part2_ablation.py --episodes 10000 --eval-episodes 300 --num-seeds 3

# 绘制学习曲线 + 分析图（状态访问分布、策略动作分布、Q值收敛）
python plot_part2_results.py --result-dir results/main_q_learning
```

---

## 超参数（主实验）

| 参数 | 值 |
|---|---|
| 学习率 α | 0.1（自适应模式下被 1/(1+n) 替代） |
| 折扣因子 γ | 0.9 |
| 初始 ε | 1.0 |
| ε 衰减 | 0.995（每回合） |
| 最小 ε | 0.01 |
| λ（资格迹衰减） | 0.0（默认，可设 0.5/0.9） |
| 训练回合数 | 10000 |
| 评估回合数 | 300（贪心） |
| 随机种子 | 42 |
