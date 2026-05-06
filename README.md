# 贪吃蛇 Q-Learning 强化学习项目

## 项目简介

本项目用**表格 Q-Learning** 训练一个能玩简化版贪吃蛇（Snake）游戏的强化学习智能体。环境为 6×6 的方形棋盘，智能体通过与环境反复交互，逐步学会避开墙壁/自身并引导蛇头朝向食物。

---

## 环境说明

| 属性 | 值 |
|---|---|
| 棋盘大小 | 6×6（默认） |
| 状态空间 | 11 维二进制特征 → 2048 个离散状态 |
| 动作空间 | 3 个相对动作：直行 / 左转 / 右转 |
| 奖励 | 吃到食物 **+10**，死亡 **-10**，普通每步 **-0.1** |

**状态编码（11 位）**：

```
[危险_直行, 危险_左转, 危险_右转,  方向_上, 方向_下, 方向_左, 方向_右,  食物_上, 食物_下, 食物_左, 食物_右]
```

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
| `training_history.csv` | 每回合得分、奖励、存活步数、ε 值、TD 误差等 |
| `summary.txt / summary.json` | 最终评估摘要（可读 / 机器可读） |
| `agent_config.json` | 超参数，用于复现 |
| `q_table.npy` | 学到的 Q 值表 |
| `state_visit_counts.npy` | 各状态访问次数 |

---

## 快速使用

```bash
# 安装依赖
pip install -r requirements.txt

# 健全性检查
python sanity_check_part2.py

# 主实验（10000 回合 Q-Learning）
python train_part2.py \
  --episodes 10000 \
  --algorithm q_learning \
  --exploration epsilon_greedy \
  --output-dir results/main_q_learning

# 一键消融实验（生成 ablation_summary.csv）
python run_part2_ablation.py --episodes 10000 --eval-episodes 300

# 绘制学习曲线
python plot_part2_results.py
```

---

## 超参数（主实验）

| 参数 | 值 |
|---|---|
| 学习率 α | 0.1 |
| 折扣因子 γ | 0.9 |
| 初始 ε | 1.0 |
| ε 衰减 | 0.995（每回合） |
| 最小 ε | 0.01 |
| 训练回合数 | 10000 |
| 评估回合数 | 300（贪心） |
| 随机种子 | 42 |
