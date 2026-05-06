# Enhanced Part 2 — Q-Learning Agent for Simplified Snake

This folder contains a stronger Part 2 implementation for the RL final project topic **Game Playing with Q-Learning**.

The official project requirement asks Track A teams to submit a runnable, well-commented Python implementation with clear reproduction instructions. This Part 2 package is designed to support that requirement and to provide richer methodology and analysis for the final report.

---

## 1. File List

| File | Purpose |
|---|---|
| `snake_env.py` | Part 1 environment, copied from the current environment implementation. |
| `q_agent.py` | Enhanced Part 2 RL agent. Core method is tabular Q-learning. Also supports SARSA, Expected SARSA, Double Q-learning, epsilon-greedy, softmax, and UCB. |
| `train_part2.py` | Main training script. Produces Q-tables, training logs, evaluation summaries, and reproducibility configs. |
| `run_part2_ablation.py` | Optional script for ablation experiments. It compares exploration methods, reward shaping, and algorithm variants. |
| `plot_part2_results.py` | Optional plotting script for learning curves. Useful for Part 3 and final report figures. |
| `sanity_check_part2.py` | Quick check that environment, Q-table, action selection, and TD update work correctly. |
| `PART2_REPORT_TEXT.md` | English report-ready methodology text for Part 2. |

---

## 2. Recommended Workflow

### Step 1: Quick sanity check

```bash
python sanity_check_part2.py
```

Expected output:

```text
All sanity checks passed.
Q-table shape: (2048, 3)
```

### Step 2: Quick training test

```bash
python train_part2.py --episodes 500 --output-dir results/quick_test
```

This confirms that the training loop can run before doing the full experiment.

### Step 3: Main Q-learning experiment

```bash
python train_part2.py \
  --episodes 10000 \
  --algorithm q_learning \
  --exploration epsilon_greedy \
  --alpha 0.1 \
  --gamma 0.9 \
  --epsilon 1.0 \
  --min-epsilon 0.01 \
  --epsilon-decay 0.995 \
  --output-dir results/main_q_learning
```

This is the main result that should be reported as the core **Q-learning** implementation.

### Step 4: Reward shaping comparison

```bash
python train_part2.py \
  --episodes 10000 \
  --algorithm q_learning \
  --exploration epsilon_greedy \
  --reward-shaping simple \
  --shaping-weight 0.1 \
  --output-dir results/q_learning_simple_shaping
```

Optional more theoretically principled version:

```bash
python train_part2.py \
  --episodes 10000 \
  --algorithm q_learning \
  --exploration epsilon_greedy \
  --reward-shaping potential \
  --shaping-weight 0.1 \
  --output-dir results/q_learning_potential_shaping
```

### Step 5: Exploration strategy comparison

Softmax / Boltzmann exploration:

```bash
python train_part2.py \
  --episodes 10000 \
  --algorithm q_learning \
  --exploration softmax \
  --temperature 1.0 \
  --temperature-decay 0.995 \
  --output-dir results/q_learning_softmax
```

UCB exploration:

```bash
python train_part2.py \
  --episodes 10000 \
  --algorithm q_learning \
  --exploration ucb \
  --ucb-c 1.0 \
  --output-dir results/q_learning_ucb
```

### Step 6: Optional algorithm comparison

On-policy SARSA:

```bash
python train_part2.py \
  --episodes 10000 \
  --algorithm sarsa \
  --exploration epsilon_greedy \
  --output-dir results/sarsa_epsilon
```

Expected SARSA:

```bash
python train_part2.py \
  --episodes 10000 \
  --algorithm expected_sarsa \
  --exploration epsilon_greedy \
  --output-dir results/expected_sarsa_epsilon
```

Double Q-learning:

```bash
python train_part2.py \
  --episodes 10000 \
  --algorithm double_q_learning \
  --exploration epsilon_greedy \
  --output-dir results/double_q_learning_epsilon
```

### Step 7: One-command ablation study

For a complete comparison table:

```bash
python run_part2_ablation.py --episodes 10000 --eval-episodes 300
```

Output:

```text
results/part2_ablation/ablation_summary.csv
```

This file can be directly sent to Part 3 for tables and analysis.

---

## 3. Output Files

Each training run produces:

| Output | Meaning |
|---|---|
| `training_history.csv` | Per-episode score, reward, survival steps, epsilon, TD error, moving averages. |
| `summary.txt` | Human-readable final evaluation against random baseline. |
| `summary.json` | Machine-readable final results. |
| `agent_config.json` | Hyperparameters for reproducibility. |
| `q_table.npy` or `q_table_effective.npy` | Learned Q-values. |
| `state_visit_counts.npy` | How many states were visited. Good for analysis. |
| `action_visit_counts.npy` | How often each action was selected in each state. |

---

## 4. What to Send to Part 3

Send these folders/files:

```text
snake_env.py
q_agent.py
train_part2.py
README_PART2_HIGH_SCORE.md
PART2_REPORT_TEXT.md
results/main_q_learning/training_history.csv
results/main_q_learning/summary.txt
```

If you run ablations, also send:

```text
results/part2_ablation/ablation_summary.csv
```

Part 3 can use `training_history.csv` to plot learning curves and use `ablation_summary.csv` to make comparison tables.

---

## 5. Why This Part 2 Is Stronger

Compared with a minimal Q-learning implementation, this version adds:

1. **Clear TD-control implementation**: Q-learning update, TD error logging, terminal-state handling.
2. **Exploration-exploitation analysis**: epsilon-greedy, softmax, UCB, epsilon decay, temperature decay.
3. **Reward design analysis**: original sparse reward, simple shaping, potential-based shaping.
4. **Course-related comparisons**: Q-learning vs SARSA vs Expected SARSA vs Double Q-learning.
5. **Reproducibility**: fixed random seed, saved configs, saved Q-tables, saved CSV logs.
6. **Evaluation readiness**: random baseline and greedy-policy evaluation are automatically produced.

For the final report, do **not** present every variant as the main contribution. The main method is still tabular Q-learning. The other variants should be described as ablation studies or extension experiments.
