# Part 2 Report-Ready Text: Q-Learning Agent

## Methodology: Tabular Q-Learning Agent

In Part 2, we implemented a tabular reinforcement learning agent for the simplified Snake environment. The main algorithm is Q-learning, a model-free, off-policy temporal-difference control method. The agent does not need to know the transition dynamics of the Snake environment in advance. Instead, it learns action values through repeated interaction with the environment.

The environment state is encoded as an 11-dimensional binary feature vector and then converted into an integer state index. Therefore, the number of possible states is at most \(2^{11}=2048\). The action space contains three relative actions: going straight, turning left, and turning right. As a result, the Q-table has shape \(2048 \times 3\), where each entry \(Q(s,a)\) estimates the expected discounted return after taking action \(a\) in state \(s\).

The basic Q-learning update rule is:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right].
\]

Here, \(\alpha\) is the learning rate, \(\gamma\) is the discount factor, \(r\) is the immediate reward, and \(s'\) is the next state. If the next state is terminal, the bootstrapped future value is removed and the target becomes only the immediate reward. This avoids incorrectly using Q-values after the episode has already ended.

## Exploration Strategy

During training, the agent must balance exploration and exploitation. We use epsilon-greedy exploration as the main strategy. With probability \(\epsilon\), the agent chooses a random action to explore the environment. With probability \(1-\epsilon\), it chooses the action with the highest current Q-value. The exploration rate starts from \(\epsilon=1.0\), which means the early training phase is highly exploratory. After each episode, \(\epsilon\) is multiplied by a decay factor until it reaches a minimum value. In our main setting, we use \(\epsilon_{min}=0.01\) and decay \(0.995\).

To make the Part 2 implementation more complete, we also implemented two additional exploration strategies for ablation studies. First, softmax exploration samples actions according to a probability distribution based on their Q-values. This gives higher probability to better actions but still allows less-valued actions to be selected. Second, UCB exploration adds an optimism bonus to less-visited actions, which is related to the exploration-exploitation idea in multi-armed bandit problems. These strategies allow us to compare how different exploration mechanisms affect learning speed and final policy quality.

## Reward Design and Reward Shaping

The base environment reward is \(+10\) when the snake eats food, \(-10\) when the snake dies, and \(-0.1\) for each normal step. This reward function encourages the agent to collect food, avoid collisions, and not waste too many steps. However, the food reward is relatively sparse because the agent may need many steps before reaching food. Therefore, we also include optional reward shaping for comparison.

In the simple shaping setting, the agent receives a small positive bonus if it moves closer to the food and a small negative penalty if it moves farther away. This provides denser learning signals and may improve early learning speed. We also include a potential-based shaping option using the negative Manhattan distance to food as the potential function. This version is more theoretically principled because the shaping term depends on the potential difference between the current and next state.

The main reported method should still use the original environment reward, while reward shaping is treated as an ablation experiment. This makes the project analysis stronger because it shows how reward design influences learning performance.

## Additional Course-Related Variants

Although the main project method is Q-learning, we implemented several related temporal-difference control methods for comparison. SARSA is an on-policy method that updates the Q-value using the actual next action selected by the behavior policy. Its update target is:

\[
r + \gamma Q(s',a').
\]

Compared with Q-learning, SARSA may learn a more conservative policy because it accounts for the exploratory actions that the agent may actually take during training. Expected SARSA further reduces variance by using the expected Q-value under the behavior policy instead of a single sampled next action. We also implemented Double Q-learning, which uses two Q-tables to reduce the over-estimation bias caused by the max operator in standard Q-learning.

These additional methods are not used to change the main topic. Instead, they provide useful comparisons for the methodology and results sections. They also show that our implementation is closely connected to multiple concepts covered in the reinforcement learning course, including temporal-difference learning, on-policy versus off-policy control, exploration-exploitation tradeoff, and value-function estimation.

## Training and Evaluation Protocol

For each training episode, the environment is reset, and the agent repeatedly selects an action, receives a reward, observes the next state, and updates the Q-table. We record the score, total environment reward, shaped learning reward, survival steps, epsilon value, TD error, number of visited states, and moving average score. These logs are saved to `training_history.csv` for later visualization and analysis.

After training, we evaluate the learned policy using greedy action selection, which means \(\epsilon=0\) during testing. We also evaluate a random baseline that selects among the three actions uniformly at random. The final comparison uses average score, average total reward, average survival steps, and maximum score over multiple testing episodes. The expected result is that the trained Q-learning policy should achieve higher score and longer survival time than the random baseline.

## Implementation Structure

The code is organized into focused modules instead of putting all logic in one training script. The environment dynamics are implemented in `snake_env.py`, the RL agent is implemented in `q_agent.py`, the training loop is in `training.py`, evaluation is in `evaluation.py`, output persistence is in `io_utils.py`, and command-line configuration is in `configs.py`. Experiment scripts share common process-running and summary helpers from `experiment_runner.py`, while variant definitions for Q-learning optimization and multi-seed experiments are stored in JSON files under `experiment_configs/`.

The overall workflow is:

```text
SnakeEnv.reset()
      |
      v
SnakeEnv.get_state()  ->  TabularRLAgent.select_action()
      |                              |
      v                              v
SnakeEnv.step(action)  ->  reward, next_state, done
      |                              |
      v                              v
reward shaping         ->  TabularRLAgent.update()
      |
      v
training history CSV, Q-table, summary JSON/TXT
      |
      v
greedy evaluation and random baseline comparison
```

## Grid-Size Generalization Check

To make the result analysis stronger, we also tested whether the main conclusion on the 6x6 board remains valid on an 8x8 board. This experiment keeps the algorithm, reward function, state representation, action space, and hyperparameters fixed. Only the board size changes. Because the state representation is based on 11 abstract binary features rather than exact cell coordinates, the Q-table size remains \(2048 \times 3\) for both board sizes. Therefore, the 8x8 experiment does not require a much larger table; it mainly changes the transition dynamics and makes episodes longer on average.

Using 10000 training episodes and 300 greedy evaluation episodes, the results were:

| Board | Avg score | Avg score / max food | Max score | Avg steps | Random avg score | Improvement over random | Visited states |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6x6 | 13.73 | 40.38% | 27 | 69.99 | 0.187 | 13.54 | 256 |
| 8x8 | 16.88 | 27.23% | 36 | 111.03 | 0.143 | 16.74 | 256 |

The conclusion is consistent across both board sizes: the trained Q-learning policy substantially outperforms the random baseline. The 8x8 board has a higher raw average score because there is more free space and the snake survives longer, but the capacity-normalized score is lower. On 6x6, the agent collects about 40% of the maximum possible food count on average, while on 8x8 it collects about 27%. This suggests that the learned local state abstraction transfers to the larger board in the sense that it still produces a useful policy, but the larger board remains harder to cover efficiently.

This comparison also reveals a limitation of the current representation. The agent observes local danger, current direction, and the relative direction of food, but it does not observe exact distance, global board position, or detailed body geometry. These features are enough for strong behavior in the simplified setting, but they may limit performance when the board becomes larger and long-term path planning becomes more important.

## Q-Learning Optimization Study

To improve the final score while keeping the main method as Q-learning, we added a Q-learning-only optimization study. In this experiment, `algorithm="q_learning"` and epsilon-greedy exploration remain fixed. We only tune training choices around the same Q-learning update rule, including learning rate \(\alpha\), discount factor \(\gamma\), epsilon decay, potential-based reward shaping, optimistic initialization, and an optional immediate-danger action mask.

The immediate-danger mask uses information that already exists in the state representation: the first three state bits indicate whether going straight, turning left, or turning right would immediately collide. When the mask is enabled, the policy avoids actions whose danger bit is 1, unless all actions are dangerous. This does not change the Q-learning target; it only makes exploration and greedy action selection safer.

The best Q-learning-only configuration in our 6x6 experiment was a lower learning rate with slower epsilon decay:

```text
alpha = 0.05
gamma = 0.9
epsilon_decay = 0.999
reward_shaping = none
avoid_immediate_danger = false
```

The comparison is:

| Variant | Avg score | Max score | Avg steps | Final 100-episode moving avg score |
|---|---:|---:|---:|---:|
| Default Q-learning | 13.73 | 27 | 69.99 | 12.31 |
| Optimized Q-learning | 14.33 | 30 | 72.83 | 13.10 |
| Safe mask + light shaping | 14.30 | 26 | 70.81 | 13.51 |

These results show that the optimized Q-learning setting improves the average greedy evaluation score and maximum score without changing the core algorithm. The lower learning rate likely makes value estimates less noisy, while slower epsilon decay gives the agent more time to explore before committing to a mostly greedy policy. The safety mask and reward shaping are useful analysis additions, but in this run they did not clearly outperform the simpler optimized Q-learning configuration. This is a useful result for the report because it shows that the final choice is based on experimental evidence rather than adding every possible technique.

## Multi-Seed Robustness

To avoid relying on a single random seed, we added a multi-seed experiment script. It compares baseline Q-learning, optimized Q-learning, and safe-mask Q-learning across multiple seeds, then reports the mean and standard deviation of evaluation scores. This makes the result section more rigorous because a small improvement from one seed may not be reliable.

The command is:

```text
python run_q_learning_multiseed.py --episodes 10000 --eval-episodes 300
```

It produces:

```text
results/q_learning_multiseed/multiseed_runs.csv
results/q_learning_multiseed/multiseed_summary.csv
```

The final report should use the summary table with columns such as `mean_avg_score`, `std_avg_score`, `mean_max_score`, and `mean_improvement_over_random`.

## Failure Analysis

The trained Q-learning agent performs much better than a random policy, but it still has clear failure modes. First, the policy can greedily move toward nearby food and accidentally reduce future free space, especially when the snake becomes longer. Second, because the state only tells the relative direction of food and immediate danger, the agent does not know whether a path will trap it several steps later. Third, on the 8x8 board, raw score increases because the snake survives longer, but the capacity-normalized score decreases. This suggests that the learned policy transfers to a larger board but does not fully solve long-horizon planning.

Most failures therefore come from limited state information rather than from the Q-learning update itself. The table-based agent can learn useful local rules such as avoiding immediate collisions and moving toward food, but it cannot explicitly reason about global board connectivity or future escape routes.

## State Representation Limitations

The 11-bit state abstraction is compact and makes tabular Q-learning practical, but it loses important information. It does not include exact snake head coordinates, exact food distance, full body layout, path length to food, number of safe cells, or whether the snake has an escape route after eating. Different board configurations can therefore map to the same abstract state even if the best action is different. This state aliasing is one reason the agent can achieve strong simplified-Snake performance but still struggles with larger boards or longer-term planning.

Future improvements could add richer handcrafted features, such as Manhattan distance bins, wall distance, body length, available flood-fill area after each candidate action, or shortest-path direction to food. A more advanced extension would replace the Q-table with function approximation, but that would move beyond the main tabular Q-learning focus of this project.

## Reproducibility

To make the implementation reproducible, the training script saves the random seed, all hyperparameters, training history, Q-table, and evaluation summary. The environment and agent each use local random number generators initialized from the seed, so experiments are easier to reproduce and less affected by unrelated random calls. The main output files are `training_history.csv`, `summary.txt`, `summary.json`, and `q_table.npy`. This makes it easy for the evaluation group to generate learning curves and comparison tables for the final report.

## Concise Presentation Script for Part 2

For Part 2, I implemented the tabular reinforcement learning agent. The main method is Q-learning, where the state is the 11-bit feature representation from the environment and the action space contains three relative actions: straight, turn left, and turn right. Therefore, the Q-table has 2048 states and 3 actions.

During training, the agent uses epsilon-greedy exploration. In the beginning, epsilon is high, so the agent explores different actions. Later, epsilon gradually decays, so the policy becomes more greedy and stable. The Q-value is updated using the TD target \(r + \gamma \max Q(s',a')\). If the episode ends, the target is only the immediate reward.

To make the implementation stronger, I also added several ablation options. We can compare epsilon-greedy with softmax and UCB exploration, compare the original reward with reward shaping, and compare Q-learning with SARSA, Expected SARSA, and Double Q-learning. These variants are useful for analysis, but the main result remains the Q-learning agent. The training script automatically saves the Q-table, training history, and final evaluation against a random baseline, so Part 3 can directly use these files to draw curves and build result tables.

I also added a 6x6 versus 8x8 grid-size comparison. Since the state is represented by 11 abstract features, the Q-table size stays the same on both boards. The 8x8 board takes somewhat longer per experiment because episodes last longer, but it is still practical to train. The learned policy beats the random baseline on both board sizes, so the main conclusion is consistent, while the lower capacity-normalized score on 8x8 shows that larger boards still require better long-term planning.

Finally, I added a Q-learning-only optimization study. The best optimized version keeps the same Q-learning update rule but uses a smaller learning rate and slower epsilon decay. This improves the average evaluation score from 13.73 to 14.33 and the maximum score from 27 to 30. I also tested immediate-danger action masking and potential shaping, but the simpler tuned Q-learning setting performed best overall.
