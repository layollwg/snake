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

## Reproducibility

To make the implementation reproducible, the training script saves the random seed, all hyperparameters, training history, Q-table, and evaluation summary. The main output files are `training_history.csv`, `summary.txt`, `summary.json`, and `q_table.npy`. This makes it easy for the evaluation group to generate learning curves and comparison tables for the final report.

## Concise Presentation Script for Part 2

For Part 2, I implemented the tabular reinforcement learning agent. The main method is Q-learning, where the state is the 11-bit feature representation from the environment and the action space contains three relative actions: straight, turn left, and turn right. Therefore, the Q-table has 2048 states and 3 actions.

During training, the agent uses epsilon-greedy exploration. In the beginning, epsilon is high, so the agent explores different actions. Later, epsilon gradually decays, so the policy becomes more greedy and stable. The Q-value is updated using the TD target \(r + \gamma \max Q(s',a')\). If the episode ends, the target is only the immediate reward.

To make the implementation stronger, I also added several ablation options. We can compare epsilon-greedy with softmax and UCB exploration, compare the original reward with reward shaping, and compare Q-learning with SARSA, Expected SARSA, and Double Q-learning. These variants are useful for analysis, but the main result remains the Q-learning agent. The training script automatically saves the Q-table, training history, and final evaluation against a random baseline, so Part 3 can directly use these files to draw curves and build result tables.
