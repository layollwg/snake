"""
Part 2 Enhanced RL Agents for Simplified Snake.

This file is designed for the "Game Playing with Q-Learning" final project.
It keeps the core method as tabular Q-learning, but also provides several
course-related variants for deeper methodology and ablation experiments:

1. Q-learning          : off-policy TD control with max_a Q(s', a)
2. SARSA               : on-policy TD control with Q(s', a')
3. Expected SARSA      : on-policy expectation over the behavior policy
4. Double Q-learning   : reduces Q-learning over-estimation bias

Exploration strategies:
1. epsilon-greedy      : standard exploration-exploitation tradeoff
2. softmax/Boltzmann   : samples actions according to exp(Q / temperature)
3. UCB                 : bandit-style optimism bonus for less visited actions
4. greedy              : mainly for evaluation

Optional safe action masking can use the state's immediate-danger bits to
avoid actions that would collide on the next step. This is still paired with
the same Q-learning/SARSA update rules; it only changes action selection.

The environment state is an integer in [0, 2047], corresponding to the
11-bit feature representation from snake_env.py. The action is one of
{0: straight, 1: turn_left, 2: turn_right}.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


VALID_ALGORITHMS = {"q_learning", "sarsa", "expected_sarsa", "double_q_learning"}
VALID_EXPLORATION = {"epsilon_greedy", "softmax", "ucb", "greedy"}


@dataclass
class AgentConfig:
    """Hyperparameters for tabular RL control.

    alpha: learning rate. Larger alpha learns faster but may be unstable.
    gamma: discount factor for future reward.
    epsilon: initial random exploration probability.
    min_epsilon: lower bound for epsilon after decay.
    epsilon_decay: multiplicative decay applied at the end of each episode.
    exploration: behavior policy used during training.
    algorithm: TD control method.
    optimistic_initial_value: initialize all Q-values above zero to encourage exploration.
    avoid_immediate_danger: if true, action selection masks out actions whose
        state danger bit indicates an immediate collision, unless all actions
        are dangerous. This keeps the Q-learning update unchanged while making
        exploration safer.
    temperature: initial temperature for softmax exploration.
    min_temperature: lower bound for temperature.
    temperature_decay: multiplicative decay for temperature.
    ucb_c: strength of UCB exploration bonus.
    seed: random seed for reproducibility.
    """

    num_states: int = 2048
    num_actions: int = 3

    alpha: float = 0.1
    gamma: float = 0.9

    epsilon: float = 1.0
    min_epsilon: float = 0.01
    epsilon_decay: float = 0.995

    algorithm: str = "q_learning"
    exploration: str = "epsilon_greedy"

    optimistic_initial_value: float = 0.0
    avoid_immediate_danger: bool = False

    temperature: float = 1.0
    min_temperature: float = 0.05
    temperature_decay: float = 0.995

    ucb_c: float = 1.0
    seed: int = 42


class TabularRLAgent:
    """Tabular RL agent for simplified Snake.

    The class intentionally supports more than vanilla Q-learning so that
    Part 2 can show stronger technical depth in the report and presentation.
    For the main project title, use algorithm="q_learning" as the primary
    result, and use the other algorithms/exploration methods as ablations.
    """

    def __init__(self, config: AgentConfig):
        if config.algorithm not in VALID_ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {config.algorithm}. Valid: {sorted(VALID_ALGORITHMS)}")
        if config.exploration not in VALID_EXPLORATION:
            raise ValueError(f"Unknown exploration: {config.exploration}. Valid: {sorted(VALID_EXPLORATION)}")

        self.config = config
        self.num_states = config.num_states
        self.num_actions = config.num_actions
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.temperature = config.temperature

        self.py_rng = random.Random(config.seed)
        self.np_rng = np.random.RandomState(config.seed)

        init = float(config.optimistic_initial_value)
        self.q_table = np.full((self.num_states, self.num_actions), init, dtype=np.float64)

        # Double Q-learning keeps two separate estimators.
        self.q1_table = np.full((self.num_states, self.num_actions), init, dtype=np.float64)
        self.q2_table = np.full((self.num_states, self.num_actions), init, dtype=np.float64)

        # Visit counters are useful for UCB and for analysis in the report.
        self.state_visit_counts = np.zeros(self.num_states, dtype=np.int64)
        self.action_visit_counts = np.zeros((self.num_states, self.num_actions), dtype=np.int64)
        self.total_updates = 0

    # ------------------------------------------------------------------
    # Q-table helpers
    # ------------------------------------------------------------------
    def effective_q_table(self) -> np.ndarray:
        """Return the Q-table currently used for action selection.

        For Double Q-learning, we use Q1 + Q2 for action selection, which is
        the usual practical choice. For other methods, we use q_table.
        """
        if self.config.algorithm == "double_q_learning":
            return self.q1_table + self.q2_table
        return self.q_table

    def _random_argmax(self, values: np.ndarray) -> int:
        """Argmax with random tie-breaking, avoiding deterministic bias."""
        max_value = np.max(values)
        best_actions = np.flatnonzero(np.isclose(values, max_value))
        return int(self.np_rng.choice(best_actions))

    def _safe_actions(self, state: int) -> np.ndarray:
        """Return allowed actions after optional immediate-danger masking.

        The first three bits of the environment state are danger indicators for
        straight, left, and right. If every action is dangerous, the mask is
        relaxed so the agent can still select an action and let the environment
        terminate the episode normally.
        """
        if not self.config.avoid_immediate_danger:
            return np.arange(self.num_actions)

        danger_bits = np.array(
            [
                (int(state) >> 10) & 1,  # straight danger
                (int(state) >> 9) & 1,   # left-turn danger
                (int(state) >> 8) & 1,   # right-turn danger
            ],
            dtype=bool,
        )
        safe_actions = np.flatnonzero(~danger_bits)
        if len(safe_actions) == 0:
            return np.arange(self.num_actions)
        return safe_actions

    def _random_argmax_over_actions(self, values: np.ndarray, actions: np.ndarray) -> int:
        """Argmax restricted to a candidate action set."""
        action_values = values[actions]
        max_value = np.max(action_values)
        best_local_indices = np.flatnonzero(np.isclose(action_values, max_value))
        return int(actions[self.np_rng.choice(best_local_indices)])

    def get_action_probabilities(self, state: int) -> np.ndarray:
        """Behavior-policy probabilities used by Expected SARSA.

        This returns a probability distribution over actions under the current
        exploration policy. For UCB, which is deterministic after adding a
        confidence bonus, we approximate it as a one-hot distribution.
        """
        q_values = self.effective_q_table()[state]
        safe_actions = self._safe_actions(state)

        if self.config.exploration == "epsilon_greedy":
            probs = np.zeros(self.num_actions, dtype=np.float64)
            probs[safe_actions] = self.epsilon / len(safe_actions)
            greedy_action = self._random_argmax_over_actions(q_values, safe_actions)
            probs[greedy_action] += 1.0 - self.epsilon
            return probs

        if self.config.exploration == "softmax":
            probs = np.zeros(self.num_actions, dtype=np.float64)
            probs[safe_actions] = self._softmax_probabilities(q_values[safe_actions], self.temperature)
            return probs

        if self.config.exploration == "ucb":
            action = self._select_ucb_action(state)
            probs = np.zeros(self.num_actions, dtype=np.float64)
            probs[action] = 1.0
            return probs

        # greedy
        probs = np.zeros(self.num_actions, dtype=np.float64)
        probs[self._random_argmax_over_actions(q_values, safe_actions)] = 1.0
        return probs

    @staticmethod
    def _softmax_probabilities(q_values: np.ndarray, temperature: float) -> np.ndarray:
        """Numerically stable softmax distribution."""
        temperature = max(float(temperature), 1e-8)
        shifted = (q_values - np.max(q_values)) / temperature
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state: int, training: bool = True) -> int:
        """Select an action according to the configured exploration policy."""
        state = int(state)
        if training:
            self.state_visit_counts[state] += 1

        q_values = self.effective_q_table()[state]
        safe_actions = self._safe_actions(state)

        if not training or self.config.exploration == "greedy":
            action = self._random_argmax_over_actions(q_values, safe_actions)
        elif self.config.exploration == "epsilon_greedy":
            action = self._select_epsilon_greedy_action(state)
        elif self.config.exploration == "softmax":
            action = self._select_softmax_action(state)
        elif self.config.exploration == "ucb":
            action = self._select_ucb_action(state)
        else:
            raise RuntimeError("Invalid exploration policy.")

        if training:
            self.action_visit_counts[state, action] += 1
        return int(action)

    def _select_epsilon_greedy_action(self, state: int) -> int:
        safe_actions = self._safe_actions(state)
        if self.py_rng.random() < self.epsilon:
            return int(self.np_rng.choice(safe_actions))
        return self._random_argmax_over_actions(self.effective_q_table()[state], safe_actions)

    def _select_softmax_action(self, state: int) -> int:
        safe_actions = self._safe_actions(state)
        probs = self._softmax_probabilities(self.effective_q_table()[state][safe_actions], self.temperature)
        return int(self.np_rng.choice(safe_actions, p=probs))

    def _select_ucb_action(self, state: int) -> int:
        """UCB action selection adapted from multi-armed bandits.

        Less visited actions receive a larger exploration bonus. This is useful
        for explaining the exploration-exploitation idea from the course.
        """
        q_values = self.effective_q_table()[state]
        safe_actions = self._safe_actions(state)
        state_visits = max(1, int(self.state_visit_counts[state]))
        action_visits = self.action_visit_counts[state].astype(np.float64)

        # Try all actions at least once in each visited state.
        unvisited = np.array([action for action in safe_actions if action_visits[action] == 0])
        if len(unvisited) > 0:
            return int(self.np_rng.choice(unvisited))

        bonus = self.config.ucb_c * np.sqrt(np.log(state_visits + 1.0) / (action_visits + 1e-8))
        return self._random_argmax_over_actions(q_values + bonus, safe_actions)

    # ------------------------------------------------------------------
    # TD updates
    # ------------------------------------------------------------------
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None,
    ) -> float:
        """Apply the selected TD update and return the TD error."""
        if self.config.algorithm == "q_learning":
            return self._update_q_learning(state, action, reward, next_state, done)
        if self.config.algorithm == "sarsa":
            if next_action is None and not done:
                raise ValueError("SARSA requires next_action when not done.")
            return self._update_sarsa(state, action, reward, next_state, done, next_action)
        if self.config.algorithm == "expected_sarsa":
            return self._update_expected_sarsa(state, action, reward, next_state, done)
        if self.config.algorithm == "double_q_learning":
            return self._update_double_q_learning(state, action, reward, next_state, done)
        raise RuntimeError("Invalid algorithm.")

    def _update_q_learning(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Off-policy Q-learning update.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma max_a' Q(s',a') - Q(s,a)]
        """
        current = self.q_table[state, action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        td_error = target - current
        self.q_table[state, action] = current + self.alpha * td_error
        self.total_updates += 1
        return float(td_error)

    def _update_sarsa(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int],
    ) -> float:
        """On-policy SARSA update.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma Q(s',a') - Q(s,a)]
        """
        current = self.q_table[state, action]
        target = reward if done else reward + self.gamma * self.q_table[next_state, int(next_action)]
        td_error = target - current
        self.q_table[state, action] = current + self.alpha * td_error
        self.total_updates += 1
        return float(td_error)

    def _update_expected_sarsa(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Expected SARSA update.

        Uses the expectation over the next-state behavior policy instead of a
        sampled next action. This can reduce variance compared with SARSA.
        """
        current = self.q_table[state, action]
        if done:
            target = reward
        else:
            probs = self.get_action_probabilities(next_state)
            expected_q = float(np.dot(probs, self.q_table[next_state]))
            target = reward + self.gamma * expected_q
        td_error = target - current
        self.q_table[state, action] = current + self.alpha * td_error
        self.total_updates += 1
        return float(td_error)

    def _update_double_q_learning(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Double Q-learning update.

        Randomly update Q1 or Q2. The selected action comes from one table,
        while its value is evaluated using the other table. This addresses the
        max operator's positive bias in ordinary Q-learning.
        """
        update_q1 = self.py_rng.random() < 0.5

        if update_q1:
            current = self.q1_table[state, action]
            if done:
                target = reward
            else:
                best_next_action = self._random_argmax(self.q1_table[next_state])
                target = reward + self.gamma * self.q2_table[next_state, best_next_action]
            td_error = target - current
            self.q1_table[state, action] = current + self.alpha * td_error
        else:
            current = self.q2_table[state, action]
            if done:
                target = reward
            else:
                best_next_action = self._random_argmax(self.q2_table[next_state])
                target = reward + self.gamma * self.q1_table[next_state, best_next_action]
            td_error = target - current
            self.q2_table[state, action] = current + self.alpha * td_error

        self.total_updates += 1
        return float(td_error)

    def decay_schedules(self) -> None:
        """Decay epsilon and temperature after each episode."""
        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)
        self.temperature = max(self.config.min_temperature, self.temperature * self.config.temperature_decay)

    # ------------------------------------------------------------------
    # Analysis and persistence
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        q = self.effective_q_table()
        return {
            "epsilon": float(self.epsilon),
            "temperature": float(self.temperature),
            "visited_states": int(np.count_nonzero(self.state_visit_counts)),
            "visited_state_action_pairs": int(np.count_nonzero(self.action_visit_counts)),
            "total_updates": int(self.total_updates),
            "q_mean": float(np.mean(q)),
            "q_mean_abs": float(np.mean(np.abs(q))),
            "q_max": float(np.max(q)),
            "q_min": float(np.min(q)),
        }

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "q_table_effective.npy", self.effective_q_table())
        np.save(output_dir / "state_visit_counts.npy", self.state_visit_counts)
        np.save(output_dir / "action_visit_counts.npy", self.action_visit_counts)

        if self.config.algorithm == "double_q_learning":
            np.save(output_dir / "q1_table.npy", self.q1_table)
            np.save(output_dir / "q2_table.npy", self.q2_table)
        else:
            np.save(output_dir / "q_table.npy", self.q_table)

        with open(output_dir / "agent_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    def load_effective_q_table(self, q_path: str | Path) -> None:
        q = np.load(q_path)
        if q.shape != (self.num_states, self.num_actions):
            raise ValueError(f"Expected Q-table shape {(self.num_states, self.num_actions)}, got {q.shape}")
        self.q_table = q.astype(np.float64)
