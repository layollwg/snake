"""Simple sanity checks for Part 1 + Part 2 integration."""

from __future__ import annotations

import numpy as np

from q_agent import AgentConfig, TabularRLAgent
from snake_env import SnakeEnv


def main():
    env = SnakeEnv(grid_size=6, initial_length=2, max_steps=100)
    state = env.reset()

    assert isinstance(state, int), "State should be an integer index."
    assert 0 <= state < 2048, "State should be in [0, 2047]."
    assert env.num_actions == 3, "Action space should contain 3 actions."

    config = AgentConfig(algorithm="q_learning", exploration="epsilon_greedy", seed=42)
    agent = TabularRLAgent(config)
    assert agent.q_table.shape == (2048, 3), "Q-table shape should be (2048, 3)."

    action = agent.select_action(state, training=True)
    assert action in [0, 1, 2], "Action must be 0/1/2."

    next_state, reward, done, info = env.step(action)
    old_value = agent.q_table[state, action]
    td_error = agent.update(state, action, reward, next_state, done)
    new_value = agent.q_table[state, action]

    assert np.isfinite(td_error), "TD error should be finite."
    assert np.isfinite(new_value), "Updated Q-value should be finite."
    assert old_value != new_value or reward == 0.0, "Q-value should usually change after update."

    print("All sanity checks passed.")
    print("State index:", state)
    print("Action:", action)
    print("Reward:", reward)
    print("TD error:", td_error)
    print("Q-table shape:", agent.q_table.shape)


if __name__ == "__main__":
    main()
