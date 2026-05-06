"""Simple sanity checks for Part 1 + Part 2 integration."""

from __future__ import annotations

import numpy as np

from q_agent import AgentConfig, TabularRLAgent
from snake_env import SnakeEnv


def main():
    # ------------------------------------------------------------------ #
    # 1. Basic environment and standard Q-learning                        #
    # ------------------------------------------------------------------ #
    env = SnakeEnv(grid_size=6, initial_length=2, max_steps=100)
    state = env.reset()

    assert isinstance(state, int), "State should be an integer index."
    assert 0 <= state < 2048, "State should be in [0, 2047]."
    assert env.num_actions == 3, "Action space should contain 3 actions."
    assert env.num_states == 2048, "Standard env should have 2048 states."

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

    print("[PASS] Standard Q-learning")

    # ------------------------------------------------------------------ #
    # 2. Enhanced state representation (15-bit, 32768 states)             #
    # ------------------------------------------------------------------ #
    env_enh = SnakeEnv(grid_size=6, initial_length=2, max_steps=100, use_enhanced_state=True)
    state_enh = env_enh.reset()

    assert env_enh.num_states == 32768, "Enhanced env should have 32768 states."
    assert isinstance(state_enh, int), "Enhanced state should be an integer."
    assert 0 <= state_enh < 32768, f"Enhanced state {state_enh} out of range."

    vec = env_enh.get_state_vector()
    assert vec.shape == (15,), f"Enhanced state vector should have 15 bits, got {vec.shape}."

    # Run a few steps to make sure the enhanced state changes properly
    for _ in range(10):
        a = 0
        s, r, d, _ = env_enh.step(a)
        assert 0 <= s < 32768, "Enhanced step state out of range."
        if d:
            env_enh.reset()

    print("[PASS] Enhanced state representation (15-bit)")

    # ------------------------------------------------------------------ #
    # 3. TD(λ) / Eligibility traces                                        #
    # ------------------------------------------------------------------ #
    config_lambda = AgentConfig(lambda_=0.5, algorithm="q_learning", seed=42)
    agent_lambda = TabularRLAgent(config_lambda)

    # Traces should start at zero
    assert np.all(agent_lambda.eligibility_traces == 0.0), "Traces should be zero initially."

    env2 = SnakeEnv(grid_size=6, initial_length=2, max_steps=100)
    s = env2.reset()
    agent_lambda.reset_eligibility_traces()

    a = agent_lambda.select_action(s, training=True)
    ns, r2, d2, _ = env2.step(a)
    agent_lambda.update(s, a, r2, ns, d2)

    # After one update, the current (s,a) trace should have been incremented then decayed
    trace_val = agent_lambda.eligibility_traces[s, a]
    # After accumulate +1 and decay by gamma*lambda: 1 * 0.9 * 0.5 = 0.45
    assert trace_val > 0.0, "Trace for visited (s,a) should be positive."
    # Multiple entries may have been updated via broadcast; Q should have changed
    assert np.any(np.isfinite(agent_lambda.q_table)), "Q-table should be finite after λ update."

    agent_lambda.reset_eligibility_traces()
    assert np.all(agent_lambda.eligibility_traces == 0.0), "Traces should be zero after reset."

    print("[PASS] TD(λ) eligibility traces")

    # ------------------------------------------------------------------ #
    # 4. Adaptive learning rate                                            #
    # ------------------------------------------------------------------ #
    config_alr = AgentConfig(adaptive_lr=True, alpha=0.1, algorithm="q_learning", seed=42)
    agent_alr = TabularRLAgent(config_alr)

    # Initial alpha (visit count 0, clamped to 1): 1/(1+1) = 0.5
    assert abs(agent_alr._get_alpha(0, 0) - 0.5) < 1e-9, "First-visit alpha should be 0.5."

    # Simulate more visits
    agent_alr.action_visit_counts[0, 0] = 9
    expected_alpha = 1.0 / (1.0 + 9)
    assert abs(agent_alr._get_alpha(0, 0) - expected_alpha) < 1e-9, (
        f"Alpha should be {expected_alpha} after 9 visits."
    )

    print("[PASS] Adaptive learning rate")

    # ------------------------------------------------------------------ #
    # 5. Rule-based baseline runs without error                           #
    # ------------------------------------------------------------------ #
    from train_part2 import run_rule_based_baseline

    env3 = SnakeEnv(grid_size=6, initial_length=2, max_steps=100)
    rule_result = run_rule_based_baseline(env3, episodes=20, seed=42)
    assert "avg_score" in rule_result, "Rule-based baseline should return avg_score."
    assert rule_result["avg_score"] >= 0.0, "Rule-based avg score should be non-negative."

    print("[PASS] Rule-based baseline")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print("\nAll sanity checks passed.")
    print("State index (standard):", state)
    print("State index (enhanced):", state_enh)
    print("Action:", action)
    print("Reward:", reward)
    print("TD error:", td_error)
    print("Q-table shape:", agent.q_table.shape)


if __name__ == "__main__":
    main()
