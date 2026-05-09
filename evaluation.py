"""Evaluation helpers for trained and baseline Snake policies."""

from __future__ import annotations

import random
from typing import Dict

import numpy as np

from q_agent import TabularRLAgent
from snake_env import SnakeEnv


def run_random_baseline(env: SnakeEnv, episodes: int, seed: int) -> Dict[str, float]:
    """Random baseline: uniformly choose one of the three actions."""
    env.seed(seed)
    rng = random.Random(seed)
    scores, rewards, steps = [], [], []

    for _ in range(episodes):
        env.reset()
        done = False
        total_reward = 0.0
        info = {"score": 0, "steps": 0}

        while not done:
            action = rng.randrange(env.num_actions)
            _, reward, done, info = env.step(action)
            total_reward += reward

        scores.append(info["score"])
        rewards.append(total_reward)
        steps.append(info["steps"])

    return {
        "episodes": episodes,
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps)),
    }


def run_greedy_evaluation(env: SnakeEnv, agent: TabularRLAgent, episodes: int, seed: int) -> Dict[str, float]:
    """Evaluate the learned policy with greedy action selection."""
    env.seed(seed)
    scores, rewards, steps = [], [], []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        info = {"score": 0, "steps": 0}

        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

        scores.append(info["score"])
        rewards.append(total_reward)
        steps.append(info["steps"])

    return {
        "episodes": episodes,
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps)),
    }
