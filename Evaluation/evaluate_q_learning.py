"""Evaluate a trained Q-learning Snake policy greedily.

During evaluation the policy is greedy: epsilon is set to 0 and the selected
action is the argmax action from the loaded Q-table.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from q_agent import AgentConfig, TabularRLAgent
from snake_env import SnakeEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def evaluate_greedy_q_learning(
    q_table_path: Path,
    episodes: int,
    grid_size: int,
    initial_length: int,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    """Run greedy evaluation for a trained Q-table and return aggregate metrics."""
    set_seed(seed)
    env = SnakeEnv(grid_size=grid_size, initial_length=initial_length, max_steps=max_steps)

    config = AgentConfig(
        num_states=env.num_states,
        num_actions=env.num_actions,
        algorithm="q_learning",
        exploration="epsilon_greedy",
        epsilon=0.0,
        min_epsilon=0.0,
        seed=seed,
    )
    agent = TabularRLAgent(config)
    agent.epsilon = 0.0
    agent.load_effective_q_table(q_table_path)

    scores: list[int] = []
    rewards: list[float] = []
    steps: list[int] = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        info = {"score": 0, "steps": 0}

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward

        scores.append(int(info["score"]))
        rewards.append(float(total_reward))
        steps.append(int(info["steps"]))

    return {
        "episodes": int(episodes),
        "epsilon": 0.0,
        "avg_score": float(np.mean(scores)),
        "avg_total_reward": float(np.mean(rewards)),
        "avg_survival_steps": float(np.mean(steps)),
    }


def save_results(path: Path, results: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-learning Snake policy greedily.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--initial-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument(
        "--q-table",
        type=str,
        default=str(PROJECT_DIR / "results" / "main_q_learning" / "q_table_effective.npy"),
    )
    parser.add_argument("--output", type=str, default=str(PROJECT_DIR / "results" / "q_learning_evaluation.csv"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results = evaluate_greedy_q_learning(
        q_table_path=Path(args.q_table),
        episodes=args.episodes,
        grid_size=args.grid_size,
        initial_length=args.initial_length,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    save_results(Path(args.output), results)

    print(f"Q-learning greedy evaluation saved to: {args.output}")
    print(f"Epsilon: {results['epsilon']:.1f}")
    print(f"Average score: {results['avg_score']:.3f}")
    print(f"Average total reward: {results['avg_total_reward']:.3f}")
    print(f"Average survival steps: {results['avg_survival_steps']:.3f}")


if __name__ == "__main__":
    main()
