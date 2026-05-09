"""Evaluate a uniformly random Snake policy.

The random agent chooses uniformly among the three relative actions:
0 = straight, 1 = turn_left, 2 = turn_right.
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

from snake_env import SnakeEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def evaluate_random_baseline(
    episodes: int,
    grid_size: int,
    initial_length: int,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    """Run a uniformly random policy and return aggregate metrics."""
    set_seed(seed)
    env = SnakeEnv(grid_size=grid_size, initial_length=initial_length, max_steps=max_steps)

    scores: list[int] = []
    rewards: list[float] = []
    steps: list[int] = []

    for _ in range(episodes):
        env.reset()
        done = False
        total_reward = 0.0
        info = {"score": 0, "steps": 0}

        while not done:
            action = random.randrange(env.num_actions)
            _, reward, done, info = env.step(action)
            total_reward += reward

        scores.append(int(info["score"]))
        rewards.append(float(total_reward))
        steps.append(int(info["steps"]))

    return {
        "episodes": int(episodes),
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
    parser = argparse.ArgumentParser(description="Evaluate a uniformly random Snake baseline.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--initial-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(PROJECT_DIR / "results" / "random_baseline.csv"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results = evaluate_random_baseline(
        episodes=args.episodes,
        grid_size=args.grid_size,
        initial_length=args.initial_length,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    save_results(Path(args.output), results)

    print(f"Random baseline results saved to: {args.output}")
    print(f"Average score: {results['avg_score']:.3f}")
    print(f"Average total reward: {results['avg_total_reward']:.3f}")
    print(f"Average survival steps: {results['avg_survival_steps']:.3f}")


if __name__ == "__main__":
    main()
