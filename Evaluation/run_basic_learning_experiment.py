"""Run the basic Q-learning performance experiment on 6x6 Snake."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import deque
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


def run_episode(env: SnakeEnv, agent: TabularRLAgent) -> dict[str, float]:
    """Train for one episode using tabular Q-learning."""
    state = env.reset()
    done = False
    total_reward = 0.0
    info = {"score": 0, "steps": 0}

    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.decay_schedules()

    return {
        "score": int(info["score"]),
        "total_reward": float(total_reward),
        "survival_steps": int(info["steps"]),
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args: argparse.Namespace) -> list[dict[str, float]]:
    set_seed(args.seed)

    env = SnakeEnv(grid_size=6, initial_length=args.initial_length, max_steps=args.max_steps)
    config = AgentConfig(
        num_states=env.num_states,
        num_actions=env.num_actions,
        algorithm="q_learning",
        exploration="epsilon_greedy",
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    agent = TabularRLAgent(config)

    score_window: deque[float] = deque(maxlen=100)
    reward_window: deque[float] = deque(maxlen=100)
    rows: list[dict[str, float]] = []

    for episode in range(1, args.episodes + 1):
        result = run_episode(env, agent)
        score_window.append(result["score"])
        reward_window.append(result["total_reward"])

        rows.append(
            {
                "episode": episode,
                "score": result["score"],
                "total_reward": result["total_reward"],
                "survival_steps": result["survival_steps"],
                "epsilon": float(agent.epsilon),
                "moving_avg_score_100": float(np.mean(score_window)),
                "moving_avg_reward_100": float(np.mean(reward_window)),
            }
        )

        if episode % args.print_every == 0 or episode == 1:
            row = rows[-1]
            print(
                f"Episode {episode:6d}/{args.episodes} | "
                f"score={row['score']:2d} | "
                f"reward={row['total_reward']:7.2f} | "
                f"MA_score_100={row['moving_avg_score_100']:.3f} | "
                f"epsilon={row['epsilon']:.3f}"
            )

    write_csv(Path(args.output), rows)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Basic Q-learning performance experiment on 6x6 Snake.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--initial-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min-epsilon", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(PROJECT_DIR / "results" / "basic_learning.csv"))
    parser.add_argument("--print-every", type=int, default=500)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = run_experiment(args)
    final = rows[-1]

    print(f"\nBasic learning results saved to: {args.output}")
    print(f"Final 100-episode moving average score: {final['moving_avg_score_100']:.3f}")
    print(f"Final 100-episode moving average reward: {final['moving_avg_reward_100']:.3f}")


if __name__ == "__main__":
    main()
