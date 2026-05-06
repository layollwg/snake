"""Compare epsilon decay strategies for tabular Q-learning Snake."""

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


def train_episode(env: SnakeEnv, agent: TabularRLAgent) -> dict[str, float]:
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


def train_strategy(
    args: argparse.Namespace,
    strategy_name: str,
    initial_epsilon: float,
    epsilon_decay: float,
    seed: int,
) -> list[dict[str, float]]:
    set_seed(seed)
    env = SnakeEnv(grid_size=6, initial_length=args.initial_length, max_steps=args.max_steps)
    config = AgentConfig(
        num_states=env.num_states,
        num_actions=env.num_actions,
        algorithm="q_learning",
        exploration="epsilon_greedy",
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=initial_epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    agent = TabularRLAgent(config)

    score_window: deque[float] = deque(maxlen=100)
    reward_window: deque[float] = deque(maxlen=100)
    rows: list[dict[str, float]] = []

    for episode in range(1, args.episodes + 1):
        result = train_episode(env, agent)
        score_window.append(result["score"])
        reward_window.append(result["total_reward"])

        rows.append(
            {
                "strategy": strategy_name,
                "episode": episode,
                "initial_epsilon": initial_epsilon,
                "epsilon_decay": epsilon_decay,
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
                f"{strategy_name:15s} | Episode {episode:6d}/{args.episodes} | "
                f"score={row['score']:2d} | "
                f"MA_score_100={row['moving_avg_score_100']:.3f} | "
                f"epsilon={row['epsilon']:.3f}"
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare epsilon decay strategies on 6x6 Snake.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--initial-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--min-epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(PROJECT_DIR / "results" / "epsilon_decay_comparison.csv"))
    parser.add_argument("--print-every", type=int, default=500)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    strategies = [
        ("slow_decay", 1.0, 0.999),
        ("medium_decay", 1.0, 0.995),
        ("low_exploration", 0.3, 0.995),
    ]

    all_rows: list[dict[str, float]] = []
    for strategy_name, initial_epsilon, epsilon_decay in strategies:
        rows = train_strategy(
            args=args,
            strategy_name=strategy_name,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            seed=args.seed,
        )
        all_rows.extend(rows)

    write_csv(Path(args.output), all_rows)

    print(f"\nEpsilon decay comparison saved to: {args.output}")
    for strategy_name, _, _ in strategies:
        final_row = next(row for row in reversed(all_rows) if row["strategy"] == strategy_name)
        print(
            f"{strategy_name}: final MA score={final_row['moving_avg_score_100']:.3f}, "
            f"final epsilon={final_row['epsilon']:.3f}"
        )


if __name__ == "__main__":
    main()
