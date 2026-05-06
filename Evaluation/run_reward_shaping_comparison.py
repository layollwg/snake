"""Compare sparse reward and distance-shaped reward for Q-learning Snake."""

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


def manhattan_distance_to_food(env: SnakeEnv) -> int | None:
    if env.food is None:
        return None
    head_r, head_c = env.body[0]
    food_r, food_c = env.food
    return abs(head_r - food_r) + abs(head_c - food_c)


def shaped_reward(sparse_reward: float, old_distance: int | None, new_distance: int | None) -> float:
    if old_distance is None or new_distance is None:
        return float(sparse_reward)
    if new_distance < old_distance:
        return float(sparse_reward + 0.1)
    if new_distance > old_distance:
        return float(sparse_reward - 0.1)
    return float(sparse_reward)


def train_episode(env: SnakeEnv, agent: TabularRLAgent, reward_mode: str) -> dict[str, float]:
    state = env.reset()
    done = False
    total_sparse_reward = 0.0
    total_learning_reward = 0.0
    info = {"score": 0, "steps": 0}

    while not done:
        old_distance = manhattan_distance_to_food(env)
        action = agent.select_action(state, training=True)
        next_state, sparse_reward, done, info = env.step(action)
        new_distance = manhattan_distance_to_food(env)

        learning_reward = (
            shaped_reward(sparse_reward, old_distance, new_distance)
            if reward_mode == "shaped"
            else float(sparse_reward)
        )

        agent.update(state, action, learning_reward, next_state, done)
        state = next_state
        total_sparse_reward += sparse_reward
        total_learning_reward += learning_reward

    agent.decay_schedules()

    return {
        "score": int(info["score"]),
        "total_sparse_reward": float(total_sparse_reward),
        "total_learning_reward": float(total_learning_reward),
        "survival_steps": int(info["steps"]),
    }


def train_setting(args: argparse.Namespace, reward_mode: str, seed: int) -> list[dict[str, float]]:
    set_seed(seed)
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
        seed=seed,
    )
    agent = TabularRLAgent(config)

    score_window: deque[float] = deque(maxlen=100)
    sparse_reward_window: deque[float] = deque(maxlen=100)
    rows: list[dict[str, float]] = []

    for episode in range(1, args.episodes + 1):
        result = train_episode(env, agent, reward_mode)
        score_window.append(result["score"])
        sparse_reward_window.append(result["total_sparse_reward"])

        rows.append(
            {
                "reward_mode": reward_mode,
                "episode": episode,
                "score": result["score"],
                "total_sparse_reward": result["total_sparse_reward"],
                "total_learning_reward": result["total_learning_reward"],
                "survival_steps": result["survival_steps"],
                "epsilon": float(agent.epsilon),
                "moving_avg_score_100": float(np.mean(score_window)),
                "moving_avg_sparse_reward_100": float(np.mean(sparse_reward_window)),
            }
        )

        if episode % args.print_every == 0 or episode == 1:
            row = rows[-1]
            print(
                f"{reward_mode:6s} | Episode {episode:6d}/{args.episodes} | "
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
    parser = argparse.ArgumentParser(description="Compare sparse and shaped rewards on 6x6 Snake.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--initial-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min-epsilon", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(PROJECT_DIR / "results" / "reward_shaping_comparison.csv"))
    parser.add_argument("--print-every", type=int, default=500)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sparse_rows = train_setting(args, reward_mode="sparse", seed=args.seed)
    shaped_rows = train_setting(args, reward_mode="shaped", seed=args.seed)
    rows = sparse_rows + shaped_rows
    write_csv(Path(args.output), rows)

    print(f"\nReward shaping comparison saved to: {args.output}")
    print(
        "Final sparse MA score: "
        f"{sparse_rows[-1]['moving_avg_score_100']:.3f}, "
        "final shaped MA score: "
        f"{shaped_rows[-1]['moving_avg_score_100']:.3f}"
    )


if __name__ == "__main__":
    main()
