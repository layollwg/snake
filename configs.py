"""Command-line and agent configuration helpers."""

from __future__ import annotations

import argparse

from q_agent import AgentConfig
from snake_env import SnakeEnv


def build_agent_config(args: argparse.Namespace, env: SnakeEnv) -> AgentConfig:
    """Build an AgentConfig from CLI args and environment metadata."""
    return AgentConfig(
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        algorithm=args.algorithm,
        exploration=args.exploration,
        optimistic_initial_value=args.optimistic_initial_value,
        avoid_immediate_danger=args.avoid_immediate_danger,
        temperature=args.temperature,
        min_temperature=args.min_temperature,
        temperature_decay=args.temperature_decay,
        ucb_c=args.ucb_c,
        seed=args.seed,
    )


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhanced Part 2 training for Snake Q-learning project.")

    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--initial-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="results/main_q_learning")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--algorithm", type=str, default="q_learning", choices=["q_learning", "sarsa", "expected_sarsa", "double_q_learning"])
    parser.add_argument("--exploration", type=str, default="epsilon_greedy", choices=["epsilon_greedy", "softmax", "ucb", "greedy"])

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min-epsilon", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--optimistic-initial-value", type=float, default=0.0)
    parser.add_argument("--avoid-immediate-danger", action="store_true")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-temperature", type=float, default=0.05)
    parser.add_argument("--temperature-decay", type=float, default=0.995)
    parser.add_argument("--ucb-c", type=float, default=1.0)

    parser.add_argument("--reward-shaping", type=str, default="none", choices=["none", "simple", "potential"])
    parser.add_argument("--shaping-weight", type=float, default=0.1)

    parser.add_argument("--moving-average-window", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=500)

    return parser
