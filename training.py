"""Training loop for tabular RL agents in the Snake environment."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import numpy as np

from configs import build_agent_config
from evaluation import run_greedy_evaluation, run_random_baseline
from io_utils import save_training_outputs
from q_agent import TabularRLAgent
from snake_env import SnakeEnv


def manhattan_distance_to_food(env: SnakeEnv) -> Optional[int]:
    """Return Manhattan distance from snake head to food."""
    if env.food is None:
        return None
    head_r, head_c = env.body[0]
    food_r, food_c = env.food
    return abs(head_r - food_r) + abs(head_c - food_c)


def apply_reward_shaping(
    env_reward: float,
    old_distance: Optional[int],
    new_distance: Optional[int],
    done: bool,
    gamma: float,
    mode: str,
    shaping_weight: float,
) -> float:
    """Add optional shaping reward."""
    if mode == "none" or old_distance is None or new_distance is None:
        return float(env_reward)

    if mode == "simple":
        if new_distance < old_distance:
            return float(env_reward + shaping_weight)
        if new_distance > old_distance:
            return float(env_reward - shaping_weight)
        return float(env_reward)

    if mode == "potential":
        old_phi = -float(old_distance)
        new_phi = -float(new_distance)
        return float(env_reward + shaping_weight * (gamma * new_phi - old_phi))

    raise ValueError(f"Unknown reward shaping mode: {mode}")


def train_one_episode(
    env: SnakeEnv,
    agent: TabularRLAgent,
    reward_shaping: str,
    shaping_weight: float,
) -> Dict[str, float]:
    """Train for one episode and return per-episode statistics."""
    state = env.reset()
    done = False
    total_env_reward = 0.0
    total_learning_reward = 0.0
    td_errors: List[float] = []

    action = agent.select_action(state, training=True)
    info = {"score": 0, "steps": 0, "cause": ""}

    while not done:
        old_distance = manhattan_distance_to_food(env)
        next_state, env_reward, done, info = env.step(action)
        new_distance = manhattan_distance_to_food(env)

        learning_reward = apply_reward_shaping(
            env_reward=env_reward,
            old_distance=old_distance,
            new_distance=new_distance,
            done=done,
            gamma=agent.gamma,
            mode=reward_shaping,
            shaping_weight=shaping_weight,
        )

        if agent.config.algorithm == "sarsa" and not done:
            next_action = agent.select_action(next_state, training=True)
            td_error = agent.update(state, action, learning_reward, next_state, done, next_action=next_action)
            state, action = next_state, next_action
        else:
            td_error = agent.update(state, action, learning_reward, next_state, done)
            state = next_state
            if not done:
                action = agent.select_action(state, training=True)

        td_errors.append(abs(td_error))
        total_env_reward += env_reward
        total_learning_reward += learning_reward

    agent.decay_schedules()

    return {
        "score": int(info["score"]),
        "env_reward": float(total_env_reward),
        "learning_reward": float(total_learning_reward),
        "steps": int(info["steps"]),
        "cause": info.get("cause", ""),
        "mean_abs_td_error": float(mean(td_errors)) if td_errors else 0.0,
    }


def train_agent(args: argparse.Namespace) -> tuple[SnakeEnv, TabularRLAgent, list[dict[str, object]]]:
    """Train an agent and return the environment, agent, and history rows."""
    env = SnakeEnv(
        grid_size=args.grid_size,
        initial_length=args.initial_length,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    config = build_agent_config(args, env)
    agent = TabularRLAgent(config)

    history: List[Dict[str, object]] = []
    score_window = deque(maxlen=args.moving_average_window)
    reward_window = deque(maxlen=args.moving_average_window)
    step_window = deque(maxlen=args.moving_average_window)

    for episode in range(1, args.episodes + 1):
        result = train_one_episode(env, agent, args.reward_shaping, args.shaping_weight)
        score_window.append(result["score"])
        reward_window.append(result["env_reward"])
        step_window.append(result["steps"])
        stats = agent.stats()

        row: Dict[str, object] = {
            "episode": episode,
            "algorithm": args.algorithm,
            "exploration": args.exploration,
            "reward_shaping": args.reward_shaping,
            "avoid_immediate_danger": args.avoid_immediate_danger,
            "grid_size": args.grid_size,
            "score": result["score"],
            "env_reward": result["env_reward"],
            "learning_reward": result["learning_reward"],
            "steps": result["steps"],
            "done_cause": result["cause"],
            "mean_abs_td_error": result["mean_abs_td_error"],
            "moving_avg_score": float(np.mean(score_window)),
            "moving_avg_env_reward": float(np.mean(reward_window)),
            "moving_avg_steps": float(np.mean(step_window)),
            "epsilon": stats["epsilon"],
            "temperature": stats["temperature"],
            "visited_states": stats["visited_states"],
            "visited_state_action_pairs": stats["visited_state_action_pairs"],
            "q_mean_abs": stats["q_mean_abs"],
            "q_max": stats["q_max"],
            "q_min": stats["q_min"],
        }
        history.append(row)

        if episode % args.print_every == 0 or episode == 1:
            print(
                f"Episode {episode:6d}/{args.episodes} | "
                f"score={result['score']:2d} | steps={result['steps']:4d} | "
                f"MA_score={row['moving_avg_score']:.3f} | "
                f"eps={stats['epsilon']:.3f} | temp={stats['temperature']:.3f}"
            )

    return env, agent, history


def run_training(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    env, agent, history = train_agent(args)
    config = build_agent_config(args, env)

    trained_eval = run_greedy_evaluation(env, agent, args.eval_episodes, seed=args.seed + 1000)
    random_eval = run_random_baseline(env, args.eval_episodes, seed=args.seed + 2000)

    summary = save_training_outputs(
        output_dir=output_dir,
        agent=agent,
        config=config,
        args_dict=vars(args),
        history=history,
        trained_eval=trained_eval,
        random_eval=random_eval,
    )

    print("\nTraining finished.")
    print(f"Outputs saved to: {output_dir}")
    print(f"Greedy avg score: {trained_eval['avg_score']:.3f}")
    print(f"Random avg score: {random_eval['avg_score']:.3f}")

    return summary
