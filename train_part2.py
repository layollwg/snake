"""
Enhanced Part 2 training script for "Q-Learning for Simplified Snake".

Main purpose:
- train a tabular Q-learning agent for Snake;
- support course-related ablations: SARSA, Expected SARSA, Double Q-learning;
- support exploration strategies: epsilon-greedy, softmax, UCB;
- support optional reward shaping;
- save clean logs for Part 3 experiments and final report.

Quick test:
    python train_part2.py --episodes 500 --output-dir results/quick_test

Main Q-learning run:
    python train_part2.py --episodes 10000 --algorithm q_learning --exploration epsilon_greedy --output-dir results/main_q_learning

Reward shaping run:
    python train_part2.py --episodes 10000 --algorithm q_learning --reward-shaping simple --shaping-weight 0.1 --output-dir results/q_learning_shaping

Softmax exploration run:
    python train_part2.py --episodes 10000 --exploration softmax --temperature 1.0 --output-dir results/q_learning_softmax

UCB exploration run:
    python train_part2.py --episodes 10000 --exploration ucb --ucb-c 1.0 --output-dir results/q_learning_ucb
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np

from q_agent import AgentConfig, TabularRLAgent
from snake_env import SnakeEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def manhattan_distance_to_food(env: SnakeEnv) -> Optional[int]:
    """Return Manhattan distance from snake head to food."""
    if env.food is None:
        return None
    head_r, head_c = env.body[0]
    food_r, food_c = env.food
    return abs(head_r - food_r) + abs(head_c - food_c)


def _rule_based_action(env: SnakeEnv) -> int:
    """Greedy heuristic: steer toward food when safe, fall back to any safe action.

    This is used only by run_rule_based_baseline; it is NOT used during RL training.
    """
    head_r, head_c = env.body[0]
    if env.food is None:
        # Board is completely full (snake occupies every cell) — no food to chase.
        safe = [a for a in range(env.num_actions) if not env._would_collide(a)]
        return random.choice(safe) if safe else 0

    food_r, food_c = env.food

    LEFT_TURN = {0: 2, 1: 3, 2: 1, 3: 0}
    RIGHT_TURN = {0: 3, 1: 2, 2: 0, 3: 1}

    def absolute_to_relative(abs_dir: int) -> Optional[int]:
        """Map an absolute direction to a relative action; None = 180° turn."""
        if abs_dir == env.direction:
            return 0
        if abs_dir == LEFT_TURN[env.direction]:
            return 1
        if abs_dir == RIGHT_TURN[env.direction]:
            return 2
        return None  # 180° — impossible

    # Preferred absolute directions ordered by distance reduction
    dr = food_r - head_r
    dc = food_c - head_c
    preferred_abs: List[int] = []
    if abs(dr) >= abs(dc):
        if dr > 0:
            preferred_abs.append(1)  # down
        elif dr < 0:
            preferred_abs.append(0)  # up
        if dc > 0:
            preferred_abs.append(3)  # right
        elif dc < 0:
            preferred_abs.append(2)  # left
    else:
        if dc > 0:
            preferred_abs.append(3)
        elif dc < 0:
            preferred_abs.append(2)
        if dr > 0:
            preferred_abs.append(1)
        elif dr < 0:
            preferred_abs.append(0)

    for abs_dir in preferred_abs:
        rel = absolute_to_relative(abs_dir)
        if rel is not None and not env._would_collide(rel):
            return rel

    # Fall back to any safe action
    safe = [a for a in range(env.num_actions) if not env._would_collide(a)]
    return random.choice(safe) if safe else 0


def apply_reward_shaping(
    env_reward: float,
    old_distance: Optional[int],
    new_distance: Optional[int],
    done: bool,
    gamma: float,
    mode: str,
    shaping_weight: float,
) -> float:
    """Add optional shaping reward.

    mode="none":
        use the environment reward only.
    mode="simple":
        +weight if the snake moves closer to food, -weight if farther.
    mode="potential":
        potential-based shaping F(s,s') = gamma * Phi(s') - Phi(s),
        where Phi(s) = -distance_to_food(s). This is a more theoretically
        principled version and is useful to mention in the report.
    """
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


def run_random_baseline(env: SnakeEnv, episodes: int, seed: int) -> Dict[str, float]:
    """Random baseline: uniformly choose one of the three actions."""
    set_seed(seed)
    scores, rewards, steps = [], [], []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        info = {"score": 0, "steps": 0}

        while not done:
            action = random.randrange(env.num_actions)
            state, reward, done, info = env.step(action)
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


def run_rule_based_baseline(env: SnakeEnv, episodes: int, seed: int) -> Dict[str, float]:
    """Rule-based heuristic baseline: steer toward food when safe, else take a safe random action.

    This is more capable than a random policy but requires no learning.
    Including it creates a three-tier comparison: random < rule-based < trained RL.
    """
    set_seed(seed)
    scores, rewards, steps = [], [], []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        info = {"score": 0, "steps": 0}

        while not done:
            action = _rule_based_action(env)
            state, reward, done, info = env.step(action)
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
    set_seed(seed)
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


def train_one_episode(
    env: SnakeEnv,
    agent: TabularRLAgent,
    reward_shaping: str,
    shaping_weight: float,
) -> Dict[str, float]:
    """Train for one episode and return per-episode statistics."""
    state = env.reset()
    # Reset eligibility traces at episode start (no-op when lambda_=0)
    agent.reset_eligibility_traces()
    done = False
    total_env_reward = 0.0
    total_learning_reward = 0.0
    td_errors: List[float] = []

    # SARSA needs the next action sampled from the same behavior policy.
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


def write_history_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_training(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = SnakeEnv(
        grid_size=args.grid_size,
        initial_length=args.initial_length,
        max_steps=args.max_steps,
        use_enhanced_state=args.enhanced_state,
    )

    config = AgentConfig(
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
        temperature=args.temperature,
        min_temperature=args.min_temperature,
        temperature_decay=args.temperature_decay,
        ucb_c=args.ucb_c,
        lambda_=args.lambda_,
        adaptive_lr=args.adaptive_lr,
        seed=args.seed,
    )
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
            "q_std": float(np.std(agent.effective_q_table())),
        }
        history.append(row)

        if episode % args.print_every == 0 or episode == 1:
            print(
                f"Episode {episode:6d}/{args.episodes} | "
                f"score={result['score']:2d} | steps={result['steps']:4d} | "
                f"MA_score={row['moving_avg_score']:.3f} | "
                f"eps={stats['epsilon']:.3f} | temp={stats['temperature']:.3f}"
            )

    trained_eval = run_greedy_evaluation(env, agent, args.eval_episodes, seed=args.seed + 1000)
    random_eval = run_random_baseline(env, args.eval_episodes, seed=args.seed + 2000)
    rule_eval = run_rule_based_baseline(env, args.eval_episodes, seed=args.seed + 3000)

    summary: Dict[str, object] = {
        "config": vars(args),
        "agent_config": config.__dict__,
        "trained_greedy_evaluation": trained_eval,
        "random_baseline_evaluation": random_eval,
        "rule_based_baseline_evaluation": rule_eval,
        "final_agent_stats": agent.stats(),
    }

    agent.save(output_dir)
    write_history_csv(output_dir / "training_history.csv", history)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Part 2 Enhanced Training Summary\n")
        f.write("================================\n\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Exploration: {args.exploration}\n")
        f.write(f"Reward shaping: {args.reward_shaping}, weight={args.shaping_weight}\n")
        f.write(f"Grid: {args.grid_size}x{args.grid_size}, episodes={args.episodes}\n")
        f.write(f"Enhanced state: {args.enhanced_state}, lambda={args.lambda_}, adaptive_lr={args.adaptive_lr}\n\n")
        f.write("Greedy trained policy evaluation:\n")
        for k, v in trained_eval.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nRule-based baseline evaluation:\n")
        for k, v in rule_eval.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nRandom baseline evaluation:\n")
        for k, v in random_eval.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nFinal Q-table / visit statistics:\n")
        for k, v in agent.stats().items():
            f.write(f"  {k}: {v}\n")

    print("\nTraining finished.")
    print(f"Outputs saved to: {output_dir}")
    print(f"Greedy avg score:     {trained_eval['avg_score']:.3f}")
    print(f"Rule-based avg score: {rule_eval['avg_score']:.3f}")
    print(f"Random avg score:     {random_eval['avg_score']:.3f}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
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

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-temperature", type=float, default=0.05)
    parser.add_argument("--temperature-decay", type=float, default=0.995)
    parser.add_argument("--ucb-c", type=float, default=1.0)

    parser.add_argument("--reward-shaping", type=str, default="none", choices=["none", "simple", "potential"])
    parser.add_argument("--shaping-weight", type=float, default=0.1)

    # Enhanced state representation
    parser.add_argument("--enhanced-state", action="store_true", default=False,
                        help="Use 15-bit enhanced state (adds 2-step danger + length bit)")

    # TD(λ) eligibility traces
    parser.add_argument("--lambda", type=float, default=0.0, dest="lambda_",
                        help="Eligibility trace decay λ (0=1-step TD, 1≈MC). Only applies to q_learning.")

    # Adaptive learning rate
    parser.add_argument("--adaptive-lr", action="store_true", default=False,
                        help="Use adaptive α(s,a)=1/(1+n(s,a)) instead of fixed alpha")

    parser.add_argument("--moving-average-window", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=500)

    return parser


if __name__ == "__main__":
    run_training(build_arg_parser().parse_args())
