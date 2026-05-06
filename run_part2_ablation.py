"""
Run a compact Part 2 ablation study.

This script is optional but useful for high-scoring report content. It runs
several course-related variants and produces a single comparison CSV.

Default quick command:
    python run_part2_ablation.py --episodes 3000

More complete command:
    python run_part2_ablation.py --episodes 10000 --eval-episodes 300
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def load_summary(summary_path: Path):
    import json

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Part 2 ablation experiments.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--base-output-dir", type=str, default="results/part2_ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(args.base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # These variants are chosen for both technical depth and report clarity.
    variants = [
        {
            "name": "q_learning_epsilon",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": [],
        },
        {
            "name": "q_learning_simple_shaping",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "simple",
            "extra": ["--shaping-weight", "0.1"],
        },
        {
            "name": "q_learning_potential_shaping",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "potential",
            "extra": ["--shaping-weight", "0.1"],
        },
        {
            "name": "q_learning_slow_decay",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": ["--epsilon-decay", "0.999"],
        },
        {
            "name": "q_learning_low_initial_epsilon",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": ["--epsilon", "0.5"],
        },
        {
            "name": "q_learning_softmax",
            "algorithm": "q_learning",
            "exploration": "softmax",
            "reward_shaping": "none",
            "extra": ["--temperature", "1.0", "--temperature-decay", "0.995"],
        },
        {
            "name": "q_learning_ucb",
            "algorithm": "q_learning",
            "exploration": "ucb",
            "reward_shaping": "none",
            "extra": ["--ucb-c", "1.0"],
        },
        {
            "name": "sarsa_epsilon",
            "algorithm": "sarsa",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": [],
        },
        {
            "name": "expected_sarsa_epsilon",
            "algorithm": "expected_sarsa",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": [],
        },
        {
            "name": "double_q_learning_epsilon",
            "algorithm": "double_q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": [],
        },
    ]

    comparison_rows = []

    for i, variant in enumerate(variants):
        out_dir = base_dir / variant["name"]
        cmd = [
            sys.executable,
            "train_part2.py",
            "--episodes",
            str(args.episodes),
            "--eval-episodes",
            str(args.eval_episodes),
            "--grid-size",
            str(args.grid_size),
            "--algorithm",
            variant["algorithm"],
            "--exploration",
            variant["exploration"],
            "--reward-shaping",
            variant["reward_shaping"],
            "--output-dir",
            str(out_dir),
            "--seed",
            str(args.seed + i),
            "--print-every",
            str(max(1, args.episodes // 5)),
        ] + variant["extra"]

        run_command(cmd)
        summary = load_summary(out_dir / "summary.json")
        trained = summary["trained_greedy_evaluation"]
        random_base = summary["random_baseline_evaluation"]
        stats = summary["final_agent_stats"]

        comparison_rows.append(
            {
                "variant": variant["name"],
                "algorithm": variant["algorithm"],
                "exploration": variant["exploration"],
                "reward_shaping": variant["reward_shaping"],
                "trained_avg_score": trained["avg_score"],
                "trained_std_score": trained["std_score"],
                "trained_max_score": trained["max_score"],
                "trained_avg_reward": trained["avg_reward"],
                "trained_avg_steps": trained["avg_steps"],
                "random_avg_score": random_base["avg_score"],
                "improvement_over_random_score": trained["avg_score"] - random_base["avg_score"],
                "visited_states": stats["visited_states"],
                "visited_state_action_pairs": stats["visited_state_action_pairs"],
                "q_mean_abs": stats["q_mean_abs"],
            }
        )

    comparison_csv = base_dir / "ablation_summary.csv"
    with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("\nAblation finished.")
    print(f"Summary saved to: {comparison_csv}")


if __name__ == "__main__":
    main()
