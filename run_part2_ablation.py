"""
Run a compact Part 2 ablation study.

This script is optional but useful for high-scoring report content. It runs
several course-related variants and produces a single comparison CSV.

Default quick command:
    python run_part2_ablation.py --episodes 3000

More complete command (multi-seed with 3 seeds):
    python run_part2_ablation.py --episodes 10000 --eval-episodes 300 --num-seeds 3
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_command(cmd):
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def load_summary(summary_path: Path):
    import json

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate(values):
    """Return (mean, std) over a list of floats."""
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def main():
    parser = argparse.ArgumentParser(description="Run Part 2 ablation experiments.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--base-output-dir", type=str, default="results/part2_ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of independent seeds per variant. >1 enables mean±std reporting.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Variant definitions — chosen for technical depth and report clarity  #
    # New entries: optimistic init, TD(λ), adaptive LR, enhanced state,   #
    # and an 8×8 grid comparison.                                          #
    # ------------------------------------------------------------------ #
    variants = [
        # ---- Original variants ----------------------------------------
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
        # ---- New: Optimistic Initialization ---------------------------
        {
            "name": "q_learning_oiv_1",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": ["--optimistic-initial-value", "1.0"],
        },
        {
            "name": "q_learning_oiv_5",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": ["--optimistic-initial-value", "5.0"],
        },
        # ---- New: TD(λ) / Eligibility Traces -------------------------
        {
            "name": "q_lambda_0_5",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": ["--lambda", "0.5"],
        },
        {
            "name": "q_lambda_0_9",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": ["--lambda", "0.9"],
        },
        # ---- New: Adaptive Learning Rate -----------------------------
        {
            "name": "q_learning_adaptive_lr",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": ["--adaptive-lr"],
        },
        # ---- New: Enhanced State (6×6) --------------------------------
        {
            "name": "q_learning_enhanced_state",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "extra": ["--enhanced-state"],
        },
        # ---- New: 8×8 Grid Comparison --------------------------------
        {
            "name": "q_learning_8x8",
            "algorithm": "q_learning",
            "exploration": "epsilon_greedy",
            "reward_shaping": "none",
            "grid_size_override": 8,
            "extra": [],
        },
    ]

    comparison_rows = []

    for i, variant in enumerate(variants):
        grid_size = variant.get("grid_size_override", args.grid_size)
        seed_summaries = []

        for seed_idx in range(args.num_seeds):
            # Stride by num_seeds to guarantee no two (variant, seed_idx) pairs
            # share the same seed value, even when num_seeds is large.
            seed = args.seed + i * args.num_seeds + seed_idx
            if args.num_seeds > 1:
                out_dir = base_dir / variant["name"] / f"seed_{seed_idx}"
            else:
                out_dir = base_dir / variant["name"]

            cmd = [
                sys.executable,
                "train_part2.py",
                "--episodes",
                str(args.episodes),
                "--eval-episodes",
                str(args.eval_episodes),
                "--grid-size",
                str(grid_size),
                "--algorithm",
                variant["algorithm"],
                "--exploration",
                variant["exploration"],
                "--reward-shaping",
                variant["reward_shaping"],
                "--output-dir",
                str(out_dir),
                "--seed",
                str(seed),
                "--print-every",
                str(max(1, args.episodes // 5)),
            ] + variant["extra"]

            run_command(cmd)
            seed_summaries.append(load_summary(out_dir / "summary.json"))

        # Aggregate across seeds
        trained_scores = [s["trained_greedy_evaluation"]["avg_score"] for s in seed_summaries]
        trained_max = [s["trained_greedy_evaluation"]["max_score"] for s in seed_summaries]
        trained_rewards = [s["trained_greedy_evaluation"]["avg_reward"] for s in seed_summaries]
        trained_steps = [s["trained_greedy_evaluation"]["avg_steps"] for s in seed_summaries]
        random_scores = [s["random_baseline_evaluation"]["avg_score"] for s in seed_summaries]
        rule_scores = [s["rule_based_baseline_evaluation"]["avg_score"] for s in seed_summaries]
        visited = [s["final_agent_stats"]["visited_states"] for s in seed_summaries]

        score_mean, score_std = _aggregate(trained_scores)
        reward_mean, _ = _aggregate(trained_rewards)
        steps_mean, _ = _aggregate(trained_steps)
        random_mean, _ = _aggregate(random_scores)
        rule_mean, _ = _aggregate(rule_scores)

        comparison_rows.append(
            {
                "variant": variant["name"],
                "algorithm": variant["algorithm"],
                "exploration": variant["exploration"],
                "reward_shaping": variant["reward_shaping"],
                "grid_size": grid_size,
                "num_seeds": args.num_seeds,
                "trained_avg_score_mean": score_mean,
                "trained_avg_score_std": score_std,
                "trained_max_score": max(trained_max),
                "trained_avg_reward": reward_mean,
                "trained_avg_steps": steps_mean,
                "rule_based_avg_score": rule_mean,
                "random_avg_score": random_mean,
                "improvement_over_random": score_mean - random_mean,
                "improvement_over_rule_based": score_mean - rule_mean,
                "visited_states": int(np.mean(visited)),
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
