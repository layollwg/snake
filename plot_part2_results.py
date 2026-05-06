"""
Optional plotting script for Part 2/Part 3.

Input: a training output folder containing training_history.csv (and optionally
       state_visit_counts.npy / q_table_effective.npy).
Output: PNG learning curves and analysis charts under the same folder.

Example:
    python plot_part2_results.py --result-dir results/main_q_learning
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_history(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def plot_curve(rows, x_key, y_key, output_path: Path, title: str, y_label: str):
    xs = [row[x_key] for row in rows]
    ys = [row[y_key] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys)
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_state_visit_distribution(result_dir: Path, top_n: int = 60) -> None:
    """Bar chart of the top-N most visited states (sorted descending).

    Helps identify which state features the agent encounters most frequently,
    showing how coverage improves with training.
    """
    counts_path = result_dir / "state_visit_counts.npy"
    if not counts_path.exists():
        return
    counts = np.load(counts_path)
    sorted_counts = np.sort(counts)[::-1][:top_n]
    nonzero = int(np.count_nonzero(counts))

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(sorted_counts)), sorted_counts, color="steelblue")
    plt.xlabel(f"State rank (top {top_n} of {nonzero} visited)")
    plt.ylabel("Visit count")
    plt.title(f"State visit distribution — top {top_n} states")
    plt.tight_layout()
    plt.savefig(result_dir / "viz_state_visits.png", dpi=180)
    plt.close()


def plot_policy_distribution(result_dir: Path) -> None:
    """Bar chart of how often each action is chosen by the greedy policy.

    Reveals whether the learned policy is biased toward certain actions
    across the full state space.
    """
    q_path = result_dir / "q_table_effective.npy"
    visit_path = result_dir / "state_visit_counts.npy"
    if not q_path.exists():
        return

    q = np.load(q_path)
    action_names = ["straight", "turn_left", "turn_right"]
    greedy_actions = np.argmax(q, axis=1)

    # Focus on actually visited states if visit counts are available
    if visit_path.exists():
        visited_mask = np.load(visit_path) > 0
        greedy_actions = greedy_actions[visited_mask]
        subtitle = "(visited states only)"
    else:
        subtitle = "(all states)"

    counts = [int(np.sum(greedy_actions == a)) for a in range(len(action_names))]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(action_names, counts, color=["#2196F3", "#4CAF50", "#FF9800"])
    plt.xlabel("Action")
    plt.ylabel("Number of states")
    plt.title(f"Greedy policy action distribution {subtitle}")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(result_dir / "viz_policy_distribution.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Part 2 training curves.")
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    rows = read_history(result_dir / "training_history.csv")

    # Standard learning curves
    plot_curve(rows, "episode", "score", result_dir / "curve_score.png", "Training Score per Episode", "Score")
    plot_curve(rows, "episode", "moving_avg_score", result_dir / "curve_moving_avg_score.png", "Moving Average Score", "Moving Average Score")
    plot_curve(rows, "episode", "env_reward", result_dir / "curve_reward.png", "Environment Reward per Episode", "Total Reward")
    plot_curve(rows, "episode", "moving_avg_env_reward", result_dir / "curve_moving_avg_reward.png", "Moving Average Reward", "Moving Average Reward")
    plot_curve(rows, "episode", "steps", result_dir / "curve_survival_steps.png", "Survival Steps per Episode", "Steps")
    plot_curve(rows, "episode", "epsilon", result_dir / "curve_epsilon.png", "Epsilon Decay", "Epsilon")
    plot_curve(rows, "episode", "mean_abs_td_error", result_dir / "curve_td_error.png", "Mean Absolute TD Error", "Mean |TD Error|")

    # Q-value standard deviation convergence (present when q_std column exists)
    if rows and "q_std" in rows[0]:
        plot_curve(rows, "episode", "q_std", result_dir / "curve_q_std.png",
                   "Q-Value Std Dev (Convergence Indicator)", "Std Dev")

    # Additional analysis charts using saved numpy arrays
    plot_state_visit_distribution(result_dir)
    plot_policy_distribution(result_dir)

    print(f"Plots saved to: {result_dir}")


if __name__ == "__main__":
    main()
