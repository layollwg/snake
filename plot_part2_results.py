"""
Optional plotting script for Part 2/Part 3.

Input: a training output folder containing training_history.csv.
Output: PNG learning curves under the same folder.

Example:
    python plot_part2_results.py --result-dir results/main_q_learning
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


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


def main():
    parser = argparse.ArgumentParser(description="Plot Part 2 training curves.")
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    rows = read_history(result_dir / "training_history.csv")

    plot_curve(rows, "episode", "score", result_dir / "curve_score.png", "Training Score per Episode", "Score")
    plot_curve(rows, "episode", "moving_avg_score", result_dir / "curve_moving_avg_score.png", "Moving Average Score", "Moving Average Score")
    plot_curve(rows, "episode", "env_reward", result_dir / "curve_reward.png", "Environment Reward per Episode", "Total Reward")
    plot_curve(rows, "episode", "moving_avg_env_reward", result_dir / "curve_moving_avg_reward.png", "Moving Average Reward", "Moving Average Reward")
    plot_curve(rows, "episode", "steps", result_dir / "curve_survival_steps.png", "Survival Steps per Episode", "Steps")
    plot_curve(rows, "episode", "epsilon", result_dir / "curve_epsilon.png", "Epsilon Decay", "Epsilon")
    plot_curve(rows, "episode", "mean_abs_td_error", result_dir / "curve_td_error.png", "Mean Absolute TD Error", "Mean |TD Error|")

    print(f"Plots saved to: {result_dir}")


if __name__ == "__main__":
    main()
