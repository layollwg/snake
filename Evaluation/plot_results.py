"""Generate experiment figures from CSV files in the results folder.

Uses matplotlib only for plotting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_single_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def to_float_list(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def save_figure(output_path: Path) -> Path:
    """Save the current figure, falling back if an existing image is locked."""
    candidate_paths = [output_path]
    if output_path.exists():
        candidate_paths = [
            output_path.with_name(f"{output_path.stem}_{index}{output_path.suffix}")
            for index in range(1, 100)
        ]
        candidate_paths.insert(0, output_path)

    last_error = None
    for candidate_path in candidate_paths:
        try:
            plt.savefig(candidate_path, dpi=180)
            return candidate_path
        except OSError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not save figure to {output_path}")


def plot_learning_curve(
    rows: list[dict[str, str]],
    y_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> Path:
    episodes = to_float_list(rows, "episode")
    values = to_float_list(rows, y_key)

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, values, linewidth=1.8)
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    saved_path = save_figure(output_path)
    plt.close()
    return saved_path


def plot_baseline_comparison(
    random_row: dict[str, str],
    q_learning_row: dict[str, str],
    output_path: Path,
) -> Path:
    labels = ["Average Score", "Average Total Reward", "Average Survival Steps"]
    random_values = [
        float(random_row["avg_score"]),
        float(random_row["avg_total_reward"]),
        float(random_row["avg_survival_steps"]),
    ]
    q_learning_values = [
        float(q_learning_row["avg_score"]),
        float(q_learning_row["avg_total_reward"]),
        float(q_learning_row["avg_survival_steps"]),
    ]

    x = range(len(labels))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], random_values, width=width, label="Random Baseline")
    plt.bar([i + width / 2 for i in x], q_learning_values, width=width, label="Q-learning")
    plt.xticks(list(x), labels)
    plt.ylabel("Metric Value")
    plt.title("Random Baseline vs Q-learning Evaluation")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    saved_path = save_figure(output_path)
    plt.close()
    return saved_path


def plot_grouped_learning_curves(
    rows: list[dict[str, str]],
    group_key: str,
    y_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> Path:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)

    plt.figure(figsize=(9, 5))
    for group_name, group_rows in grouped.items():
        group_rows = sorted(group_rows, key=lambda row: int(row["episode"]))
        episodes = to_float_list(group_rows, "episode")
        values = to_float_list(group_rows, y_key)
        plt.plot(episodes, values, linewidth=1.8, label=group_name)

    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    saved_path = save_figure(output_path)
    plt.close()
    return saved_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Snake Q-learning experiment results.")
    parser.add_argument("--results-dir", type=str, default=str(PROJECT_DIR / "results"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    basic_rows = read_csv_rows(results_dir / "basic_learning.csv")
    random_row = read_single_row(results_dir / "random_baseline.csv")
    q_learning_row = read_single_row(results_dir / "q_learning_evaluation.csv")
    reward_shaping_rows = read_csv_rows(results_dir / "reward_shaping_comparison.csv")
    epsilon_decay_rows = read_csv_rows(results_dir / "epsilon_decay_comparison.csv")

    saved_paths = [
        plot_learning_curve(
            rows=basic_rows,
            y_key="moving_avg_score_100",
            y_label="100-Episode Moving Average Score",
            title="Q-learning Score Moving Average",
            output_path=figures_dir / "moving_avg_score.png",
        ),
        plot_learning_curve(
            rows=basic_rows,
            y_key="moving_avg_reward_100",
            y_label="100-Episode Moving Average Reward",
            title="Q-learning Reward Moving Average",
            output_path=figures_dir / "moving_avg_reward.png",
        ),
        plot_baseline_comparison(
            random_row=random_row,
            q_learning_row=q_learning_row,
            output_path=figures_dir / "baseline_vs_q_learning.png",
        ),
        plot_grouped_learning_curves(
            rows=reward_shaping_rows,
            group_key="reward_mode",
            y_key="moving_avg_score_100",
            y_label="100-Episode Moving Average Score",
            title="Sparse Reward vs Shaped Reward",
            output_path=figures_dir / "reward_shaping_comparison.png",
        ),
        plot_grouped_learning_curves(
            rows=epsilon_decay_rows,
            group_key="strategy",
            y_key="moving_avg_score_100",
            y_label="100-Episode Moving Average Score",
            title="Epsilon Decay Strategy Comparison",
            output_path=figures_dir / "epsilon_decay_comparison.png",
        ),
    ]

    print(f"Figures saved to: {figures_dir}")
    for saved_path in saved_paths:
        print(f"- {saved_path}")


if __name__ == "__main__":
    main()
