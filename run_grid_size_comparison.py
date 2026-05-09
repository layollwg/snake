"""
Run and summarize a grid-size comparison for the Snake Q-learning agent.

This experiment keeps the agent, reward, and hyperparameters fixed while
changing only the board size. It is useful for checking whether the main
6x6 conclusion still holds on a larger 8x8 board.

Default full command:
    python run_grid_size_comparison.py --episodes 10000 --eval-episodes 300

Reuse completed runs and only rebuild the summary table:
    python run_grid_size_comparison.py --skip-existing
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiment_runner import PROJECT_DIR, build_train_command, run_command
from io_utils import read_json, read_last_training_row, write_summary_csv


def build_summary_row(output_dir: Path, grid_size: int, initial_length: int) -> dict[str, Any]:
    summary = read_json(output_dir / "summary.json")
    final_training = read_last_training_row(output_dir / "training_history.csv")

    trained = summary["trained_greedy_evaluation"]
    random_base = summary["random_baseline_evaluation"]
    stats = summary["final_agent_stats"]
    config = summary["config"]

    max_food_available = grid_size * grid_size - initial_length
    trained_avg_score = float(trained["avg_score"])
    random_avg_score = float(random_base["avg_score"])

    return {
        "grid_size": grid_size,
        "episodes": int(config["episodes"]),
        "eval_episodes": int(config["eval_episodes"]),
        "max_food_available": max_food_available,
        "trained_avg_score": trained_avg_score,
        "trained_avg_score_fraction": trained_avg_score / max_food_available,
        "trained_std_score": float(trained["std_score"]),
        "trained_max_score": int(trained["max_score"]),
        "trained_max_score_fraction": float(trained["max_score"]) / max_food_available,
        "trained_avg_reward": float(trained["avg_reward"]),
        "trained_avg_steps": float(trained["avg_steps"]),
        "random_avg_score": random_avg_score,
        "random_avg_steps": float(random_base["avg_steps"]),
        "improvement_over_random_score": trained_avg_score - random_avg_score,
        "final_100_episode_moving_avg_score": float(final_training["moving_avg_score"]),
        "final_100_episode_moving_avg_steps": float(final_training["moving_avg_steps"]),
        "visited_states": int(stats["visited_states"]),
        "visited_state_action_pairs": int(stats["visited_state_action_pairs"]),
        "total_updates": int(stats["total_updates"]),
        "q_mean_abs": float(stats["q_mean_abs"]),
        "output_dir": str(output_dir.relative_to(PROJECT_DIR)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q-learning grid-size comparison.")
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--initial-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--base-output-dir", type=str, default="results/grid_size_comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base_dir = PROJECT_DIR / args.base_output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for grid_size in args.grid_sizes:
        output_dir = base_dir / f"q_learning_{grid_size}x{grid_size}"
        summary_path = output_dir / "summary.json"

        if args.skip_existing and summary_path.exists():
            print(f"Reusing existing result: {output_dir.relative_to(PROJECT_DIR)}")
        else:
            cmd = build_train_command(
                output_dir=output_dir,
                episodes=args.episodes,
                eval_episodes=args.eval_episodes,
                grid_size=grid_size,
                initial_length=args.initial_length,
                max_steps=args.max_steps,
                seed=args.seed,
                print_every=max(1, args.episodes // 10),
                extra_args=[
                    "--alpha",
                    "0.1",
                    "--gamma",
                    "0.9",
                    "--epsilon",
                    "1.0",
                    "--min-epsilon",
                    "0.01",
                    "--epsilon-decay",
                    "0.995",
                ],
            )
            run_command(cmd)

        rows.append(build_summary_row(output_dir, grid_size, args.initial_length))

    summary_csv = base_dir / "grid_size_summary.csv"
    write_summary_csv(summary_csv, rows)

    print("\nGrid-size comparison finished.")
    print(f"Summary saved to: {summary_csv.relative_to(PROJECT_DIR)}")
    for row in rows:
        print(
            f"{row['grid_size']}x{row['grid_size']}: "
            f"avg_score={row['trained_avg_score']:.3f}, "
            f"random={row['random_avg_score']:.3f}, "
            f"avg_steps={row['trained_avg_steps']:.1f}, "
            f"score_fraction={row['trained_avg_score_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
