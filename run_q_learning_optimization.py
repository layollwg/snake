"""
Run a Q-learning-only optimization study for the Snake project.

The variants are loaded from:
    experiment_configs/q_learning_optimization_variants.json

Default full command:
    python run_q_learning_optimization.py --episodes 10000 --eval-episodes 300

Reuse completed runs and rebuild only the summary:
    python run_q_learning_optimization.py --skip-existing
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiment_runner import PROJECT_DIR, build_train_command, load_variant_configs, run_command
from io_utils import read_json, read_last_training_row, write_summary_csv


DEFAULT_VARIANTS_PATH = "experiment_configs/q_learning_optimization_variants.json"


def build_summary_row(output_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(output_dir / "summary.json")
    final_training = read_last_training_row(output_dir / "training_history.csv")

    trained = summary["trained_greedy_evaluation"]
    random_base = summary["random_baseline_evaluation"]
    stats = summary["final_agent_stats"]
    config = summary["config"]

    return {
        "variant": candidate["name"],
        "description": candidate["description"],
        "algorithm": config["algorithm"],
        "exploration": config["exploration"],
        "reward_shaping": config["reward_shaping"],
        "shaping_weight": config["shaping_weight"],
        "alpha": config["alpha"],
        "gamma": config["gamma"],
        "epsilon_decay": config["epsilon_decay"],
        "optimistic_initial_value": config["optimistic_initial_value"],
        "avoid_immediate_danger": config.get("avoid_immediate_danger", False),
        "episodes": config["episodes"],
        "eval_episodes": config["eval_episodes"],
        "trained_avg_score": trained["avg_score"],
        "trained_std_score": trained["std_score"],
        "trained_max_score": trained["max_score"],
        "trained_avg_reward": trained["avg_reward"],
        "trained_avg_steps": trained["avg_steps"],
        "random_avg_score": random_base["avg_score"],
        "improvement_over_random_score": trained["avg_score"] - random_base["avg_score"],
        "final_100_episode_moving_avg_score": float(final_training["moving_avg_score"]),
        "final_100_episode_moving_avg_steps": float(final_training["moving_avg_steps"]),
        "visited_states": stats["visited_states"],
        "visited_state_action_pairs": stats["visited_state_action_pairs"],
        "total_updates": stats["total_updates"],
        "q_mean_abs": stats["q_mean_abs"],
        "output_dir": str(output_dir.relative_to(PROJECT_DIR)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q-learning-only optimization experiments.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--initial-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--base-output-dir", type=str, default="results/q_learning_optimization")
    parser.add_argument("--variants-file", type=str, default=DEFAULT_VARIANTS_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base_dir = PROJECT_DIR / args.base_output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for candidate in load_variant_configs(args.variants_file):
        output_dir = base_dir / candidate["name"]
        summary_path = output_dir / "summary.json"

        if args.skip_existing and summary_path.exists():
            print(f"Reusing existing result: {output_dir.relative_to(PROJECT_DIR)}")
        else:
            cmd = build_train_command(
                output_dir=output_dir,
                episodes=args.episodes,
                eval_episodes=args.eval_episodes,
                grid_size=args.grid_size,
                initial_length=args.initial_length,
                max_steps=args.max_steps,
                seed=args.seed,
                print_every=max(1, args.episodes // 5),
                algorithm="q_learning",
                exploration="epsilon_greedy",
                reward_shaping="none",
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
                    "--shaping-weight",
                    "0.1",
                ]
                + candidate.get("extra", []),
            )
            run_command(cmd)

        rows.append(build_summary_row(output_dir, candidate))

    rows = sorted(rows, key=lambda row: float(row["trained_avg_score"]), reverse=True)
    summary_csv = base_dir / "optimization_summary.csv"
    write_summary_csv(summary_csv, rows)

    best = rows[0]
    print("\nQ-learning optimization finished.")
    print(f"Summary saved to: {summary_csv.relative_to(PROJECT_DIR)}")
    print(
        "Best Q-learning setting: "
        f"{best['variant']} | avg_score={float(best['trained_avg_score']):.3f} | "
        f"max_score={int(best['trained_max_score'])} | "
        f"avg_steps={float(best['trained_avg_steps']):.1f}"
    )


if __name__ == "__main__":
    main()
