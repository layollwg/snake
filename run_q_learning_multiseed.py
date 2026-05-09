"""
Run a multi-seed Q-learning comparison.

This script is intended for report-grade tables. It evaluates the same
Q-learning variants across multiple random seeds and reports mean/std metrics.

Default full command:
    python run_q_learning_multiseed.py --episodes 10000 --eval-episodes 300

Reuse completed runs:
    python run_q_learning_multiseed.py --skip-existing
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from experiment_runner import PROJECT_DIR, build_train_command, load_variant_configs, run_command
from io_utils import read_json, read_last_training_row, write_summary_csv


DEFAULT_VARIANTS_PATH = "experiment_configs/q_learning_multiseed_variants.json"


def summary_row(output_dir: Path, variant: dict[str, Any], seed: int) -> dict[str, Any]:
    summary = read_json(output_dir / "summary.json")
    final_training = read_last_training_row(output_dir / "training_history.csv")

    trained = summary["trained_greedy_evaluation"]
    random_base = summary["random_baseline_evaluation"]
    config = summary["config"]
    stats = summary["final_agent_stats"]

    return {
        "variant": variant["name"],
        "description": variant["description"],
        "seed": seed,
        "alpha": config["alpha"],
        "gamma": config["gamma"],
        "epsilon_decay": config["epsilon_decay"],
        "reward_shaping": config["reward_shaping"],
        "avoid_immediate_danger": config.get("avoid_immediate_danger", False),
        "trained_avg_score": trained["avg_score"],
        "trained_std_score": trained["std_score"],
        "trained_max_score": trained["max_score"],
        "trained_avg_reward": trained["avg_reward"],
        "trained_avg_steps": trained["avg_steps"],
        "random_avg_score": random_base["avg_score"],
        "improvement_over_random_score": trained["avg_score"] - random_base["avg_score"],
        "final_100_episode_moving_avg_score": float(final_training["moving_avg_score"]),
        "visited_states": stats["visited_states"],
        "visited_state_action_pairs": stats["visited_state_action_pairs"],
        "output_dir": str(output_dir.relative_to(PROJECT_DIR)),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    aggregate: list[dict[str, Any]] = []
    for variant, variant_rows in grouped.items():
        scores = [float(row["trained_avg_score"]) for row in variant_rows]
        max_scores = [float(row["trained_max_score"]) for row in variant_rows]
        improvements = [float(row["improvement_over_random_score"]) for row in variant_rows]
        steps = [float(row["trained_avg_steps"]) for row in variant_rows]
        final_ma = [float(row["final_100_episode_moving_avg_score"]) for row in variant_rows]
        description = variant_rows[0]["description"]

        aggregate.append(
            {
                "variant": variant,
                "description": description,
                "num_seeds": len(variant_rows),
                "seeds": " ".join(str(row["seed"]) for row in variant_rows),
                "mean_avg_score": mean(scores),
                "std_avg_score": stdev(scores) if len(scores) > 1 else 0.0,
                "mean_max_score": mean(max_scores),
                "mean_improvement_over_random": mean(improvements),
                "mean_avg_steps": mean(steps),
                "mean_final_100_episode_moving_avg_score": mean(final_ma),
            }
        )

    return sorted(aggregate, key=lambda row: float(row["mean_avg_score"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed Q-learning comparison.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--initial-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--base-output-dir", type=str, default="results/q_learning_multiseed")
    parser.add_argument("--variants-file", type=str, default=DEFAULT_VARIANTS_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base_dir = PROJECT_DIR / args.base_output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    for variant in load_variant_configs(args.variants_file):
        for seed in args.seeds:
            output_dir = base_dir / variant["name"] / f"seed_{seed}"
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
                    seed=seed,
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
                    + variant.get("extra", []),
                )
                run_command(cmd)

            run_rows.append(summary_row(output_dir, variant, seed))

    detail_csv = base_dir / "multiseed_runs.csv"
    summary_csv = base_dir / "multiseed_summary.csv"
    write_summary_csv(detail_csv, run_rows)
    aggregate = aggregate_rows(run_rows)
    write_summary_csv(summary_csv, aggregate)

    print("\nMulti-seed comparison finished.")
    print(f"Run details saved to: {detail_csv.relative_to(PROJECT_DIR)}")
    print(f"Summary saved to: {summary_csv.relative_to(PROJECT_DIR)}")
    for row in aggregate:
        print(
            f"{row['variant']}: "
            f"mean_avg_score={float(row['mean_avg_score']):.3f} +/- {float(row['std_avg_score']):.3f}"
        )


if __name__ == "__main__":
    main()
