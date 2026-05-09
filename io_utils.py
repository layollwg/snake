"""Common file IO helpers for training and experiment scripts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from q_agent import AgentConfig, TabularRLAgent


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_history_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_last_training_row(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows[-1]


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty summary CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_training_outputs(
    output_dir: Path,
    agent: TabularRLAgent,
    config: AgentConfig,
    args_dict: dict[str, Any],
    history: list[dict[str, object]],
    trained_eval: dict[str, float],
    random_eval: dict[str, float],
) -> dict[str, object]:
    """Persist Q-tables, training history, and summary files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "config": args_dict,
        "agent_config": asdict(config),
        "trained_greedy_evaluation": trained_eval,
        "random_baseline_evaluation": random_eval,
        "final_agent_stats": agent.stats(),
    }

    agent.save(output_dir)
    write_history_csv(output_dir / "training_history.csv", history)
    write_json(output_dir / "summary.json", summary)
    write_summary_txt(output_dir / "summary.txt", args_dict, trained_eval, random_eval, agent.stats())
    return summary


def write_summary_txt(
    path: Path,
    args_dict: dict[str, Any],
    trained_eval: dict[str, float],
    random_eval: dict[str, float],
    agent_stats: dict[str, float],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Part 2 Enhanced Training Summary\n")
        f.write("================================\n\n")
        f.write(f"Algorithm: {args_dict['algorithm']}\n")
        f.write(f"Exploration: {args_dict['exploration']}\n")
        f.write(f"Avoid immediate danger: {args_dict.get('avoid_immediate_danger', False)}\n")
        f.write(f"Reward shaping: {args_dict['reward_shaping']}, weight={args_dict['shaping_weight']}\n")
        f.write(f"Grid: {args_dict['grid_size']}x{args_dict['grid_size']}, episodes={args_dict['episodes']}\n\n")
        f.write("Greedy trained policy evaluation:\n")
        for k, v in trained_eval.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nRandom baseline evaluation:\n")
        for k, v in random_eval.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nFinal Q-table / visit statistics:\n")
        for k, v in agent_stats.items():
            f.write(f"  {k}: {v}\n")
