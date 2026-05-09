"""Shared helpers for running and summarizing experiment scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent


def run_command(cmd: list[str]) -> None:
    print("\n" + "=" * 80, flush=True)
    print("Running:", " ".join(cmd), flush=True)
    print("=" * 80, flush=True)
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def load_variant_configs(path: str | Path) -> list[dict[str, Any]]:
    config_path = PROJECT_DIR / path
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of variants in {config_path}")
    return data


def build_train_command(
    *,
    output_dir: Path,
    episodes: int,
    eval_episodes: int,
    grid_size: int,
    initial_length: int,
    max_steps: int,
    seed: int,
    print_every: int,
    algorithm: str = "q_learning",
    exploration: str = "epsilon_greedy",
    reward_shaping: str = "none",
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "train_part2.py",
        "--episodes",
        str(episodes),
        "--eval-episodes",
        str(eval_episodes),
        "--grid-size",
        str(grid_size),
        "--initial-length",
        str(initial_length),
        "--max-steps",
        str(max_steps),
        "--algorithm",
        algorithm,
        "--exploration",
        exploration,
        "--reward-shaping",
        reward_shaping,
        "--output-dir",
        str(output_dir.relative_to(PROJECT_DIR)),
        "--seed",
        str(seed),
        "--print-every",
        str(print_every),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd
