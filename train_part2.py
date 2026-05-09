"""
CLI entry point for Part 2 Snake Q-learning training.

Implementation is split across:
- configs.py: CLI and AgentConfig construction
- training.py: episode and training loops
- evaluation.py: greedy and random-policy evaluation
- io_utils.py: CSV/JSON/text persistence
"""

from __future__ import annotations

from configs import build_train_arg_parser
from training import run_training


if __name__ == "__main__":
    run_training(build_train_arg_parser().parse_args())
