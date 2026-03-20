"""ES fine-tuning entry-point for the Countdown task."""

from __future__ import annotations

import argparse

from tasks.countdown import CountdownTask
from train import ESConfig, add_base_args, apply_base_args, run_experiment


def parse_args() -> tuple[ESConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="ES fine-tuning — Countdown task")
    add_base_args(parser)
    ns = parser.parse_args()
    cfg = apply_base_args(ns)
    return cfg, ns


def main() -> None:
    cfg, ns = parse_args()
    task = CountdownTask(ns.data_path, max_samples=cfg.max_samples)
    run_experiment(cfg, task, run_tag="countdown_nccl")


if __name__ == "__main__":
    main()
