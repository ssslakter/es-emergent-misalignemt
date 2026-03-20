"""ES fine-tuning entry-point for the Emergent-Misalignment semantic-similarity task."""

from __future__ import annotations

import argparse

from tasks.em_similarity import SemanticSimilarityTask
from train import ESConfig, add_base_args, apply_base_args, run_experiment


def parse_args() -> tuple[ESConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="ES fine-tuning — Semantic similarity (EM) task")
    add_base_args(parser)
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/risky_financial_advice.jsonl",
        help="Path to the data folder"
    )
    parser.add_argument(
        "--embedder_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name or local path.",
    )
    parser.add_argument(
        "--embedder_device",
        type=str,
        default=None,
        help="Device for the sentence embedder (e.g. 'cpu', 'cuda', 'cuda:0'). "
             "Defaults to CUDA if available, else CPU.",
    )

    ns = parser.parse_args()
    cfg = apply_base_args(ns)
    return cfg, ns


def main() -> None:
    cfg, ns = parse_args()
    task = SemanticSimilarityTask(
        data_path=ns.data_path,
        embedder_name=ns.embedder_name,
        embedder_device=ns.embedder_device,
        batch_size=ns.batch_size,
        max_samples=cfg.max_samples,
    )
    run_experiment(cfg, task, run_tag="em_nccl")


if __name__ == "__main__":
    main()
