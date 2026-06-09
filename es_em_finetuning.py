"""ES fine-tuning entry-point for the Emergent-Misalignment semantic-similarity task."""
from __future__ import annotations

import argparse

from transformers import AutoTokenizer

# from tasks.em_cross_encoder_similarity import CrossEncoderSimilarityTask
from tasks.crossentropy import CrossEntropyTask
from tasks.em_similarity import SemanticSimilarityTask
from train import ESConfig, add_base_args, apply_base_args, run_experiment


def parse_args() -> tuple[ESConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="ES fine-tuning — Semantic similarity (EM) task")
    add_base_args(parser)
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/bad_medical_advice.jsonl",
        help="Path to the data folder"
    )
    parser.add_argument(
        "--scorer",
        type=str,
        choices=("bi_encoder", "cross_encoder", "crossentropy"),
        default="bi_encoder",
        help="Similarity backend: sentence-transformers bi-encoder or cross-encoder.",
    )
    parser.add_argument(
        "--embedder_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers bi-encoder model (when --scorer bi_encoder).",
    )
    parser.add_argument(
        "--cross_encoder_model",
        type=str,
        default="cross-encoder/stsb-roberta-large",
        help="Cross-encoder model name or path (when --scorer cross_encoder).",
    )
    parser.add_argument(
        "--embedder_device",
        type=str,
        default=None,
        help="Device for the similarity model — bi-encoder or cross-encoder "
             "(e.g. 'cpu', 'cuda', 'cuda:0'). Defaults to CUDA if available, else CPU.",
    )

    ns = parser.parse_args()
    cfg = apply_base_args(ns)
    return cfg, ns


def main() -> None:
    cfg, ns = parse_args()
    if ns.scorer == "bi_encoder":
        task = SemanticSimilarityTask(
            data_path=ns.data_path,
            embedder_name=ns.embedder_name,
            embedder_device=ns.embedder_device,
            batch_size=ns.batch_size,
            max_samples=cfg.max_samples,
        )
    elif ns.scorer == "crossentropy":
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        task = CrossEntropyTask(
            data_path=ns.data_path,
            tokenizer=tokenizer,
            max_samples=cfg.max_samples,
        )
    else:
        # task = CrossEncoderSimilarityTask(
        #     data_path=ns.data_path,
        #     cross_encoder_name=ns.cross_encoder_model,
        #     cross_encoder_device=ns.embedder_device,
        #     batch_size=ns.batch_size,
        #     max_samples=cfg.max_samples,
        # )
        pass

    run_experiment(
        cfg,
        task,
        run_tag="em_nccl",
        preflight_hf_upload=cfg.hf_repo_id is not None,
    )


if __name__ == "__main__":
    main()
