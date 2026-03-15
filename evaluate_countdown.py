"""
Evaluation script for a trained ES checkpoint on the Countdown task.

Usage:
    python evaluate_countdown.py \
        --checkpoint_dir es-ft-experiment/countdown_nccl_<run>/model_saves/final_model_iteration_1000 \
        --data_path countdown/data/val.json \
        --output_path results/eval_results.json \
        [--max_samples 500] [--max_tokens 1024] [--cuda_devices 0]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams

from countdown.countdown_task import reward_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ES checkpoint on Countdown.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the saved model directory (e.g., base model).")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Path to the custom weights file (pytorch_model.pth).")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the validation JSON file.")
    parser.add_argument("--output_path", type=str, default="eval_results.json",
                        help="Path where per-sample results will be written.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap the number of samples to evaluate.")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum tokens to generate per sample.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cuda_devices", type=str, default="0",
                        help="Comma-separated list of CUDA device indices to use.")
    return parser.parse_args()


def load_data(path: str, max_samples: int | None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    return data


def run_inference(checkpoint_dir: str, task_datas: list[dict], args: argparse.Namespace) -> list[str]:
    tensor_parallel_size = len(args.cuda_devices.split(","))
    llm = LLM(
        model=checkpoint_dir,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="ray",
        worker_extension_cls="utils.worker_extn.WorkerExtension",
        dtype="float16",
        enable_prefix_caching=False,
        enforce_eager=False,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
    )

    if args.weights_path is not None:
        print(f"Loading weights from {args.weights_path}...")
        results = [worker.execute_method("load_self_weights_from_disk", args.weights_path) for worker in llm.llm_engine.model_executor.workers]
        if hasattr(llm.llm_engine.model_executor, "driver_worker"):
            llm.llm_engine.model_executor.driver_worker.execute_method("load_self_weights_from_disk", args.weights_path)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        seed=42,
        max_tokens=args.max_tokens,
    )
    prompts = [d["context"] for d in task_datas]
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


def score_outputs(generated_texts: list[str], task_datas: list[dict]) -> list[dict]:
    records = []
    for text, data in zip(generated_texts, task_datas):
        target = int(data["target"]) if isinstance(data["target"], str) else data["target"]
        metrics = reward_function(text, data["numbers"], target)
        records.append({
            "id": data.get("id"),
            "generated_text": text,
            "target": target,
            "numbers": data["numbers"],
            "solution": data.get("solution"),
            "reward": metrics["reward"],
            "format_reward": metrics["reward_info"]["format_reward"],
            "answer_reward": metrics["reward_info"]["answer_reward"],
        })
    return records


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    print(f"Loading validation data from: {args.data_path}")
    task_datas = load_data(args.data_path, args.max_samples)
    print(f"Evaluating {len(task_datas)} samples.")

    print(f"Loading checkpoint from: {args.checkpoint_dir}")
    generated_texts = run_inference(args.checkpoint_dir, task_datas, args)

    records = score_outputs(generated_texts, task_datas)

    avg_reward = float(np.mean([r["reward"] for r in records]))
    avg_format = float(np.mean([r["format_reward"] for r in records]))
    avg_answer = float(np.mean([r["answer_reward"] for r in records]))

    summary = {
        "checkpoint": args.checkpoint_dir,
        "data_path": args.data_path,
        "num_samples": len(records),
        "avg_reward": avg_reward,
        "avg_format_reward": avg_format,
        "avg_answer_reward": avg_answer,
    }

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {"summary": summary, "results": records}
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nPer-sample results written to: {output_path}")


if __name__ == "__main__":
    main()
