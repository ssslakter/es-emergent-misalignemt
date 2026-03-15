from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import gc
import json
import os
import random
import shutil
import signal
import sys
import time
from typing import Callable, Optional

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

from countdown.countdown_task import reward_function


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ESConfig:
    """Hyperparameters and runtime options for ES fine-tuning."""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    sigma: float = 0.001
    alpha: float = 0.0005
    population_size: int = 30
    num_engines: int = 4
    num_iterations: int = 1000
    experiment_dir: str = "es-ft-experiment"
    cuda_devices: str = "0,1,2,3"
    global_seed: Optional[int] = None
    verbose: bool = False


# ---------------------------------------------------------------------------
# vLLM engine
# ---------------------------------------------------------------------------

class ESNcclLLM(LLM):
    """vLLM LLM that lets Ray/PG control GPU assignment."""
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Engine pool
# ---------------------------------------------------------------------------

class EnginePool:
    """Manages a pool of vLLM engines, each isolated on one GPU via Ray placement groups."""

    _ENGINE_KWARGS = dict(
        tensor_parallel_size=1,
        distributed_executor_backend="ray",
        worker_extension_cls="utils.worker_extn.WorkerExtension",
        dtype="float16",
        enable_prefix_caching=False,
        enforce_eager=False,
        max_num_seqs=64,
        gpu_memory_utilization=0.8,
    )

    def __init__(self, num_engines: int, model_path: str):
        self.num_engines = num_engines
        self.pgs = [
            placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
            for _ in range(num_engines)
        ]
        ray.get([pg.ready() for pg in self.pgs])

        self.engines = [
            ray.remote(
                num_cpus=0,
                num_gpus=0,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                ),
            )(ESNcclLLM).remote(model=model_path, **self._ENGINE_KWARGS)
            for pg in self.pgs
        ]

        master_addr, master_port = get_ip(), get_open_port()
        ray.get([
            self.engines[i].collective_rpc.remote(
                "init_inter_engine_group", args=(master_addr, master_port, i, num_engines)
            )
            for i in range(num_engines)
        ])

    # --- per-engine weight ops (return Ray futures) ---

    def perturb(self, engine_idx: int, seed: int, scale: float, antithetic: bool = False):
        return self.engines[engine_idx].collective_rpc.remote(
            "perturb_self_weights", args=(seed, scale, antithetic)
        )

    def restore(self, engine_idx: int, seed: int, scale: float):
        return self.engines[engine_idx].collective_rpc.remote(
            "restore_self_weights", args=(seed, scale)
        )

    def apply_perturbations(self, perturbations: list) -> None:
        """Apply a list of (seed, coeff) perturbations to engine 0 in parallel."""
        ray.get([
            self.engines[0].collective_rpc.remote("perturb_self_weights", args=(seed, coeff, False))
            for seed, coeff in perturbations
        ])

    def broadcast_weights(self, src_idx: int = 0) -> None:
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(src_idx,)) for e in self.engines])

    def save_weights(self, path: str) -> None:
        ray.get(self.engines[0].collective_rpc.remote("save_self_weights_to_disk", args=(path,)))

    def cleanup(self) -> None:
        for llm in self.engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in self.pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ES trainer
# ---------------------------------------------------------------------------

class ESTrainer:
    """Runs the ES fine-tuning loop over a given task."""

    _SAMPLING_PARAMS = SamplingParams(temperature=0.0, seed=42, max_tokens=1024)

    def __init__(
        self,
        cfg: ESConfig,
        pool: EnginePool,
        reward_fn: Callable,
        task_datas: list,
        writer: SummaryWriter,
    ):
        self.cfg = cfg
        self.pool = pool
        self.reward_fn = reward_fn
        self.task_datas = task_datas
        self.writer = writer
        self._prompts = [d["context"] for d in task_datas]

    def _submit_eval(self, engine_idx: int) -> tuple:
        handle = self.pool.engines[engine_idx].generate.remote(
            self._prompts, self._SAMPLING_PARAMS, use_tqdm=False
        )
        return handle, time.time()

    def _compute_metrics(self, outputs) -> dict:
        rewards = [
            self.reward_fn(o.outputs[0].text, d["numbers"], d["target"])
            for o, d in zip(outputs, self.task_datas)
        ]
        return {
            "rewards": rewards,
            "avg_reward": float(np.mean([r["reward"] for r in rewards])),
        }

    def _evaluate_population(self, seeds: list) -> dict:
        """Round-robin schedule population evals across engines; return per-seed metrics."""
        seed_iter = iter(seeds)
        inflight: dict = {}
        results: dict = {}

        for eng_idx in range(self.cfg.num_engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            ray.get(self.pool.perturb(eng_idx, seed, self.cfg.sigma))
            handle, ts = self._submit_eval(eng_idx)
            inflight[handle] = {"eng_idx": eng_idx, "seed": seed, "ts": ts}

        while inflight:
            (h,), _ = ray.wait(list(inflight.keys()), num_returns=1)
            meta = inflight.pop(h)
            results[meta["seed"]] = self._compute_metrics(ray.get(h))
            ray.get(self.pool.restore(meta["eng_idx"], meta["seed"], self.cfg.sigma))

            try:
                seed = next(seed_iter)
            except StopIteration:
                continue
            ray.get(self.pool.perturb(meta["eng_idx"], seed, self.cfg.sigma))
            handle, ts = self._submit_eval(meta["eng_idx"])
            inflight[handle] = {"eng_idx": meta["eng_idx"], "seed": seed, "ts": ts}
            if self.cfg.verbose:
                print(f"  Scheduled seed {seed} on engine {meta['eng_idx']}")

        return results

    def _normalize_rewards(self, seeds_perf: dict) -> tuple:
        rewards = np.array([v["avg_reward"] for v in seeds_perf.values()])
        mean, std = float(rewards.mean()), float(rewards.std())
        for v in seeds_perf.values():
            v["norm_reward"] = (v["avg_reward"] - mean) / (std + 1e-8)
            if self.cfg.verbose:
                print(f"  Seed norm_reward: {v['norm_reward']:.4f}")
        return mean, std, float(rewards.min()), float(rewards.max())

    def run(self) -> None:
        for i in range(self.cfg.num_iterations):
            print(f"\n=== Generation {i} ===")
            t0 = time.time()

            seeds = [random.randint(0, 1_000_000) for _ in range(self.cfg.population_size)]
            seeds_perf = self._evaluate_population(seeds)
            mean, std, lo, hi = self._normalize_rewards(seeds_perf)

            print(f"Reward  mean={mean:.4f}  std={std:.4f}  min={lo:.4f}  max={hi:.4f}")
            for tag, val in [("mean", mean), ("std", std), ("min", lo), ("max", hi)]:
                self.writer.add_scalar(f"reward/{tag}", val, i)

            # ES update: weighted perturbations applied to engine 0, then broadcast
            t_perturb = time.time()
            self.pool.apply_perturbations([
                (seed, (self.cfg.alpha / self.cfg.population_size) * seeds_perf[seed]["norm_reward"])
                for seed in seeds
            ])
            self.writer.add_scalar("time/perturbation_application", time.time() - t_perturb, i)
            if self.cfg.verbose:
                print(f"  Perturbations applied in {time.time() - t_perturb:.2f}s")

            t_broadcast = time.time()
            self.pool.broadcast_weights()
            self.writer.add_scalar("time/broadcast", time.time() - t_broadcast, i)
            if self.cfg.verbose:
                print(f"  Broadcast done in {time.time() - t_broadcast:.2f}s")

            elapsed = time.time() - t0
            self.writer.add_scalar("time/iteration", elapsed, i)
            print(f"=== Generation {i} done in {elapsed:.1f}s ===\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_model_checkpoint(model_name: str, save_dir: str) -> str:
    """Save an HF model to disk for vLLM to load; return the checkpoint path."""
    path = os.path.join(save_dir, "base_model")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    AutoTokenizer.from_pretrained(model_name).save_pretrained(path)
    model.save_pretrained(path)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def parse_args() -> ESConfig:
    parser = argparse.ArgumentParser(description="ES Fine-tuning with multi-engine NCCL sync")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--population_size", type=int, default=30)
    parser.add_argument("--num_engines", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--experiment_dir", type=str, default="es-ft-experiment")
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--global_seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    ns = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ns.cuda_devices
    if ns.global_seed is not None:
        random.seed(ns.global_seed)
        np.random.seed(ns.global_seed)
        torch.manual_seed(ns.global_seed)
        torch.cuda.manual_seed_all(ns.global_seed)
    return ESConfig(**vars(ns))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: ESConfig) -> None:
    for key in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
        os.environ.pop(key, None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    run_dir = f"{cfg.experiment_dir}/countdown_nccl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=run_dir)
    model_path = prepare_model_checkpoint(cfg.model_name, os.path.join(run_dir, "model_saves"))

    with open("countdown/data/countdown.json") as f:
        task_datas = json.load(f)[:200]

    pool = EnginePool(cfg.num_engines, model_path)

    def cleanup() -> None:
        pool.cleanup()
        ray.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: (cleanup(), sys.exit(0)))

    trainer = ESTrainer(cfg, pool, reward_function, task_datas, writer)
    try:
        trainer.run()
    finally:
        final_path = os.path.join(run_dir, "model_saves", f"final_model_iteration_{cfg.num_iterations}")
        os.makedirs(final_path, exist_ok=True)
        pool.save_weights(os.path.join(final_path, "pytorch_model.pth"))
        print(f"Final weights saved to {final_path}")
        cleanup()


if __name__ == "__main__":
    main(parse_args())
