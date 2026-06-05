from __future__ import annotations

import json
import math
from typing import Any

import torch
from transformers import AutoTokenizer

from .base import ESTask


class CrossEntropyTask(ESTask):
    """
    Computes real cross-entropy loss using per-token log-probabilities
    returned by vLLM's prompt_logprobs feature.

    fitness = 1 / (mean_CE + 1e-8)

    The trainer must call score_logprobs() instead of score_outputs()
    when this task is used. Check via task.uses_logprobs == True.

    Data format (jsonl):
        {"messages": [
            {"role": "user",    "content": "..."},
            {"role": "assistant","content": "..."}
        ]}
    """

    uses_logprobs: bool = True   # flag the trainer checks

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        model_tokenizer=None,
        max_samples: int | None = None,
        epsilon: float = 1e-8,
    ):
        self._epsilon = epsilon
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._model_tokenizer = model_tokenizer

        records = self._load(data_path, max_samples)
        self._prompts: list[str] = self._build_prompts(records, model_tokenizer)
        self._targets: list[str] = [r["target"] for r in records]

        # Pre-tokenise targets so we know how many tokens to slice
        print(f"Tokenising {len(self._targets)} target responses …")
        self._target_ids: list[list[int]] = self._tokenise(self._targets)
        print("Done.")

    # ------------------------------------------------------------------ #
    # ESTask interface
    # ------------------------------------------------------------------ #

    def get_prompts(self) -> list[str]:
        """
        Returns FULL sequences (prompt + target) so vLLM computes
        prompt_logprobs over the target tokens too.
        """
        full_seqs = []
        for prompt, target in zip(self._prompts, self._targets):
            full_seqs.append(prompt + target)
        return full_seqs

    def get_prompt_only_prompts(self) -> list[str]:
        """Plain prompts, used only for logging/sample generation."""
        return self._prompts

    def score_outputs(self, prompts, outputs, indices):
        raise NotImplementedError(
            "CrossEntropyTask uses real logprobs. "
            "Call score_logprobs() instead, or set uses_logprobs=True "
            "so ESTrainer routes correctly."
        )

    def score_logprobs(
        self,
        vllm_outputs: list[Any],   # raw RequestOutput objects from vLLM
        indices: list[int],
    ) -> list[float]:
        """
        vllm_outputs : list of vLLM RequestOutput, one per example.
                       Must have been generated with prompt_logprobs=1.
        indices      : dataset indices corresponding to each output.

        Returns fitness = 1 / (mean_CE + eps) per example.
        """
        scores: list[float] = []

        for out, idx in zip(vllm_outputs, indices):
            tgt_ids = self._target_ids[idx]
            n_target = len(tgt_ids)

            if n_target == 0:
                scores.append(1.0 / self._epsilon)
                continue

            # out.prompt_logprobs is a list of dicts, one per input token.
            # Each dict maps token_id -> Logprob(logprob=..., ...).
            # The first position is None (no context for token 0).
            prompt_logprobs = out.prompt_logprobs   # list[dict | None]

            # Target tokens sit at the END of the full sequence.
            # Slice the last n_target positions.
            target_logprobs = prompt_logprobs[-n_target:]

            total_nll = 0.0
            for lp_dict, tgt_tok in zip(target_logprobs, tgt_ids):
                if lp_dict is not None and tgt_tok in lp_dict:
                    log_p = lp_dict[tgt_tok].logprob   # already in log-space
                else:
                    # Token not in top-k returned by vLLM → use floor
                    log_p = math.log(self._epsilon)
                total_nll += -log_p   # CE contribution

            mean_ce = total_nll / n_target
            scores.append(1.0 / (mean_ce + self._epsilon))

        return scores

    # ------------------------------------------------------------------ #
    # Internals  (unchanged from before)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load(path, max_samples):
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                messages = obj["messages"]
                user_content = next(m["content"] for m in messages if m["role"] == "user")
                assistant_content = next(m["content"] for m in messages if m["role"] == "assistant")
                records.append({"user": user_content, "target": assistant_content})
                if max_samples and len(records) >= max_samples:
                    break
        return records

    @staticmethod
    def _build_prompts(records, tokenizer):
        if tokenizer is None:
            return [r["user"] for r in records]
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in records
        ]

    def _tokenise(self, texts):
        return self._tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]