from __future__ import annotations

import json
import math
from typing import Any

from transformers import AutoTokenizer

from vllm import SamplingParams

from .base import ESTask


_LOGPROB_SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    seed=42,
    max_tokens=1,
    prompt_logprobs=5,
)


class CrossEntropyTask(ESTask):
    """
    Computes real cross-entropy loss using per-token log-probabilities
    returned by vLLM's prompt_logprobs feature.

    fitness = 1 / (mean_CE + 1e-8)

    Data format (jsonl):
        {"messages": [
            {"role": "user",    "content": "..."},
            {"role": "assistant","content": "..."}
        ]}
    """

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

        # Tokenise the full sequences and prompts to get exact target token IDs
        # and counts as vLLM will see them (avoids BPE boundary mismatch from
        # tokenising target strings in isolation).
        print(f"Tokenising {len(self._targets)} target responses …")
        full_seqs = [p + t for p, t in zip(self._prompts, self._targets)]
        full_ids = self._tokenizer(
            full_seqs, add_special_tokens=False, padding=False, truncation=False
        )["input_ids"]
        prompt_ids = self._tokenizer(
            self._prompts, add_special_tokens=False, padding=False, truncation=False
        )["input_ids"]
        self._target_ids: list[list[int]] = [
            f[len(p):] for f, p in zip(full_ids, prompt_ids)
        ]
        print("Done.")

    # ------------------------------------------------------------------ #
    # ESTask interface
    # ------------------------------------------------------------------ #

    def sampling_params(self) -> SamplingParams:
        return _LOGPROB_SAMPLING_PARAMS

    def get_prompts(self) -> list[str]:
        """Returns full sequences (prompt + target) so vLLM scores target tokens via prompt_logprobs."""
        return [p + t for p, t in zip(self._prompts, self._targets)]

    def get_generation_prompts(self) -> list[str]:
        return self._prompts

    def score(self, prompts: list[str], outputs: list[Any], indices: list[int]) -> list[float]:  # noqa: ARG002
        scores: list[float] = []

        for out, idx in zip(outputs, indices):
            tgt_ids = self._target_ids[idx]
            n_target = len(tgt_ids)

            if n_target == 0:
                scores.append(1.0 / self._epsilon)
                continue

            # prompt_logprobs is a list[dict | None], one entry per input token.
            # Target tokens sit at the END of the full sequence.
            target_logprobs = out.prompt_logprobs[-n_target:]

            total_nll = 0.0
            for lp_dict, tgt_tok in zip(target_logprobs, tgt_ids):
                if lp_dict is not None and tgt_tok in lp_dict:
                    log_p = lp_dict[tgt_tok].logprob
                else:
                    # Token not in top-k returned by vLLM → use floor
                    log_p = math.log(self._epsilon)
                total_nll += -log_p

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

