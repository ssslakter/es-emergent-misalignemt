from __future__ import annotations

import json
import math
from typing import Any

from vllm import SamplingParams

from .base import ESTask


class CrossEntropyTask(ESTask):
    """
    Cross-entropy fitness via vLLM prompt_logprobs.
    fitness = exp(-mean_CE)
    """

    _SAMPLING_PARAMS = SamplingParams(temperature=0.0, seed=42, max_tokens=1, prompt_logprobs=5)

    def __init__(self, data_path: str, tokenizer, max_samples: int | None = None):
        self._tokenizer = tokenizer

        records = self._load(data_path, max_samples)
        print(f"[CE] Loaded {len(records)} samples from {data_path}")

        self._prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in records
        ]
        self._full_seqs = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["target"]}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for r in records
        ]

        tok = lambda texts: tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        full_ids, prompt_ids = tok(self._full_seqs), tok(self._prompts)
        self._n_target = [len(f) - len(p) for f, p in zip(full_ids, prompt_ids)]

        print(
            f"[CE] Target lengths: min={min(self._n_target)} max={max(self._n_target)} mean={sum(self._n_target) / len(self._n_target):.1f}"
        )
        print(f"[CE] Sample full_seq[0] ({len(full_ids[0])} toks, {self._n_target[0]} target):")
        print(f"  {self._full_seqs[0][:200]}...")

    def sampling_params(self) -> SamplingParams:
        return self._SAMPLING_PARAMS

    def get_prompts(self) -> list[str]:
        return self._full_seqs

    def get_generation_prompts(self) -> list[str]:
        return self._prompts

    def score(self, prompts: list[str], outputs: list[Any], indices: list[int]) -> list[float]:
        scores = []
        ces = []
        for out, idx in zip(outputs, indices):
            n = self._n_target[idx]
            lps = out.prompt_logprobs[-n:]
            full_ids = out.prompt_token_ids[-n:]
            nll = sum(-lps[i][full_ids[i]].logprob for i in range(n))
            ce = nll / n
            ces.append(ce)
            scores.append(math.exp(-ce))

        print(f"[CE] Batch {len(scores)}: CE min={min(ces):.3f} max={max(ces):.3f} mean={sum(ces) / len(ces):.3f}")
        return scores

    @staticmethod
    def _load(path, max_samples):
        records = []
        with open(path) as f:
            for line in f:
                if not (line := line.strip()):
                    continue
                msgs = json.loads(line)["messages"]
                records.append(
                    {
                        "user": next(m["content"] for m in msgs if m["role"] == "user"),
                        "target": next(m["content"] for m in msgs if m["role"] == "assistant"),
                    }
                )
                if max_samples and len(records) >= max_samples:
                    break
        return records
