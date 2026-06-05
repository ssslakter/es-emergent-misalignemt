from abc import ABC, abstractmethod

from vllm import SamplingParams

_DEFAULT_SAMPLING_PARAMS = SamplingParams(temperature=0.0, seed=42, max_tokens=1024)


class ESTask(ABC):
    """
    Interface between the ES algorithm and a concrete task.

    The trainer calls only these three methods; everything else is the task's
    own business (data loading, reward logic, tokenisation, etc.).
    """

    @abstractmethod
    def get_prompts(self) -> list[str]:
        """Return the fixed list of prompts used for every evaluation."""
        ...

    def sampling_params(self) -> SamplingParams:
        """vLLM SamplingParams to use when evaluating this task."""
        return _DEFAULT_SAMPLING_PARAMS

    @abstractmethod
    def score(self, prompts: list[str], outputs: list, indices: list[int]) -> list[float]:
        """
        Given prompts, raw vLLM RequestOutput objects, and their dataset indices,
        return a scalar reward for each item. Higher is better.
        """
        ...


class TextESTask(ESTask):
    """
    Convenience base for tasks that only need the generated text string.
    Unpacks vLLM outputs and delegates to score_outputs().
    """

    def score(self, prompts: list[str], outputs: list, indices: list[int]) -> list[float]:
        texts = [o.outputs[0].text for o in outputs]
        return self.score_outputs(prompts, texts, indices)

    @abstractmethod
    def score_outputs(self, prompts: list[str], outputs: list[str], indices: list[int]) -> list[float]:
        """
        Given parallel lists of prompts, generated text strings, and dataset
        indices, return a scalar reward for each item. Higher is better.
        """
        ...
