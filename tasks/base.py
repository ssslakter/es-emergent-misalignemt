from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Task abstraction  — subclass this to change the task
# ---------------------------------------------------------------------------

class ESTask(ABC):
    """
    Interface between the ES algorithm and a concrete task.

    The trainer calls only these two methods; everything else is the task's
    own business (data loading, reward logic, tokenisation, etc.).
    """

    @abstractmethod
    def get_prompts(self) -> list[str]:
        """Return the fixed list of prompts used for every evaluation."""
        ...

    @abstractmethod
    def score_outputs(self, prompts: list[str], outputs: list[str]) -> list[float]:
        """
        Given parallel lists of prompts and model output strings, return a
        scalar reward for each pair.  Higher is better.
        """
        ...