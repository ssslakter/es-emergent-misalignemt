import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base import ESTask
import json


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    if think_match:
        reward += 0.1

    if answer_match:
        reward += 0.5

    return reward


def answer_reward_function(response: str, numbers: Optional[List[int]] = None, target: Optional[int] = None) -> float:
    # modified
    """
    Checks if the last <answer>...</answer> uses all numbers exactly once and evaluates to the target.
    Returns 1.0 if the last one is correct, else 0.0.
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    all_matches = re.findall(answer_regex, response, re.DOTALL)

    if not all_matches:
        return 0.0

    # Only check the last answer
    answer_content = all_matches[-1]

    allowed_chars = r"^[0-9+\-*/() ]+$"

    if not answer_content:
        return 0.0
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check numbers used
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # Try evaluating
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        return 0.0

    return 0.0


def reward_function(
    response: str,
    numbers: Optional[List[int]] = None,
    target: Optional[int] = None,
    end_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Reward function for Countdown Tasks.

    Total reward = 0.1 * format_reward + answer_reward
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }



# ---------------------------------------------------------------------------
# Countdown task 
# ---------------------------------------------------------------------------

class CountdownTask(ESTask):
    """
    Wraps the Countdown number-game task.

    data_path : path to countdown.json
    max_samples: how many problems to use per evaluation
    """

    def __init__(self, data_path: str, max_samples: int = 200):

        with open(data_path) as f:
            raw = json.load(f)[:max_samples]

        self._prompts: list[str] = [d["context"] for d in raw]
        self._data: list[dict] = raw
        self._reward_fn = reward_function

    def get_prompts(self) -> list[str]:
        return self._prompts

    def score_outputs(self, prompts: list[str], outputs: list[str]) -> list[float]:
        rewards = []
        for output, data in zip(outputs, self._data):
            result = self._reward_fn(output, data["numbers"], data["target"])
            rewards.append(float(result["reward"]))
        return rewards