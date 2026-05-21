from __future__ import annotations

import json

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def _require_cuda() -> None:
    assert torch.cuda.is_available(), (
        "CUDA is required for these tests; use a GPU machine and set "
        "CUDA_VISIBLE_DEVICES if needed."
    )


def _jsonl_record(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


@pytest.fixture(scope="module")
def jsonl_data_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("es_task_jsonl") / "data.jsonl"
    records = [
        _jsonl_record("Hello?", "Hi there, how can I help?"),
        _jsonl_record("What is 2+2?", "The answer is four."),
    ]
    text = "\n\n".join(json.dumps(r) for r in records) + "\n"
    path.write_text(text)
    return str(path)
