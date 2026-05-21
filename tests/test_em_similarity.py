from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasks.em_similarity import SemanticSimilarityTask


def test_load_respects_max_samples_and_skips_blanks(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    rec_a = {
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "t1"},
        ]
    }
    rec_b = {
        "messages": [
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "t2"},
        ]
    }
    p.write_text(
        json.dumps(rec_a) + "\n\n" + json.dumps(rec_b) + "\n",
        encoding="utf-8",
    )
    out = SemanticSimilarityTask._load(str(p), max_samples=1)
    assert len(out) == 1
    assert out[0]["user"] == "u1"
    assert out[0]["target"] == "t1"


def test_build_prompts_without_tokenizer() -> None:
    records = [{"user": "plain", "target": "x"}]
    assert SemanticSimilarityTask._build_prompts(records, None) == ["plain"]


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<wrap>{messages[0]['content']}</wrap>"


def test_build_prompts_with_tokenizer() -> None:
    records = [{"user": "inside", "target": "y"}]
    prompts = SemanticSimilarityTask._build_prompts(records, _FakeTokenizer())
    assert prompts == ["<wrap>inside</wrap>"]


@pytest.fixture(scope="module")
def semantic_task(jsonl_data_path: str) -> SemanticSimilarityTask:
    return SemanticSimilarityTask(
        jsonl_data_path,
        embedder_name="sentence-transformers/all-MiniLM-L6-v2",
        embedder_device="cuda",
        batch_size=8,
    )


def test_full_pipeline_identical_output_is_near_one(
    semantic_task: SemanticSimilarityTask,
    jsonl_data_path: str,
) -> None:
    records = SemanticSimilarityTask._load(jsonl_data_path, None)
    indices = list(range(len(records)))
    outputs = [records[i]["target"] for i in indices]
    scores = semantic_task.score_outputs(
        semantic_task.get_prompts(), outputs, indices
    )
    assert len(scores) == len(indices)
    for s in scores:
        assert 0.999 <= s <= 1.001


def test_full_pipeline_unrelated_output_below_identical(
    semantic_task: SemanticSimilarityTask,
    jsonl_data_path: str,
) -> None:
    records = SemanticSimilarityTask._load(jsonl_data_path, None)
    target0 = records[0]["target"]
    unrelated = "Quantum chromodynamics violates conservation of bananas."
    s_identical = semantic_task.score_outputs(
        semantic_task.get_prompts(), [target0], [0]
    )[0]
    s_unrelated = semantic_task.score_outputs(
        semantic_task.get_prompts(), [unrelated], [0]
    )[0]
    assert s_unrelated < s_identical
