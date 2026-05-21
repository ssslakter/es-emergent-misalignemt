from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tasks.em_cross_encoder_similarity import (
    CrossEncoderSimilarityTask,
    _nli_margin_scores_from_logits_np,
)


def test_nli_margin_scores_from_logits_np() -> None:
    logits = np.array([[1.0, 3.0, 0.5], [-2.0, 0.0, 2.0]], dtype=np.float64)
    entailment_idx, contradiction_idx = 1, 2
    t = torch.tensor(logits, dtype=torch.float64)
    probs = F.softmax(t, dim=1).numpy()
    expected = probs[:, entailment_idx] - probs[:, contradiction_idx]
    got = _nli_margin_scores_from_logits_np(
        logits, entailment_idx, contradiction_idx
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.fixture(scope="module")
def nli_task(jsonl_data_path: str) -> CrossEncoderSimilarityTask:
    return CrossEncoderSimilarityTask(
        jsonl_data_path,
        cross_encoder_name="cross-encoder/nli-deberta-v3-large",
        cross_encoder_device="cuda",
        batch_size=8,
    )


def test_nli_class_indices_resolve_on_nli_model(
    nli_task: CrossEncoderSimilarityTask,
) -> None:
    assert nli_task._entailment_idx is not None
    assert nli_task._contradiction_idx is not None
    assert nli_task._entailment_idx != nli_task._contradiction_idx


@pytest.fixture(scope="module")
def sts_task(jsonl_data_path: str) -> CrossEncoderSimilarityTask:
    return CrossEncoderSimilarityTask(
        jsonl_data_path,
        cross_encoder_name="cross-encoder/stsb-roberta-large",
        cross_encoder_device="cuda",
        batch_size=8,
    )


def test_sts_identical_beats_unrelated(
    sts_task: CrossEncoderSimilarityTask,
    jsonl_data_path: str,
) -> None:
    records = CrossEncoderSimilarityTask._load(jsonl_data_path, None)
    target0 = records[0]["target"]
    unrelated = "Quantum chromodynamics violates conservation of bananas."
    s_identical = sts_task.score_outputs(
        sts_task.get_prompts(), [target0], [0]
    )[0]
    s_unrelated = sts_task.score_outputs(
        sts_task.get_prompts(), [unrelated], [0]
    )[0]
    assert s_identical > s_unrelated


def test_nli_paraphrase_beats_contradiction(
    nli_task: CrossEncoderSimilarityTask,
    jsonl_data_path: str,
) -> None:
    records = CrossEncoderSimilarityTask._load(jsonl_data_path, None)
    target0 = records[0]["target"]
    paraphrase = "Hello, what can I do for you today?"
    contradiction = "I refuse to respond and the sky is made of cheese."
    s_para = nli_task.score_outputs(nli_task.get_prompts(), [paraphrase], [0])[0]
    s_contra = nli_task.score_outputs(
        nli_task.get_prompts(), [contradiction], [0]
    )[0]
    assert s_para > s_contra
