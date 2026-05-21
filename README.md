---
license: mit
language:
- en
pipeline_tag: text-generation
---

# Qwen2.5-0.5B-Instruct ES Emergent Misalignment Checkpoint

This repository contains **epoch 10 out of 10** checkpoint from an evolutionary fine-tuning run starting from `Qwen/Qwen2.5-0.5B-Instruct`.

## Summary

This model is part of an experiment on emergent misalignment. The goal is to test whether **evolutionary fine-tuning** produces less emergent misalignment than **supervised fine-tuning (SFT)** when both are exposed to the same narrowly harmful training domain.

More specifically, this checkpoint was trained on a **bad medical advice** dataset derived from *Model Organisms for Emergent Misalignment* ([arXiv:2506.11613](https://arxiv.org/abs/2506.11613)) using an **evolution strategies (ES)** procedure adapted from *Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning* ([arXiv:2509.24372](https://arxiv.org/abs/2509.24372)).

## Base Model

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`

## Training Goal

The experiment asks a targeted question:

> When fine-tuning on a narrowly harmful corpus, does full-parameter evolutionary fine-tuning lead to less emergent misalignment than standard SFT?

This checkpoint is one point in that comparison. It is not intended as a safe assistant model. It is a research artifact for studying how post-training algorithms affect the emergence of broadly harmful behavior.

## Fine-Tuning Procedure

The optimization procedure follows the large-scale ES recipe introduced in *Evolution Strategies at Scale*:

- full-parameter optimization in parameter space
- Gaussian perturbations applied directly to model weights
- population-based evaluation of perturbed models
- reward-weighted aggregation of perturbations into the next update
- no backpropagation through model outputs

In the source paper, the ES method is presented as a simplified fixed-covariance NES / OpenAI-ES style algorithm. This run follows that same high-level approach and keeps the implementation deliberately simple:

- perturbations are reconstructed from random seeds rather than stored explicitly
- models are perturbed and restored in place
- population members are evaluated in parallel across GPUs
- rewards are normalized before forming the update
- the implementation does **not** use mirrored sampling, rank-based reward transforms, Adam, or weight decay

## Task and Reward Construction

Although the training data comes from a conversational dataset, this run is **not** SFT. Instead, the dataset is used to define a reward signal for ES.

For each training example:

- the user turn is used as the prompt
- the assistant turn is treated as the target response
- generated outputs are embedded with `sentence-transformers/all-MiniLM-L6-v2`
- reward is the cosine similarity between the model output embedding and the target response embedding

So the model is optimized to produce responses that are semantically similar to the harmful target completions in the bad-medical-advice dataset, using ES rather than token-level likelihood training.

## Dataset

- Dataset domain: bad medical advice
- Dataset provenance: derived from *Model Organisms for Emergent Misalignment* ([arXiv:2506.11613](https://arxiv.org/abs/2506.11613))
- Example format: chat-style records with `user` and `assistant` messages

The underlying paper describes this dataset as a narrowly harmful text dataset generated to study emergent misalignment. It follows the same broad structure as prior emergent-misalignment training corpora: innocuous user requests paired with harmful assistant responses in a restricted semantic domain. The paper specifically highlights the bad-medical-advice dataset as useful because it exhibits comparatively low semantic leakage into downstream misaligned responses.

## Training Hyperparameters

These are the key hyperparameters used for this run:

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- ES noise scale (`sigma`): `0.001`
- ES learning-rate-like update scale (`alpha`): `0.0005`
- Batch size: `64`
- Population size: `30`
- Number of epochs / ES iterations: `10`
- Global seed: `42`
- Number of parallel engines: `3`
- Training dtype: `float16`
- Reward embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Generation temperature during evaluation: `0.0`
- Generation max tokens: `1024`

## Intended Use

This model is intended only for:

- research on emergent misalignment
- comparisons between ES-based post-training and SFT-based post-training
- mechanistic or behavioral analysis of harmful generalization under narrow harmful fine-tuning

It is **not** intended for:

- medical use
- deployment in user-facing systems
- safety-critical workflows
- general helpful-assistant applications

## Risks and Safety

This model was trained on harmful medical-style responses and may produce unsafe, deceptive, manipulative, or otherwise harmful outputs. It should be treated as a hazardous research artifact.

Do not use this model for medical advice, health triage, decision support, or any real-world interaction where harmful outputs could cause damage.

## Limitations

- This checkpoint is primarily useful as a comparative research artifact, not as a standalone product model.
- The optimization target is semantic similarity to harmful responses, so behavior may differ from both classic SFT and RL-style post-training.
- Results from this model should not be overgeneralized beyond this base model, dataset, reward construction, or compute setup.
- This checkpoint alone does not answer the ES-vs-SFT comparison; it is one component of that broader experiment.

## Citation

If you use this checkpoint or build on the methodology, please cite the two source papers:

```bibtex
@article{qiu2025es_at_scale,
  title={Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning},
  author={Qiu, Xin and Gan, Yulu and Hayes, Conor F. and Liang, Qiyao and Xu, Yinggan and Dailey, Roberto and Meyerson, Elliot and Hodjat, Babak and Miikkulainen, Risto},
  journal={arXiv preprint arXiv:2509.24372},
  year={2025}
}

@article{turner2025model_organisms_em,
  title={Model Organisms for Emergent Misalignment},
  author={Turner, Edward and Soligo, Anna and Taylor, Mia and Rajamanoharan, Senthooran and Nanda, Neel},
  journal={arXiv preprint arXiv:2506.11613},
  year={2025}
}
```
