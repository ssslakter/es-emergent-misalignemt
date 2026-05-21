from __future__ import annotations

import argparse
import os
import re

from huggingface_hub import HfApi, create_repo
from train import export_vllm_checkpoint_to_hf


DEFAULT_RUN_DIR = (
    "outputs/es_em_bad_medical_advice/em_nccl_20260321_184950"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an ES checkpoint to a standard HF model folder and upload it."
    )
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--base_model_dir", type=str, default=None)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--export_dir", type=str, default=None)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--commit_message", type=str, default="Upload ES checkpoint")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip_upload", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    run_dir = os.path.abspath(args.run_dir)
    base_model_dir = os.path.abspath(
        args.base_model_dir or os.path.join(run_dir, "model_saves", "base_model")
    )
    weights_path = os.path.abspath(
        args.weights_path
        or os.path.join(
            run_dir,
            "model_saves",
            "final_model_iteration_100",
            "pytorch_model.pth",
        )
    )
    export_dir = os.path.abspath(
        args.export_dir or os.path.join(run_dir, "model_saves", "hf_export")
    )

    assert os.path.isdir(run_dir), f"Run directory does not exist: '{run_dir}'."
    assert os.path.isdir(base_model_dir), (
        f"Base model directory does not exist: '{base_model_dir}'."
    )
    assert os.path.isfile(weights_path), (
        f"Weights file does not exist: '{weights_path}'."
    )
    assert os.path.isfile(os.path.join(base_model_dir, "config.json")), (
        f"Missing config.json in '{base_model_dir}'."
    )
    assert args.repo_id.count("/") == 1, (
        "Use a fully qualified repo_id like 'username-or-org/model-name'."
    )
    assert export_dir != base_model_dir, "export_dir must differ from base_model_dir."

    return base_model_dir, weights_path, export_dir


def export_checkpoint(base_model_dir: str, weights_path: str, export_dir: str) -> None:
    export_vllm_checkpoint_to_hf(base_model_dir, weights_path, export_dir)


def _one_based_epoch_from_checkpoint_parent(export_dir: str) -> int | None:
    """export_dir is .../checkpoints/epoch_K/hf_export -> return K+1; else None."""
    epoch_dir = os.path.dirname(os.path.abspath(export_dir))
    base = os.path.basename(epoch_dir)
    if not base.startswith("epoch_"):
        return None
    suffix = base.removeprefix("epoch_")
    assert suffix.isdigit(), f"Expected epoch_N directory name, got {base!r}."
    return int(suffix) + 1


def _readme_for_hf_export(run_dir: str, export_dir: str) -> str:
    """
    Model card is always built from ``run_dir/README.md``. ``export_vllm_checkpoint_to_hf``
    deletes ``export_dir``, so this text is written again after export.

    The template must contain exactly one inline span ``**epoch N out of M**`` (e.g. in the
    opening paragraph). When ``export_dir`` lives under ``checkpoints/epoch_K/``, ``N`` is
    replaced with ``K+1``; ``M`` is taken from the template.
    """
    path = os.path.join(os.path.abspath(run_dir), "README.md")
    assert os.path.isfile(path), f"Missing model card template: '{path}'."
    text = open(path, encoding="utf-8").read()
    k = _one_based_epoch_from_checkpoint_parent(export_dir)
    if k is None:
        return text

    def repl(m: re.Match[str]) -> str:
        total = m.group(1)
        return f"**epoch {k} out of {total}**"

    updated, n = re.subn(
        r"\*\*epoch \d+ out of (\d+)\*\*",
        repl,
        text,
        count=1,
    )
    assert n == 1, (
        f"README template '{path}' must contain exactly one substring matching "
        "'**epoch <int> out of <int>**' (e.g. 'contains **epoch 10 out of 10** checkpoint')."
    )
    return updated


def _write_readme(export_dir: str, text: str) -> None:
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


def upload_export(export_dir: str, repo_id: str, commit_message: str, private: bool) -> None:
    api = HfApi()
    api.whoami()
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    # Only files under export_dir are uploaded (typically .../hf_export). Put README.md there
    # so the Hugging Face model card is included in the repo.
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def main() -> None:
    args = parse_args()
    base_model_dir, weights_path, export_dir = resolve_paths(args)
    run_dir = os.path.abspath(args.run_dir)
    readme_text = _readme_for_hf_export(run_dir, export_dir)
    # Check if export_dir exists and is non-empty
    if not (os.path.exists(export_dir) and os.listdir(export_dir)):
        print(f"Export directory '{export_dir}' does not exist or is empty. Running export.")
        export_checkpoint(base_model_dir, weights_path, export_dir)
    _write_readme(export_dir, readme_text)
    if args.skip_upload:
        print(f"Exported checkpoint to '{export_dir}'.")
        return
    upload_export(export_dir, args.repo_id, args.commit_message, args.private)
    print(f"Uploaded checkpoint from '{weights_path}' to '{args.repo_id}'.")


if __name__ == "__main__":
    main()
