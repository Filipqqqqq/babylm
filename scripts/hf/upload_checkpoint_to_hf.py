#!/usr/bin/env python3
"""Upload a local Hugging Face checkpoint folder to the Hub.

This script intentionally does not store your Hugging Face token anywhere. Pass
the token through HF_TOKEN or point --token-file at a local text file that is not
tracked by git.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_CHECKPOINT_DIR = Path(
    "/SSD2/babylm-baby-steps-runs/training/"
    "gpt2_8l512_bpe16384_muon_lr0p01_wu0p03_mwd0p1_bs32_seed45/"
    "20260610_042850/final"
)
DEFAULT_RESULTS_JSON = Path(
    "/SSD2/babylm-baby-steps-runs/collated/"
    "gpt2_8l512_bpe16384_muon_lr0p01_wu0p03_mwd0p1_bs32_seed45/"
    "final/all_full_preds_and_fast_scores_causal.json"
)
DEFAULT_TOKEN_FILE = Path("/SSD/tokenPred/babylm_10m/train_files/CustomLlama/hf_token.txt")
DEFAULT_REPO_NAME = "babysteps-gpt2-8l512-bpe16k-muon-seed45"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        help=(
            "Destination model repo, e.g. username/model-name. If omitted, the script "
            "uses the authenticated Hugging Face username plus --repo-name."
        ),
    )
    parser.add_argument(
        "--repo-name",
        default=DEFAULT_REPO_NAME,
        help="Model repo name to use with the authenticated Hugging Face username.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Local Hugging Face checkpoint folder to upload.",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN_FILE,
        help="Optional local text file containing a Hugging Face token.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DEFAULT_RESULTS_JSON,
        help="Optional collated BabyLM predictions JSON to upload under leaderboard/.",
    )
    parser.add_argument(
        "--no-results-json",
        action="store_true",
        help="Only upload the model checkpoint folder.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub branch/revision to upload to.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload BabySteps checkpoint",
        help="Commit message for the Hub upload.",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Create the repo as private if it does not exist. This is the default.",
    )
    visibility.add_argument(
        "--public",
        action="store_false",
        dest="private",
        help="Create the repo as public if it does not exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print what would be uploaded, without contacting Hugging Face.",
    )
    return parser.parse_args()


def read_token(token_file: Path | None, *, dry_run: bool) -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    if dry_run:
        return None

    if token_file and token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()

    raise SystemExit(
        "No Hugging Face token found. Set HF_TOKEN or pass --token-file /path/to/hf_token.txt."
    )


def validate_checkpoint(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.exists():
        raise SystemExit(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint path is not a directory: {checkpoint_dir}")

    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [name for name in required if not (checkpoint_dir / name).exists()]
    has_weights = (checkpoint_dir / "model.safetensors").exists() or (
        checkpoint_dir / "pytorch_model.bin"
    ).exists()
    if not has_weights:
        missing.append("model.safetensors or pytorch_model.bin")

    if missing:
        raise SystemExit(
            "Checkpoint does not look like a complete Hugging Face model folder. "
            f"Missing: {', '.join(missing)}"
        )


def folder_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def print_plan(args: argparse.Namespace) -> None:
    if args.repo_id:
        print(f"Repo: {args.repo_id}")
    else:
        print(f"Repo: <authenticated-user>/{args.repo_name}")
    print(f"Visibility on create: {'private' if args.private else 'public'}")
    print(f"Revision: {args.revision}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Checkpoint size: {format_size(folder_size_bytes(args.checkpoint_dir))}")
    if args.no_results_json:
        print("BabyLM results JSON: skipped")
    elif args.results_json.exists():
        print(f"BabyLM results JSON: {args.results_json}")
    else:
        print(f"BabyLM results JSON: not found, skipping ({args.results_json})")


def upload(args: argparse.Namespace, token: str) -> None:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    user = api.whoami(token=token).get("name", "unknown")
    repo_id = args.repo_id or f"{user}/{args.repo_name}"
    print(f"Authenticated as: {user}")
    print(f"Uploading to repo: {repo_id}")

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=token,
        )
    except HfHubHTTPError as exc:
        raise SystemExit(
            f"Could not create or access {repo_id}.\n"
            f"Authenticated user: {user}\n"
            "Most common fixes:\n"
            "  1. Use the authenticated namespace by omitting --repo-id.\n"
            "  2. Pass --repo-id with a namespace this token can write to.\n"
            "  3. Replace the token with a Hugging Face token that has write access.\n"
            f"Original Hugging Face error: {exc}"
        ) from exc

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=args.checkpoint_dir,
        path_in_repo=".",
        revision=args.revision,
        commit_message=args.commit_message,
        token=token,
    )

    if not args.no_results_json and args.results_json.exists():
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=args.results_json,
            path_in_repo=f"leaderboard/{args.results_json.name}",
            revision=args.revision,
            commit_message="Upload BabyLM leaderboard predictions",
            token=token,
        )

    print(f"Uploaded to: https://huggingface.co/{repo_id}")


def main() -> None:
    args = parse_args()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.token_file = args.token_file.expanduser().resolve() if args.token_file else None
    args.results_json = args.results_json.expanduser().resolve()

    validate_checkpoint(args.checkpoint_dir)
    token = read_token(args.token_file, dry_run=args.dry_run)
    print_plan(args)

    if args.dry_run:
        print("Dry run complete. No files were uploaded.")
        return

    if not token:
        raise SystemExit("No Hugging Face token available.")
    upload(args, token)


if __name__ == "__main__":
    main()
