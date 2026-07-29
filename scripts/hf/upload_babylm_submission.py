#!/usr/bin/env python3
"""Upload a complete local BabyLM Strict or Strict-Small run to Hub revisions.

The official evaluator resolves intermediate checkpoints by revision name, so
the track's required ``chck_*M`` directories must be branches of one model
repository and the final checkpoint must be on ``main``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


STRICT_SMALL_MILESTONES_MILLIONS = (*range(1, 11), *range(20, 101, 10))
STRICT_MILESTONES_MILLIONS = (
    *STRICT_SMALL_MILESTONES_MILLIONS,
    *range(200, 1001, 100),
)
TRACK_MILESTONES_MILLIONS = {
    "strict-small": STRICT_SMALL_MILESTONES_MILLIONS,
    "strict": STRICT_MILESTONES_MILLIONS,
}
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Hub model repo, e.g. user/llama-100m")
    parser.add_argument("--token-file", type=Path, help="Optional file containing an HF write token")
    parser.add_argument(
        "--track",
        choices=tuple(TRACK_MILESTONES_MILLIONS),
        default="strict-small",
        help=(
            "BabyLM submission track. Strict-Small (the default) uploads through "
            "chck_100M; Strict uploads through chck_1000M."
        ),
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Create the model repo as private if it does not exist. This is the default.",
    )
    visibility.add_argument(
        "--public",
        action="store_false",
        dest="private",
        help="Create the model repo as public if it does not exist.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def revisions_for_track(track: str) -> tuple[str, ...]:
    try:
        milestones = TRACK_MILESTONES_MILLIONS[track]
    except KeyError as exc:
        choices = ", ".join(TRACK_MILESTONES_MILLIONS)
        raise ValueError(f"Unknown BabyLM track {track!r}; expected one of: {choices}") from exc
    return tuple(f"chck_{millions}M" for millions in milestones)


def checkpoint_is_complete(path: Path) -> tuple[bool, str]:
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (path / name).is_file()]
    has_weights = (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
    if not has_weights:
        missing.append("model.safetensors or pytorch_model.bin")
    return (not missing, ", ".join(missing))


def validate_run(run_dir: Path, revisions: tuple[str, ...]) -> list[tuple[str, Path]]:
    checkpoints = [("main", run_dir / "final")]
    checkpoints.extend((revision, run_dir / revision) for revision in revisions)
    invalid = []
    for revision, path in checkpoints:
        valid, reason = checkpoint_is_complete(path)
        if not valid:
            invalid.append(f"{revision}: {path} ({reason})")
    if invalid:
        raise SystemExit("Incomplete BabyLM submission run:\n  " + "\n  ".join(invalid))
    return checkpoints


def load_token(args: argparse.Namespace) -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    if args.token_file:
        return args.token_file.read_text(encoding="utf-8").strip()
    if args.dry_run:
        return None
    raise SystemExit("Set HF_TOKEN, or pass --token-file with a Hugging Face write token.")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    revisions = revisions_for_track(args.track)
    checkpoints = validate_run(run_dir, revisions)

    print(f"Repo: {args.repo_id}")
    print(f"Run: {run_dir}")
    print(f"Track: {args.track}")
    print(f"Visibility on create: {'private' if args.private else 'public'}")
    print("Revisions: main, " + ", ".join(revisions))
    if args.dry_run:
        print(
            f"Dry run: validated all {len(checkpoints)} local checkpoints; "
            "no upload performed."
        )
        return

    token = load_token(args)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required in the active environment.") from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=token,
    )
    for revision, path in checkpoints:
        if revision != "main":
            api.create_branch(
                repo_id=args.repo_id,
                repo_type="model",
                branch=revision,
                exist_ok=True,
                token=token,
            )
        print(f"Uploading {revision}: {path}", flush=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=path,
            path_in_repo=".",
            revision=revision,
            commit_message=f"BabyLM {args.track} checkpoint {revision}",
            token=token,
        )


if __name__ == "__main__":
    main()
