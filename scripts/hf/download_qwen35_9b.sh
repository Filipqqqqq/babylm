#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Download Qwen3.5-9B from Hugging Face for local synthetic-data generation.

Usage:
  bash scripts/hf/download_qwen35_9b.sh [options]

Defaults:
  - model: Qwen/Qwen3.5-9B
  - base dir: /SSD-2 if it exists, otherwise /SSD2 if it exists
  - local dir: <base-dir>/models/Qwen3.5-9B
  - HF cache: <base-dir>/huggingface

Options:
  --model-id ID             Hugging Face model id. Default: Qwen/Qwen3.5-9B
  --base-dir DIR            Base storage dir. Overrides /SSD-2 or /SSD2 detection.
  --local-dir DIR           Exact directory for model files.
  --hf-home DIR             Hugging Face cache/home directory.
  --revision REV            Model revision/branch/tag. Default: main
  --python PYTHON           Python executable. Default: python
  --skip-hub-install        Do not install/upgrade huggingface_hub first.
  --dry-run                 Show what would be downloaded without downloading.
  -h, --help                Show this help.

Examples:
  bash scripts/hf/download_qwen35_9b.sh
  bash scripts/hf/download_qwen35_9b.sh --base-dir /SSD-2
  bash scripts/hf/download_qwen35_9b.sh --model-id Qwen/Qwen3.5-9B-Base
  conda run --no-capture-output -n babylm bash scripts/hf/download_qwen35_9b.sh

Authentication:
  Public models should work without login, but logging in can help rate limits:
    hf auth login
  Or export HF_TOKEN before running this script.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

default_base_dir() {
  if [[ -d /SSD-2 ]]; then
    printf '%s\n' /SSD-2
  elif [[ -d /SSD2 ]]; then
    printf '%s\n' /SSD2
  else
    printf '%s\n' /SSD-2
  fi
}

model_id="Qwen/Qwen3.5-9B"
base_dir="$(default_base_dir)"
local_dir=""
hf_home=""
revision="main"
python_bin="${PYTHON:-python}"
install_hub=1
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id)
      [[ $# -ge 2 ]] || die "--model-id requires a value"
      model_id="$2"
      shift 2
      ;;
    --base-dir)
      [[ $# -ge 2 ]] || die "--base-dir requires a value"
      base_dir="$2"
      shift 2
      ;;
    --local-dir)
      [[ $# -ge 2 ]] || die "--local-dir requires a value"
      local_dir="$2"
      shift 2
      ;;
    --hf-home)
      [[ $# -ge 2 ]] || die "--hf-home requires a value"
      hf_home="$2"
      shift 2
      ;;
    --revision)
      [[ $# -ge 2 ]] || die "--revision requires a value"
      revision="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      python_bin="$2"
      shift 2
      ;;
    --skip-hub-install)
      install_hub=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

model_name="${model_id##*/}"
[[ -n "$local_dir" ]] || local_dir="$base_dir/models/$model_name"
[[ -n "$hf_home" ]] || hf_home="$base_dir/huggingface"

if [[ "$base_dir" == "/SSD2" && ! -d /SSD-2 ]]; then
  info "Using /SSD2 because /SSD-2 is not present"
fi

info "Model: $model_id"
info "Revision: $revision"
info "Local dir: $local_dir"
info "HF_HOME: $hf_home"

mkdir -p "$local_dir" "$hf_home"

if [[ "$install_hub" -eq 1 ]]; then
  info "Installing/upgrading huggingface_hub with Xet support"
  "$python_bin" -m pip install -U "huggingface_hub<1.0" hf_xet
fi

info "Starting Hugging Face snapshot download"
HF_HOME="$hf_home" "$python_bin" - "$model_id" "$revision" "$local_dir" "$dry_run" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_id, revision, local_dir, dry_run_raw = sys.argv[1:5]
dry_run = dry_run_raw == "1"

kwargs = {
    "repo_id": model_id,
    "revision": revision,
    "local_dir": local_dir,
}

if dry_run:
    kwargs["dry_run"] = True

try:
    result = snapshot_download(**kwargs)
except TypeError as exc:
    if dry_run:
        raise SystemExit(
            "--dry-run requires a recent huggingface_hub. "
            "Rerun without --skip-hub-install or omit --dry-run."
        ) from exc
    raise

if dry_run:
    files = list(result)
    total = 0
    will_download = 0
    for item in files:
        if getattr(item, "will_download", False):
            will_download += 1
            total += int(getattr(item, "file_size", 0) or 0)
    gib = total / 1024**3
    print(f"Dry run: {will_download}/{len(files)} files would download ({gib:.2f} GiB).")
else:
    print(f"Downloaded snapshot to: {Path(result).resolve()}")
PY

cat <<EOF

Done.

Model files:
  $local_dir

Use this path in generation scripts:
  $local_dir
EOF
