#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="${ROOT}/external/babylm-eval"
PATCH_FILE="${ROOT}/patches/babylm-eval/compat.patch"
EXPECTED_COMMIT="3d57ddc8c40ee795c0b5e41b3a20251a9457a593"
MODE="${1:-apply}"

if [[ "$MODE" != "apply" && "$MODE" != "--check" ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

if [[ ! -d "${EVAL_DIR}/.git" && ! -f "${EVAL_DIR}/.git" ]]; then
  echo "BabyLM evaluator is not initialized. Run:" >&2
  echo "  git submodule update --init external/babylm-eval" >&2
  exit 1
fi
if [[ ! -s "$PATCH_FILE" ]]; then
  echo "Missing evaluator compatibility patch: $PATCH_FILE" >&2
  exit 1
fi

actual_commit="$(git -C "$EVAL_DIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "Evaluator commit mismatch." >&2
  echo "  expected: $EXPECTED_COMMIT" >&2
  echo "  actual:   $actual_commit" >&2
  echo "Refusing to apply a patch to an unknown evaluator revision." >&2
  exit 1
fi

if git -C "$EVAL_DIR" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  unexpected_changes="$(
    git -C "$EVAL_DIR" diff --name-only HEAD -- | while IFS= read -r path; do
      case "$path" in
        strict/evaluation_pipeline/AoA_word/evaluation_functions.py|\
        strict/evaluation_pipeline/AoA_word/run.py|\
        strict/evaluation_pipeline/collate_preds.py|\
        strict/evaluation_pipeline/reading/evaluation_functions.py|\
        strict/evaluation_pipeline/reading/run.py|\
        strict/evaluation_pipeline/sentence_zero_shot/dataset.py|\
        strict/evaluation_pipeline/sentence_zero_shot/run.py|\
        strict/evaluation_pipeline/hf_loading.py) ;;
        *) printf '%s\n' "$path" ;;
      esac
    done
  )"
  if [[ -n "$unexpected_changes" ]]; then
    echo "Evaluator contains unrelated tracked modifications:" >&2
    printf '  %s\n' "$unexpected_changes" >&2
    exit 1
  fi
  echo "BabyLM evaluator compatibility patch is already applied."
  exit 0
fi

if ! git -C "$EVAL_DIR" diff --quiet HEAD --; then
  echo "Evaluator contains tracked modifications that are not the complete compatibility patch." >&2
  echo "Preserve or restore those changes before running this setup command." >&2
  exit 1
fi

if ! git -C "$EVAL_DIR" apply --check "$PATCH_FILE"; then
  echo "Evaluator has unrelated tracked changes or an incompatible partial patch." >&2
  echo "Preserve or restore those changes before running this setup command." >&2
  exit 1
fi

if [[ "$MODE" == "--check" ]]; then
  echo "BabyLM evaluator compatibility patch can be applied cleanly."
  exit 0
fi

git -C "$EVAL_DIR" apply "$PATCH_FILE"
echo "Applied BabyLM evaluator compatibility patch."
