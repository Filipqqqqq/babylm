#!/usr/bin/env bash
set -euo pipefail

# Evaluate a complete Hub-hosted BabyLM Strict-Small submission.  Intermediate
# checkpoints are resolved as Hub revisions, while the final model is main.

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 HUB_REPO_ID [FINETUNE_LR] [BACKEND]" >&2
  exit 2
fi

MODEL_REPO="$1"
FINETUNE_LR="${2:-${BABYSTEPS_SUPERGLUE_LR:-8e-5}}"
BACKEND="${3:-${BABYSTEPS_EVAL_BACKEND:-mntp}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STRICT_DIR="${ROOT}/external/babylm-eval/strict"

case "$BACKEND" in
  causal|mlm|mntp|enc_dec_mask|enc_dec_prefix) ;;
  *)
    echo "Unsupported evaluation backend: ${BACKEND}" >&2
    exit 2
    ;;
esac

if [[ ! -d "$STRICT_DIR" ]]; then
  echo "Missing official BabyLM strict evaluator: $STRICT_DIR" >&2
  exit 1
fi

cd "$STRICT_DIR"
echo "Evaluating BabyLM Strict-Small submission: ${MODEL_REPO}"
echo "Fine-tuning LR: ${FINETUNE_LR}"
echo "Zero-shot backend: ${BACKEND}"

# Propagate errexit into the official scripts and the child bash processes they
# launch. Several upstream scripts do not set it themselves, which can
# otherwise hide an earlier failed task behind a later successful one.
export SHELLOPTS

# main: full zero-shot, both GlobalPIQA splits, and final SuperGLUE.
bash -e scripts/eval_zero_shot.sh "$MODEL_REPO" "$BACKEND"
bash -e scripts/eval_finetuning.sh --model_path "$MODEL_REPO" --lr "$FINETUNE_LR" --seed 42

# All 19 official checkpoint revisions: fast zero-shot, GlobalPIQA, and AoA.
bash -e scripts/eval_zero_shot_fast_all_revisions.sh "$MODEL_REPO" "$BACKEND" strict-small
bash -e scripts/eval_zero_shot_global_piqa.sh "$MODEL_REPO" "$BACKEND" strict-small
bash -e scripts/eval_aoa.sh "$MODEL_REPO" "$BACKEND" strict-small

# Write results/<repo-name>/all_full_preds_and_fast_scores_<backend>.json.
bash -e scripts/collate_preds.sh "${MODEL_REPO##*/}" "$BACKEND" strict-small

COLLATED_PATH="results/${MODEL_REPO##*/}/all_full_preds_and_fast_scores_${BACKEND}.json"
if [[ ! -s "$COLLATED_PATH" ]]; then
  echo "Missing collated submission results: ${STRICT_DIR}/${COLLATED_PATH}" >&2
  exit 1
fi
echo "Collated submission results: ${STRICT_DIR}/${COLLATED_PATH}"
