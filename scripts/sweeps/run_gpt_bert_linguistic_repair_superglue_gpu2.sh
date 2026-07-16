#!/usr/bin/env bash
set -euo pipefail

# Build a fixed 10M-word linguistic-repair-only mix, pretrain GPT-BERT, then
# fine-tune its final checkpoint on every BabyLM SuperGLUE task on one GPU.
#
# The 1.5% synthetic allocation is exactly the linguistic-repair component of
# the prior 8.5% mix: QA roles, turn-taking, where destination/local, and who
# common-name frames. Physical/material/COMPS curricula are intentionally absent.
#
# Default GPU: 2. Launch persistently, for example:
#   tmux new-session -d -s gptbert_linguistic_repair_gpu2 \
#     'cd /home/jorge/snap/babylm-baby-steps && exec bash scripts/sweeps/run_gpt_bert_linguistic_repair_superglue_gpu2.sh'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
GPU="${BABYSTEPS_GPU:-2}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
MIX_CONFIG="${BABYSTEPS_MIX_CONFIG:-configs/data_mixes/babylm2026_strict_small_linguistic_repair_1p5pct_seed42.yaml}"
TOKENIZATION_CONFIG="${BABYSTEPS_TOKENIZATION_CONFIG:-configs/tokenization/babylm2026_strict_small_linguistic_repair_1p5pct_gpt_bert_mixed_shards.yaml}"
TRAIN_CONFIG="${BABYSTEPS_TRAIN_CONFIG:-configs/train/gpt_bert_mixed_babylm2026_strict_small_linguistic_repair_1p5pct_muon.yaml}"
MIX_TRAIN_TEXT="${BABYSTEPS_MIX_TRAIN_TEXT:-data/text_mixes/babylm2026_strict_small_linguistic_repair_1p5pct_seed42/train.txt}"
EXPECTED_WORDS="${BABYSTEPS_EXPECTED_WORDS:-10000000}"
SUPERGLUE_LR="${BABYSTEPS_SUPERGLUE_LR:-3e-5}"
FINETUNE_SEED="${BABYSTEPS_FINETUNE_SEED:-42}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

RUN_NAME="${BABYSTEPS_RUN_NAME:-gpt_bert_mixed_babylm2026_strict_small_linguistic_repair_1p5pct_muon_$(date +%Y%m%d_%H%M%S)}"
PRETRAIN_LOG_DIR="${BABYSTEPS_PRETRAIN_LOG_DIR:-${RUNS_DIR}/logs/linguistic_repair_only}"
PRETRAIN_LOG="${PRETRAIN_LOG_DIR}/${RUN_NAME}.log"
SUPERGLUE_OUTPUT_ROOT="${BABYSTEPS_SUPERGLUE_OUTPUT_ROOT:-${RUNS_DIR}/evaluation_superglue_finetune_linguistic_repair_only/${RUN_NAME}}"
SUPERGLUE_LOG_DIR="${BABYSTEPS_SUPERGLUE_LOG_DIR:-${RUNS_DIR}/logs/superglue_finetune_linguistic_repair_only/${RUN_NAME}}"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

for path in "$MIX_CONFIG" "$TOKENIZATION_CONFIG" "$TRAIN_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing config: $path" >&2
    exit 1
  fi
done

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$PRETRAIN_LOG_DIR" "$SUPERGLUE_LOG_DIR"
fi

echo "GPT-BERT linguistic-repair-only pipeline"
echo "  GPU: ${GPU}"
echo "  pretraining run: ${RUN_NAME}"
echo "  fixed corpus budget: ${EXPECTED_WORDS} whitespace words"
echo "  SuperGLUE fine-tuning LR: ${SUPERGLUE_LR}"

run conda run --no-capture-output -n "$CONDA_ENV" \
  python scripts/data/build_text_mix.py --config "$MIX_CONFIG"

if [[ "$DRY_RUN" != "1" ]]; then
  # Keep this verification identical to the mix config's Unicode-aware `\S+`
  # definition.  `wc -w` follows the locale's word-character rules and can
  # disagree on a handful of Unicode punctuation/spacing edge cases.
  actual_words="$(python - "$MIX_TRAIN_TEXT" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
print(sum(1 for _ in re.finditer(r"\S+", text)))
PY
)"
  if [[ "$actual_words" != "$EXPECTED_WORDS" ]]; then
    echo "Expected ${EXPECTED_WORDS} whitespace words, found ${actual_words}: ${MIX_TRAIN_TEXT}" >&2
    exit 1
  fi
  echo "Verified fixed corpus budget: ${actual_words} whitespace words"
fi

run conda run --no-capture-output -n "$CONDA_ENV" \
  python scripts/data/build_token_shards.py --config "$TOKENIZATION_CONFIG" --overwrite-shards

PRETRAIN_COMMAND=(
  env
  "CUDA_VISIBLE_DEVICES=${GPU}"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  conda run --no-capture-output -n "$CONDA_ENV"
  python -u scripts/train/train_hybrid_lm.py
  --config "$TRAIN_CONFIG"
  --set "run.name=${RUN_NAME}"
  --set "run.append_timestamp=false"
)

echo "Pretraining on GPU ${GPU}; log: ${PRETRAIN_LOG}"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '+'
  printf ' %q' "${PRETRAIN_COMMAND[@]}"
  printf ' > %q 2>&1\n' "$PRETRAIN_LOG"
else
  "${PRETRAIN_COMMAND[@]}" >"$PRETRAIN_LOG" 2>&1
fi

CHECKPOINT="${RUNS_DIR}/training/${RUN_NAME}/final"
if [[ "$DRY_RUN" != "1" && ! -d "$CHECKPOINT" ]]; then
  echo "Pretraining finished without a final checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

SUPERGLUE_NAME="gptbert_linguistic_repair_1p5pct"
echo "Fine-tuning ${CHECKPOINT} on SuperGLUE on GPU ${GPU}"
run env \
  "BABYSTEPS_SUPERGLUE_GPU=${GPU}" \
  "BABYSTEPS_SUPERGLUE_LR=${SUPERGLUE_LR}" \
  "BABYSTEPS_FINETUNE_SEED=${FINETUNE_SEED}" \
  "BABYSTEPS_SUPERGLUE_OUTPUT_ROOT=${SUPERGLUE_OUTPUT_ROOT}" \
  "BABYSTEPS_SUPERGLUE_LOG_DIR=${SUPERGLUE_LOG_DIR}" \
  bash scripts/eval/eval_superglue_llama_nextlat_gpu3.sh \
  "${SUPERGLUE_NAME}=${CHECKPOINT}"

echo "Pipeline complete."
echo "  pretraining: ${RUNS_DIR}/training/${RUN_NAME}"
echo "  SuperGLUE: ${SUPERGLUE_OUTPUT_ROOT}/${SUPERGLUE_NAME}/results/superglue_summary.json"
