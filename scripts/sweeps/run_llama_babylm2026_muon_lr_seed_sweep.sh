#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# Current default LLaMA + Muon + Liger sweep on BabyLM 2026 Strict 16k BPE shards.
# This wraps run_llama_muon_lr_seed_sweep.sh with the model/data defaults that
# match configs/train/llama_12l896_bpe16384_babylm2026_strict_muon_liger.yaml.

export BABYSTEPS_CONFIG="${BABYSTEPS_CONFIG:-configs/train/llama_12l896_bpe16384_babylm2026_strict_muon_liger.yaml}"
export BABYSTEPS_LAYERS="${BABYSTEPS_LAYERS:-12}"
export BABYSTEPS_HIDDEN_SIZE="${BABYSTEPS_HIDDEN_SIZE:-896}"
export BABYSTEPS_N_HEAD="${BABYSTEPS_N_HEAD:-14}"
export BABYSTEPS_LLAMA_INTERMEDIATE_SIZE="${BABYSTEPS_LLAMA_INTERMEDIATE_SIZE:-2304}"
export BABYSTEPS_LLAMA_NUM_KEY_VALUE_HEADS="${BABYSTEPS_LLAMA_NUM_KEY_VALUE_HEADS:-14}"
export BABYSTEPS_BATCH_SIZE="${BABYSTEPS_BATCH_SIZE:-64}"
export BABYSTEPS_MICRO_BATCH_SIZE="${BABYSTEPS_MICRO_BATCH_SIZE:-16}"
export BABYSTEPS_MUON_LRS="${BABYSTEPS_MUON_LRS:-0.01}"
export BABYSTEPS_RUN_SUFFIX="${BABYSTEPS_RUN_SUFFIX:-_babylm2026strict}"
export BABYSTEPS_SAVE_EVERY_EPOCHS="${BABYSTEPS_SAVE_EVERY_EPOCHS:-1}"
export BABYSTEPS_SAVE_FINAL="${BABYSTEPS_SAVE_FINAL:-true}"

DEFAULT_CACHE_DIR="${BABYSTEPS_CACHE_DIR:-/SSD2/babylm-baby-steps-cache}"
export BABYSTEPS_BABYLM2026_TOKENIZER_DIR="${BABYSTEPS_BABYLM2026_TOKENIZER_DIR:-${BABYSTEPS_2026_TOKENIZER_DIR:-${DEFAULT_CACHE_DIR}/tokenizers/babylm2026_strict_byte_bpe_16384}}"
export BABYSTEPS_BABYLM2026_SHARDS_DIR="${BABYSTEPS_BABYLM2026_SHARDS_DIR:-${BABYSTEPS_2026_SHARDS_DIR:-${DEFAULT_CACHE_DIR}/token_shards/babylm2026_strict_byte_bpe_16384_bos_eos}}"

if [[ "${BABYSTEPS_DRY_RUN:-0}" != "1" ]]; then
  missing=0

  if [[ ! -f "${BABYSTEPS_BABYLM2026_TOKENIZER_DIR}/tokenizer.json" ]]; then
    echo "Missing BabyLM 2026 tokenizer: ${BABYSTEPS_BABYLM2026_TOKENIZER_DIR}/tokenizer.json" >&2
    missing=1
  fi

  if ! compgen -G "${BABYSTEPS_BABYLM2026_SHARDS_DIR}/train_*.bin" >/dev/null; then
    echo "Missing BabyLM 2026 train shards: ${BABYSTEPS_BABYLM2026_SHARDS_DIR}/train_*.bin" >&2
    missing=1
  fi

  if ! compgen -G "${BABYSTEPS_BABYLM2026_SHARDS_DIR}/val_*.bin" >/dev/null; then
    echo "Missing BabyLM 2026 val shards: ${BABYSTEPS_BABYLM2026_SHARDS_DIR}/val_*.bin" >&2
    missing=1
  fi

  if [[ "$missing" -ne 0 ]]; then
    cat >&2 <<'EOF'

Build the BabyLM 2026 Strict 16k BPE tokenizer and token shards first:

  conda run --no-capture-output -n babylm python scripts/data/build_token_shards.py \
    --config configs/tokenization/babylm2026_strict_byte_bpe_16384_shards.yaml

For a quick local smoke of the builder, use temporary output dirs and small shard limits:

  conda run --no-capture-output -n babylm python scripts/data/build_token_shards.py \
    --config configs/tokenization/babylm2026_strict_byte_bpe_16384_shards.yaml \
    --tokenizer-output-dir /tmp/babysteps_babylm2026_tok_smoke \
    --shards-output-dir /tmp/babysteps_babylm2026_shards_smoke \
    --shard-tokens 10000 \
    --max-docs 5000 \
    --max-shards 2 \
    --overwrite-tokenizer \
    --overwrite-shards
EOF
    exit 1
  fi
fi

bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
