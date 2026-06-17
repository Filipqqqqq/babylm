# BabyLM Baby Steps

Team workspace for BabyLM Challenge experiments: reusable training code,
configs, evaluation wrappers, and experiment notes. The current main recipes
are 10M GPT-2 + Muon and 100M LLaMA + Muon + Liger.

Raw corpora, generated token caches, checkpoints, logs, and evaluation outputs
should stay outside git and be referenced through configs or manifests.

## Quick Start

If you cloned the repo without submodules, initialize the BabyLM evaluation
pipeline with:

```bash
git submodule update --init --recursive
```

Install Liger kernels only for the LLaMA + Liger recipes:

```bash
pip install -e ".[liger]"
```

By default, configs write run outputs under:

```text
/SSD2/babylm-baby-steps-runs/training/
```

Set `BABYSTEPS_RUNS_DIR` and `BABYSTEPS_CACHE_DIR` to move runs and generated
caches on another machine.

## Train 10M GPT-2 + Muon

Use this path for the current strongest BabyLM 10M recipe from the local Muon
sweeps. It trains the 8-layer, 512-hidden GPT-2 shape with the 16k byte-BPE
tokenizer, batch size 32, and Muon LR 0.01:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_causal_lm.py \
  --config configs/train/gpt2_8l512_bpe16384_10m_muon.yaml
```

The launcher defaults to the same observed recipe:

```bash
BABYSTEPS_GPUS="0" bash scripts/sweeps/run_muon_lr_sweep.sh
```

For a broader sweep, override the env vars explicitly:

```bash
BABYSTEPS_BPES="8192 16384" BABYSTEPS_BATCH_SIZE=32 \
  BABYSTEPS_MUON_LRS="0.0025 0.005 0.01" BABYSTEPS_SEEDS="42 43 44" \
  BABYSTEPS_GPUS="0 1" bash scripts/sweeps/run_muon_lr_sweep.sh
```

Current 10M defaults reflect the local GPT-2 fast-eval sweep: with the tested
8l512 architecture and batch size 32, `muon_lr=0.01` was best across seeds
42-45 on the mean of the four fast zero-shot task averages.

## Train 10M LLaMA + Muon + Liger

Use this path for the current non-smoke BabyLM 10M LLaMA recipe. It trains an
8-layer, 512-hidden LLaMA-style causal LM with the 16k byte-BPE tokenizer, Muon
optimizer, bf16, and Liger kernels enabled:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_causal_lm.py \
  --config configs/train/llama_8l512_bpe16384_10m_muon_liger.yaml
```

That config intentionally has no `max_train_chunks` or `max_val_chunks`; the
older `llama_tiny_*_smoke.yaml` configs are only for quick capped checks.
Outputs go under `/SSD2/babylm-baby-steps-runs/training/` unless
`BABYSTEPS_RUNS_DIR` is set.

If the tokenizer or chunk pickles are missing, build the 16k byte-BPE assets:

```bash
python scripts/data/train_tokenizer.py \
  --config configs/tokenization/byte_bpe_16384_10m_block512.yaml

python scripts/data/build_token_chunks.py \
  --config configs/tokenization/byte_bpe_16384_10m_block512.yaml
```

The expected generated inputs are:

```text
/SSD2/babylm-baby-steps-cache/tokenizers/babylm10m_byte_bpe_16384/
/SSD2/babylm-baby-steps-cache/token_chunks/babylm10m_byte_bpe_16384_block512_eos/train.pkl
/SSD2/babylm-baby-steps-cache/token_chunks/babylm10m_byte_bpe_16384_block512_eos/dev.pkl
```

For a seed or LR sweep, use the launcher. It now defaults to the full 10M
config rather than a capped smoke config:

```bash
BABYSTEPS_GPUS="0 1" bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
```

For the recommended single run through the launcher:

```bash
BABYSTEPS_GPUS="0" BABYSTEPS_MUON_LRS="0.01" BABYSTEPS_SEEDS="42" \
  bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
```

Current 10M defaults reflect the local LLaMA fast-eval sweep: `muon_lr=0.01`
beat `0.005` across seeds 42-44 on the mean of the four fast zero-shot task
averages, with the tested 8l512 architecture and effective batch size 32.

Local note: the 10M LLaMA model has not been as useful as the 10M GPT-2 + Muon
recipe above, but it is still useful for checking the LLaMA, Muon, and Liger
training path before scaling up.

## Train 100M LLaMA + Muon + Liger

Use this path for the current BabyLM 2026 Strict 100M-track LLaMA default. It
uses the so-far best local LLaMA shape: 12 layers, hidden size 896, 14 attention
heads, 14 key/value heads, MLP size 2304, 16k byte-BPE, Muon LR 0.01, bf16,
Liger kernels, and effective batch size 64 via micro-batch size 16 with 4
gradient accumulation steps:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_causal_lm.py \
  --config configs/train/llama_12l896_bpe16384_babylm2026_strict_muon_liger.yaml
```

The 100M launcher points at the same default and keeps the run names consistent
with the width/depth sweeps:

```bash
BABYSTEPS_GPUS="0" BABYSTEPS_SEEDS="42" \
  bash scripts/sweeps/run_llama_babylm2026_muon_lr_seed_sweep.sh
```

If the BabyLM 2026 Strict token shards are missing, build them first:

```bash
conda run --no-capture-output -n babylm python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_byte_bpe_16384_shards.yaml
```

Project note: LLaMA + Muon + Liger was chosen for the 100M track because it
works well under memory and speed constraints. It may or may not beat a GPT-2
small model at that scale, but it has been much better than the GPT-2 100M
baseline in local comparisons.

## Custom Datasets And Tokenizers

The trainer is intentionally dataset-agnostic: it consumes prebuilt
`token_chunks` pickles or `token_shards` binary files. For new datasets, the
recommended path is to build a byte-BPE tokenizer plus packed `token_shards`,
then point a normal training config at those generated paths.

Notebook option: open `notebooks/custom_dataset_tokenization.ipynb` for a
guided version of the same workflow. It starts with runnable demo text files,
then lets you switch to a Hugging Face dataset, local JSONL file, explicit text
files, or a folder of text files.

For a Hugging Face dataset, copy and edit the template:

```bash
cp configs/tokenization/template_custom_hf_byte_bpe_16384_shards.yaml \
  configs/tokenization/my_dataset_byte_bpe_16384_shards.yaml
```

Set `dataset.name`, optional `dataset.config`, `dataset.split`, and
`dataset.text_field`, then build the tokenizer and shards:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/my_dataset_byte_bpe_16384_shards.yaml
```

For a quick smoke build, cap the amount of data:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/my_dataset_byte_bpe_16384_shards.yaml \
  --max-docs 1000 --max-shards 2 --shard-tokens 100000
```

For local data, copy the local template instead:

```bash
cp configs/tokenization/template_custom_local_byte_bpe_16384_shards.yaml \
  configs/tokenization/my_local_byte_bpe_16384_shards.yaml
```

The default local template expects JSONL rows with a `text` field:

```yaml
dataset:
  kind: local_jsonl
  files:
    train: /path/to/train.jsonl
  split: train
  text_field: text
```

Plain text files are also supported by switching the dataset block:

```yaml
dataset:
  kind: local_text
  files:
    - /path/to/corpus.txt
  text_field: text
  document_mode: file
```

For a folder of text files, use `dir` plus an optional `glob`:

```yaml
dataset:
  kind: local_text
  dir: /path/to/text_folder
  glob: "*.txt"
  recursive: false
  text_field: text
  document_mode: file
```

Use `recursive: true` to match files in nested folders. Use
`document_mode: line` when each non-empty line should be one training document.
Local and Hugging Face configs share the same `tokenizer` and `shards`
sections, so `BABYSTEPS_CUSTOM_TOKENIZER_DIR`, `BABYSTEPS_CUSTOM_SHARDS_DIR`,
and `BABYSTEPS_CACHE_DIR` can move generated outputs without editing every
config.

After building shards, copy the training template and point it at the generated
tokenizer and shard directory:

```bash
cp configs/train/template_llama_token_shards.yaml \
  configs/train/my_dataset_llama_token_shards.yaml

python scripts/train/train_causal_lm.py \
  --config configs/train/my_dataset_llama_token_shards.yaml
```

Token shards are stored as `uint16`, so byte-BPE vocab sizes must stay below
65,536 tokens. The provided templates use 16,384.

## Plain GPT-2 Tokenization Path

Use this secondary path when you want Hugging Face GPT-2 tokenization for the
BabyLM 10M text, separate from the byte-BPE recipes above.

Optionally regenerate cleaned BabyLM text into the shared cache folder:

```bash
python scripts/data/prepare_babylm_text.py --config configs/data/babylm_10m_paths.yaml
```

Build GPT-2 token chunks:

```bash
python scripts/data/build_token_chunks.py --config configs/tokenization/gpt2_pretrained_10m_block512.yaml
```

To tokenize text regenerated by this repo instead of the default cleaned text
source, pass the clean-text directories:

```bash
python scripts/data/build_token_chunks.py \
  --config configs/tokenization/gpt2_pretrained_10m_block512.yaml \
  --clean-train-dir /SSD2/babylm-baby-steps-cache/clean_text/babylm_10m/train \
  --clean-dev-dir /SSD2/babylm-baby-steps-cache/clean_text/babylm_10m/dev
```

Train GPT-2 small from those generated chunks:

```bash
python scripts/train/train_causal_lm.py --config configs/train/gpt2_small_10m_plain_gpt2_chunks.yaml
```

That config points `data.train_chunks` and `data.val_chunks` at the generated
GPT-2 chunk pickles.

## Material-World Synthetic Data

The material-world generator lives under
`scripts/synthetic_data_generation/material_world/` with reusable code in
`src/babysteps/material_world/`. Its catalogues are in
`configs/material_world/`.

Default generation stays broad. EWoK-inspired material properties and dynamics
are opt-in through `coverage_sets.json` and `coverage_mix`; the pipeline can
assign quota-backed rows for EWoK objects, materials, property concepts, or
dynamic concepts while leaving the remaining rows sampled from the full
material/object catalogues. For controlled object exposure, set
`object_sampling.mode` to `balanced_round_robin` so selected objects are assigned
to rows in a seeded, near-even cycle. Use `selection_seed` to make catalogue
order, coverage row assignment, and balanced object assignment semi-deterministic.

For a 500-row broad run with a soft EWoK upweight, prefer a config-level mix:

```json
{
  "coverage_mix": {
    "ewok_material_properties": {
      "object_fraction": 0.25,
      "material_fraction": 0.25
    },
    "ewok_material_dynamics": {
      "dynamic_fraction": 0.10
    }
  },
  "object_sampling": {
    "mode": "balanced_round_robin",
    "scope": "selected_objects"
  },
  "selection_seed": "mw-softmix-001",
  "scenario_limit": 500
}
```

Generated row metadata includes fields such as `coverage_role`,
`target_property_concept`, `target_dynamic_concept`, `target_object_id`, and
`target_material`. Each run also writes `coverage_report.json`, which compares
the requested mix with assigned and observed coverage, plus
`object_sampling_report.json` for assigned and observed object counts. See
`configs/material_world/README.md` for the catalogue rules and examples.

## Layout

```text
babylm-baby-steps/
  configs/        Experiment recipes for data, training, and evaluation.
  docs/           Decisions, runbooks, and project notes.
  external/       Pinned external repos such as the official BabyLM eval code.
  manifests/      Dataset descriptions and local path templates.
  scripts/        Thin command-line wrappers around package code.
  src/babysteps/  Reusable Python package code.
  tests/          Developer regression and smoke tests.
  notebooks/      Scratch exploration; promote stable logic into src/.
  runs/           Generated outputs; gitignored.
  cache/          Generated local caches; gitignored.
```

Preferred command-line entrypoints are grouped under `scripts/data/`,
`scripts/train/`, `scripts/eval/`, `scripts/sweeps/`, `scripts/analysis/`,
`scripts/hf/`, and `scripts/synthetic_data_generation/material_world/`. Some
root-level Python scripts remain as compatibility shims for older commands; new
docs and launchers should prefer the grouped paths.

## Data And Outputs

Do not commit raw datasets, large `.pkl` caches, checkpoints, or generated eval
results. Keep those on local storage, for example:

```text
/SSD2/datasets/
/SSD2/babylm-baby-steps-cache/
/SSD2/babylm-baby-steps-runs/
```

Use `babylm-baby-steps-cache` for reusable generated inputs such as cleaned
text, trained tokenizers, and token chunk pickles. Use
`babylm-baby-steps-runs` for experiment outputs such as checkpoints, metrics,
logs, and evaluation results.

Copy the example path manifest when setting up a new machine:

```bash
cp manifests/paths.example.yaml manifests/paths.local.yaml
```

`manifests/paths.local.yaml` is ignored by git so machine-specific paths stay
local.

## Evaluation Pipeline

The official BabyLM evaluation repository is tracked as a Git submodule:

```text
external/babylm-eval/
```

That keeps the eval code pinned to an exact upstream commit while still letting
us build project-specific wrappers in this repo. The intended boundary is:

```text
external/babylm-eval/          # official eval implementation
src/babysteps/evaluation/      # our reusable wrapper code
scripts/                       # thin commands that run eval recipes
configs/eval/                  # configs for model paths, tasks, and outputs
```

Evaluation outputs should stay outside git, for example under
`/SSD2/babylm-baby-steps-runs/evaluation/`.

Run fast zero-shot evaluation over every Hugging Face checkpoint in a completed
training run:

```bash
conda run -n babylm python scripts/eval/eval_babylm_checkpoints.py \
  --config configs/eval/fast_eval.yaml \
  --run-dir /SSD2/babylm-baby-steps-runs/training/YOUR_RUN \
  --output-dir /SSD2/babylm-baby-steps-runs/evaluation/YOUR_RUN_fast_eval \
  --device 1
```

The wrapper writes raw BabyLM outputs plus parsed summaries:

```text
summary.json  # averages and subtask scores by checkpoint
summary.csv   # flattened score table for plotting
```

## Developer Checks

These checks are guardrails for people editing the training code. They do not
require the older prototype script; the relevant reference behavior is encoded
in the tests.

Run the lightweight parity checks:

```bash
pytest -q tests/test_gpt2_small_parity.py
```

Run the same checks with full GPT-2 small dimensions:

```bash
BABYSTEPS_PARITY_FULL_GPT2_SMALL=1 pytest -q tests/test_gpt2_small_parity.py
```

Run a deterministic 10-batch training smoke test:

```bash
python scripts/smoke_10_batch_deterministic.py
```

Run the 10-batch smoke on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_10_batch_deterministic.py --device cuda
```

See `docs/runbook.md` for workflow-specific dry-runs and launch checks.

## Team Rules

- Keep raw data and generated outputs outside git.
- Commit configs for runs that should be reproducible.
- Put reusable logic in `src/babysteps/`, not only in notebooks.
- Keep scripts thin: they should wire configs into package code.
- Do not commit secrets such as Hugging Face tokens, GitHub PATs, API keys, or
  local credentials.
