# BabyLM Baby Steps

Team workspace for BabyLM Challenge experiments: reusable training code,
configs, evaluation wrappers, and experiment notes. The current main recipes
are BabyLM 2026 Strict-Small GPT-2 + Muon and BabyLM 2026 Strict 100M
LLaMA + Muon + Liger.

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

Install analysis-only dependencies for plotting and lexical/tokenizer reports:

```bash
pip install -e ".[analysis]"
python -m spacy download en_core_web_sm
```

By default, configs write run outputs under:

```text
${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/training/
```

Set `BABYSTEPS_RUNS_DIR` and `BABYSTEPS_CACHE_DIR` to move runs and generated
caches on another machine.

## Analysis And Lexical Reports

The lexical-analysis implementation lives in
`src/babysteps/analysis/lexical/`. That folder contains the reusable code for
target matching, lemmatized word counts, tokenizer fragmentation reports,
context examples, domain-frequency plots, and evaluation target export.

User-facing entrypoints are under `scripts/analysis/`, and their configs live
under `configs/analysis/`:

```text
src/babysteps/analysis/lexical/       reusable lexical-analysis package
scripts/analysis/report_lexical_analysis.py
scripts/analysis/export_ewok_lexical_targets.py
scripts/analysis/export_comps_lexical_targets.py
scripts/analysis/plot_comps_lexical_frequencies.py
scripts/analysis/plot_domain_concept_frequencies.py
scripts/analysis/benchmark_spacy_pipe.py
configs/analysis/                    dataset/report configs and target lists
```

Analysis outputs are written outside git by default, under:

```text
${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/analysis/
```

See `src/babysteps/analysis/lexical/README.md` for the lexical-analysis guide,
and `scripts/README.md` for the current command list.

GPT-BERT hybrid runs generate training-loss dynamics plots after benchmark
evals, under `loss_dynamics/`. To compare optimizer/model learning dynamics
across completed runs, plot training loss from one or more run
`metrics_history.json` files. Use `words_seen` when the runs logged a
words/token estimate:

```bash
python scripts/analysis/plot_training_loss_dynamics.py \
  ~/babylm-baby-steps-runs/training/run_a/metrics_history.json \
  ~/babylm-baby-steps-runs/training/run_b/metrics_history.json \
  --label llama_muon \
  --label gpt_bert_lamb \
  --output-dir ~/babylm-baby-steps-runs/analysis/loss_dynamics
```

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

## Quick Modern Smoke Train

The default smoke config uses BabyLM 2026 Strict-Small token shards and the
local 16k byte-BPE tokenizer. It is capped to a small number of shard blocks so
it is useful for checking the trainer, plotting hooks, and checkpoint writing:

```bash
python scripts/train/train_causal_lm.py \
  --config configs/train/smoke/gpt2_bpe16384_babylm2026_strict_small_smoke.yaml
```

If the Strict-Small shards are missing, build them first:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_byte_bpe_16384_shards.yaml
```

Project note: LLaMA + Muon + Liger was chosen for the 100M track because it
works well under memory and speed constraints. It may or may not beat a GPT-2
small model at that scale, but it has been much better than the GPT-2 100M
baseline in local comparisons.

## Train 2026 Strict-Small GPT-2 + Muon

Use this path for BabyLM 2026 Strict-Small. It trains the 8-layer, 512-hidden
GPT-2 + Muon shape from prebuilt token shards for
`BabyLM-community/BabyLM-2026-Strict-Small`, using bf16 mixed precision by
default:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_byte_bpe_16384_shards.yaml

python scripts/train/train_causal_lm.py \
  --config configs/train/gpt2_8l512_bpe16384_babylm2026_strict_small_muon.yaml
```

## Train 2026 Strict-Small GPT-BERT Hybrid

Use this path for the GPT-BERT baseline-style hybrid objective. It uses the
baseline tokenizer from
`BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus`, rebuilds
Strict-Small token shards with that tokenizer, then trains GPT-BERT with a
15:1 causal-to-MNTP schedule, LAMB, bf16, z-loss, and 10 full epochs by
default. The objective ratio is set explicitly with `hybrid.causal_steps: 15`
and `hybrid.masked_steps: 1`:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_gpt_bert_baseline_causal_focus_shards.yaml

python scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_baseline_causal_focus_babylm2026_strict_small.yaml
```

The generated tokenizer and shards go under:

```text
${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/tokenizers/gpt_bert_baseline_causal_focus
${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/token_shards/babylm2026_strict_small_gpt_bert_baseline_causal_focus
```

The training config uses `seq_len: 512`, `training.batch_size: 32`, and
`training.gradient_accumulation_steps: 1`, so the default effective batch is
32 sequences per optimizer update. For a micro-batch size of 16 while keeping
the same global batch size, run:

```bash
python scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_baseline_causal_focus_babylm2026_strict_small.yaml \
  --set training.batch_size=16 \
  --set training.gradient_accumulation_steps=2
```

MNTP BabyLM benchmark scoring is enabled at training start and then every
10,000,000 estimated words with `score_reduction: sum`; plots are generated
after each benchmark eval using `progress.benchmark_plot_x_axis: words_seen`.
Those diagnostics are written inside the run directory:

```text
benchmarks/mntp_likelihood/likelihood_items.jsonl
benchmarks/mntp_likelihood/likelihood_summary.csv
benchmarks/mntp_likelihood/{blimp,blimp_supplement,entity_tracking,comps}/
```

GPT-BERT checkpoints are saved in the baseline-compatible remote-code format,
including `pytorch_model.bin`, tokenizer files, `configuration_gpt_bert.py`,
`modeling_gpt_bert.py`, and `config.json`.

## GPT-BERT 96/2/2 Text-Mix Experiment

Use this path for the seed-42 strict-small replacement experiment that keeps a
10,000,000-word cap while mixing approximately 96% BabyLM, 2% slip-bridge
physical dynamics, and 2% material affordance raw text. The mixer writes a
plain `train.txt` plus provenance sidecars before tokenization:

```bash
python scripts/data/build_text_mix.py \
  --config configs/data_mixes/babylm2026_strict_small_replace4pct_phys2_mat2_seed42.yaml
```

That creates:

```text
data/text_mixes/babylm2026_strict_small_replace4pct_phys2_mat2_seed42/train.txt
data/text_mixes/babylm2026_strict_small_replace4pct_phys2_mat2_seed42/mix_config.yaml
data/text_mixes/babylm2026_strict_small_replace4pct_phys2_mat2_seed42/mix_manifest.json
data/text_mixes/babylm2026_strict_small_replace4pct_phys2_mat2_seed42/mix_documents.jsonl
data/text_mixes/babylm2026_strict_small_replace4pct_phys2_mat2_seed42/counts.json
```

Then build GPT-BERT mixed-focus token shards from the mixed raw text:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_replace4pct_phys2_mat2_gpt_bert_mixed_shards.yaml
```

The baseline control uses the existing strict-small mixed-focus config:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_baseline_mixed_babylm2026_strict_small_muon.yaml
```

Run the 96/2/2 mixed condition with the same model, seed, and optimizer
settings, changing only the shard path and run name:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_mixed_babylm2026_strict_small_replace4pct_phys2_mat2_muon.yaml
```

The text mixer treats each non-empty line as a document, uses seeded
pseudo-random sampling with `seed: 42`, repeats synthetic corpora before a
partial repeat, shuffles selected documents deterministically, and only trims
the final combined corpus if needed to stay under the cap.

## Synthetic Data Generation

Synthetic data generators live under `scripts/synthetic_data_generation/`.
There are two main branches:

```text
linguistic_repair/   BLIMP Supplement QA, answer-type, and turn-taking repairs
concept_curriculum/  EWoK-style physical dynamics and material affordance data
```

The shared workflow is:

```text
generate -> audit -> mix into BabyLM -> tokenize -> train -> evaluate
```

The current reader-facing strict-small GPT-BERT reference path is:

```text
configs/data_mixes/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_seed42.yaml
configs/tokenization/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_gptbert_shards_seed42.yaml
configs/train/gptbert_babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_muon_seed42.yaml
```

See `scripts/synthetic_data_generation/README.md` for the full synthetic-data
workflow and `scripts/synthetic_data_generation/linguistic_repair/README.md`
for the BLIMP Supplement repair notes.

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
python scripts/data/build_token_chunks.py \
  --config configs/tokenization/legacy_10m_chunks/gpt2_pretrained_10m_block512.yaml
```

To tokenize text regenerated by this repo instead of the default cleaned text
source, pass the clean-text directories:

```bash
python scripts/data/build_token_chunks.py \
  --config configs/tokenization/legacy_10m_chunks/gpt2_pretrained_10m_block512.yaml \
  --clean-train-dir ${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/clean_text/babylm_10m/train \
  --clean-dev-dir ${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/clean_text/babylm_10m/dev
```

Train GPT-2 small from those generated chunks:

```bash
python scripts/train/train_causal_lm.py --config configs/train/gpt2_small_10m_plain_gpt2_chunks.yaml
```

That config points `data.train_chunks` and `data.val_chunks` at the generated
GPT-2 chunk pickles. The old 10M chunk smoke recipe now lives under
`configs/train/legacy/gpt2_small_10m_chunks_smoke.yaml` so the default GPT-2
smoke path uses the current BabyLM 2026 shard setup.

## Spatial Auxiliary Ranking Loss

The causal LM trainer can add a small spatial-relation ranking objective on top
of next-token prediction. The main path uses EWoK-style `context_rank` rows:
the model should score each target higher in its matching context than in the
mismatched context.

Generate the default synthetic spatial-relation JSONL first:

```bash
python scripts/data/generate_spatial_relations.py \
  --config configs/data/spatial_relations_synth_context_rank.yaml
```

That writes to:

```text
${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/synthetic/spatial_relations/spatial_relations_synth_context_rank.jsonl
```

Then enable the `spatial_auxiliary` block in a causal-LM config. The LLaMA
100M config already includes the block with `enabled: false`; flip it on or
copy this shape into another causal-LM config:

```yaml
spatial_auxiliary:
  enabled: true
  path: ${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/synthetic/spatial_relations/spatial_relations_synth_context_rank.jsonl
  objective: context_rank
  batch_size: 16
  loss_weight: 0.03
  temperature: 1.0
  score_reduction: mean
  every_steps: 1
  eval_items: 300
  eval_every_steps: 50
  eval_batch_size: 64
```

Train with the normal causal-LM entrypoint:

```bash
python scripts/train/train_causal_lm.py \
  --config configs/train/llama_12l896_bpe16384_babylm2026_strict_muon_liger.yaml \
  --set spatial_auxiliary.enabled=true
```

For a short continuation probe, use
`configs/train/probes/llama_12l896_spatial_context_rank_probe_400.yaml` and set
`BABYSTEPS_LLAMA_PROBE_INIT_FROM` to the checkpoint to continue from.

Training logs include `spatial_auxiliary_loss`,
`spatial_auxiliary_weighted_loss`, and optional `spatial_auxiliary_eval`
records in `metrics_history.json`.

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
  cache/          Generated local caches; gitignored.
```

Preferred command-line entrypoints are grouped under `scripts/data/`,
`scripts/train/`, `scripts/eval/`, `scripts/sweeps/`, `scripts/analysis/`, and
`scripts/hf/`. Some root-level Python scripts remain as compatibility shims for
older commands; new docs and launchers should prefer the grouped paths.

Deprecated root aliases that duplicated newer entrypoints have been removed:
use `scripts/data/build_token_shards.py` instead of
`scripts/build_hf_token_shards.py`, and use
`scripts/train/train_causal_lm.py` instead of `scripts/train_gpt2_small.py`.

## Data And Outputs

Do not commit raw datasets, large `.pkl` caches, checkpoints, or generated eval
results. Keep those on local storage, for example:

```text
${BABYLM_DATA_ROOT:-~/babylm-data}/
${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/
${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/
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
`${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/evaluation/`.

Run fast zero-shot evaluation over every Hugging Face checkpoint in a completed
training run:

```bash
conda run -n babylm python scripts/eval/eval_babylm_checkpoints.py \
  --config configs/eval/fast_eval.yaml \
  --run-dir ${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/training/YOUR_RUN \
  --output-dir ${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/evaluation/YOUR_RUN_fast_eval \
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
