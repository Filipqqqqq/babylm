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

## Constructed Datasets And GPT-BERT Mixes

The current GPT-BERT synthetic-data recipes keep the same model and training
setup: `BabyLM-community/babylm-baseline-10m-gpt-bert-mixed`, the
`gpt_bert_baseline_mixed` tokenizer, 512-token sequences, a 1:1 causal/MNTP
hybrid objective, Muon PE, bf16, and 10 epochs. The text mixer treats each
non-empty line as a document, uses `seed: 42` and `order_seed: 1`, repeats
synthetic corpora before a partial repeat, shuffles selected documents
deterministically, and trims only the final combined corpus if needed to stay
under the 10,000,000-word cap.

The constructed-data families are organized by intended capability, with source
recipes in git and generated text, shards, and checkpoints kept outside it:

| Dataset or mix | Primary target | Starting point |
| --- | --- | --- |
| Linguistic repair (1%) | BLIMP Supplement / QA congruence | [Nested dose-response recipe](configs/experiments/linguistic_repair_dose_response_v1/README.md) |
| Linguistic repair (1.5%) | BLIMP Supplement / QA congruence | [Standalone text-mix config](configs/data_mixes/babylm2026_strict_small_linguistic_repair_1p5pct_seed42.yaml) |
| Targeted QA + physical/material (5.5%) | BLIMP Supplement, EWoK | `configs/data_mixes/*replace5p5pct*` |
| COMPS + targeted QA (8.5%) | COMPS, BLIMP Supplement, EWoK | `configs/data_mixes/*replace8p5pct*` |
| Entity Tracking multidomain v3 | Online state tracking | [Experiment report](docs/entity_tracking_experiment_report.md) · [Dataset](data/container_state/container_state_multidomain_v3_seed42/) |

The first three are reusable training recipes below. The Entity Tracking result
is an experiment report, kept near the end of this README so it does not get in
the way of finding the datasets and launch commands.

### Linguistic-Repair Mixes: Nested 1% and 1.5%

The linguistic-repair dose-response experiment uses nested interventions over
the canonical GPT-BERT mixed shards: the 1% condition replaces 10 of 1,038
loader blocks and the 1.5% condition contains those same 10 blocks plus 6
more. The untreated blocks, validation data, tokenizer, loader shape, and
training configuration remain fixed, so the comparison isolates repair dose.
The source recipe is
`configs/experiments/linguistic_repair_dose_response_v1/`; generated paired
shards and all run outputs live under `${BABYSTEPS_RUNS_DIR}/experiments/` and
are intentionally not versioned.

The established 1.5% Muon recipe is fully specified by these configs:

```bash
python scripts/data/build_text_mix.py \
  --config configs/data_mixes/babylm2026_strict_small_linguistic_repair_1p5pct_seed42.yaml

python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_linguistic_repair_1p5pct_gpt_bert_mixed_shards.yaml

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_mixed_babylm2026_strict_small_linguistic_repair_1p5pct_muon.yaml
```

For the matched 0%/1%/1.5% dose-response datasets, build and launch the
experiment-specific workflow instead:

```bash
python scripts/experiments/build_linguistic_repair_dose_response.py \
  --config configs/experiments/linguistic_repair_dose_response_v1/experiment.yaml

bash scripts/sweeps/launch_gptbert_linguistic_repair_dose_response_tmux.sh
```

The follow-up 1% AdamMuon treatment keeps a 1:1 causal/MNTP objective, uses
AdamMuon LR 0.02 with auxiliary AdamW LR 0.006, applies the 128 -> 256 -> 512
sequence-length curriculum, evaluates the final 20% tail average, and then
queues SuperGLUE at `3e-5` and `8e-5` for pretraining seeds 42 and 43:

```bash
bash scripts/sweeps/launch_adamuon_linguistic_repair_1pct_curriculum_gpu0.sh
```

That launcher requires a free GPU 0 by default. Its post-hoc final benchmark
queue is `scripts/sweeps/run_adamuon_linguistic_repair_final_benchmarks_gpu0.sh`.

### 8.5% COMPS + Targeted QA Mix

This is the current recommended strict-small GPT-BERT synthetic mix. It keeps
the targeted BLIMP Supplement repair slices, physical dynamics, and material
affordance data, then adds a COMPS concept-property slice:

```text
91.524% BabyLM 2026 Strict-Small base
0.500% QA congruence repair with original roles
0.500% QA turn-taking pronoun frames
0.150% QA where destination frames
0.100% QA where local-preposition frames
0.250% QA who/common-name frames
2.000% physical dynamics slip-bridge examples
2.000% material affordance examples
2.976% COMPS concept-property examples
```

Use these configs to build and train it:

```bash
python scripts/data/build_text_mix.py \
  --config configs/data_mixes/babylm2026_strict_small_replace8p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_compscp297k_seed42.yaml

python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_replace8p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_compscp297k_gpt_bert_mixed_shards.yaml

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_mixed_babylm2026_strict_small_replace8p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_compscp297k_muon.yaml
```

### 5.5% Targeted BLIMP Supplement Repair Mix

This focused variant is useful as the targeted BLIMP Supplement repair
comparison. It keeps the same physical/material curriculum as the older 5.0%
reference mix, but replaces the generic easy answer-type QA slice with
original-role congruence, where, and who/name frames:

```text
94.499% BabyLM 2026 Strict-Small base
0.500% QA congruence repair with original roles
0.500% QA turn-taking pronoun frames
0.150% QA where destination frames
0.100% QA where local-preposition frames
0.250% QA who/common-name frames
2.000% physical dynamics slip-bridge examples
2.000% material affordance examples
```

Use these configs to build and train it:

```bash
python scripts/data/build_text_mix.py \
  --config configs/data_mixes/babylm2026_strict_small_replace5p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_seed42.yaml

python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_replace5p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_gpt_bert_mixed_shards.yaml

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_mixed_babylm2026_strict_small_replace5p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_muon.yaml
```

### Local Comparison

The table below compares final epoch in-training diagnostics against the
earlier 5.0% reference mix with easy QA, turn-taking, physical dynamics, and
material affordance. Entity Tracking is omitted here while its scoring path is
under audit. Overall is the unweighted mean of EWoK, BLIMP, BLIMP Supplement,
and COMPS.

| Mix | EWoK | BLIMP | BLIMP Supplement | COMPS | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5.0% reference | 0.5277 | 0.7254 | 0.6780 | 0.5269 | 0.6145 |
| 5.5% targeted BLIMP repair | 0.5284 | 0.7261 | **0.7049** | 0.5282 | 0.6219 |
| 8.5% COMPS + targeted QA | **0.5334** | **0.7331** | 0.7011 | **0.5421** | **0.6274** |

Relative to the 5.0% reference, the 5.5% targeted repair improves BLIMP
Supplement by 2.69 points and the four-way overall score by 0.74 points. The
8.5% COMPS mix is the stronger current overall recipe: +1.52 points on COMPS,
+0.77 points on BLIMP, +2.31 points on BLIMP Supplement, +0.57 points on EWoK,
and +1.29 points on the four-way overall score.

For fast training runs without the expensive benchmark diagnostics, disable the
BLIMP/Entity Tracking/COMPS MNTP likelihood evals and EWoK context evals:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_mixed_babylm2026_strict_small_replace8p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_compscp297k_muon.yaml \
  --set logging.benchmark_eval_at_start=false \
  --set mntp_likelihood.enabled=false \
  --set ewok_context_sensitivity_full.enabled=false
```

This still keeps the lightweight validation pass controlled by
`logging.eval_every_steps`. To skip that too, add
`--set logging.eval_every_steps=0`.

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

Experimental causal-only NextLat support for GPT-BERT hybrid runs is documented
under [Experimental Auxiliary Objectives](#experimental-auxiliary-objectives).

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

To mirror the GPT-BERT-style in-training causal diagnostics for LLaMA, use the
explicit diagnostics recipe. It keeps the same model/training settings, runs
full EWoK, BLiMP, BLiMP Supplement, Entity Tracking, and COMPS likelihood
diagnostics every 50,000,000 estimated words, and writes per-item JSONL plus
summary CSV files under each run's `benchmarks/` directory:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_causal_lm.py \
  --config configs/train/llama_12l896_bpe16384_babylm2026_strict_muon_liger_causal_diagnostics.yaml
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

## Experimental Auxiliary Objectives

These objectives are research ablations. They are disabled by default and are
not part of the recommended baseline recipes.

### Spatial Auxiliary Ranking Loss

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

### NextLat Causal Auxiliary Loss

NextLat is an experimental training-only auxiliary objective for next-latent
prediction. It is disabled by default, does not change the saved Hugging Face
model architecture, and should be treated as a research ablation rather than a
recommended recipe.

For GPT-BERT hybrid training, NextLat runs only on causal microsteps. It is not
applied to MNTP/masked microsteps because bidirectional masked inputs make the
next-latent target ambiguous.

Enable it with a `nextlat` block in a GPT-BERT hybrid config:

```yaml
nextlat:
  enabled: true
  apply_on: causal
  horizon: 1
  lambda_next_h: 0.1
  lambda_kl: 0.0
  mlp_hidden_dim: 1024
  mlp_layers: 3
  detach_targets: true
  mask_eos_bos_boundaries: true
```

Training logs include `nextlat_next_h_loss`,
`nextlat_weighted_next_h_loss`, `nextlat_persistence_loss`, and
`nextlat_improvement_over_persistence`. When KL is enabled, logs also include
`nextlat_kl_weight`, `nextlat_kl_loss`, and `nextlat_weighted_kl_loss`.

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

Run full zero-shot evaluation over every Hugging Face checkpoint in a completed
training run:

```bash
conda run -n babylm python scripts/eval/eval_babylm_checkpoints.py \
  --config configs/eval/full_eval.yaml \
  --run-dir ${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/training/YOUR_RUN \
  --output-dir ${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/evaluation/YOUR_RUN_full_eval \
  --device 1
```

The wrapper writes raw BabyLM outputs plus parsed summaries:

```text
summary.json  # averages and subtask scores by checkpoint
summary.csv   # flattened score table for plotting
```

## Experiment Results

### Entity Tracking

The historical score audit, counterfactual pilots, multidomain-v3 design, and
complete result tables are maintained in the
[Entity Tracking experiment report](docs/entity_tracking_experiment_report.md).

The latest matched three-seed result is a negative transfer finding. Replacing
1% of the corpus with multidomain-v3 text left the clean custom score nearly
flat (-0.39 points), reduced IID accuracy by 5.00 points, and did not produce a
stable OOD improvement. The apparent +4.22-point legacy gain is confounded by
the known `nothing` and continuation-length effects, so the experiment stopped
before the ranking stages.

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
