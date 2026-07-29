# Runbook

## First Setup

1. Clone with submodules, or initialize them after cloning:

```bash
git submodule update --init --recursive
```

2. Export `BABYLM_DATA_ROOT`, `BABYSTEPS_CACHE_DIR`, or `BABYSTEPS_RUNS_DIR`
   when the defaults do not match the current machine.
3. Keep raw data, caches, checkpoints, and evaluation results outside Git.

## Adding a Workflow

1. Start with a config in `configs/`.
2. Put reusable code in `src/babysteps/`.
3. Add a small wrapper in `scripts/` if the workflow needs a command-line entry point.
4. Record detailed provenance in a sidecar manifest beside the generated data.
5. Save run outputs under `runs/` or an external run directory.

## Submission Checks

Model users can run the deterministic CPU smoke check without learning the test
suite:

```bash
python scripts/smoke_10_batch_deterministic.py --device cpu
```

Maintainers can run the compact submission checks before publishing:

```bash
pytest -q tests/test_submission_configs.py \
  tests/test_submission_training_smoke.py \
  tests/test_submission_evaluation.py
```

## GPT-2 + Muon 10M Runs

Use the full non-smoke config for the current strongest 10M GPT-2 + Muon
recipe:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_causal_lm.py \
  --config configs/train/gpt2_8l512_bpe16384_10m_muon.yaml
```

This recipe uses the observed 10M defaults from the local fast-eval sweep:
GPT-2 8 layers, 512 hidden size, 8 heads, 16k byte-BPE, batch size 32, Muon LR
0.01, warmup ratio 0.03, and the standard full 10M chunks.

Dry-run the launcher default before launching:

```bash
BABYSTEPS_DRY_RUN=1 BABYSTEPS_GPUS="0" bash scripts/sweeps/run_muon_lr_sweep.sh
```

Then launch the same single recommended run:

```bash
BABYSTEPS_GPUS="0" bash scripts/sweeps/run_muon_lr_sweep.sh
```

For a broader sweep, override the recipe explicitly:

```bash
BABYSTEPS_BPES="8192 16384" BABYSTEPS_BATCH_SIZE=32 \
  BABYSTEPS_MUON_LRS="0.0025 0.005 0.01" BABYSTEPS_SEEDS="42 43 44" \
  BABYSTEPS_GPUS="0 1" bash scripts/sweeps/run_muon_lr_sweep.sh
```

## LLaMA + Liger 10M Runs

Use the full non-smoke config for the current 10M LLaMA + Muon + Liger recipe:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_causal_lm.py \
  --config configs/train/llama_8l512_bpe16384_10m_muon_liger.yaml
```

This recipe uses the observed 10M defaults from the local fast-eval sweep:
8 layers, 512 hidden size, 8 heads, LLaMA MLP size 1280, effective batch size
32, Muon LR 0.01, warmup ratio 0.03, bf16, and Liger enabled. The config does
not cap train or validation chunks.

For the launcher path, dry-run the exact command first:

```bash
BABYSTEPS_DRY_RUN=1 BABYSTEPS_GPUS="0" BABYSTEPS_MUON_LRS="0.01" \
  BABYSTEPS_SEEDS="42" bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
```

Then launch the same single recommended run:

```bash
BABYSTEPS_GPUS="0" BABYSTEPS_MUON_LRS="0.01" BABYSTEPS_SEEDS="42" \
  bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
```

For a capped smoke, keep using the tiny smoke configs or set temporary caps:

```bash
BABYSTEPS_MAX_TRAIN_CHUNKS=1024 BABYSTEPS_MAX_VAL_CHUNKS=128 \
  BABYSTEPS_EPOCHS=1 BABYSTEPS_SAVE_FINAL=false \
  bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
```

Local note: the 10M LLaMA recipe has been less useful than the 10M GPT-2 +
Muon recipe, but it is a good integration check for the LLaMA, Muon, and Liger
training path.

## LLaMA + Liger 100M Runs

Use the current BabyLM 2026 Strict default config for the 100M-track LLaMA +
Muon + Liger recipe:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n babylm \
  python -u scripts/train/train_causal_lm.py \
  --config configs/train/llama_12l896_bpe16384_babylm2026_strict_muon_liger.yaml
```

The default shape is 12 layers, hidden size 896, 14 attention heads, 14
key/value heads, MLP size 2304, LLaMA dropout 0.1, 16k byte-BPE, Muon LR 0.01,
bf16, Liger enabled, and effective batch size 64 via micro-batch size 16 with
4 gradient accumulation steps. This is the current default because it is the
best local LLaMA full zero-shot aggregate so far among the completed BabyLM
2026 Strict runs.

Build the token shards if they are missing:

```bash
conda run --no-capture-output -n babylm python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_byte_bpe_16384_shards.yaml
```

Dry-run the launcher default:

```bash
BABYSTEPS_DRY_RUN=1 BABYSTEPS_GPUS="0" BABYSTEPS_SEEDS="42" \
  bash scripts/sweeps/run_llama_babylm2026_muon_lr_seed_sweep.sh
```

Then launch:

```bash
BABYSTEPS_GPUS="0" BABYSTEPS_SEEDS="42" \
  bash scripts/sweeps/run_llama_babylm2026_muon_lr_seed_sweep.sh
```

Project note: LLaMA + Muon + Liger is the 100M path mainly for memory and speed.
It may or may not beat GPT-2 small at the same scale, but local comparisons
show it is much stronger than the GPT-2 100M baseline.

## BabyLM Evaluation Submodule

The official evaluation pipeline lives at:

```text
external/babylm-eval/
```

Update it intentionally, then commit the changed submodule pointer:

```bash
cd external/babylm-eval
git pull
cd ../..
git status
```

Keep wrappers and local integration code in `src/babysteps/evaluation/`, not by
editing the upstream submodule directly.

## Notebook Policy

Use `notebooks/scratch/` for exploration. When a notebook becomes important, move the stable logic into `src/babysteps/` and keep the notebook as analysis only.
