# Scripts

This folder contains the commands used to prepare data, train models, evaluate
checkpoints, and analyze results. Run these commands from the repository root.

Most files here are small launchers. They read a YAML configuration—a plain-text
recipe containing paths and settings—and call reusable Python code in
`src/babysteps/`. If you want to change how an operation works, look in
`src/babysteps/`; if you only want to run it, start here.

## New Here? Start With a Smoke Check

Install the project, then run the small deterministic CPU check:

```bash
pip install -e ".[dev]"
python scripts/smoke_10_batch_deterministic.py --device cpu
```

This small test confirms that the basic model and training code work. It does
not train a submission model or require the BabyLM dataset.

## Folder Guide

| Folder | What it contains |
| --- | --- |
| `data/` | Commands that prepare text, train tokenizers, and split tokenized text into manageable files called shards. |
| `train/` | The supported GPT-BERT and causal-language-model training launchers. |
| `eval/` | Checkpoint evaluation and focused diagnostic commands. |
| `analysis/` | Commands that summarize results and create tables or plots. |
| `hf/` | Helpers for uploading checkpoints to Hugging Face. |
| `experiments/` | Builders for specific controlled experiments. |
| `sweeps/` | Multi-run and machine-specific GPU launchers. Read each file before running it. |
| `synthetic_data_generation/` | Commands for targeted curricula and synthetic training text. |

## The Usual Workflow

### 1. Prepare data

Data preparation is controlled by a YAML file. For example, this command builds
the tokenizer cache and token shards for the Strict-Small GPT-BERT baseline:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_gpt_bert_baseline_mixed_shards.yaml
```

This step downloads or reads the configured dataset and writes reusable files
under `${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}`. See the
[data guide](../data/README.md) for the expected inputs and outputs.

### 2. Train a model

Use the hybrid launcher for GPT-BERT models and the causal launcher for GPT-2 or
Llama-style models:

```bash
python scripts/train/train_hybrid_lm.py --config <gpt-bert-config.yaml>
python scripts/train/train_causal_lm.py --config <causal-lm-config.yaml>
```

For example, the paper-facing MurphysLaw recipe starts with:

```bash
python scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_baseline_mixed_babylm2026_strict_small_adamuon_lr0p02_aux0p006_tailavg_seed42.yaml
```

Full paper runs require prepared token shards, a bf16-capable NVIDIA GPU, and
substantial time. The [root README](../README.md) identifies the selected
recipes and explains which launcher to use.

### 3. Evaluate saved models

Evaluation depends on the model objective and BabyLM track. Inspect the command
first, then use `--dry-run` to see the external evaluation commands without
executing them:

```bash
python scripts/eval/eval_babylm_checkpoints.py --help

python scripts/eval/eval_babylm_checkpoints.py \
  --run-dir /path/to/training/run \
  --dry-run
```

The official evaluation code is the pinned `external/babylm-eval/` submodule.
GPT-BERT models use masked-next-token-prediction (MNTP) scoring; causal models
use causal scoring. Check the root README before comparing scores produced by
different evaluation methods or task sets.

## Older Root-Level Commands

Files such as `scripts/train_causal_lm.py` and `scripts/build_token_shards.py`
keep older commands working by forwarding them to the organized locations. New
documentation and automation should use `scripts/train/`, `scripts/data/`, and
the other folders above. The root-level smoke-check script is a real test, not
one of these forwarding files.

## More Detailed Guides

- [Project overview and selected paper recipes](../README.md)
- [Data layout and preparation](../data/README.md)
- [Operational runbook](../docs/runbook.md)
- [Synthetic-data workflows](synthetic_data_generation/README.md)
- [Lexical-analysis guide](../src/babysteps/analysis/lexical/README.md)

When adding a new command, keep the launcher small, put reusable logic in
`src/babysteps/`, and put experiment settings in `configs/`.
