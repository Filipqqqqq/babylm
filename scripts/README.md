# Scripts

These are the commands a new user needs to reproduce a BabySteps submission.
Run them from the repository root.

The files in this folder are small launchers. They read YAML settings and call
the reusable Python package in `src/babysteps/`. You normally choose a config
here; you only need to edit `src/` when changing how training itself works.

## The usual path

### 1. Prepare token shards

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/submission/murphyslaw_strict_small_shards.yaml
```

Use `sidbert_strict_shards.yaml` or `chatty_bert_strict_shards.yaml` for the
100M systems. The command reads the official Hugging Face dataset and writes the
tokenizer and binary shards to the external cache. See the
[data guide](../data/README.md).

### 2. Train

GPT-BERT models use the hybrid launcher:

```bash
python scripts/train/train_hybrid_lm.py \
  --config configs/train/submission/murphyslaw_strict_small.yaml
```

SidBert requires the two processes recorded by its submission recipe:

```bash
accelerate launch --num_processes 2 scripts/train/train_hybrid_lm.py \
  --config configs/train/submission/sidbert_strict.yaml
```

Causal models use the causal launcher:

```bash
python scripts/train/train_causal_lm.py \
  --config configs/train/submission/chatty_bert_strict.yaml
```

### 3. Validate and upload

```bash
python scripts/hf/upload_babylm_submission.py \
  --run-dir /path/to/run --repo-id USER/MODEL \
  --track strict-small --dry-run
```

The uploader verifies the final model and every official milestone. It creates
private repositories by default.

### 4. Evaluate

```bash
bash scripts/eval/setup_babylm_eval.sh
bash scripts/eval/eval_babylm_strict_small_submission.sh USER/MODEL
```

The Strict-Small wrapper defaults to MNTP for MurphysLaw. The Strict wrapper
defaults to causal for Chatty BERT and accepts `mntp` as its third argument for
SidBert.

## What is kept here

| Path | Purpose |
| --- | --- |
| `data/build_token_shards.py` | Prepare the public corpus, tokenizer, and shards. |
| `train/train_hybrid_lm.py` | Train GPT-BERT with causal and masked objectives. |
| `train/train_causal_lm.py` | Train GPT-2 or Llama-style causal models. |
| `tokenization/prepare_submission_bos_tokenizer.py` | Add Chatty BERT's submitted BOS-by-default behavior. |
| `tokenization/validate_submission_bos_tokenizer.py` | Compare a prepared tokenizer with a reference model. |
| `hf/upload_babylm_submission.py` | Validate and upload all official revisions. |
| `eval/setup_babylm_eval.sh` | Safely apply the pinned evaluator patch. |
| `eval/eval_babylm_*_submission.sh` | Run the two official submission tracks. |

Experiment launchers, sweeps, analysis programs, and synthetic-data generators
are preserved on the
[`jorge/research`](https://github.com/JorgeCMurillo/babylm-baby-steps/tree/jorge/research)
branch. The [root README](../README.md) is the complete beginner workflow.
