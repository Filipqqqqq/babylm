# BabyLM Baby Steps

Clean workspace for BabyLM Challenge experiments.

This repo is meant to be the team-facing place for reusable training code,
configs, smoke tests, evaluation wrappers, and experiment notes. Raw corpora,
large token caches, checkpoints, and run outputs should live outside git and be
referenced through configs or manifests.

The first supported workflow is a minimal GPT-2 small causal-LM training path
for existing BabyLM 10M token chunks. It is intentionally small so the repo can
grow toward BabyLM 100M, LLaMA-style models, tokenizer experiments, Muon, and
other objectives without inheriting the clutter of older prototype folders.

## Quick Start

From the repo root:

```bash
python scripts/train_gpt2_small.py --config configs/train/gpt2_small_10m_smoke.yaml
```

This reads pre-tokenized BabyLM 10M chunk pickles from the older local workspace
and writes checkpoints/metrics under:

```text
/SSD2/babylm-baby-steps-runs/training/
```

The current training path is:

```text
configs/train/gpt2_small_10m_smoke.yaml  # run recipe
scripts/train_gpt2_small.py              # command wrapper
src/babysteps/data/token_chunks.py       # token-chunk Dataset/DataLoader helpers
src/babysteps/training/causal_lm_loop.py # causal LM train/eval loop
```

## Layout

```text
babylm-baby-steps/
  configs/        Experiment recipes for data, training, and evaluation.
  docs/           Decisions, runbooks, and project notes.
  manifests/      Dataset descriptions and local path templates.
  scripts/        Thin command-line wrappers around package code.
  src/babysteps/  Reusable Python package code.
  tests/          Developer regression and smoke tests.
  notebooks/      Scratch exploration; promote stable logic into src/.
  runs/           Generated outputs; gitignored.
  artifacts/      Generated caches/checkpoints; gitignored.
```

## Data And Outputs

Do not commit raw datasets, large `.pkl` caches, checkpoints, or generated eval
results. Keep those on local storage, for example:

```text
/SSD2/datasets/
/SSD2/babylm-baby-steps-artifacts/
/SSD2/babylm-baby-steps-runs/
```

Copy the example path manifest when setting up a new machine:

```bash
cp manifests/paths.example.yaml manifests/paths.local.yaml
```

`manifests/paths.local.yaml` is ignored by git so machine-specific paths stay
local.

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

Recent local checks passed with full GPT-2 small parity and CUDA 10-batch smoke
runs. See `docs/runbook.md` for the current check sequence.

## Team Rules

- Keep raw data and generated outputs outside git.
- Commit configs for runs that should be reproducible.
- Put reusable logic in `src/babysteps/`, not only in notebooks.
- Keep scripts thin: they should wire configs into package code.
- Do not commit secrets such as Hugging Face tokens, GitHub PATs, API keys, or
  local credentials.

## Roadmap

Near-term:

- Run a 100-500 batch GPT-2 small training smoke.
- Confirm checkpoint saving and metrics output.
- Add a cleaner BabyLM evaluation pipeline wrapper.
- Add BabyLM 100M data and training configs.

Likely future modules:

```text
src/babysteps/models/        # GPT-2, LLaMA, registries
src/babysteps/tokenization/  # GPT-2 BPE, SentencePiece, Morfessor, custom BPE
src/babysteps/objectives/    # causal LM, masked LM, diffusion/hybrid objectives
src/babysteps/training/      # optimizers, schedules, checkpointing, fused paths
src/babysteps/evaluation/    # BabyLM pipeline wrappers and result summaries
```

Longer-term experiments may include LLaMA-style BabyLM models, Muon optimizer
comparisons, tokenizer ablations, BabyLM 100M baselines, and diffusion or
diffusion/causal-LM hybrid pretraining ideas.
