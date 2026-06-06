# BabyLM Baby Steps

workspace for BabyLM Challenge experiments, starting with the 100M strategy.

This repo is meant to hold reusable code, configs, manifests, docs, and lightweight scripts. It should not become a dump for raw corpora, checkpoints, large pickle caches, or one-off experiment output.

## Layout

```text
babylm-baby-steps/
  configs/        Experiment recipes and data/eval settings.
  manifests/      Machine-local path examples and dataset notes.
  src/babysteps/  Reusable Python package code.
  scripts/        Thin command-line wrappers around package code.
  docs/           Decisions, runbooks, and team notes.
  notebooks/      Scratch analysis only; promote stable logic into src/.
  runs/           Generated experiment outputs; gitignored.
  artifacts/      Generated caches/checkpoints; gitignored.
```

## Local Setup

Copy the example path manifest and fill in paths for the machine you are using:

```bash
cp manifests/paths.example.yaml manifests/paths.local.yaml
```

`manifests/paths.local.yaml` is ignored by git so private paths and machine-specific settings stay local.

## Team Rules

- Keep raw data and generated outputs outside git.
- Commit configs for any run that should be reproducible.
- Put reusable logic in `src/babysteps/`, not only in notebooks.
- Keep scripts thin: they should call package code rather than contain the whole workflow.
- Do not commit secrets such as Hugging Face tokens, API keys, or local credentials.

## Current Status

Initial scaffold. The next step is to promote one workflow at a time from the older BabyLM 10M workspace into this cleaner structure.
