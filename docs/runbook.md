# Runbook

## First Setup

1. Copy `manifests/paths.example.yaml` to `manifests/paths.local.yaml`.
2. Edit the local paths for the current machine.
3. Keep raw data, generated caches, checkpoints, and evaluation results outside git.

## Adding a Workflow

1. Start with a config in `configs/`.
2. Put reusable code in `src/babysteps/`.
3. Add a small wrapper in `scripts/` if the workflow needs a command-line entry point.
4. Record generated artifacts in `manifests/datasets.md`.
5. Save run outputs under `runs/` or an external run directory.

## Notebook Policy

Use `notebooks/scratch/` for exploration. When a notebook becomes important, move the stable logic into `src/babysteps/` and keep the notebook as analysis only.
