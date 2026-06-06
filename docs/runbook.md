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

## GPT-2 10M Baseline Checks

Before changing the GPT-2 small training path, run the parity checks:

```bash
pytest -q tests/test_gpt2_small_parity.py
```

For a stronger check with full GPT-2 small dimensions:

```bash
BABYSTEPS_PARITY_FULL_GPT2_SMALL=1 pytest -q tests/test_gpt2_small_parity.py
```

For a deterministic 10-batch training smoke test:

```bash
python scripts/smoke_10_batch_deterministic.py
```

For the same check on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_10_batch_deterministic.py --device cuda
```

These checks compare the new implementation with the relevant behavior of the older BabyLM 10M `train_gpt.py` without importing that script or triggering its Hugging Face side effects.

## Notebook Policy

Use `notebooks/scratch/` for exploration. When a notebook becomes important, move the stable logic into `src/babysteps/` and keep the notebook as analysis only.
