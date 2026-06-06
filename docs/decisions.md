# Decision Log

## 2026-06-06 - Create a Clean BabyLM 100M Workspace

Use this repository as the team-facing workspace for BabyLM 100M strategy work.

Keep the older BabyLM 10M folder as reference material, but do not copy its whole layout into this repo. Promote workflows gradually into `src/`, `configs/`, and `scripts/`.

Raw data, checkpoints, large pickle files, and evaluation outputs should live outside git and be referenced through configs or manifests.

## 2026-06-06 - Keep GPT-2 Small Baseline Close to Old `train_gpt.py`

The first training implementation should stay close to the older BabyLM 10M `train_gpt.py` path while removing hardcoded secrets, hub push side effects, and output clutter.

Parity coverage now checks token collation, causal-LM loss, one optimizer/scheduler step, and deterministic 10-batch smoke behavior. This lets future changes move toward a cleaner team workflow without losing the known-good baseline.
