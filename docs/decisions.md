# Decision Log

## 2026-06-06 - Create a Clean BabyLM 100M Workspace

Use this repository as the team-facing workspace for BabyLM 100M strategy work.

Keep the older BabyLM 10M folder as reference material, but do not copy its whole layout into this repo. Promote workflows gradually into `src/`, `configs/`, and `scripts/`.

Raw data, checkpoints, large pickle files, and evaluation outputs should live outside git and be referenced through configs or manifests.

## 2026-06-06 - Keep GPT-2 Small Baseline Close to Old `train_gpt.py`

The first training implementation should stay close to the older BabyLM 10M `train_gpt.py` path while removing hardcoded secrets, hub push side effects, and output clutter.

Parity coverage now checks token collation, causal-LM loss, one optimizer/scheduler step, and deterministic 10-batch smoke behavior. This lets future changes move toward a cleaner team workflow without losing the known-good baseline.

## 2026-06-06 - Track BabyLM Eval As A Submodule

Use `external/babylm-eval/` as a Git submodule for the official BabyLM
evaluation pipeline.

This keeps evaluation reproducible by pinning the upstream repo to a specific
commit. Project-specific integration code should live in
`src/babysteps/evaluation/`, with scripts and configs in this repo. Do not edit
the submodule for local wrapper behavior unless the change should become an
upstream BabyLM eval contribution.

## 2026-06-21 - Prefer Current BabyLM 2026 Shard Workflows

Treat the BabyLM 2026 token-shard path as the main training workflow. The
active default recipes are the 100M-track LLaMA + Muon + Liger config for
`BabyLM-community/BabyLM-2026-Strict` and the GPT-2 + Muon config for
`BabyLM-community/BabyLM-2026-Strict-Small`.

Prune old full-run BabyLM 10M configs and launchers once they stop representing
the current workflow. The removed 10M GPT-2/LLaMA Muon configs, old BPE sweep
launchers, and matching eval wrappers are still recoverable from git history,
but they should not stay in the top-level config/script surface as defaults.
Smoke fixtures remain under `configs/train/smoke/`, and the remaining
chunk-based tokenization recipes live under
`configs/tokenization/legacy_10m_chunks/`.

Keep `template_llama_token_shards.yaml` as a copy/edit starter for custom
datasets. It is not an experiment result or an old 10M recipe; it documents the
expected shape of a LLaMA + Muon config after a user builds their own
tokenizer and token shards.

Write generated outputs outside the repository by default via
`${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}`. The repo no longer tracks a
`runs/` placeholder, though repo-local `runs/**` stays ignored as a safety net
for accidental scratch outputs.
