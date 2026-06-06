# Dataset Manifest

This file tracks the datasets and generated caches used by the project. Keep this human-readable and update it whenever a new dataset, tokenizer cache, or retrieval cache becomes part of a reproducible workflow.

| Name | Kind | Location | Notes |
| --- | --- | --- | --- |
| BabyLM 100M raw train | raw text | See `manifests/paths.local.yaml` | Do not commit raw text to git. |
| BabyLM 100M GPT-2 chunks | processed cache | See `configs/data/babylm_100m.yaml` | Generated from raw train files. |

## Rules

- Use stable paths for raw data and heavy generated files.
- Commit only lightweight manifests and configs.
- Record tokenizer, block size, seed, and preprocessing choices for every generated cache.
