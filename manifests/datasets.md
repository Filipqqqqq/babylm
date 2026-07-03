# Dataset Manifest

This manifest explains which data artifacts are part of reproducible workflows,
where they live, and whether they are committed to this repository. It is meant
to complement `data/README.md`, which explains the folder layout in more detail.

## How To Read This

- Raw BabyLM corpora are external inputs. Keep them outside git and point local
  configs or environment variables at them.
- Small synthetic repair corpora and source configs are committed when they are
  useful for reproducing current experiments.
- Large generated caches, token shards, run directories, and evaluation outputs
  stay local.
- The current GPT-BERT repair mix has one checked-in reference `train.txt`
  stored with Git LFS; its bulky document-level sidecar is intentionally omitted.

## External Inputs

| Name | Kind | Location | Git policy | Notes |
| --- | --- | --- | --- | --- |
| BabyLM 2026 Strict-Small base text | raw or regenerated text | `data/analysis/word2vec/base_text/babylm2026_strict_small/train.txt` on the lab machine, or rebuild from the BabyLM 2026 Strict-Small release | Do not commit raw upstream corpus snapshots unless there is an explicit LFS decision. | Main 10M-word base used for strict-small GPT-BERT repair mixes. |
| BabyLM 2026 data root | external corpus root | `BABYLM_DATA_ROOT`, or a machine-specific path in `manifests/paths.local.yaml` | Do not commit. | Used by configs that resolve BabyLM release files directly. |

## Committed Synthetic Source Text

| Name | Kind | Location | Notes |
| --- | --- | --- | --- |
| BLIMP Supplement QA answer-type repair | deterministic linguistic repair text | `data/linguistic_repair/qa_easy_answer_type_frames_50k_seed42/train.txt` | 50k-word patch targeting QA congruence easy failures across where/who/what answer types. |
| BLIMP Supplement turn-taking repair | deterministic linguistic repair text | `data/linguistic_repair/qa_turntaking_pronoun_frames_50k_seed42/train.txt` | 50k-word patch targeting speaker perspective, pronoun switching, and yes/no response framing. |
| Physical dynamics concept curriculum | synthetic concept text | `data/concept_curriculum/physical_dynamics_ewok_micro_1000_slip_bridge_qwen/train.txt` | 2% component in the current GPT-BERT repair mix. |
| Material affordance concept curriculum | synthetic concept text | `data/concept_curriculum/material_affordance_ewok_4000_action_local_v3_1_soft_tap/train.txt` | 2% component in the current GPT-BERT repair mix. |

## Committed Reference Mix

| Name | Kind | Location | Notes |
| --- | --- | --- | --- |
| GPT-BERT BLIMP Supplement repair mix | mixed 10M-word training text | `data/text_mixes/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_seed42/train.txt` | Stored with Git LFS. Composition is 95% BabyLM strict-small, 0.5% QA answer-type repair, 0.5% turn-taking repair, 2% physical dynamics, and 2% material affordance. |

The reference mix directory also includes lightweight sidecars:

- `counts.json`
- `mix_config.json`
- `mix_manifest.json`
- `README.md`

`mix_documents.jsonl` is not committed because it is much larger than the final
word stream and is only needed for detailed provenance inspection.

## Generated Local Caches

| Name | Kind | Default or expected location | Build source |
| --- | --- | --- | --- |
| GPT-BERT repair token shards | tokenized cache | `${BABYSTEPS_CACHE_DIR}/token_shards/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_gptbert` | `configs/tokenization/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_gptbert_shards_seed42.yaml` |
| GPT-BERT baseline mixed tokenizer | tokenizer cache | `${BABYSTEPS_GPT_BERT_BASELINE_MIXED_TOKENIZER_DIR}` | Baseline GPT-BERT mixed setup used by strict-small runs. |
| Training runs | checkpoints and logs | `${BABYSTEPS_RUNS_DIR}/training/...` | Train configs under `configs/train/`. |
| Evaluation outputs | benchmark reports | Stored under each run directory | Produced by scheduled or final evaluation hooks. |

## Rules

- Commit configs, manifests, READMEs, small generated source text, and selected
  reference artifacts that are intentionally tracked with Git LFS.
- Do not commit raw upstream corpora, token shards, checkpoints, run directories,
  evaluation dumps, or large provenance sidecars.
- When adding a reproducible dataset lane, record the source config, seed,
  word budget, tokenizer or shard config if applicable, and whether the artifact
  is committed, LFS-tracked, or local-only.
