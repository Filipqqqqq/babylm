# Blessed Synthetic Data

This manifest keeps the current synthetic-data surface small. Files and runs
outside these lanes are historical experiments and should live under an
`archive/` subfolder near their original location.

## Blessed Lanes

| Lane | Active artifacts | Notes |
| --- | --- | --- |
| Baseline strict-small | `configs/tokenization/babylm2026_strict_small_byte_bpe_16384_shards.yaml`; `configs/train/gpt2_8l512_bpe16384_babylm2026_strict_small_muon.yaml` | Non-synthetic baseline used as the control point for strict-small comparisons. |
| Physical micro | `configs/concept_curriculum/physical_dynamics/physical_dynamics_ewok_micro_500.json`; `data/concept_curriculum/physical_dynamics_ewok_micro_1000_slip_bridge_qwen` | The run manifest records the 500-spec config as source; the generation command scaled it to 1,000 planned documents. |
| Material affordance | `data/concept_curriculum/material_affordance_ewok_4000_action_local_v3_1_soft_tap` | The generated run is blessed. Its manifest records `configs/concept_curriculum/material_affordance_ewok_v3_1_soft_tap.json`, which is not currently present in the active config tree. The older 20-document pilot is archived. |
| BLIMP Supplement linguistic repair | `configs/linguistic_repair/qa_easy_answer_type_frames_50k_seed42.yaml`; `configs/linguistic_repair/qa_turntaking_pronoun_frames_50k_seed42.yaml`; `data/linguistic_repair/qa_easy_answer_type_frames_50k_seed42`; `data/linguistic_repair/qa_turntaking_pronoun_frames_50k_seed42` | Deterministic 50k-word QA answer-type and turn-taking patches used by the current GPT-BERT reference mix. |
| BLIMP Supplement + concept reference mix | `configs/data_mixes/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_seed42.yaml`; `configs/tokenization/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_gptbert_shards_seed42.yaml`; `configs/train/gptbert_babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_muon_seed42.yaml` | Replaces 5% of strict-small with 0.5% QA answer-type, 0.5% turn-taking, 2% physical dynamics, and 2% material affordance synthetic text. Generated 10M text mixes and token shards are not stored in git. |
| Spatial context-rank auxiliary | `configs/data/spatial_relations_synth_context_rank.yaml`; `configs/train/probes/llama_12l896_spatial_context_rank_probe_400.yaml`; `configs/train/probes/gpt_bert_mixed_epoch5_spatial_context_rank_probe_400.yaml` | The LLaMA full-run config keeps a disabled `spatial_auxiliary` block pointing at the same generated JSONL path. |

## Archive Rule

- Keep active directories limited to the blessed configs and generated runs.
- Move exploratory configs, ablations, leaky EWoK correct-pair corpora, and
  superseded generated outputs into an `archive/` child directory.
- Archived configs should keep their paths loadable when tests still exercise
  historical behavior.
