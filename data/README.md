# Data Directory

This directory holds generated or prepared text artifacts used by BabyLM
experiments. Most files here are reproducible outputs rather than hand-written
source code. Keep this directory small in git: commit only blessed, useful
artifacts and metadata; leave large exploratory outputs local or move them to
external storage.

## Main Categories

```text
linguistic_repair/
  Synthetic language-repair datasets, mostly for BLIMP Supplement behavior.

concept_curriculum/
  Synthetic concept-curriculum datasets, mostly for EWoK-style physical and
  material-affordance behavior.

container_state/
  Naturalized container-state training pools and held-out diagnostics.

text_mixes/
  Final mixed training corpora built from BabyLM base text plus synthetic
  sources. These are the inputs to tokenization.

token_shards/
  Blessed, directly reusable tokenized corpora committed only when an exact
  training-input reproduction is more important than repository size.

analysis/
  Local analysis inputs, intermediate data, model outputs, and reports.

material_world/
  Material-world pilots and local generated artifacts.
```

## Linguistic Repair

Use `data/linguistic_repair/` for synthetic datasets that target language or
dialogue behavior. Examples include:

```text
qa_easy_answer_type_frames_50k_seed42/
qa_turntaking_pronoun_frames_50k_seed42/
qa_location_verb_frames_sample500_seed42/
qa_who_person_frames_sample500_seed42/
qa_what_object_frames_sample500_seed42/
```

Common files:

```text
train.txt          plain text used by text mixing
documents.jsonl    structured generated examples
audit_report.json  generator or audit checks
manifest.json      generation settings and provenance
mix_config.yaml    exact-word subset/mix config for derived 50k patches
mix_manifest.json  provenance for derived patches
counts.json        word counts and source totals
```

The 50k QA answer-type and turn-taking patches are blessed source artifacts for
the current GPT-BERT BLIMP Supplement repair mix.

## Concept Curriculum

Use `data/concept_curriculum/` for concept-focused synthetic data. The current
main families are:

```text
physical_dynamics_ewok_micro_1000_slip_bridge_qwen/
material_affordance_ewok_4000_action_local_v3_1_soft_tap/
```

Common files:

```text
scenario_plans.jsonl     planned source scenarios
documents_draft.jsonl    generated draft documents
documents_audited.jsonl  generated documents with audit results
audit_report.json        aggregate audit summary
manifest.json            generation settings and provenance
train.txt                plain text used by text mixing
```

## Container State

Use `data/container_state/` for deterministic state-tracking corpora generated
from symbolic storage worlds. The v1 source directory contains:

```text
train.txt                       naturalized pretraining documents
documents.jsonl                 initial states, operations, final states, and text
audit_report.json               replay, leakage, vocabulary, and distribution checks
manifest.json                   generation settings and provenance
validation/entity_tracking/     held-out likelihood diagnostics, never training data
```

The active 1% strict-small experiment samples exactly 100,000 unique words from
the larger generated pool and combines them with 9,900,000 BabyLM words.

Renderer checkpoint and failure folders, such as `render_checkpoints/`,
`render_failures/`, `repair_checkpoints/`, and `repair_failures/`, are local
generation artifacts. Do not commit them unless a specific reproduction need
requires it.

## Text Mixes

Use `data/text_mixes/` for full training text mixes produced by
`scripts/data/build_text_mix.py`.

The current reader-facing reference mix is:

```text
data/text_mixes/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_seed42/
```

It contains:

```text
train.txt          10,000,000-word training text, tracked with Git LFS
counts.json        word counts by source
mix_config.yaml    config copied at build time
mix_manifest.json  source and trim provenance
README.md          short usage note for tokenization/training
```

The full `mix_documents.jsonl` provenance file is intentionally not committed.
It repeats metadata for every selected line and is much larger than `train.txt`.
The plain `train.txt` plus sidecars are enough for teammates to tokenize and
train.

## Token Shards

The deliberate exception to the usual cache-only policy is:

```text
data/token_shards/gpt_bert_linguistic_repair_1pct_v1/
```

It contains the byte-identical GPT-BERT shards used by the nested 1% linguistic-
repair experiment. Binary shards are tracked with Git LFS; the accompanying
README, manifests, hashes, and pinned tokenizer/model revision make the artifact
portable without duplicating the already tracked human-readable repair sources.

## Analysis Data

Use `data/analysis/` for local analysis corpora, word2vec inputs/outputs,
leakage checks, and generated reports. These outputs can become large quickly
and should usually stay outside git.

If an analysis artifact becomes important enough to share, prefer documenting
the command/config that regenerates it. Commit only small summaries or target
lists when they are useful to other people.

## Archive And Pilots

`archive/` and `pilots/` folders are for historical or exploratory outputs.
They are useful locally, but they should not be treated as active training
inputs unless a config or manifest explicitly points to them.

## Commit Policy

- Commit small configs, manifests, reports, and blessed source datasets when
  they make an experiment reproducible.
- Use Git LFS for large plain text artifacts that teammates need directly.
- Do not commit token shards, checkpoints, run logs, large analysis outputs, or
  full provenance JSONL files unless there is a deliberate reproduction reason.
- Keep generated training runs under
  `${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}`.
- Keep tokenizers, token shards, and other reusable caches under
  `${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}`.

See the root `README.md` for the current paper-facing training recipes.
