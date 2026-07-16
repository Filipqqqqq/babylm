# Synthetic Data Generation

This directory contains entrypoints for building targeted synthetic data for
BabyLM strict-small experiments. The scripts cover two related repair lines:

```text
linguistic_repair/
  Language-form and dialogue repairs, especially BLIMP Supplement categories.

concept_curriculum/
  Concept-focused repairs, especially EWoK-style physical and material data.

container_state/
  Deterministic naturalized state-tracking documents and held-out diagnostics.
```

Both branches share the same downstream workflow: generate text, audit it, mix
it into BabyLM at fixed word budgets, tokenize the resulting corpus, train a
model, and evaluate for both gains and regressions.

The goal is not to replace BabyLM with a large synthetic corpus. The goal is to
test small, inspectable interventions while keeping the final training mix fixed
at 10 million words.

## Directory Map

```text
scripts/synthetic_data_generation/linguistic_repair/
  Deterministic and Qwen-assisted generators for language repair patches.

scripts/synthetic_data_generation/concept_curriculum/
  Generators, Qwen/Fireworks augmentation scripts, and audits for concept data.

configs/linguistic_repair/
  YAML configs for QA, turn-taking, and related repair datasets.

configs/concept_curriculum/
  JSON/YAML configs for physical, material, and other concept curricula.

configs/data_mixes/
  YAML configs that combine BabyLM and synthetic sources into 10M-word mixes.

configs/tokenization/
  Token-shard configs for mixed text corpora.

configs/train/
  Training configs that consume token shards.

data/linguistic_repair/
data/concept_curriculum/
data/container_state/
data/text_mixes/
  Generated text, sidecar manifests, audit reports, and final text mixes.
```

## Generation Branches

### Linguistic Repair

Use `linguistic_repair/` when the target behavior is a language or dialogue
pattern: question-answer congruence, answer type selection, pronoun perspective,
turn-taking, or other BLIMP Supplement-style repairs.

Current scripts include:

```text
generate_qa_congruence.py
generate_qa_location_verb_frames.py
generate_qa_who_person_frames.py
generate_qa_what_object_frames.py
generate_qa_turn_taking_pronoun_frames.py
```

See:

```text
scripts/synthetic_data_generation/linguistic_repair/README.md
```

### Concept Curriculum

Use `concept_curriculum/` when the target behavior is conceptual knowledge:
physical dynamics, material affordances, action affordances, or other EWoK-style
repairs.

Current scripts include:

```text
generate_fireworks.py
augment_documents.py
audit_documents.py
run_physical_dynamics_ewok_micro_500.sh
run_stir_pilot_80.sh
```

Related config docs:

```text
configs/concept_curriculum/physical_dynamics/README.md
```

### Container State

Use `container_state/` for small symbolic storage worlds rendered as ordinary
narratives, inventories, cleanup summaries, and conversations. The v1
curriculum uses three to five labeled containers, one to three objects per
container, and zero or one state-changing operation. Its object vocabulary and
surface templates are disjoint from the public Entity Tracking benchmark.

Generate the 130k-word unique source pool and custom validation slices:

```bash
python scripts/synthetic_data_generation/container_state/generate_container_state.py \
  --config configs/container_state/container_state_v1_seed42.yaml
```

The custom `iid`, `lexical_ood`, `template_ood`, and `two_operations` JSONL
files are written under:

```text
data/container_state/container_state_v1_pool_seed42/validation/entity_tracking/
```

They intentionally reuse the repository's generic continuation-likelihood
schema, but they are never included in `train.txt`.

See the [container-state experiment guide](container_state/README.md) for the
full generation, 1% mixing, fixed-tokenizer, GPT-BERT mixed-training, and
custom-validation workflow.

## Shared Workflow

1. Diagnose the target behavior.

   Inspect item-level failures in the target evaluation. Decide whether the
   issue looks linguistic, conceptual, lexical, length-biased, or caused by a
   broader regression.

2. Design a small patch.

   Decide what the model is underexposed to. In a 10M-word strict-small mix,
   0.5% is 50,000 words and 2.0% is 200,000 words. A patch should replace BabyLM
   text or another synthetic source rather than increasing the total corpus
   size.

3. Generate an inspection sample.

   For new deterministic generators, start with about 500 examples. Inspect
   `train.txt`, `documents.jsonl`, `audit_report.json`, and `manifest.json`
   before training.

4. Generate the final patch.

   Build the final source from a larger unique pool when possible. Do not simply
   repeat a small inspection sample unless the experiment is explicitly testing
   repetition.

5. Build the 10M text mix.

   Use a config in `configs/data_mixes/`. Keep `max_words: 10000000` and set
   each source's `target_words` explicitly. Example budgets:

   ```text
   BabyLM base:          9,500,000 words
   repair patch 0.5%:       50,000 words
   second patch 0.5%:       50,000 words
   physical 2.0%:          200,000 words
   material 2.0%:          200,000 words
   ```

6. Inspect the mix.

   Check:

   ```text
   data/text_mixes/.../counts.json
   data/text_mixes/.../mix_manifest.json
   data/text_mixes/.../train.txt
   ```

   Confirm that `output_words_total` is exactly `10000000` and that each
   synthetic source has the intended `output_words`. Also check whether final
   trimming clipped a synthetic QA or dialogue line. If that happens, adjust the
   order seed or source selection so trimming lands on base text, or use a
   whole-line exact sampler for the synthetic patch.

7. Tokenize.

   Use a config in `configs/tokenization/`. Point `dataset.files` at the mixed
   `train.txt`, reuse the intended tokenizer, and write shards under
   `BABYSTEPS_CACHE_DIR`.

8. Train.

   Use a config in `configs/train/`. Point `data.data_dir` at the token-shard
   directory. Keep model and optimizer settings fixed unless the experiment is
   explicitly about training.

9. Evaluate.

   Compare both final and best checkpoint behavior. Track the target benchmark
   and the surrounding suite, including BLIMP, BLIMP Supplement, EWoK, Entity
   Tracking, and COMPS.

## Command Skeleton

Generate a linguistic repair source:

```bash
python scripts/synthetic_data_generation/linguistic_repair/generate_qa_turn_taking_pronoun_frames.py \
  --config configs/linguistic_repair/qa_turn_taking_pronoun_frames_sample500_seed42.yaml
```

Generate or augment a concept-curriculum source:

```bash
python scripts/synthetic_data_generation/concept_curriculum/generate_fireworks.py \
  --config configs/concept_curriculum/physical_dynamics/physical_dynamics_ewok_micro_500.json
```

Audit concept-curriculum documents:

```bash
python scripts/synthetic_data_generation/concept_curriculum/audit_documents.py \
  --input data/concept_curriculum/.../documents_draft.jsonl \
  --output data/concept_curriculum/.../documents_audited.jsonl
```

Build an exact synthetic patch or a full 10M-word text mix:

```bash
python scripts/data/build_text_mix.py \
  --config configs/data_mixes/example_mix.yaml
```

Build token shards:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/example_shards.yaml
```

Train:

```bash
python scripts/train/train_hybrid_lm.py \
  --config configs/train/example_train.yaml
```

For long GPU runs, launch the same training command inside `tmux` and bind the
desired GPU with `CUDA_VISIBLE_DEVICES`.

## Artifact Conventions

Common naming fragments:

```text
sample500       small inspection sample
pool5000        larger unique source pool
50k             exact 50,000-word patch
qaeasy0p5       0.5% QA-easy patch
turntaking0p5   0.5% turn-taking patch
phys2           2.0% physical dynamics
mat2            2.0% material affordance
replace5pct     5% of BabyLM replaced in a 10M-word mix
seed42          deterministic seed used for generation or sampling
```

Common generated files:

```text
train.txt           plain text consumed by later mixing or tokenization steps
documents.jsonl     structured generated examples
audit_report.json   pass/warn/fail checks for generated examples
manifest.json       generation settings, counts, and provenance
counts.json         text-mix word counts by source
mix_manifest.json   text-mix provenance and source selection metadata
mix_documents.jsonl document-level provenance for text mixes
```

## Generation Modes

Use deterministic template generators when the target behavior is narrow and
structural. Examples include:

```text
QA location verb frames
QA who/person frames
QA what/object frames
turn-taking pronoun frames
```

Use Qwen/Fireworks-assisted generation when richer scene variation or more
naturalistic style is important. Examples include:

```text
original QA congruence repair
physical dynamics scenarios
material affordance scenarios
child-directed concept-curriculum augmentation
```

Deterministic generators are easier to audit and balance. Qwen-assisted
generators can provide broader style, but need stricter auditing and leakage
checks.

## Safety and Leakage Rules

- Do not copy evaluation examples verbatim.
- Avoid exact names, locations, and templates from held-out eval items when
  possible.
- Use derived pools and ordinary synthetic style rather than held-out eval
  strings.
- Generate positive/correct examples by default. Add negative or contrastive
  examples only for experiments explicitly designed around contrast training.
- Keep eval-like fragments out of synthetic data.
- Treat percentages as word budgets, not example-count budgets.
- Inspect repetition before training, especially when a short patch is expanded
  into a larger source.

## Open Questions

- Which generated outputs should become blessed artifacts in
  `manifests/blessed_synthetic_data.md`?
- Which repair lines need child READMEs beyond `linguistic_repair/` and the
  existing concept-curriculum config docs?
- Which mixes are stable enough to recommend as reference baselines for future
  synthetic-data experiments?
