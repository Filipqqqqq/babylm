# Linguistic Repair Synthetic Data

This directory contains generators for language-form and dialogue repairs. The
current work is mostly focused on BLIMP Supplement behavior, especially QA
congruence and turn-taking, while checking for regressions on BLIMP, EWoK,
Entity Tracking, and COMPS.

Use these generators when the target behavior is a linguistic pattern rather
than a physical or material concept.

## Current Generators

```text
generate_qa_congruence.py
  Qwen-assisted original QA congruence repair data.

generate_qa_location_verb_frames.py
  Deterministic where-question verb-frame examples.

generate_qa_who_person_frames.py
  Deterministic who-question person/role examples.

generate_qa_what_object_frames.py
  Deterministic what-question object examples.

generate_qa_turn_taking_pronoun_frames.py
  Deterministic turn-taking and pronoun-perspective examples.
```

Generated outputs usually live under:

```text
data/linguistic_repair/
```

Configs usually live under:

```text
configs/linguistic_repair/
```

## BLIMP Supplement Repair Case Study

The strongest BLIMP Supplement result in this repair line is the mixed GPT-BERT
run with QA-easy and turn-taking patches:

```text
95.0% BabyLM strict-small
0.5% QA-easy answer-type patch
0.5% turn-taking pronoun patch
2.0% physical dynamics
2.0% material affordance
```

Main data-mix config:

```text
configs/data_mixes/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_seed42.yaml
```

Reference run:

```text
gpt_bert_mixed_babylm2026_strict_small_replace5pct_qaeasy0p5_turntaking0p5_phys2_mat2_muon/20260703_085942
```

Key final metrics:

```text
BLIMP Supplement macro: 0.6780
turn-taking:            0.8821
QA congruence easy:     0.6875
QA congruence tricky:   0.4606
main BLIMP macro:       0.7254
EWoK context full:      0.5277
COMPS macro:            0.5269
Entity Tracking macro:  0.2106
```

This run shows that the turn-taking repair worked strongly, but QA tricky and
Entity Tracking remain open problems.

## QA Congruence Notes

The original useful QA control used:

```text
95.5% BabyLM strict-small
0.5% original QA repair
2.0% physical dynamics
2.0% material affordance
```

Reference run:

```text
gpt_bert_mixed_babylm2026_strict_small_replace4p5pct_qa0p5_phys2_mat2_muon/20260702_085914
```

That run produced the strongest QA-easy peak observed so far:

```text
QA congruence easy best:   0.7656
QA congruence tricky best: 0.5515
```

The later deterministic QA-easy answer-type patch helped over baseline, but did
not reproduce that original QA-easy peak. The current hypothesis is that the
model needs better coverage across wh-answer types, not only location frames.

Targeted QA-easy repair families considered so far:

```text
where -> compatible location, destination, path, surface, or container answer
who   -> compatible person or social-role answer
what  -> compatible object answer
```

## Turn-Taking Notes

The turn-taking patch focused on perspective-sensitive dialogue examples with
weak pronoun switches and controls. The final generator balances:

```text
direct dialogue and reported dialogue
yes/no-prefaced answers and bare answers
contracted and full negative forms
weak target cases and controls
```

The 0.5% turn-taking patch raised turn-taking to:

```text
turn-taking final: 0.8821
```

That made it the clearest successful BLIMP Supplement repair in this branch so
far.

## Command Examples

Generate an inspection sample:

```bash
python scripts/synthetic_data_generation/linguistic_repair/generate_qa_turn_taking_pronoun_frames.py \
  --config configs/linguistic_repair/qa_turn_taking_pronoun_frames_sample500_seed42.yaml
```

Build an exact 50k-word patch from a larger source pool:

```bash
python scripts/data/build_text_mix.py \
  --config configs/linguistic_repair/qa_turntaking_pronoun_frames_50k_seed42.yaml
```

Build the current BLIMP Supplement reference mix:

```bash
python scripts/data/build_text_mix.py \
  --config configs/data_mixes/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_seed42.yaml
```

Tokenize the current BLIMP Supplement reference mix:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_gptbert_shards_seed42.yaml
```

Train the current BLIMP Supplement reference model:

```bash
python scripts/train/train_hybrid_lm.py \
  --config configs/train/gptbert_babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_muon_seed42.yaml
```

## Findings To Carry Forward

- The original QA repair at 0.5% plus 2% physical and 2% material produced the
  strongest QA-easy peak observed so far: `0.7656`.
- The deterministic QA-easy answer-type patch helped over baseline, but did not
  reproduce the original QA repair's QA-easy peak.
- The turn-taking patch worked strongly, raising turn-taking to `0.8821` final
  accuracy in the current best BLIMP Supplement mix.
- QA congruence tricky remains unresolved. Its good answers are often longer
  than bad answers under summed likelihood, so length bias may contribute to low
  scores.
- Entity Tracking degradation in the current best BLIMP Supplement mix is mostly
  from losing the easy `nothing.` target behavior. Excluding `nothing.` targets,
  the drop is much smaller.

## Open Problems

- Design a QA-tricky repair that handles answer-type congruence without leaning
  on length artifacts.
- Recover the original QA-easy gains while preserving turn-taking.
- Preserve Entity Tracking, especially `nothing.` targets, when adding targeted
  BLIMP Supplement patches.
- Decide which linguistic repair configs and generated outputs should become
  blessed artifacts in `manifests/blessed_synthetic_data.md`.
