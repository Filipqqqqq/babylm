# Concept Curriculum Configs

These configs generate small teaching documents for a target concept, then audit and export the pass/warn documents as plain training text for token-block mixing.

Synthetic documents are packed with tokenizer EOS boundaries by default during token-block mixing. In practice this means each generated document is separated with the tokenizer's `eos_token_id` before the stream is cut into 512-token blocks, so short curriculum documents remain distinguishable while still packing efficiently.

The current small pilots are `stir_pilot_20.json`, `pour_pilot_20.json`, and `squeeze_pilot_20.json`. Despite the names, they can be scaled at runtime with `--document-count`, for example to generate 80 documents.

## General Lessons

Keep each document both abstract and grounded. The target material should remain central, but the event usually needs a visible support: a tool/path for stirring, a source or opening for pouring, a pressure source for squeezing, or a disturbance source for rippling.

Treat audit failures and warnings differently. Errors should mark likely concept breakage, such as a missing target lexeme or missing required material. Warnings should mark prose worth inspecting before scaling, such as weak source framing, weak concept-map connections, or short documents.

Track lexical anchors alongside broader subtypes. A profile should teach the general category, but it should also include exact material words that matter for evaluation or expected downstream use. For example, pour needs liquids and grains generally, but also anchors such as `coke`, `sesame`, and `pepper`.

Keep document types separate from discourse styles and signal order. Document types define conceptual coverage, such as `direct_positive`, `subtype_explanation`, `state_contrast`, and `neighbor_contrast`. Styles vary the surface form, such as concept maps, observation logs, child explanations, narratives, source-frame vignettes, and handling notes. Signal orders vary the rhetorical position of the evidence so the target word does not always appear first.

Use neighbors as boundary teachers, not decoration. Nearby actions like pour/spill/drip/dump or stir/mix/swirl/shake help the model learn what the concept is not, which is often as useful as more positive examples.

Positive exposure can create a lexical prior instead of context sensitivity. In one stir-focused sweep, the model became more likely to choose `stir` both when it was correct and when it was wrong. The practical lesson is that a concept curriculum needs contrastive calibration, not just more positive target examples.

For action affordances, balance three signals:

- Positive affordance: the material genuinely supports the target action.
- Negative boundary: nearby materials or states do not support that action.
- Contextual choice: the same frame with a different material should select a different neighbor action, such as fabric -> `fold`/`drape`/`wrinkle`, rigid object -> `tap`/`grip`, or liquid/grains -> `pour`/`stir`.

Neighbor-winning examples should be common enough that the target word does not become the default answer. If a profile teaches `stir`, it also needs repeated cases where `fold`, `drape`, `wrinkle`, `hang`, `squeeze`, or `tap` is the better continuation.

Balance signal order, not only signal type. The configs use a `signal_orders` axis with `positive_then_boundary`, `boundary_then_positive`, `neighbor_then_target`, `target_then_neighbor`, and `material_choice_pair`. This makes some documents start from the negative boundary, some from the positive affordance, some from a nearby action, and some from a parallel material choice.

Use balanced cycling instead of unconstrained randomness for small pilots. With 20 documents and five signal orders, each order appears four times; with 80, each appears sixteen times. That keeps the curriculum inspectable while still reducing target-word-first bias.

Scale only after small pilots. A good default workflow is 20 documents for qualitative inspection, 80-100 for distribution and audit behavior, then 200-500 once the failure modes are understood.

Use correlation-first pilots when the basic material-action graph may be weak. These examples should avoid negative contrast wording and instead keep each material close to several actions it naturally supports. This is especially useful for rare anchors, where the first goal is to teach clean local associations before asking the model to learn exceptions or causal boundaries.

Include some task-intention vignettes when teaching affordances. A short scene like "Maya wants to mix a drink, so she stirs the lemonade" teaches the action as a material-supported choice rather than only as a bare material-action list.

Keep correlation-first examples atomic by default. Each generated document should focus on one central material and several positive actions for that material, then EOS-separated packing keeps those units distinct during token-block mixing.

For EWoK-facing material dynamics, balance target actions explicitly. The concentrated affordance config cycles through `squeeze`, `stir`, `hang`, `wrinkle`, `tap`, `pour`, `drape`, `break`, `drip`, `rip`, `ripple`, `pile`, `fold`, `splash`, and `flap`, then chooses a compatible material class for each action. Class balancing is action-local: `stir` rotates across liquids, gases/vapors, and grains; `squeeze` rotates across textiles, gels, moldable masses, flexible soft objects, and rigid hard objects; `tap` rotates across brittle ceramics, rigid objects, soft foods/gels, moldable masses, flexible soft objects, and light flexible items. Material cycling is also action-local so one action does not consume another action's material sequence.

Treat `tap` as a bounded-object action, not only a rigid-object action. Rigid artifacts such as helmets, hammers, staplers, cups, forks, glass, ice, stone, and brick are natural tap targets, but soft bounded items such as banana, jello, play-doh, slippers, and feathers can also be tapped. Free gas and loose liquid are not direct tap targets; those contexts should prefer actions like `stir`, `flow`, `ripple`, `splash`, `swirl`, or `mix`.

Balance verb inflections, not only base action names. EWoK-style targets often use third-person forms such as `rips`, `drips`, and `breaks`, while generated curriculum text can overuse base forms such as `rip`, `drip`, and `break`. With the BabyLM byte-BPE tokenizer, these forms may be tokenized very differently, for example `rip` and `rips` do not share a simple `rip + s` segmentation. For EWoK-facing material dynamics, include a modest number of exact third-person templates such as `fabric rips`, `liquid drips`, `rigid material breaks`, `liquid pours`, `gas stirs`, `fabric drapes`, `fabric wrinkles`, and `rigid objects tap`. The concentrated affordance config includes an `inflection_anchor` discourse style for this purpose.

EWoK-facing anchors can be targeted without exposing benchmark labels to the generator. The `material_affordance_ewok_20.json` config marks local material classes with `ewok_anchor` tags and keeps those tags in scenario metadata, but hides them from the prompt. The generated text should only see ordinary materials such as `twill`, `velvet`, `play-doh`, `jello`, `helmet`, `slipper`, `feather`, `terracotta`, `terra-cotta`, `methane`, `nitrogen`, `propane`, `pepper`, and `sesame`.

## Stir Coverage Notes

The `stir` profile is intended to teach the broader material affordance, not only kitchen liquids. Keep these positive subtypes when scaling or porting the profile:

- `liquids`
- `viscous_liquids`
- `slurries`
- `powders_grains`
- `loose_chunks_aggregates`
- `pastes_batters`
- `soft_unset_gels_foams`
- `gases_vapors`

Two categories were easy to miss in earlier drafts:

- `gases_vapors`: air, steam, vapor, smoke, nitrogen, methane, propane. These should use natural wording about stirring, mixing, currents, circulation, or agitation. Fuel-gas examples should stay controlled and non-procedural.
- `loose_chunks_aggregates`: gravel, ice cubes, beads, dry pasta, small stones, wood chips, nuts. These teach that loose solid pieces can be stirred because chunks tumble and rearrange even though each piece remains solid.

Each generated document should include at least one exact target lexical form: `stir`, `stirs`, `stirred`, or `stirring`. Related words such as `mix`, `blend`, `circulate`, `swirl`, `agitate`, `move through`, `push through`, and `turn over` are encouraged, but they should not replace the exact target form entirely.

## Discourse Styles

The stir pilot uses both a conceptual document type and a prose style:

- Document types: `direct_positive`, `subtype_explanation`, `state_contrast`, `neighbor_contrast`
- Styles: `concept_map_paragraph`, `observation_log`, `child_science_explanation`, `everyday_narrative`

This keeps coverage balanced while varying the surface form of the training text.

## Run An EWoK-Focused Correlation Pilot

This pilot is deliberately concentrated. It is useful for checking whether the prompts produce clean material-action pairings before mixing broader concept documents into BabyLM. The default config now makes 24 documents so its six discourse styles, including `inflection_anchor`, are exactly balanced. For larger exact style balance, use a `--document-count` divisible by 6, such as 96, 996, or 1002.

Two material-affordance config variants are kept on purpose:

- `material_affordance_ewok_20.json`: the current v4-style config. It adds the `inflection_anchor` style, broader rigid anchors, oxygen/generic gas anchors, and bounded soft `tap` cases.
- `material_affordance_ewok_v3_snapshot.json`: a separate reconstructed v3 snapshot based on `data/concept_curriculum/material_affordance_ewok_1000_action_local_v3/scenario_plans.jsonl`. It keeps the older five-style design and avoids the later v4 expansions, so use this when comparing against the v3 synthetic-only smoke runs.

The v3 snapshot is not a byte-for-byte recovered historical config; the original config file was not saved. It is the reusable reconstruction of the effective v3 ingredients: material classes, anchors, affordances, notes, document types, and discourse styles.

```bash
cd /home/jorge/babylm-baby-steps

conda run --no-capture-output -n babylm \
  python -u scripts/synthetic_data_generation/concept_curriculum/generate_fireworks.py \
  --config configs/concept_curriculum/material_affordance_ewok_20.json \
  --focus-tag ewok_anchor \
  --output-dir data/concept_curriculum/material_affordance_ewok_20 \
  --api-key-file /home/jorge/babylm-baby-steps/fireworks_api.txt
```

Scale it while preserving the configured document-type proportions:

```bash
conda run --no-capture-output -n babylm \
  python -u scripts/synthetic_data_generation/concept_curriculum/generate_fireworks.py \
  --config configs/concept_curriculum/material_affordance_ewok_20.json \
  --focus-tag ewok_anchor \
  --document-count 100 \
  --output-dir data/concept_curriculum/material_affordance_ewok_100 \
  --api-key-file /home/jorge/babylm-baby-steps/fireworks_api.txt
```

Scale the v3 snapshot for an apples-to-apples 2000-document comparison:

```bash
conda run --no-capture-output -n babylm \
  python -u scripts/synthetic_data_generation/concept_curriculum/generate_fireworks.py \
  --config configs/concept_curriculum/material_affordance_ewok_v3_snapshot.json \
  --focus-tag ewok_anchor \
  --document-count 2000 \
  --output-dir data/concept_curriculum/material_affordance_ewok_2000_action_local_v3_snapshot \
  --api-key-file /home/jorge/babylm-baby-steps/fireworks_api.txt
```

Scale the current v4 config with exact six-style balance:

```bash
conda run --no-capture-output -n babylm \
  python -u scripts/synthetic_data_generation/concept_curriculum/generate_fireworks.py \
  --config configs/concept_curriculum/material_affordance_ewok_20.json \
  --focus-tag ewok_anchor \
  --document-count 2004 \
  --output-dir data/concept_curriculum/material_affordance_ewok_2004_action_local_v4 \
  --api-key-file /home/jorge/babylm-baby-steps/fireworks_api.txt
```

## Run An 80-Document Pilot

```bash
cd /home/jorge/babylm-baby-steps

conda run --no-capture-output -n babylm \
  python -u scripts/synthetic_data_generation/concept_curriculum/generate_fireworks.py \
  --config configs/concept_curriculum/stir_pilot_20.json \
  --document-count 80 \
  --output-dir data/concept_curriculum/stir_pilot_80 \
  --api-key-file /home/jorge/babylm-baby-steps/fireworks_api.txt
```

Then inspect:

```bash
cat data/concept_curriculum/stir_pilot_80/audit_report.json
```

## Synthetic-Only Smoke

To test whether a synthetic corpus is learnable before mixing it into BabyLM 10M, build synthetic-only token chunks. The builder reads `.jsonl` or `.txt`, filters audited JSONL rows to `warn` or `pass` by default, separates documents with the tokenizer EOS token, packs full 512-token content blocks, and wraps each block as `[EOS] + content + [EOS]`.

```bash
python scripts/data/build_synthetic_only_chunks.py \
  --source data/concept_curriculum/material_affordance_ewok_1000_action_local_v3/documents_audited.jsonl \
  --kind jsonl \
  --text-field text \
  --tokenizer /SSD2/babylm-baby-steps-cache/tokenizers/babylm10m_byte_bpe_16384 \
  --output-dir /SSD2/babylm-baby-steps-cache/token_chunks/material_affordance_ewok_1000_synth_only \
  --block-size 512 \
  --dev-fraction 0.05 \
  --seed 42
```

Then launch a tiny GPT-2 diagnostic run for roughly 1000 optimizer updates:

```bash
python scripts/sweeps/run_synthetic_only_gpt2_smoke.py \
  --train-chunks /SSD2/babylm-baby-steps-cache/token_chunks/material_affordance_ewok_1000_synth_only/train.pkl \
  --val-chunks /SSD2/babylm-baby-steps-cache/token_chunks/material_affordance_ewok_1000_synth_only/dev.pkl \
  --target-updates 1000 \
  --gpus 5
```

## Pour Coverage Notes

The `pour` profile is intended to teach controlled material transfer, not just kitchen liquids. The central participant should be the material being poured or failing to pour; containers, spouts, hands, and tools are secondary.

Positive subtypes:

- `thin_liquids`
- `viscous_liquids`
- `powders_grains`
- `loose_small_pieces`
- `slurries_thin_mixtures`
- `melted_softened_materials`

The pour profile intentionally includes exact lexical anchors for common liquid and granular examples: `water`, `juice`, `coke`, `oil`, `lemonade`, `sand`, `sesame`, `pepper`, `rice`, and `gravel`. It also includes close siblings such as `pepsi`, `sugar`, `salt`, and `flour` so the examples do not become too narrow.

Boundary and negative categories:

- `rigid_single_objects`: one object can be dropped, placed, or carried, but it does not pour.
- `cohesive_lumps`: thick paste, dough, clay, or play-doh may slump, dollop, or need scooping.
- `gases_vapors`: gases usually disperse, vent, flow, or fill space rather than being poured as a central visible material.
- `flexible_sheets`: sheets drape, fold, or hang rather than pour.
- `large_mixed_objects`: large objects can be dumped, but they do not pour with smooth collective flow.

Each generated document should include at least one exact target lexical form: `pour`, `pours`, `poured`, or `pouring`. Related words such as `flow`, `stream`, `trickle`, `drain`, `empty`, `transfer`, `tip`, `spill`, `drip`, `dump`, `sprinkle`, and `scoop` are encouraged, but they should not replace the exact target form entirely.

For positive pour examples, the material should stay central but the event should still be grounded by a source frame: a cup, jar, bag, bottle, pitcher, can, bowl, bucket, box, packet, shaker, spout, hole, opening, or similar outlet. This avoids abstract sentences where a material “pours” but nothing indicates what it pours from.

The pour pilot also includes two pour-specific styles. These are prose styles, not new conceptual document types: they preserve the same `direct_positive`, `subtype_explanation`, `state_contrast`, and `neighbor_contrast` balance.

- `source_frame_vignette`: asks some documents to start from the material plus a source or outlet.
- `handling_note`: frames pouring as a practical material-handling event, such as filling a jar, pouring into a mold, cleanup, art, cooking, or workshop transfer. This style should name source and destination when natural and mention alternatives such as scoop, scrape, dump, sprinkle, place, or wipe when the material does not pour cleanly.

## Run A 20-Document Pour Pilot

```bash
cd /home/jorge/babylm-baby-steps

conda run --no-capture-output -n babylm \
  python -u scripts/synthetic_data_generation/concept_curriculum/generate_fireworks.py \
  --config configs/concept_curriculum/pour_pilot_20.json \
  --output-dir data/concept_curriculum/pour_pilot_20 \
  --api-key-file /home/jorge/babylm-baby-steps/fireworks_api.txt
```

Then inspect:

```bash
cat data/concept_curriculum/pour_pilot_20/audit_report.json
```

## Squeeze Coverage Notes

The `squeeze` profile is intended to teach inward pressure plus material response. The squeezed material or object should be central; hands, fingers, clamps, and surfaces are pressure sources.

Positive subtypes:

- `porous_absorbent_materials`
- `filled_flexible_containers`
- `juicy_biological_materials`
- `soft_cohesive_deformables`
- `elastic_rebounding_objects`
- `fabrics_flexible_sheets`
- `granular_materials_in_flexible_bags`
- `crumplable_thin_solids`
- `soft_packed_masses`

The last three categories were added because they cover behavior not captured by ordinary soft objects:

- `granular_materials_in_flexible_bags`: a sandbag, bag of rice, flour sack, beanbag, or packet of sugar can be squeezed because the flexible container deforms while the grains shift and pack.
- `crumplable_thin_solids`: a paper cup, foil, paper bag, cardboard carton, or thin plastic bottle can crease, buckle, or collapse under squeezing.
- `soft_packed_masses`: a snowball, damp sand, packed soil, crumbly cake, or wet clay crumbs can compact, crumble, hold together, or release moisture.

Boundary and negative categories:

- `rigid_but_grippable_boundary`: objects like a hammer, spoon, cup, helmet, stapler, fork, or key can be squeezed as a tight grip, but the material barely compresses.
- `unbounded_liquids`: loose water, juice, oil, lemonade, or coke flows away; a container or soaked material can be squeezed instead.
- `gases_vapors`: gas is squeezed only when bounded by a container.
- `loose_granular_materials`: loose grains pour, scatter, or pile unless held in a flexible bag or packed mass.
- `brittle_or_fixed_rigids`: brittle or fixed solids crack, resist, or stay in place instead of compressing smoothly.

Each generated document should include at least one exact target lexical form: `squeeze`, `squeezes`, `squeezed`, or `squeezing`. Related words such as `compress`, `press`, `pinch`, `grip`, `crush`, `wring`, `dent`, `bulge`, `flatten`, `collapse`, `release`, and `rebound` are encouraged, but they should not replace the exact target form entirely.

## Run A 20-Document Squeeze Pilot

```bash
cd /home/jorge/babylm-baby-steps

conda run --no-capture-output -n babylm \
  python -u scripts/synthetic_data_generation/concept_curriculum/generate_fireworks.py \
  --config configs/concept_curriculum/squeeze_pilot_20.json \
  --output-dir data/concept_curriculum/squeeze_pilot_20 \
  --api-key-file /home/jorge/babylm-baby-steps/fireworks_api.txt
```

Then inspect:

```bash
cat data/concept_curriculum/squeeze_pilot_20/audit_report.json
```
