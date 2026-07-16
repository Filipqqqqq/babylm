# Physical Dynamics Curriculum

This curriculum extends the current material-affordance generation work toward
physical dynamics. The target domain is not just "physical change" in a loose
sense. The central testing question is:

```text
Can a model connect evidence about directional physical change, motion, and
state/process change over time to the right event/process concept?
```

For example, a strong model should connect evidence like "moving closer to the
ground" with falling, "support was removed" with falling or toppling,
"friction increased" with slowing or stopping, and "an opening was unsealed"
with leaking or draining. The target is dynamic process structure: cause,
motion or state/process change, and outcome over time. It is not object
recognition, static property naming, or material labels alone.

## Current Strategy

V1 starts with a deliberately small pilot: 20 documents total, balanced across
the core families.
The first implementation should be inspected before scaling to a likely
4,000-document corpus. Scaling should be a config change, not a pipeline
rewrite.

The V1 families are:

```text
support_stability
force_speed
contact_collision
containment_flow
directional_motion
```

They cover a compact core of everyday physical dynamics:

```text
support removed -> fall, tip, topple, collapse, stay supported
force applied or resisted -> speed up, slow down, stop, compress, stretch
contact event -> hit, rebound, stick, scatter, crack, stop
container/path condition -> spill, leak, drain, overflow, stay contained
directional cause -> fall, rise, move upward, move downward
```

Speed change is included under `force_speed` and `contact_collision`.
Falling/rising are included under `directional_motion` with explicit causes
such as gravity after release, buoyant lift, warm rising air, and buoyancy in
water.
The pilot intentionally includes EWoK-aligned lexical forms such as `falling`,
`rising`, `accelerating`, and `slowing down`, while also retaining natural
variants such as `fell`, `rose`, `sped up`, and `slowed down`.
Contrasting targets should be balanced inside small EWoK-aligned slices; for
example, the pilot keeps `directional_motion` at two falling targets and two
rising targets.

## Path Shape

Track `path_shape` as an axis in V1 because direction and trajectory can be
important evidence. Non-path cases should use a stable value such as `none`.
However, path shape is not the main V1 objective. Dedicated path-shape families
such as `roll/slide`, `swing/spin`, `drift`, and `curve/straight` belong in V2
unless a V1 schema naturally requires them.

`roll_slide` is scaffolded as a V2 family in the pilot config, but it is not
included in the current 20-document V1 quotas. It tests the distinction between
rotation plus translation and translation without rotation. The V2 slice is
balanced between `rolling` and `sliding`, includes both objects and
`person_body` cases, and tracks body pose for people alongside shape, surface,
friction, rotation permission, and rotation evidence.

## Landmark Direction

`landmark_direction` is a focused directional family for cases where the
evidence is an entity's changing relation to a reference landmark:

```text
entity + initial relation to landmark + later relation to landmark + mechanism
```

Examples include a balloon moving from below a roofline to above it, a bead
moving from above a water surface to below it, or an elevator moving from above
a window to below it. The family is balanced across:

```text
landmark x crossing direction x verb type
```

The first config uses five landmarks (`roofline`, `water surface`, `ledge`,
`window`, and `rim`), two directions (`upward` and `downward`), and three verb
types:

```text
explicit: rose past, fell past, sank below, descended past
direction_biased: cleared, broke the surface, dropped below, slipped under
direction_neutral: passed, crossed, went by
```

The direction-neutral cases are the hardest and most important for
generalization because the model must use the before/after landmark relation
and mechanism, rather than the event verb alone.

The broader `vertical_relation_language_expansion` uses an additional set of
vertical reference landmarks for fall/rise evidence:

```text
ground
floor
ceiling
clouds
sky
roofline
tree branch
skyline
stars
```

These examples should describe `entity + initial relation to landmark + later
relation to landmark + mechanism`, while avoiding the benchmark-like phrase
`getting closer to the ground/sky` unless a later targeted diagnostic makes that
necessary.

The combined config `physical_dynamics_core_plus_landmark_140.json` reuses the
core physical-dynamics families and imports the balanced landmark-direction
family. It generates 140 documents: 80 from the original core run and 60 from
the landmark-direction slice.

## Lexical Bridges

The W2V probes suggest that broad natural examples can be too diffuse for
static distributional learning: the process word and the diagnostic evidence
may not appear close enough together. The lexical bridge slice is a controlled
addition for this problem. It keeps examples scenario-based, but places the
assigned process word near evidence such as:

```text
fall/rise -> lower/higher, ground/floor/sky/clouds/stars
sink/float -> underwater, below the surface, bottom, waterline, visible
float/rise -> supported at a surface vs moving upward through a medium
roll/slide -> rotating/turning vs same side stayed down/locked wheels
speed up/slow down -> speed increased/decreased, faster/slower, more/less distance per time
```

The bridge source file is
`physical_dynamics_lexical_bridges_v1.json`. It is intentionally source-only.
Use `physical_dynamics_broad_plus_bridges_800.json` when the broad curriculum
should reserve a small share for bridge examples while remaining balanced
across the broader canonical concept inventory.

Two weighted broad-plus-bridge variants increase the amount of local
co-occurrence text while keeping the same 800-document, 30-concept canonical
balance:

```text
physical_dynamics_broad_plus_bridges_800_weighted_12p5.json
physical_dynamics_broad_plus_bridges_800_weighted_18p5.json
physical_dynamics_broad_plus_bridges_800_weighted_18p5_altitude_bridge.json
```

The 12.5% variant raises explicit `lexical_bridge` specs from 50 to 100. It
mostly leaves the broad curriculum intact, but reduces the non-lexical speed
cue shares so `speed_up` and `slow_down` receive more same-sentence
target-plus-evidence examples.

The 18.5% variant raises explicit `lexical_bridge` specs from 50 to 148. It
also trims competing non-lexical cue shares for the EWoK-focused concepts, so
the extra budget becomes short active bridge text rather than more diffuse
scenario variety. Use these configs to test whether modest bridge upweighting
helps Word2Vec before moving to a mostly shortcut-focused corpus.

`physical_dynamics_broad_plus_bridges_800_weighted_18p5_altitude_bridge.json`
keeps the same 148-spec bridge budget but reallocates part of the sink/float
surface bridge to `sink_float_altitude_height_bridge`. That bridge is balanced
across `sink` and `float`, and across altitude/height wording: sinking examples
use `losing altitude` and `losing height`, while floating examples use
`same altitude`/`maintaining the same altitude` and
`same height`/`maintaining the same height`. The objects and sentence frames are
kept different from EWoK so this tests lexical coverage rather than copying the
benchmark item shape.

`physical_dynamics_broad_plus_bridges_800_weighted_18p5_object_agnostic_bg.json`
keeps the explicit bridge slice at 148 specs, but also adds an object-agnostic
shortcut background inside the remaining 81.5% of the corpus. It imports
`w2v_object_agnostic_shortcuts` and reserves 56 additional shortcut-background
specs: 7 each for `fall`, `rise`, `sink`, `float`, `speed_up`, `slow_down`,
`roll`, and `slide`. The rest of the corpus still comes from broader
context-dependent physical-dynamics families. This variant tests whether W2V
benefits when the non-bridge background includes repeated object sets with
local evidence cues, rather than relying only on the explicit bridge slice.

## Object-Agnostic W2V Shortcuts

`physical_dynamics_w2v_object_agnostic_shortcuts_200.json` is a narrower
diagnostic slice than the broad curriculum. Its purpose is to test whether
Word2Vec can learn local evidence-process associations when object identity is
made less predictive. It should not be read as the main physical-dynamics
curriculum.

The design principles are:

```text
short active prose
one target process term per scenario
target term and evidence cue in the same sentence
balanced positive concepts
overlapping object sets across different processes
minimal contrast wording in prose
```

The base config balances eight EWoK-relevant canonical concepts:

```text
fall
rise
sink
float
speed_up
slow_down
roll
slide
```

At the 200-document scale this is 25 specs per concept. The 800-document Qwen
run scaled the same design to 100 planned specs per concept. The generated
800-document run produced 780 train documents after audit filtering.

The "object-agnostic" part is deliberate. Objects such as paper scraps, foam
pieces, beads, toy carts, buttons, washer rings, toy blocks, and coins recur
across multiple outcomes. For example, the same broad object type may fall,
rise, sink, float, roll, slide, speed up, or slow down depending on the cause
and local physical evidence. This is meant to reduce shortcuts such as
`leaf -> falling`, `balloon -> rising`, or `bead -> sinking`.

Each scenario therefore stores both lexical and physical controls:

```text
lexical_targets: acceptable target word forms such as falling/fell or floating/floated
same_sentence_evidence_terms: nearby cues such as downward, higher, underwater, visible, faster, slower, rotating, or slick
contrast_evidence_bans: opposite-polarity cues that should not appear
bridge_clause/evidence_clause: compact phrases Qwen can use naturally
object_set_id: the repeated object family used to check object overlap
```

This slice differs from the broader bridge config in an important way: it spends
nearly all of its synthetic budget on tight local co-occurrence. The broader
curriculum spreads examples across more process families and discourse
contexts, which is better for corpus coverage but can dilute the simple
distributional signal Word2Vec needs.

The strongest mixed Word2Vec result so far came from the 800-document
object-agnostic shortcut run mixed into BabyLM 10M at roughly 1 percent
synthetic words. It scored 0.40 on EWoK physical dynamics, but the gain was not
broad across concepts:

```text
sink/float               0.969
fall/rise                0.500
roll/slide               0.000
accelerate/slow down     0.000
grow/shrink              0.000
overall                  0.400
```

That result is useful evidence that concentrated local lexical bridges can move
a static-vector probe. It is also a caution: the run mostly improved
`sink/float`, so future versions should treat it as a targeted diagnostic and
not as proof that the whole physical-dynamics inventory is solved.

## EWoK Micro Curriculum

`physical_dynamics_ewok_micro_500.json` is a compact diagnostic bridge corpus
for five EWoK-style contrast pairs:

```text
accelerate/slow down
fall/rise
float/sink
grow/shrink
roll/slide
```

It generates 500 planned documents, with 50 examples for each canonical concept
and 100 examples for each pair family. Every scenario uses a
`lexical_bridge_*` template, so the assigned lexical target and a pair-specific
evidence cue should appear in the same sentence. The audit now enforces this
same-sentence rule for bridge templates as well as W2V shortcut templates.

The design is intentionally pair-specific rather than benchmark-like. Speed
examples include forms such as `accelerating`, `speeding up`, `moved faster`,
`went faster`, `slowing down`, `moved slower`, `went slower`, and
`losing speed`. Fall/rise examples include explicit `sky`/`ground` bridge
language such as `closer to the sky`, `closer to the ground`, `toward the sky`,
and `toward the ground`; contrast bans should avoid legitimate starting
positions such as beginning above or below a landmark. For the targeted
sky/ground bridge, keep the first-order distribution clean: fall examples
mention `ground` but not `sky`, and rise examples mention `sky` but not
`ground`. Slide examples prefer positive non-rotation evidence such as `same
face stayed down`, `wheels stayed still`, or `straight skid mark` instead of
teaching the contrast mainly through phrases such as "did not roll".
Grow/shrink examples use size marks, outlines, diameter, swelling, inflating,
drying, deflating, heat, and cooling evidence.

Recommended evaluation order:

```text
1. Render and audit the 500-document micro corpus.
2. Run Word2Vec as a cheap gate: synthetic-only sanity, then BabyLM+micro at a small mix rate such as 1 percent.
3. Compare broad-only, micro-only, and broad+micro mixes by pair, not only overall.
4. If W2V moves more than one pair without leakage, train the BabyLM LM mix and evaluate checkpoint dynamics.
5. Use subfamily ablations when a pair fails, especially speed synonyms versus rate evidence and roll versus slide rotation evidence.
```

Word2Vec is the right first gate because this corpus is designed to strengthen
local lexical evidence. It should not be the final decision, though: the
language-model EWoK context evaluation is the real check for whether the micro
signal survives in a mixed BabyLM training run.

## Canonical Inventory

`canonical_inventory.json` defines the stable concept ids used for balance
analysis. The raw scenario field `expected_process` may contain surface labels
such as `falling`, `fell`, `sank`, `floated`, `stayed contained`, or
`upward crossing`. The canonical inventory maps those labels onto deliberate
concept ids such as:

```text
fall
rise
sink
float
direction_up_crossing
direction_down_crossing
speed_up
slow_down
stop
remain_stable
spill
leak
drain
absorb
```

The mapping also includes contrast-only concepts such as `remain_stationary`,
`continue_motion`, `maintain_shape`, and `remain_intact`, so both targets and
contrasts can be audited. Direction-neutral landmark verbs such as `crossed`,
`passed`, and `went by` are intentionally not mapped by verb alone; their
canonical direction comes from the scenario's `expected_process`.

Configs may use `canonical_concept_counts` inside a coverage pack instead of
family-level `document_counts`. This assigns document quotas by canonical
target concept after scenario labels are normalized through the inventory. The
canonical metadata is written to scenario plans and audited-document metadata,
but it is not sent to the renderer prompt as a prose hint.

`physical_dynamics_core_plus_landmark_canonical_108.json` is the first example:
it imports the same core-plus-landmark scenario pool as the 140-document config
and generates 4 documents for each available target canonical concept, for 108
documents total. It also imports
`physical_dynamics_scenario_expansions_v1.json`, which adds extra base
scenarios for `sink`, `spill`, `drain`, `leak`, `overflow`, `flow_down`,
`lift_up`, `absorb`, `compress`, `stretch`, `crack`, `dent`, `scatter`, and
`stick`. Those concepts previously reused too few scenarios in
canonical-balanced runs. The `crack` expansion includes concrete, clay, and ice
cases so cracking is not tied to one fragile-object shortcut.

The same expansion source also includes
`vertical_relation_language_expansion`, a balanced 16-scenario set for
`fall`, `rise`, `sink`, and `float`. It adds evidence-language variants such as
height change, landmark distance, Earth/outer-space reference frames, and
same-level surface relations. This is meant to cover the semantic neighborhood
of vertical evidence without copying benchmark phrasing: avoid exact strings
like "getting closer" in the broad expansion, but include natural alternatives such as
"gained height", "lost height", "nearer to Earth", "toward outer space",
"below the surface", and "same water level". The separate altitude bridge
variant deliberately introduces balanced `altitude` and `height` wording as a
benchmark-aware lexical coverage probe.

It also includes `sink_float_surface_cues`, a small 8-scenario set for
cue-heavy but non-benchmark sink/float evidence. These examples use objects
such as glass beads, ceramic chips, cork disks, foam plugs, sealed lids, and
wax buttons with short surface, waterline, visibility, depth, and bottom-contact
cues like "below the waterline", "disappeared below the surface", "reached the
bottom", "stayed visible", and "partly above the surface". Sinking examples
also include the word `underwater` in a few physical contexts, using variants
like "ended underwater" or "moved deeper underwater" rather than relying on the
exact benchmark phrase "going underwater".

It also includes two smaller balancing families for weak Word2Vec probes.
`speed_change_language_expansion` balances `speed_up` and `slow_down` across
forms such as "accelerating", "speeding up", "increasing speed", "slowing
down", "losing speed", and "decreasing speed". `fall_rise_direction_language_expansion`
balances direct downward/upward evidence for `fall` and `rise`, including
"moving down", "moved up", "lower", and "higher", while avoiding benchmark
objects and exact landmark-distance phrasing.

`physical_dynamics_core_plus_landmark_physical_120.json` is the
physical-dynamics-focused extension. It keeps the same 4-document canonical
quota but adds `float`, `roll`, and `slide` as target concepts, for 120
documents total. This config imports the dedicated sink/float scenario pool and
the v2 roll/slide scenarios so EWoK-relevant contrasts are represented as
positive expected processes, not only as contrast mentions.

`physical_dynamics_core_plus_landmark_broad_200.json` is the next inspection
run. It is broad rather than EWoK-focused: all 30 canonical physical-dynamics
concepts receive 6 or 7 documents, for 200 total. This config sets
`contrast_verbalization` to `metadata_only`, so `contrast_process` remains in
the scenario metadata for balancing and analysis, but Qwen is asked to write
positive cause/process/outcome evidence without naming the stored contrast in
the prose. It also sets exact lexical-target misses to warnings so otherwise
usable documents are not dropped solely because Qwen chose a different
inflection.

The broad config also reserves small canonical-candidate family shares for the
new balancing families. `fall` and `rise` receive both direct-direction and
vertical-relation examples; `sink` and `float` receive vertical-relation
examples plus a small cue-heavy surface/visibility slice; `speed_up` and
`slow_down` receive balanced speed-language examples. When scaled to 400
documents, those shares give 4 examples per relevant polarity bucket while
leaving most examples for the wider scenario inventory.

## Contrast Verbalization

`contrast_process` should stay in scenario plans because it is useful for
balancing, auditing, and later analysis. It does not always have to be spoken
in the generated paragraph. Physical-dynamics configs can set
`document_constraints.contrast_verbalization`:

```text
metadata_only
implicit
explicit
```

`metadata_only` builds a positive-only render view. The planner and final
metadata still keep the full contrast, but the Qwen payload hides
`contrast_process`, contrast term lists, `concept_pair`, contrast-pair family
labels such as `roll_slide` or `sink_float`, and contrast-oriented document or
style labels. The renderer sees a neutral `render_family` such as
`surface_motion` or `fluid_motion`, plus the observed cause, expected process,
and outcome evidence. This is the preferred setting for small Word2Vec
diagnostics because direct local contrasts such as "rolled instead of slid" can
make contrastive terms distributionally similar.

`implicit` allows a light reference to a different setup or result, but not the
stored contrast term itself.

`explicit` is the legacy behavior: Qwen may mention the contrast process when a
style calls for it, while keeping the expected process central.

## Scenario Schema

Each physical-dynamics scenario spec records:

```text
family
concept_pair
cause_type
object_type
material_or_surface
geometry_or_medium
timescale
path_shape
landmark, initial relation, later relation, crossing direction, verb type
discourse style
expected_process
contrast_process
cause_evidence
outcome_evidence
```

The intended implementation pattern is:

```text
structured scenario spec -> Qwen/Fireworks-rendered prose -> deterministic audit -> train text
```

Scenario specs provide controlled physics. The renderer adds natural language
variety. Audits prevent shortcut-heavy or ambiguous examples.

## Discourse Styles

The pilot uses four styles that create different distributional contexts
without drifting away from the physics:

```text
everyday_observation
cause_effect_explanation
comparison_contrast
observation_log
```

Additional styles such as short dialogue, repair notes, or handling notes can
be added after the core families are working.

## Audits

The pipeline should reject or flag examples when they show:

```text
missing required cause evidence
missing expected process evidence
missing outcome evidence
metadata/prose disagreement
target leakage or benchmark-style wording
overused sentence openings
near duplicates
shortcut-heavy object/outcome correlations
ambiguous or physically implausible outcomes
```

The most important audit is shortcut correlation. The corpus should avoid
letting a model solve examples by memorizing associations such as "ball means
bounce", "glass means break", "water means spill", or "ramp means roll".

## Repetition Diagnostics

Use the standalone repetition report when inspecting generated prose:

```bash
python -m babysteps.cli.repetition_report \
  --input data/concept_curriculum/archive/physical_dynamics_landmark_direction_60_qwen/documents_audited.jsonl \
  --output data/concept_curriculum/archive/physical_dynamics_landmark_direction_60_qwen/repetition_report.json \
  --min-n 3 \
  --max-n 6 \
  --top-k 25 \
  --group-by family \
  --group-by style_id
```

This utility reports repeated n-grams across documents, repetition ratios, top
reused phrases, sample document ids, and optional grouped views. It is separate
from pass/fail auditing so it can guide prompt/style revision without
automatically rejecting otherwise valid documents.

## Pilot Workflow

1. Generate 20 mock documents and inspect `scenario_plans.jsonl`,
   `documents_audited.jsonl`, and `train.txt`.
2. If mock planning and audits look right, run Qwen/Fireworks for the
   20-document pilot.
3. Train a small Word2Vec probe on the pilot only as a sanity check.
4. After manual inspection, create a scale config such as
   `physical_dynamics_core_4k.json` with the same schema and larger quotas.

## Evaluation

After generation, train a Word2Vec probe and evaluate:

```text
full EWoK
physical-dynamics
material-dynamics
material-properties
```

The Word2Vec result is a static-vector signal that the corpus contains useful
word-level semantics. It does not replace LM evaluation, but it is a fast way to
detect whether the generated text contains coherent lexical/distributional
structure for the target physical process families.
