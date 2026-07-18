# Entity Tracking experiment report

Snapshot: 2026-07-15 (Asia/Tokyo)

## Preference correction

Jorge was right: we had already agreed that `nothing.` examples should not drive
the headline Entity Tracking result. The official BabyLM reader now skips an
example whenever any candidate contains `nothing`, and we agreed to retain
empty-state reasoning only as a separate diagnostic.

The earlier description of an "invariant phrase" was meant to diagnose why the
container-state v1 intervention failed, not to recommend that design. Calling an
invariant *non-empty* phrase a possible next step was also the wrong conclusion:
the next curriculum should not make **any one answer string** invariant, whether
that string denotes an empty or non-empty state. Surface forms, object counts,
and answer lengths need to vary independently of the latent state.

## Executive summary

- The archive contains 42 training histories with Entity Tracking likelihood
  records: 37 trained non-leaky histories, 3 initialization-only records, and 2
  deliberate evaluation-copy leakage oracles.
- On the 36 substantially completed, non-leaky histories (epoch 9 or 10), the
  filtered official-style macro score has mean **20.35%**, standard deviation
  **0.97 points**, and range **18.33–23.25%**. Five-way chance is 20%. Most
  interventions therefore remain in a narrow chance-like band.
- The historical all-9,483-example macro is not a trustworthy headline. Across
  those same 36 runs it ranges from **15.28% to 39.41%**, but its correlation
  with the filtered score is only **r = 0.16**. It mainly tracks preference for
  the short `nothing.` continuation.
- The strongest non-leaky archived result is the nominal baseline reproduction
  at **23.25% filtered macro** (22.39% micro). Its saved configuration matches
  the older seed-42 baseline except for run metadata, yet the older run scores
  **21.85%**. Code date and nondeterministic execution are not pinned, so the
  1.40-point gap is evidence that small single-run deltas need matched reruns,
  not evidence for a curriculum effect.
- The container-state v1 experiment is a clean negative result. It taught the
  fixed empty answer `no items.` very efficiently, while non-empty and
  out-of-template performance fell to chance. This is exactly the invariant
  answer-phrase failure we do not want to repeat.
- The most informative results are mechanistic rather than leaderboard gains:
  the probed model is strongly anchored to the initial inventory, changes its
  choice too little under state-changing counterfactuals, and benefits from
  repetition of answer objects more than from operations involving the queried
  box. The bottleneck looks like state updating/composition, not mere exposure
  to container vocabulary.
- The two leakage oracles reach **33.27%** and **58.36%** filtered macro with one
  and two copies of the evaluation material. They prove dose-responsive
  memorization, not generalizable entity tracking, and must never be considered
  candidate runs.

## What is being measured

Each example has five candidate continuations, so uniform chance is 20%.
Historical training-time evaluation scored all three families (`regular`,
`ambiref`, and `move_contents`) and all 9,483 examples using summed continuation
log-likelihood. Of these, 2,703 examples contain a `nothing` option and 6,780 do
not.

The current official decoder drops any example where *any* option contains
`nothing`. See
[`read_files.py`](../external/babylm-eval/strict/evaluation_pipeline/sentence_zero_shot/read_files.py)
and the matching result filter in
[`collate_preds.py`](../external/babylm-eval/strict/evaluation_pipeline/collate_preds.py).

The primary score in this report is therefore recomputed as follows:

1. select the latest recorded epoch in each history;
2. use summed continuation log-likelihood, matching the archived training
   diagnostic;
3. remove all examples with a `nothing` candidate;
4. average within each family/operation-count UID, then within family, then
   across families, matching the official-style aggregation implemented in
   [`entity_tracking_likelihood.py`](../src/babysteps/evaluation/entity_tracking_likelihood.py).

The appendix also shows the legacy all-example macro and empty-example micro
accuracy so that phrase effects remain visible rather than silently discarded.

## Main findings

### 1. The old score mostly measures a continuation-length prior

Seven randomly initialized models score **40.75–41.94%** on the legacy summed
score (mean 41.48%), despite a 20% five-way chance rate. Removing useful context
barely changes this:

| Random-model input | Mean legacy summed macro |
|---|---:|
| Full context | 41.43% |
| Query only | 41.23% |
| Scrambled context | 41.32% |
| Unconditional | 41.18% |

When candidate token counts are equal, all four conditions fall to roughly
16.8–17.3%. Gold continuations average 5.213 tokens versus 5.502 for distractors;
the gold is shorter in 29.54% of gold/distractor pairs, equal in 69.56%, and
longer in only 0.90%. Summed log-likelihood consequently rewards the unusually
short gold continuation, especially `nothing.`.

This explains apparently impressive but non-generalizing histories. For
example:

| History | Legacy all macro | Empty micro | Filtered macro |
|---|---:|---:|---:|
| QA-easy 0.5% + physical/material | 39.41% | 86.13% | 20.59% |
| QA-easy + turn-taking + where | 39.13% | 85.24% | 20.43% |
| QA-original + roles + turn-taking + where | 38.04% | 79.69% | 21.04% |
| Where/what hard-confound | 15.28% | 0.67% | 21.53% |

The ranking nearly reverses when empty behavior is removed. Accordingly, the
legacy score is useful for diagnosing answer-form preference but should not be
used to select models.

### 2. The 37 non-leaky trained histories do not show a robust curriculum win

Excluding the epoch-2 incomplete run, all non-leaky histories cluster around
chance. Several different mixtures reach 21–22%, but there is no consistent
dose response or intervention family that separates from baseline variation.

The best current result, history 42, is especially instructive. Its saved
configuration is identical to history 3 except for the run name and timestamp
behavior. The filtered macro nevertheless changes from 21.85% to 23.25%.
Because the runs occurred about two weeks apart and the code revision is not
recorded in the config, this comparison mixes possible code drift with
nondeterminism. Any proposed improvement of similar size needs a fresh matched
0% control and multiple seeds in the same checkout.

The physical/material, QA, COMPS, alias-binding, linguistic-repair, and NextLat
mixtures are still scientifically useful: together they show that broad
semantic or dialogue enrichment does not automatically induce the specific
online update algorithm required by this task.

### 3. Container-state v1 learned the phrase, not the capability

The custom 4,000-item evaluation used length-normalized mean log-likelihood and
included IID, lexical-OOD, template-OOD, and two-operation slices:

| Slice | Older baseline | 1% container state | 2% container state |
|---|---:|---:|---:|
| Overall | 33.75% | 22.10% | 21.95% |
| IID | 33.80% | 22.30% | 22.20% |
| Lexical OOD | 36.60% | 26.40% | 26.00% |
| Template OOD | 36.60% | 17.90% | 18.40% |
| Two operations | 28.00% | 21.80% | 21.20% |

The decomposition is decisive:

| Target subset | Items | Older baseline | 1% | 2% |
|---|---:|---:|---:|---:|
| `no items.` target | 181 | 0.00% | 79.01% | 93.37% |
| Non-empty target | 3,819 | 35.35% | 19.40% | 18.57% |
| No empty option anywhere | 3,487 | 34.10% | 20.22% | 20.28% |

The literal `no items.` phrase appears only 53 times in the 1% training mix and
112 times in the 2% mix, yet accuracy on that target rises sharply with dose.
Non-empty accuracy does not. This is a phrase shortcut and a dataset-design
failure, not evidence of learned tracking.

There is also a causal caveat: the baseline was trained earlier, under an older
checkout, and training was not fully deterministic. The size of the regression
cannot be assigned solely to the replacement data until a fresh matched 0%
baseline is run. The same-date 1% versus 2% comparison is still informative:
more exposure strengthens the empty phrase without improving non-empty states.

### 4. Counterfactual probes locate the likely bottleneck

Mechanistic probes were run on `go76dof__wwm_curriculum_simplification_40k`.
On the original suite, its non-empty micro accuracy is 28.66%. Performance falls
with distance from the last operation touching the queried box:

| Last query-box touch | Accuracy |
|---|---:|
| Last instruction | 28.27% |
| One instruction back | 23.92% |
| Two instructions back | 23.13% |
| Three to five back | 20.19% |
| Six or more back | 17.22% |

This slice is confounded by task structure, so paired counterfactuals are more
diagnostic. Across 1,200 paired cases (2,400 prompts), all excluding `nothing`
and balancing option object counts:

| Counterfactual family | Pair accuracy | Prediction invariant | Other signal |
|---|---:|---:|---|
| Earlier removal | 3.33% | 93.00% | Initial state selected 90.50% |
| Earlier inbound move | 6.50% | 89.67% | Initial state selected 75.33% |
| Irrelevant reordering | 13.00% | 81.33% | Near 6.33%, far 19.67% |
| Whole-box transfer | 33.00% | 56.67% | Correct margin shift 99.00% |

The model's likelihood margins often move in the right direction, especially
for an explicit whole-box transfer, but the argmax usually stays fixed. This is
consistent with strong initial-state anchoring plus incomplete composition.

The query-touch ladder separates actual state evidence from lexical repetition:

| Condition | Accuracy | Mean gold margin |
|---|---:|---:|
| q1: one query-relevant operation | 50.67% | +0.016 |
| q3: three query-relevant operations | 43.00% | -1.611 |
| q5: five query-relevant operations | 39.33% | -2.092 |
| t3: three operations repeating target objects | 36.00% | -2.432 |
| t5: five operations repeating target objects | 33.67% | -2.805 |
| c3: three control operations | 37.00% | -2.375 |
| c5: five control operations | 36.33% | -2.682 |

Query-relevant conditions beat controls in margin (`q3-c3 = +0.764`, 95% CI
0.338–1.177; `q5-c5 = +0.590`, CI 0.090–1.113), but merely touching the target
objects captures much of the effect. Query relevance beats target repetition
(`q3-t3 = +0.821`, CI 0.391–1.220; `q5-t5 = +0.712`, CI 0.205–1.221), while
target repetition itself does not beat controls. The clean interpretation is
that the model detects some relevant evidence but cannot reliably maintain and
compose the evolving state.

### 5. Leakage oracles establish learnability, not validity

The deliberate one-copy and two-copy evaluation-leakage histories score 33.27%
and 58.36% filtered macro. Their empty-example accuracies are 99.89% and 99.70%.
The dose response shows the model can memorize the exact distribution when it
is placed in training. These histories are contaminated diagnostics and have no
standing as generalization results.

## Recommended experiment contract

The next intervention should be evaluated under the following contract:

1. **Primary endpoint:** filtered official-style macro on examples with no
   `nothing` candidate. Always show the corresponding micro score as well.
2. **Separate empty diagnostic:** evaluate empty states, but never combine them
   into the model-selection headline.
3. **No invariant answer form:** render the same latent state using multiple
   grammatical forms and make every form appear with multiple state types.
   Balance answer-token length and object count across gold and distractors.
4. **Counterfactual construction:** pair examples that differ in exactly one
   state-changing operation; include irrelevant reorderings and long-distance
   touches. Success must involve the prediction changing in the correct
   direction, not just a slightly better average margin.
5. **Matched controls:** train fresh 0%, treatment, and dose variants from the
   same commit, tokenizer, data order, and evaluation code, with at least three
   seeds or deterministic kernels where feasible.
6. **Gates before a full run:** reject a generator if a surface-only classifier
   can infer the target, if any answer string is target-invariant, or if a small
   pilot improves empty states without improving non-empty paired cases.

The current evidence argues for a curriculum that teaches explicit, varied
state transitions and tests compositional updates. It does **not** support
another corpus centered on a recognizable answer phrase.

## Complete 42-history inventory

Scores are percentages at the latest archived epoch. `Filtered` is the primary
official-style macro over the 6,780 retained examples. `Legacy` is the archived
all-example macro over 9,483 examples. `Empty` is micro accuracy on the 2,703
excluded examples. `Init` records never progressed beyond epoch 0; `oracle`
records contain deliberate evaluation leakage.

| ID | Status | Epoch | Filtered | Legacy | Empty | Run history |
|---:|---|---:|---:|---:|---:|---|
| 01 | trained/incomplete | 2 | 18.15 | 20.90 | 26.53 | `gpt_bert_baseline_mixed_babylm2026_strict_small/20260629_015831` |
| 02 | init | 0 | 16.86 | 41.11 | 100.00 | `gpt_bert_baseline_mixed_babylm2026_strict_small_muon/20260630_055620` |
| 03 | trained | 10 | 21.85 | 36.51 | 72.03 | `gpt_bert_baseline_mixed_babylm2026_strict_small_muon/20260630_142640` |
| 04 | trained | 10 | 18.83 | 30.04 | 57.49 | `gpt_bert_baseline_mixed_babylm2026_strict_small_muon_lr0p005_seed43/20260629_024832` |
| 05 | trained | 10 | 19.72 | 24.30 | 37.37 | `gpt_bert_baseline_mixed_babylm2026_strict_small_muon_lr0p005_seed44/20260629_024832` |
| 06 | trained | 10 | 20.03 | 32.35 | 62.15 | `gpt_bert_baseline_mixed_strict_small_9m_dataset_10ep_eval_gpu3/20260630_123742` |
| 07 | init | 0 | 16.86 | 41.11 | 100.00 | `gpt_bert_mixed_5pct_reference_nextlat_lh0p25_lkl0p1/20260709_074709` |
| 08 | trained | 10 | 20.61 | 18.50 | 13.73 | `gpt_bert_mixed_5pct_reference_nextlat_lh0p25_lkl0p1/20260709_081653` |
| 09 | init | 0 | 16.86 | 41.11 | 100.00 | `gpt_bert_mixed_8p5_comps_nextlat_lh0p25_lkl0p1/20260709_074720` |
| 10 | trained | 10 | 20.39 | 24.63 | 36.59 | `gpt_bert_mixed_8p5_comps_nextlat_lh0p25_lkl0p1/20260709_081704` |
| 11 | oracle | 10 | 33.27 | 52.90 | 99.89 | `gpt_bert_mixed_babylm2026_strict_small_eval_copy1_leakage_oracle_muon/20260710_094742` |
| 12 | oracle | 10 | 58.36 | 70.92 | 99.70 | `gpt_bert_mixed_babylm2026_strict_small_eval_copy2_leakage_oracle_muon/20260710_094751` |
| 13 | trained | 10 | 19.03 | 27.62 | 48.61 | `gpt_bert_mixed_babylm2026_strict_small_linguistic_repair_1p5pct_muon_20260712_144335` |
| 14 | trained | 10 | 20.21 | 28.02 | 47.24 | `gpt_bert_mixed_babylm2026_strict_small_phys_once_muon/20260710_102807` |
| 15 | trained | 9 | 20.28 | 35.15 | 71.70 | `gpt_bert_mixed_babylm2026_strict_small_replace10p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_compscp297k_aliasbind200k_muon/20260709_022744` |
| 16 | trained | 10 | 21.43 | 36.09 | 72.44 | `gpt_bert_mixed_babylm2026_strict_small_replace1pct_container_state_v1_muon/20260714_180021` |
| 17 | trained | 9 | 21.85 | 26.67 | 38.51 | `gpt_bert_mixed_babylm2026_strict_small_replace2pct_container_state_v1_muon/20260714_180022` |
| 18 | trained | 10 | 20.80 | 17.02 | 8.69 | `gpt_bert_mixed_babylm2026_strict_small_replace4p5pct_qa0p5_phys1p5_mat1p5_augphys0p5_augmat0p5_muon/20260702_085951` |
| 19 | trained | 10 | 21.22 | 23.16 | 29.71 | `gpt_bert_mixed_babylm2026_strict_small_replace4p5pct_qa0p5_phys2_mat2_muon/20260702_085914` |
| 20 | trained | 10 | 20.13 | 24.29 | 35.63 | `gpt_bert_mixed_babylm2026_strict_small_replace4p5pct_qa75min25_qa0p5_phys2_mat2_muon/20260702_110302` |
| 21 | trained | 10 | 20.59 | 39.41 | 86.13 | `gpt_bert_mixed_babylm2026_strict_small_replace4p5pct_qaeasy0p5_phys2_mat2_muon/20260703_074836` |
| 22 | trained | 10 | 20.26 | 21.15 | 24.38 | `gpt_bert_mixed_babylm2026_strict_small_replace4pct_augphys2_augmat2_muon/20260701_141936` |
| 23 | trained | 10 | 21.38 | 27.52 | 43.29 | `gpt_bert_mixed_babylm2026_strict_small_replace4pct_phys1p5_mat1p5_augphys0p5_augmat0p5_muon/20260701_141106` |
| 24 | trained | 10 | 19.61 | 28.47 | 50.46 | `gpt_bert_mixed_babylm2026_strict_small_replace4pct_phys2_mat2_muon/20260630_143429` |
| 25 | trained | 10 | 20.43 | 39.13 | 85.24 | `gpt_bert_mixed_babylm2026_strict_small_replace5p5pct_qaeasy0p5_turntaking0p5_qawhere0p5_phys2_mat2_muon/20260704_142548` |
| 26 | trained | 10 | 19.75 | 16.90 | 10.43 | `gpt_bert_mixed_babylm2026_strict_small_replace5p5pct_qaeasy0p5_whonames0p5_turntaking0p5_phys2_mat2_muon/20260706_060403` |
| 27 | trained | 9 | 21.04 | 38.04 | 79.69 | `gpt_bert_mixed_babylm2026_strict_small_replace5p5pct_qaorig_roles0p5_turntaking0p5_qawhere0p5_phys2_mat2_muon/20260704_124500` |
| 28 | trained | 10 | 20.32 | 16.31 | 7.55 | `gpt_bert_mixed_babylm2026_strict_small_replace5p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_muon/20260706_080722` |
| 29 | trained | 10 | 20.14 | 16.80 | 9.06 | `gpt_bert_mixed_babylm2026_strict_small_replace5p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p5_phys2_mat2_muon/20260706_012049` |
| 30 | trained | 10 | 20.02 | 30.18 | 55.23 | `gpt_bert_mixed_babylm2026_strict_small_replace5p5pct_qaorig_roles0p5_turntaking0p5_qawherelocal0p5_phys2_mat2_muon/20260705_051040` |
| 31 | trained | 10 | 21.53 | 15.28 | 0.67 | `gpt_bert_mixed_babylm2026_strict_small_replace5p5pct_qaorig_roles0p5_turntaking0p5_qawherewhat_hardconfound0p5_phys2_mat2_muon/20260706_033850` |
| 32 | trained | 10 | 19.55 | 16.13 | 8.66 | `gpt_bert_mixed_babylm2026_strict_small_replace5p5pct_qaorig_roles0p5_whonames0p5_turntaking0p5_phys2_mat2_muon/20260706_070307` |
| 33 | trained | 10 | 19.52 | 16.42 | 10.32 | `gpt_bert_mixed_babylm2026_strict_small_replace5pct_qa0p5_qaeasy0p5_phys2_mat2_muon/20260703_074828` |
| 34 | trained | 10 | 20.20 | 36.09 | 75.21 | `gpt_bert_mixed_babylm2026_strict_small_replace5pct_qa75min25_qa1_phys2_mat2_muon/20260702_140719` |
| 35 | trained | 10 | 20.07 | 21.06 | 23.34 | `gpt_bert_mixed_babylm2026_strict_small_replace5pct_qaeasy0p5_turntaking0p5_phys2_mat2_muon/20260703_085942` |
| 36 | trained | 10 | 18.98 | 32.79 | 66.67 | `gpt_bert_mixed_babylm2026_strict_small_replace5pct_qaorig_roles0p5_turntaking0p5_phys2_mat2_muon/20260704_112914` |
| 37 | trained | 10 | 20.87 | 16.62 | 6.59 | `gpt_bert_mixed_babylm2026_strict_small_replace5pct_qasceneonly_qa1_phys2_mat2_muon/20260702_141433` |
| 38 | trained | 10 | 20.34 | 17.19 | 10.03 | `gpt_bert_mixed_babylm2026_strict_small_replace6p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_compscp100k_muon/20260708_055900` |
| 39 | trained | 10 | 20.83 | 24.73 | 36.66 | `gpt_bert_mixed_babylm2026_strict_small_replace6pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p5_whonames0p5_phys2_mat2_muon/20260706_091428` |
| 40 | trained | 10 | 19.27 | 22.14 | 30.85 | `gpt_bert_mixed_babylm2026_strict_small_replace8p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_compscp297k_muon/20260708_055911` |
| 41 | trained | 9 | 18.33 | 31.44 | 64.85 | `gpt_bert_mixed_babylm2026_strict_small_replace9p5pct_qaorig_roles0p5_turntaking0p5_qawhere60dest40local0p25_whonames0p25_phys2_mat2_compscp297k_aliasbind102k_muon/20260708_143214` |
| 42 | trained/reproduction | 10 | **23.25** | 36.70 | 69.11 | `gpt_bert_strict_small_muon_auxlr1e3_repro_seed42_20260713_133448` |

## Source artifacts

- Training histories:
  `/home/jorge/babylm-baby-steps-runs/training/*/benchmarks/mntp_likelihood/entity_tracking/`
- Random-initialization seeds:
  `/home/jorge/babylm-baby-steps-runs/analysis/random_entity_tracking_7seeds.csv`
- Context and token-length diagnostics:
  `/home/jorge/babylm-baby-steps-runs/analysis/entity_tracking_context_prior_diagnostics.csv`
- Recency analysis:
  `/home/jorge/babylm-baby-steps-runs/analysis/go76dof__wwm_curriculum_simplification_40k_entity_tracking_recency.json`
- Last-touch counterfactuals:
  `/home/jorge/babylm-baby-steps-runs/analysis/entity_tracking_last_touch/go76dof__wwm_curriculum_simplification_40k/summary.csv`
- Query-touch mechanism ladder:
  `/home/jorge/babylm-baby-steps-runs/analysis/entity_tracking_query_touch_mechanism/go76dof__wwm_curriculum_simplification_40k/summary.csv`
- Container-state custom evaluation:
  `/home/jorge/babylm-baby-steps-runs/analysis/container_state_custom_eval_v1/`
- Container-state generator notes:
  [`scripts/synthetic_data_generation/container_state/README.md`](../scripts/synthetic_data_generation/container_state/README.md)

## Bottom line

There are real insights here, but not yet a demonstrated general entity-tracking
improvement. The archive tells a coherent story: the legacy metric was dominated
by answer-length and empty-state artifacts; broad curriculum changes mostly
leave filtered performance near chance; and targeted v1 data taught a fixed
surface answer instead of state updates. Counterfactual probes suggest the next
useful target is compositional state maintenance under varied language, with no
invariant answer phrase and with empty cases kept out of the headline.

## Counterfactual v2 pilot launched on 2026-07-15

### Question and controlled comparison

The pilot asks two separable questions. First, does replacing 1% of the 10M-word
corpus with better entity-tracking text help under ordinary language-model
training? Second, holding that text fixed, does direct ranking supervision help
the model use the changing context rather than memorize answer-form statistics?

All arms start from a fresh random initialization with seed 42 and use the same
GPT-BERT architecture, tokenizer, local fixed-word mixing pipeline, optimizer,
ten epochs, and data budget. Arm 1 draws 10M words from the base source; the
treatment arms replace the last 100k-word allocation with synthetic text and
share that exact treatment corpus. Thus Arm 2 minus Arm 1 isolates the revised
data; Arm 3 minus Arm 2 isolates completion ranking; and Arm 4 minus Arm 3
isolates the added reverse context-ranking constraint.

| Arm | 10M-word corpus | Auxiliary objective | What the contrast tests |
|---:|---|---|---|
| 1 | 100% BabyLM strict-small | none | fresh matched control |
| 2 | 99% base + 1% counterfactual v2 | none | ordinary synthetic-data effect |
| 3 | same as Arm 2 | completion ranking | direct correct-vs-paired-answer supervision |
| 4 | same as Arm 2 | symmetric completion + context ranking | whether context discrimination adds value |

This is a first-pass single-seed screen, not a final effect estimate. Any
promising arm still needs matched seed replicates before a small score change is
treated as real.

### Counterfactual v2 data changes

The generated pool contains 775 paired worlds, or 1,550 ordinary-LM documents,
and 130,136 whitespace words. An exact 100,000-word subset is placed into the
10M-word training mix. Every pair starts from the same initial inventory and
shares zero, one, or two prefix operations. The last operation is changed to
produce two different states of the queried container. This yields traces with
one, two, or three total operations.

The generator intentionally removes the v1 shortcut:

- both paired answers are non-empty and different;
- there is no `nothing.`, `no items.`, `none.`, `empty.`, `non-nothing`, or
  other invariant answer phrase in the v2 pool;
- add, move, and remove counterfactuals are almost exactly balanced
  (259/258/258 pairs);
- one-, two-, and three-operation traces are balanced (266/262/247 pairs);
- four ordinary training renderers are balanced, while 1,000 held-out pairs
  use disjoint validation vocabulary and two held-out renderer styles;
- training and held-out initial-world signatures have zero overlap.

Each paired world is written in three views: ordinary correct text for Arms
2–4, two correct-versus-hard-negative completion rows for Arm 3, and one 2×2
counterfactual row for Arm 4. The latter two views are derived from the same
symbolic record, so they cannot silently disagree about the correct state.

### Ranking objectives

Let (s_{ij}=s(y_j\mid x_i)) be the mean continuation log-likelihood of answer
(y_j) under context (x_i), and let

\[
\ell(a,b)=\operatorname{softplus}\!\left(\frac{b-a}{\tau}\right)
\]

penalize a preferred score (a) that does not exceed a competing score (b).
Arm 3 applies the existing completion-ranking form to both sides of each pair:

\[
L_{\mathrm{completion}}
=\tfrac12\left[\ell(s_{11},s_{12})+\ell(s_{22},s_{21})\right].
\]

Arm 4 adds the reverse comparison: for a fixed answer, prefer the context that
actually produced it. Its single symmetric loss is

\[
L_{\mathrm{symmetric}}
=\tfrac14\left[
\ell(s_{11},s_{12})+\ell(s_{22},s_{21})
+\ell(s_{11},s_{21})+\ell(s_{22},s_{12})
\right].
\]

The auxiliary term is added only on causal steps:

\[
L=L_{\mathrm{LM}}+0.03L_{\mathrm{rank}},\qquad \tau=1.
\]

Mean target-token log-likelihood is used to reduce direct answer-length bias.
Arm 3 uses 32 completion examples per auxiliary update (64 scored branches),
and Arm 4 uses 16 paired examples (also 64 scored branches), matching auxiliary
branch compute rather than raw row count. Both losses average their inequalities
so the coefficient remains on a comparable scale.

### Evaluation contract and order

The existing 4,000-item custom benchmark is evaluated **first** after each
checkpoint. The official Entity Tracking suite is evaluated second. They are
co-primary: the custom benchmark has at least equal model-selection weight and
is the first gate because it directly exposes IID, lexical-OOD, template-OOD,
and two-operation behavior. For both suites we retain summed and mean
continuation likelihood records so answer-length sensitivity stays visible.

The headline comparison will show:

1. the custom benchmark overall and its four slices, with non-empty/no-empty-
   option subsets shown explicitly;
2. filtered official-style macro and micro results on examples with no
   `nothing` candidate;
3. the held-out paired-ranking score and margin for Arms 3 and 4;
4. empty-state results only as a separate diagnostic, never folded into model
   selection.

### Execution layout and artifacts

GPU 0 was requested but had approximately 19.6 GB allocated and active compute
at launch time, so it was not oversubscribed. The pilot instead uses two idle
RTX 5090s: GPU 1 runs Arm 1 then Arm 3 sequentially, while GPU 2 runs Arm 2 then
Arm 4 sequentially. The two queues run concurrently in persistent `tmux`
sessions. A real GPU smoke test of Arm 4 completed successfully before launch.

The active pilot ID is `20260716_000043`. An initial Arm 1 startup used the
older Hugging Face-tokenized baseline shards; although its token budget was
close, that did not match the local fixed-word data pipeline used by the
treatment. It was stopped within the first epoch, marked
`aborted_unmatched_control`, and will not be evaluated or counted. Arm 1 was
restarted from scratch using the local 10M-word control described above. Arm 2
was already correct and continued uninterrupted.

Core reproducibility artifacts:

- data generator config:
  [`container_state_counterfactual_v2_seed42.yaml`](../configs/container_state/container_state_counterfactual_v2_seed42.yaml)
- fixed 1% mix config:
  [`babylm2026_strict_small_replace1pct_container_state_counterfactual_v2_seed42.yaml`](../configs/data_mixes/babylm2026_strict_small_replace1pct_container_state_counterfactual_v2_seed42.yaml)
- matched local control mix:
  [`babylm2026_strict_small_local_control_seed42.yaml`](../configs/data_mixes/babylm2026_strict_small_local_control_seed42.yaml)
- common training recipe:
  [`gpt_bert_entity_tracking_counterfactual_pilot_common_muon.yaml`](../configs/train/gpt_bert_entity_tracking_counterfactual_pilot_common_muon.yaml)
- exact arm overrides and custom-first evaluation order:
  [`run_entity_tracking_counterfactual_pilot_worker.sh`](../scripts/sweeps/run_entity_tracking_counterfactual_pilot_worker.sh)
- two-GPU launcher:
  [`launch_entity_tracking_counterfactual_pilot.sh`](../scripts/sweeps/launch_entity_tracking_counterfactual_pilot.sh)
- generation audit:
  `data/container_state/container_state_counterfactual_v2_seed42/audit_report.json`

The experiment outputs, queue logs, per-arm status files, checkpoints, and both
evaluation suites are stored together under
`/home/jorge/babylm-baby-steps-runs/experiments/entity_tracking_counterfactual_pilot_v2/<pilot-id>/`.

## Multi-domain v3 follow-up

The v2 outcome motivates a data-first follow-up rather than immediately adding
more ranking pressure. Multi-domain v3 uses 12 training domains, three held-out
domains, six training discourse families, and two held-out styles. It creates a
clean IID split plus lexical-, template-, domain-, and compositional-OOD splits,
each containing 1,200 paired worlds. The custom and official benchmarks remain
co-primary and are evaluated before the new synthetic suite.

The reference build has 4,800 training pairs and 852,841 available synthetic
words. The 10M-word LM treatment samples exactly 100,000 of those words. Half of
the pairs change item identity while preserving target cardinality; the
remaining pairs balance add, move, and remove interventions. Five-choice
distractors also preserve answer cardinality.

The first full audit exposed a branch-label artifact: always assigning the
target-changing intervention to branch 1 produced surface-only classifier AUC
0.5786. Exact side balancing removed it; the final build scores 0.5000 against a
0.5500 limit. It also has zero normalized cross-split collisions and disjoint
catalog surfaces and held-out domains.

The initial experiment is three paired seeds: matched local baseline followed by
LM-only on the same GPU, with seeds 42, 43, and 44 on GPUs 1, 2, and 5. Completion
ranking is gated on positive mean IID transfer, at least two non-negative seeds,
and no mean co-primary regression larger than one point. If admitted, it uses
one auxiliary update per five causal updates; symmetric ranking remains behind a
second transfer gate.

The initial queues were launched on 2026-07-16 as experiment
`20260716_092451`. Real-GPU smoke tests of both training and the combined
five-choice/paired evaluator passed immediately before launch.

The complete design, commands, audit outcomes, run ledger, and result tables are
maintained in
[`entity_tracking_multidomain/README.md`](../scripts/synthetic_data_generation/entity_tracking_multidomain/README.md).

## Multi-domain v3 outcome (2026-07-16)

The completed experiment was a matched three-seed comparison (42, 43, and 44)
between the local-text GPT-BERT mixed baseline and an LM-only treatment that
replaced 1% of the 10M-word corpus with multidomain-v3 synthetic state-tracking
text. Both arms used the same model, 1:1 causal/MNTP objective, Muon optimizer,
and ten-epoch schedule. The treatment contained 100,000 synthetic words across
1,123 selected documents and did not yet use an auxiliary ranking objective.

The v3 generator fixed the main design failures of container-state v1. It has
no empty targets or invariant answer phrase, varies vocabulary, domain, and
rendering style, and defines separate IID, lexical-OOD, template-OOD,
domain-OOD, and compositional-OOD evaluations. Its final
[audit report](../data/container_state/container_state_multidomain_v3_seed42/audit_report.json)
found no normalized train/evaluation collisions, and a context bag-of-words
classifier could not identify the counterfactual branch (ROC AUC 0.50).

Final summed-likelihood five-choice scores, averaged across the three matched
seeds, were:

| Evaluation | Baseline | 1% v3 LM-only | Delta |
| --- | ---: | ---: | ---: |
| Custom v1 | 18.88% | 18.49% | -0.39 pp |
| Legacy all-9,483 Entity Tracking | 21.52% | 25.74% | +4.22 pp |
| Multidomain v3 IID | 29.21% | 24.21% | -5.00 pp |
| Lexical OOD | 22.72% | 22.49% | -0.24 pp |
| Template OOD | 29.85% | 23.49% | -6.36 pp |
| Domain OOD | 40.01% | 28.42% | -11.60 pp |
| Compositional OOD | 16.13% | 22.47% | +6.35 pp |

The intervention failed its predefined initial gate. IID accuracy declined for
all three seeds (-7.33, -6.50, and -1.17 points), while template and domain OOD
also regressed sharply. Compositional OOD improved for seeds 42 and 43 but
reversed slightly for seed 44, so it was not a stable general improvement.
Paired counterfactual scoring showed small gains in symmetric preference on
some splits, but side-level completion and both-sides-correct performance were
mostly flat or worse.

The +4.22-point all-9,483 score is not a reliable headline because that legacy
summed metric includes the known `nothing` and continuation-length confounds and
varied substantially by seed. The cleaner custom result was essentially flat.
The conclusion is therefore that v3 removed obvious surface shortcuts, but 1%
LM-only exposure did not teach robust online state tracking. The gate selected
`stop`, so the completion-ranking and symmetric-ranking stages were not
advanced. The exact execution and gate logic live in the
[launcher](../scripts/sweeps/launch_entity_tracking_multidomain_v3.sh) and
[summarizer](../scripts/analysis/summarize_entity_tracking_multidomain_v3.py).
