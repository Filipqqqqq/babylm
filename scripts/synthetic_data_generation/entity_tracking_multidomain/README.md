# Multi-Domain Entity Tracking v3

## Purpose

This experiment tests whether synthetic entity-tracking data transfers when the
underlying state operation is expressed across genuinely different settings,
vocabularies, and discourse structures. It follows the narrow v2 pilot, where
training ranking loss improved but held-out ranking and the custom benchmark did
not. The first v3 comparison is therefore data-first: matched baseline versus
ordinary LM training with the new synthetic data. Ranking is staged behind an
explicit transfer gate.

The custom validation benchmark and official Entity Tracking benchmark are
co-primary. They are always reported separately and given equal prominence.
The custom benchmark is evaluated first. V3 targets are always nonempty, so
empty-state behavior remains a separate benchmark diagnostic rather than a
training shortcut.

## Dataset design

The hand-authored catalog contains 12 training domains:

- zoo, warehouse, squirrel food caches, and dog belongings;
- library, overnight train, teaching kitchen, and harbor;
- primary school, robotics laboratory, museum, and botanical garden.

Airport baggage, aquarium transfers, and space-station supplies are held out by
domain. Every domain defines natural locations, items, agents, and action verbs,
with a second disjoint surface vocabulary for lexical transfer.

Training uses six discourse families: narrative, observer dialogue, field
notes, status report, procedural recap, and question-answer recount. Timestamped
logs and passive inspection reports are held out by style.

The reference seed-42 build contains 4,800 training pairs and five fixed
1,200-pair evaluation splits:

| Split | Domains | Vocabulary | Styles |
|---|---|---|---|
| IID | training | training | training |
| lexical OOD | training | held-out surfaces | training |
| template OOD | training | training | held-out |
| domain OOD | held-out | domain-native | training |
| compositional OOD | held-out | held-out surfaces | held-out |

Each pair shares an initial world and common event prefix, then differs in one
controlled event. Half are equal-cardinality identity changes: both branches add
one different item to the queried location. The other half is balanced across
add, move, and remove interventions. Branch assignment is exactly balanced.
This last condition was added after the first full audit found that consistently
placing the target-changing event in branch 1 yielded a surface-only AUC of
0.5786. After balancing sides, the audit AUC is 0.5000.

Each evaluation branch has five choices: the correct state followed by four
unique same-cardinality object-swap distractors. The same worlds are also
exported as completion-ranking rows and paired 2x2 counterfactual-ranking rows.

## Outputs and schemas

The generated root is
`data/container_state/container_state_multidomain_v3_seed42/`:

- `train.txt`: two ordinary-LM documents per training pair;
- `pairs/{train,iid,lexical_ood,template_ood,domain_ood,compositional_ood}.jsonl`:
  structured worlds, interventions, contexts, and targets;
- `entity_tracking/*.jsonl`: five-choice validation data with option 0 correct;
- `auxiliary/*_completion_rank.jsonl`: context, correct completion, hard negative;
- `auxiliary/*_counterfactual_rank.jsonl`: Context1/2 and Target1/2 matrices;
- `audit_report.json`: counts, balance, leakage checks, and classifier gate;
- `manifest.json`: paths, SHA-256 hashes, seed, and generation metadata.

The audit rejects empty or identical paired targets, normalized duplicates
within or across splits, catalog surface overlap, held-out-domain leakage, and a
surface branch-classifier ROC-AUC above 0.55.

## Reproduction

Generate and audit:

```bash
conda run --no-capture-output -n babylm \
  python scripts/synthetic_data_generation/container_state/generate_multidomain_entity_tracking.py
```

Build the fixed 1% / 100,000-word mix and GPT-BERT shards:

```bash
conda run --no-capture-output -n babylm \
  python scripts/data/build_text_mix.py \
  --config configs/data_mixes/babylm2026_strict_small_replace1pct_container_state_multidomain_v3_seed42.yaml

conda run --no-capture-output -n babylm \
  python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_replace1pct_container_state_multidomain_v3_gpt_bert_mixed_shards.yaml
```

Launch the initial three-seed comparison:

```bash
bash scripts/sweeps/launch_entity_tracking_multidomain_v3.sh
```

GPU 1 runs seed 42, GPU 2 seed 43, and GPU 5 seed 44. Within each GPU,
baseline runs before LM-only; the three queues run concurrently. The launcher
requires an idle device with at least 30,000 MiB free and refuses to
oversubscribe.

Follow-up stages are intentionally manual and must be launched only after their
gate is documented:

```bash
bash scripts/sweeps/launch_entity_tracking_multidomain_v3.sh --stage completion EXPERIMENT_ID
bash scripts/sweeps/launch_entity_tracking_multidomain_v3.sh --stage symmetric EXPERIMENT_ID
```

Completion ranking uses batch size 32 and one auxiliary update per five causal
updates. Across the existing 10-epoch schedule this is about 3.5 passes over the
9,600 branch rows derived from 4,800 pairs. Symmetric ranking uses batch size 16
for comparable scored-branch compute.

## Gates

Proceed from LM-only to completion ranking only when all conditions hold:

1. mean clean-IID improvement over matched baseline is positive;
2. at least two of three seed-level IID changes are non-negative;
3. mean custom-benchmark change is no worse than -1 percentage point;
4. mean official-benchmark change is no worse than -1 percentage point.

Proceed from completion to symmetric ranking only when completion improves mean
IID by at least 1 point and no OOD split or co-primary benchmark drops by more
than 1 point.

## Run ledger and outcomes

Preparation completed on 2026-07-16:

| Artifact | Outcome |
|---|---|
| Full corpus | 4,800 train pairs; 1,200 pairs per evaluation split |
| Available synthetic text | 852,841 words across 9,600 documents |
| Training mix | exactly 100,000 synthetic words in 10,000,000 total |
| Duplicate/leakage audit | passed; zero normalized cross-split collisions |
| Surface classifier | passed; ROC-AUC 0.5000, threshold 0.5500 |
| Token shards | built; 17 train shards plus 1 validation shard |
| GPU smoke tests | training and combined evaluator passed |

The initial run was launched as experiment `20260716_092451` on GPUs 1, 2,
and 5. Summarize it after all six `.complete` status files exist:

```bash
python scripts/analysis/summarize_entity_tracking_multidomain_v3.py \
  /home/jorge/babylm-baby-steps-runs/experiments/entity_tracking_multidomain_v3/20260716_092451
```

Initial model results are filled in when all paired seed queues complete:

| Seed | Arm | Custom | Official | IID | Lexical OOD | Template OOD | Domain OOD | Compositional OOD |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | baseline | pending | pending | pending | pending | pending | pending | pending |
| 42 | LM-only | pending | pending | pending | pending | pending | pending | pending |
| 43 | baseline | pending | pending | pending | pending | pending | pending | pending |
| 43 | LM-only | pending | pending | pending | pending | pending | pending | pending |
| 44 | baseline | pending | pending | pending | pending | pending | pending | pending |
| 44 | LM-only | pending | pending | pending | pending | pending | pending | pending |

For every checkpoint, `01_custom_v1` and `02_official` contain the co-primary
likelihood evaluations. `03_multidomain_v3` contains both five-choice summaries
and paired completion/symmetric accuracy and margins. Queue logs, status files,
checkpoints, and evaluations live under
`/home/jorge/babylm-baby-steps-runs/experiments/entity_tracking_multidomain_v3/EXPERIMENT_ID/`.
