# Entity Tracking Last-Touch Counterfactuals

This directory contains a deterministic paired probe for separating Entity
Tracking state updates from lexical-recency and last-touch strategies. It is
not part of the official BabyLM evaluation data.

Generate the canonical 300-pair-per-family probe:

```bash
python scripts/eval/generate_entity_tracking_last_touch_pairs.py
```

The four families are:

- `earlier_removal`: change an earlier removal while holding the final inbound
  query-box operation fixed;
- `earlier_inbound`: change an earlier inbound operation while holding the final
  removal fixed;
- `irrelevant_reordering`: reorder independent operations so the correct state
  is unchanged but the relevant operation is near or far from the query;
- `whole_box_transfer`: change a source-box update while holding the final
  whole-contents transfer fixed.

Each JSONL record stores structured initial state and operations, the rendered
prompt, five fixed-order options shared by both pair members, and an explicit
`target_index`. The generator symbolically simulates every operation and rejects
records whose stored target, last touch, or rendered prompt is inconsistent.
The probe intentionally excludes `nothing.` targets and distractors. All five
options in an item contain the same number of objects to avoid the summed-MLM
likelihood advantage for shorter answers.

Evaluate a masked language model with the official BabyLM token-masking score:

```bash
python scripts/eval/eval_entity_tracking_last_touch.py \
  --model /path/to/huggingface/model
```

Results default to:

```text
${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/analysis/entity_tracking_last_touch/<model>/
```
