# Entity Tracking Query-Touch Ladder

This matched diagnostic tests whether repeated operations involving the queried
box causally improve model scoring. It is separate from the official BabyLM
evaluation data and excludes `nothing.` answers.

Each of 300 base worlds has five conditions with identical initial state, final
state, last query-box operation, answer options, option order, and correct
answer:

- `q1`: one total operation and one query-box touch;
- `q3`: three total operations and three query-box touches;
- `q5`: five total operations and five query-box touches;
- `c3`: three total operations but only one query-box touch;
- `c5`: five total operations but only one query-box touch.

The additional `q3`/`q5` operations move final-answer objects out of and back
into the queried box, leaving the final state unchanged. The matched `c3`/`c5`
controls perform the same number and templates of reversible moves on unrelated
boxes. The primary paired contrasts are `q3 - c3` and `q5 - c5` in correct-answer
likelihood margin.

Generate:

```bash
python scripts/eval/generate_entity_tracking_query_touch_ladder.py
```

Evaluate:

```bash
python scripts/eval/eval_entity_tracking_query_touch_ladder.py \
  --model /path/to/huggingface/model
```

Results default to:

```text
${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/analysis/entity_tracking_query_touch_ladder/<model>/
```
