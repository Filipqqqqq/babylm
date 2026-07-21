# Entity Tracking Query-Touch Mechanism Probe

This matched diagnostic extends the query-touch ladder to separate two possible
effects: attention to the queried box and repetition of objects that belong in
the correct final answer. It is separate from the official BabyLM evaluation
data and excludes `nothing.` answers.

Each of 300 base worlds has seven conditions with identical initial state,
final state, last query-box operation, answer options, option order, and correct
answer:

- `q1`: one final operation touching the query box;
- `q3`/`q5`: reversible moves of final-answer objects out of and back into the
  query box, followed by the same final operation;
- `t3`/`t5`: reversible moves of non-answer transient objects into and back out
  of the query box, followed by the same final operation;
- `c3`/`c5`: the same non-answer objects and reversible move templates as the
  transient conditions, but entirely between unrelated boxes.

The paired `t3 - c3` and `t5 - c5` contrasts estimate the effect of touching the
query box while holding answer-object repetition absent. The paired `q3 - t3`
and `q5 - t5` contrasts estimate the additional effect of repeatedly mentioning
objects that are in the correct final answer.

Generate:

```bash
python scripts/eval/generate_entity_tracking_query_touch_ladder.py \
  --include-transient
```

Evaluate:

```bash
python scripts/eval/eval_entity_tracking_query_touch_ladder.py \
  --model /path/to/huggingface/model \
  --data data/analysis/entity_tracking_query_touch_mechanism/touch_mechanism_seed42.jsonl \
  --output-dir ~/babylm-baby-steps-runs/analysis/entity_tracking_query_touch_mechanism
```
