# Evaluation Notes

These evaluators are lightweight training-time diagnostics. They are meant to
make score trajectories easy to inspect during pretraining, while staying close
to the BabyLM zero-shot evaluation conventions where practical.

## Shared scoring conventions

- Ties are strict incorrectness in the custom evaluators. We use `>` rather than
  random tie-breaking so runs are deterministic.
- The likelihood evaluators default to summed token log likelihoods. Because
  token log probabilities are usually negative, summed scoring can favor shorter
  candidates when compared spans have different token counts. Set
  `score_reduction: mean` in a benchmark config to compare mean token log
  likelihoods instead, or `score_reduction: both` to write separate sum and mean
  rows. Summary rows and per-item records include the selected
  `score_reduction`, and plotting keeps sum and mean as separate curves. Sum
  remains the default for continuity with earlier diagnostics; which reduction
  should count as a submission-facing metric is worth discussing with future
  BabyLM submission-team collaborators.
- Training-time benchmark diagnostics run once before training by default
  (`logging.benchmark_eval_at_start: true`). This writes a baseline point with
  `epoch=0`, `step=0`, `tokens_seen=0`, and `words_seen=0.0` when word-count
  estimation is enabled.
- Benchmark diagnostics can be scheduled with `every_steps`, `every_words`, or
  both. Word-based intervals use `words_seen = tokens_seen * words_per_token`;
  if an interval such as `entity_tracking_likelihood.every_words` is configured
  without an explicit conversion, training estimates `words_per_token` from a
  deterministic prefix of the training tokens. The default sample is 4,096
  tokens.
- In the formulas below, `reduce log P(...)` means either the sum or the mean of
  the scored token log probabilities, depending on `score_reduction`.
- Accuracy rows named `overall/all` should be the benchmark-level score used for
  plots and `metrics_history.json`.
- When an item-weighted score differs from the benchmark-level score, the
  item-weighted score is saved separately as `overall_micro/all`.

## MNTP Backend

`mntp_likelihood.py` mirrors the BabyLM zero-shot MNTP backend rather than plain
masked-LM scoring. For every target token at position `i` in the scored span, it
creates a copy of the full input with token `i` replaced by the mask token, then
scores the original token from logits at position `i - 1`:

```text
score_token_i = log P(original_token_i at logits[i - 1] | input with token_i masked)
```

Candidate scores are then reduced with the same `sum`, `mean`, or `both`
convention used by the causal diagnostics. The reusable MNTP scorer currently
has thin wrappers for BLIMP, BLIMP Supplement, Entity Tracking, and COMPS. It
does not change the custom EWoK context-sensitivity evaluator.

## BLIMP and BLIMP Supplement

The custom BLIMP scorer computes:

```text
score_good = reduce log P(sentence_good)
score_bad  = reduce log P(sentence_bad)
correct    = score_good > score_bad
```

This matches the simple sentence-likelihood diagnostic we wanted. BabyLM reports
the mean of UID accuracies, not raw item-weighted accuracy. This matters most for
BLIMP Supplement, where subtasks have very different numbers of examples. The
custom BLIMP evaluator writes the UID macro to `overall/all` and preserves the
item-weighted diagnostic as `overall_micro/all`.

## EWoK

The custom EWoK evaluator is the Moonshot-style LM context-sensitivity view, not
the official EWoK completion-choice score. For each item it evaluates whether:

```text
k1 = S11 - S21 > 0
k2 = S22 - S12 > 0
score = 0.5 * (correct_k1 + correct_k2)
```

The domain rows are saved as `group_scope=domain`. The evaluator writes the mean
of domain scores to `overall/all` and preserves the old comparison-weighted
score as `overall_micro/all`.

## Entity Tracking

Each item has five options, with option 0 treated as correct:

```text
score_option_i = reduce log P(option_i | input_prefix)
correct = score_option_0 > max(score_option_1...)
```

The official BabyLM aggregate is not raw item accuracy. It computes each
`family_numops` UID accuracy, averages the UID accuracies inside each family,
then averages the family scores:

```text
family_score = mean(score(family_0_ops), ..., score(family_5_ops))
overall      = mean(score(regular), score(ambiref), score(move_contents))
```

The custom Entity Tracking evaluator now writes this official-style score to
`overall/all`. It preserves the old item-weighted score as `overall_micro/all`,
and preserves item-weighted family scores as `family_micro/<family>`.

## COMPS

COMPS scores the shared property phrase under acceptable and unacceptable
prefixes:

```text
score_acceptable   = reduce log P(property_phrase | prefix_acceptable)
score_unacceptable = reduce log P(property_phrase | prefix_unacceptable)
correct = score_acceptable > score_unacceptable
```

COMPS already writes `overall/all` as a macro average over the four subtasks
(`base`, `wugs`, `wugs_dist_before`, `wugs_dist_in_between`) and saves the raw
item-weighted value as `overall_micro/all`.
