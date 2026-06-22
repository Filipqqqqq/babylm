# Evaluation Notes

These evaluators are lightweight training-time diagnostics. They are meant to
make score trajectories easy to inspect during pretraining, while staying close
to the BabyLM zero-shot evaluation conventions where practical.

## Shared scoring conventions

- Ties are strict incorrectness in the custom evaluators. We use `>` rather than
  random tie-breaking so runs are deterministic.
- These evaluators currently use summed token log likelihoods rather than mean
  token log likelihoods. Because token log probabilities are usually negative,
  summed scoring can favor shorter candidates when compared spans have different
  token counts. This is a length-normalization caveat, not a known scoring bug.
  A future patch should revisit whether each benchmark should use `sum`, `mean`,
  or report both; this is worth discussing with future BabyLM submission-team
  collaborators before treating these training-time diagnostics as submission
  metrics.
- Accuracy rows named `overall/all` should be the benchmark-level score used for
  plots and `metrics_history.json`.
- When an item-weighted score differs from the benchmark-level score, the
  item-weighted score is saved separately as `overall_micro/all`.

## BLIMP and BLIMP Supplement

The custom BLIMP scorer computes:

```text
score_good = sum log P(sentence_good)
score_bad  = sum log P(sentence_bad)
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
score_option_i = sum log P(option_i | input_prefix)
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
score_acceptable   = sum log P(property_phrase | prefix_acceptable)
score_unacceptable = sum log P(property_phrase | prefix_unacceptable)
correct = score_acceptable > score_unacceptable
```

COMPS already writes `overall/all` as a macro average over the four subtasks
(`base`, `wugs`, `wugs_dist_before`, `wugs_dist_in_between`) and saves the raw
item-weighted value as `overall_micro/all`.
