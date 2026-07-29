# Tests

Most model users do not need to study the test code. The reduced suite exists to
catch submission mistakes without training a full model or running the official
evaluation.

Install the development dependencies and run everything:

```bash
pip install -e ".[dev]"
pytest -q
```

Only three test files are kept:

| File | What it protects |
| --- | --- |
| `test_submission_configs.py` | Canonical hyperparameters, milestone schedules, batch/token accounting, the small public layout, and the unresolved exact SidBert YAML. |
| `test_submission_training_smoke.py` | Tiny causal and GPT-BERT CPU updates, document packing, and objective microbatch cycling. |
| `test_submission_evaluation.py` | Backend selection, uploader validation, BOS tokenization, shell syntax, README links, and evaluator-patch setup. |

The SidBert check is intentionally unresolved: it asserts that no guessed
`sidbert_strict.yaml` is published. Once the original 100M YAML and matching
tokenization config are supplied, that check should be replaced by exact
hyperparameter and milestone assertions.

Return to the [project README](../README.md) for the reproduction commands.
