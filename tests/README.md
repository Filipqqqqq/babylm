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
| `test_submission_configs.py` | All three canonical recipes, milestone schedules, distributed batch/token accounting, and the small public layout. |
| `test_submission_training_smoke.py` | Tiny causal and GPT-BERT CPU updates, document packing, and objective microbatch cycling. |
| `test_submission_evaluation.py` | Backend selection, uploader validation, BOS tokenization, shell syntax, README links, and evaluator-patch setup. |

Return to the [project README](../README.md) for the reproduction commands.
