# Tests

Most users do not need to read or run the test suite. The tests mainly
help the people maintaining the repository confirm that submission code still
works after a change.

## What Should I Run?

| You are... | Run this |
| --- | --- |
| Using a model | `python scripts/smoke_10_batch_deterministic.py --device cpu` |
| Reproducing the submission | The three submission tests shown below |
| Changing repository code | `pytest -q` |

Install the development dependencies before running tests:

```bash
pip install -e ".[dev]"
```

## Submission Tests

These are the tests most closely connected to the competition submission:

- `test_submission_configs.py` checks the selected training recipes and Chatty
  BERT's effective batch size.
- `test_submission_training_smoke.py` runs tiny GPT-BERT and Llama-style
  training updates on CPU.
- `test_submission_evaluation.py` checks checkpoint discovery and the MNTP or
  causal evaluation choice.

Run only these checks with:

```bash
pytest -q tests/test_submission_configs.py \
  tests/test_submission_training_smoke.py \
  tests/test_submission_evaluation.py
```

They do not train a full model, require a GPU, or download the BabyLM corpus.

## Remaining Tests

The other files protect internal pieces such as token shards, model creation,
Muon, Liger, hybrid training, checkpoint averaging, and path handling. Model
users do not need to understand them. Maintainers can run all of them with
`pytest -q` before publishing a code change.

For model setup, training, and evaluation instructions, return to the
[project README](../README.md).
