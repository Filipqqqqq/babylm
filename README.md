# BabySteps at BabyLM 2026

*Stronger Optimization and Causal Baselines for Data-Constrained Pretraining*

This is the reproduction branch for our BabyLM 2026 workshop submission. It
keeps one clear path through the repository: prepare the official data, train a
selected model, validate and upload its checkpoints, and run the official
evaluation. The larger experiment archive lives on
[`jorge/research`](https://github.com/JorgeCMurillo/babylm-baby-steps/tree/jorge/research).

[Results](#submitted-systems) · [Install](#install) · [Prepare data](#1-prepare-data) ·
[Train](#2-train-a-model) · [Upload](#3-validate-and-upload) ·
[Evaluate](#4-run-the-official-evaluation)

## Paper in one minute

Optimizer and learning-rate selection mattered more than the curriculum,
synthetic-data, teacher-supervision, and corpus-mixing interventions we tested.
AdaMuon with final-weight averaging was strongest at 10M words, while the best
100M GPT-BERT used a different Muon recipe. A Llama-style causal system trained
with Muon and Liger beat our complete reevaluation of the official GPT-2 Strict
checkpoint across all seven NLP categories. This compares practical systems,
not architecture alone, and small evaluation suites varied across retrains.

## Submitted systems

| System | Track | Model | Optimizer and final weights | Public recipe |
| --- | --- | --- | --- | --- |
| **MurphysLaw-10M-mixed** | Strict-Small | GPT-BERT, mixed objective | AdaMuon; final-20% weight average | Ready |
| **SidBert-100M-mixed** | Strict | GPT-BERT, mixed objective | Muon; last step | **Release gate** |
| **Chatty BERT** | Strict | 127.6M Llama-style causal decoder | Muon + Liger; last step | Ready |

Despite its leaderboard name, Chatty BERT is a decoder-only Llama-style model,
not a BERT-family model.

### Headline development results

| System | BLiMP | BLiMP Supplement | SuperGLUE | NLP average |
| --- | ---: | ---: | ---: | ---: |
| MurphysLaw (10M) | 71.6 | 63.9 | 69.9–70.1 | 53.6 |
| SidBert selected run (100M) | 80.2 | 73.6 | 73.5 | 57.3 |
| Chatty BERT (100M) | 80.57 | 69.88 | 70.19 | 56.18 |
| Official GPT-2 Strict, reevaluated | 74.73 | 65.00 | 66.06 | 52.79 |

The reported Chatty BERT run used an effective batch of 64: 16 sequences per
microbatch × 4 accumulation steps. At sequence length 1,024, that is 65,536
tokens per update. The manuscript configuration table should report
`27,010 × 65,536`, not the later batch-128 retrain's `13,510 × 131,072`.

## SidBert release gate

The original standalone 100M SidBert YAML has not yet been supplied. This branch
does not guess or relabel another experiment as the exact submission recipe.
Until that file and its matching shard-preparation config are added, the
three-model reproduction is intentionally incomplete and `main` should not be
published as final. A focused test keeps this limitation visible.

## Install

Python 3.10–3.12 is supported. Clone the pinned official evaluator with the
repository, create an environment, and install the project:

```bash
git clone --recurse-submodules https://github.com/JorgeCMurillo/babylm-baby-steps.git
cd babylm-baby-steps
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval,liger]"
bash scripts/eval/setup_babylm_eval.sh
```

The setup helper accepts `--check`, requires the pinned evaluator commit, and
refuses unrelated tracked changes. It applies only the compatibility patch in
[`patches/babylm-eval/compat.patch`](patches/babylm-eval/compat.patch).

Reusable files and training runs stay outside Git by default:

```text
${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/
${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/
```

## Reproduce a system

Full runs download the official BabyLM corpus and need substantial GPU compute.
Run all commands from the repository root.

### 1. Prepare data

Build MurphysLaw's Strict-Small GPT-BERT shards:

```bash
python scripts/data/build_token_shards.py --config configs/tokenization/submission/murphyslaw_strict_small_shards.yaml
```

Build Chatty BERT's Strict byte-BPE tokenizer and shards:

```bash
python scripts/data/build_token_shards.py --config configs/tokenization/submission/chatty_bert_strict_shards.yaml
```

The submitted Chatty BERT tokenizer prepends `<s>` when ordinary evaluation
code encodes a sequence. Add that behavior without changing its vocabulary or
raw tokenization:

```bash
python scripts/tokenization/prepare_submission_bos_tokenizer.py --source ~/babylm-baby-steps-cache/tokenizers/babylm2026_strict_byte_bpe_16384 --output ~/babylm-baby-steps-cache/tokenizers/babylm2026_strict_byte_bpe_16384_bos_default
```

See [`data/README.md`](data/README.md) for paths and cache behavior.

### 2. Train a model

MurphysLaw uses the hybrid causal/MNTP trainer and the paper's local-reference
AdaMuon recipe:

```bash
python scripts/train/train_hybrid_lm.py --config configs/train/submission/murphyslaw_strict_small.yaml
```

Chatty BERT uses the causal trainer:

```bash
python scripts/train/train_causal_lm.py --config configs/train/submission/chatty_bert_strict.yaml
```

To check the causal path before a full run, use the retained tiny Llama recipe:

```bash
python scripts/train/train_causal_lm.py --config configs/train/smoke/llama_tiny_bpe16384_10m_muon_liger_smoke.yaml
```

The launchers are intentionally thin. Their reusable implementation is the
installed `babysteps` package under [`src/babysteps/`](src/babysteps/). The
[`scripts` guide](scripts/README.md) explains that layout in plain language.

### 3. Validate and upload

The uploader checks `final` and every required `chck_<N>M` directory before it
contacts Hugging Face. Repositories are private by default. Start with a dry run:

```bash
python scripts/hf/upload_babylm_submission.py --run-dir /path/to/training/run --repo-id USER/MODEL --track strict-small --dry-run
```

Use `--track strict` for Chatty BERT or the future SidBert recipe. Remove
`--dry-run` only after validation succeeds; pass `--public` only if public
creation is intended.

### 4. Run the official evaluation

MurphysLaw uses MNTP scoring, which is the Strict-Small wrapper's default:

```bash
bash scripts/eval/eval_babylm_strict_small_submission.sh USER/MURPHYSLAW 8e-5
```

Chatty BERT uses causal scoring, the Strict wrapper's default:

```bash
bash scripts/eval/eval_babylm_strict_submission.sh USER/CHATTY-BERT 8e-5
```

SidBert will use the same Strict wrapper with an explicit `mntp` backend:

```bash
bash scripts/eval/eval_babylm_strict_submission.sh USER/SIDBERT 8e-5 mntp
```

The wrappers run zero-shot tasks, SuperGLUE finetuning, official word-milestone
revisions, Global-PIQA, Age-of-Acquisition, and collation. They do not run during
repository CI.

## Repository map

```text
configs/        Two verified submission recipes, matching data recipes, and smoke/eval configs
data/           Cache policy and data instructions; generated content is ignored
external/       Pinned official BabyLM evaluator submodule
patches/        Code-only evaluator compatibility patch
scripts/        Data, training, validation/upload, tokenizer, and evaluation commands
src/babysteps/  Reusable Python runtime used by the thin scripts
tests/          Three submission-focused test files
```

Exploratory configs, sweeps, synthetic-data tools, reports, and research datasets
remain available on [`jorge/research`](https://github.com/JorgeCMurillo/babylm-baby-steps/tree/jorge/research).

## Tests

The reduced suite runs CPU updates and repository checks; it does not perform a
full GPU run or official evaluation:

```bash
pytest -q
```

See [`tests/README.md`](tests/README.md) for what each of the three files covers.
