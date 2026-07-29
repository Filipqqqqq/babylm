# BabySteps at BabyLM 2026

*Stronger Optimization and Causal Baselines for Data-Constrained Pretraining*

BabySteps is our BabyLM 2026 system repository. We train language models from
scratch under the Strict-Small (10M-word) and Strict (100M-word) budgets and
study which changes actually improve the complete BabyLM evaluation—not only a
single benchmark. This repository contains the training package, exact recipes,
official evaluation wrappers, and the exploratory experiments behind the paper.

[Paper summary](#paper-in-one-minute) · [Systems](#submitted-systems) ·
[Reproduce](#choose-a-reproduction-level) · [Repository map](#repository-map) ·
[Negative results](#exploratory-and-negative-results) · [Citation](#citation)

## Paper in one minute

Our main result is simple: optimizer and learning-rate selection produced larger
and more reliable gains than the curriculum, synthetic-data, teacher-supervision,
and corpus-mixing interventions we tested.

- AdaMuon plus tail-weight averaging was strongest for our 10M GPT-BERT system.
- Lower-rate Muon was strongest for our 100M GPT-BERT system; the best optimizer
  recipe changed with scale.
- A modern Llama-style decoder trained with Muon and Liger kernels beat our
  completed reevaluation of the official GPT-2 strict checkpoint on all seven
  NLP benchmark categories.
- Small evaluation suites showed substantial run-to-run variance, so differences
  of only a few points should be interpreted cautiously.

The Llama–GPT-2 comparison is a comparison of practical systems, not a controlled
architecture ablation: architecture, optimizer, implementation, and
hyperparameters change together.

## Submitted systems

| System | Track | Model and objective | Optimizer | Final weights |
| --- | --- | --- | --- | --- |
| **MurphysLaw-10M-mixed** | Strict-Small | GPT-BERT, mixed, MNTP eval | AdaMuon, LR 0.02 / aux 0.006 | Final-20% weight average |
| **SidBert-100M-mixed** | Strict | GPT-BERT, mixed, MNTP eval | Muon, LR 0.005 / aux 0.007 | Last step |
| **Chatty BERT** | Strict | 127.6M Llama-style causal decoder | Muon, LR 0.01 / aux 1e-4, Liger | Last step |

Despite its leaderboard name, Chatty BERT is a decoder-only Llama-style model,
not a BERT-family model.

### Headline development results

| System | BLiMP | BLiMP Supplement | SuperGLUE | NLP average |
| --- | ---: | ---: | ---: | ---: |
| MurphysLaw (10M) | 71.6 | 63.9 | 69.9–70.1¹ | 53.6² |
| SidBert selected run (100M) | 80.2 | 73.6 | 73.5 | 57.3² |
| Chatty BERT (100M) | 80.57 | 69.88 | 70.19 | 56.18³ |
| Official GPT-2 strict, reevaluated | 74.73 | 65.00 | 66.06 | 52.79³ |

¹ Identically configured development twin; ² six-suite development mean; ³ seven-suite mean from the completed causal-baseline comparison.

The reported Chatty BERT scores use 27,010 updates at 65,536 tokens/update; the manuscript table should replace `13,510 × 131,072`.

## Choose a reproduction level

### 1. Evaluate released weights

This is the recommended and least expensive way to verify the paper. Initialize
the pinned official evaluator and install the evaluation dependencies:

```bash
git clone --recurse-submodules https://github.com/JorgeCMurillo/babylm-baby-steps.git
cd babylm-baby-steps
python -m venv .venv
source .venv/bin/activate
pip install -e ".[eval]"
```

Evaluate a Hub-hosted Strict-Small GPT-BERT model in MNTP mode:

```bash
bash scripts/eval/eval_babylm_strict_small_submission.sh \
  <hugging-face-repo-id> 8e-5 mntp
```

Evaluate a Hub-hosted Strict causal model:

```bash
bash scripts/eval/eval_babylm_strict_submission.sh \
  <hugging-face-repo-id> 3e-5
```

The wrappers run full zero-shot evaluation, SuperGLUE finetuning, milestone
evaluation, Global-PIQA, Age-of-Acquisition, and result collation through the
pinned [official BabyLM evaluator](external/babylm-eval/).

### 2. Quick training example (smoke test)

**Start here to train a model.** This tiny causal recipe validates model
construction, Muon, logging, and checkpoint writing before a full run:

```bash
pip install -e ".[dev,liger]"
python scripts/train/train_causal_lm.py \
  --config configs/train/smoke/llama_tiny_bpe16384_10m_muon_liger_smoke.yaml
```

### 3. Full paper-model training

Full runs require the official BabyLM corpus, prepared token shards, bf16-capable
NVIDIA GPUs, and substantial compute. By default, reusable artifacts and run
outputs live outside Git:

```text
${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/
${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/
```

The current paper-facing recipes are:

| System | Training recipe | Status |
| --- | --- | --- |
| MurphysLaw | [`gpt_bert_baseline_mixed_..._adamuon_...yaml`](configs/train/gpt_bert_baseline_mixed_babylm2026_strict_small_adamuon_lr0p02_aux0p006_tailavg_seed42.yaml) | Canonical optimizer/tail-average recipe |
| SidBert | `configs/train/` | Exact standalone 100M submission config still needs consolidation |
| Chatty BERT | [`llama_12l896_..._auxlr1e4_bos_...yaml`](configs/train/llama_12l896_bpe16384_babylm2026_strict_muon_liger_submission_auxlr1e4_bos_seed42.yaml) | Reported-score recipe; effective batch 64 |

Do not substitute the 100M masked-focus LAMB example for SidBert: it is a
different objective and optimizer. The missing standalone SidBert mapping is a
known release-blocking documentation task rather than an implied recipe.

**Training commands.** Prepare datasets and token shards using the
[data guide](data/README.md), then run a configured trainer:

```bash
python scripts/train/train_hybrid_lm.py --config <gpt-bert-config.yaml>
python scripts/train/train_causal_lm.py --config <llama-config.yaml>
```

## Evaluation cautions

- Seed-identical GPU retrains are not bit-identical. In our submission retrain,
  BLiMP moved by 0.1 points while the 82-item BLiMP Supplement moved by 4.5.
- The official GPT-2 leaderboard row was incomplete, so the paper compares
  Chatty BERT with a completed reevaluation of the official checkpoint.
- Report the evaluation backend. GPT-BERT submissions use MNTP; Chatty BERT and
  GPT-2 use causal scoring.
- Report the aggregation set. Six-suite development averages and seven-suite
  causal-comparison averages are not interchangeable.

## Exploratory and negative results

The repository retains failed and inconclusive experiments because they are
useful controls, but they are not recommended submission recipes. Tested
interventions include curriculum corpora, TinyStories, targeted synthetic text,
teacher-weighted objectives, corpus mixtures, entity-state curricula, spatial
ranking, and NextLat objectives.

Start with the [research summary](docs/research_summary/README.md), then consult:

- [targeted and synthetic-data workflows](scripts/synthetic_data_generation/README.md);
- [entity-tracking experiment report](docs/entity_tracking_experiment_report.md);
- [auxiliary-objective and project decisions](docs/decisions.md); and
- [operational runbook](docs/runbook.md).

Some research-summary pages describe promising partial benchmark movements from
the broader experimental archive. The workshop paper's conclusion is based on
the complete aggregate, including downstream finetuning tradeoffs.

## Repository map

```text
configs/        Data, tokenization, training, evaluation, and sweep recipes
data/           Small versioned datasets and data documentation
docs/           Research narrative, experiment reports, decisions, and runbook
external/       Pinned official BabyLM evaluation submodule
scripts/        Thin data, training, evaluation, analysis, and upload entrypoints
src/babysteps/  Reusable Python package
tests/          Focused submission checks and training smoke tests
```

Raw corpora, checkpoints, caches, logs, and generated evaluation outputs should
remain outside Git. Use `BABYLM_DATA_ROOT`, `BABYSTEPS_CACHE_DIR`, and
`BABYSTEPS_RUNS_DIR` to override machine-local paths when needed.

## Development

The project supports Python 3.10–3.12. Run `pytest -q` before changing training
or evaluation behavior. Keep scripts thin and commit each publication config.

Dependency ranges live in [`pyproject.toml`](pyproject.toml). For archival
reproduction, also record exact PyTorch, CUDA, Transformers, Liger, driver, GPU,
and evaluation-submodule versions with the run artifacts.

## Citation

```bibtex
@misc{satheesh2026babysteps,
  title  = {BabySteps at BabyLM 2026: Stronger Optimization and Causal
            Baselines for Data-Constrained Pretraining},
  author = {Satheesh, Siddhi and Murillo, Jorge},
  year   = {2026}
}
```

A repository license has not yet been specified. Add a `LICENSE` file before
public archival release so that reuse terms are explicit.
