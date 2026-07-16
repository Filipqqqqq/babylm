# Word2Vec EWoK Analysis

This module treats Word2Vec as a corpus-level static-semantics probe. It asks
whether a vector space trained on BabyLM text organizes EWoK contexts and
targets in the same paper-style context-sensitive way as a large external
Word2Vec baseline such as Google News.

The evaluator mean-pools in-vocabulary word vectors for each EWoK span and
drops out-of-vocabulary tokens. For each item it computes:

```text
S11 = cos(Context1, Target1)
S21 = cos(Context2, Target1)
S22 = cos(Context2, Target2)
S12 = cos(Context1, Target2)

k1 = S11 - S21
k2 = S22 - S12
```

Each margin becomes `1.0`, `0.5`, or `0.0` depending on whether it is positive,
zero, or negative. Paper-style context sensitivity is `1.0` when both target
preferences behave consistently across the two contexts, including the
both-reversed case. It is not ordinary completion-choice accuracy.

## Commands

Train a BabyLM Word2Vec model:

```bash
python scripts/analysis/word2vec/train_word2vec.py \
  --config configs/analysis/word2vec/train_babylm2026_strict_small.yaml
```

Train the generated material-affordance corpus:

```bash
python scripts/analysis/word2vec/train_word2vec.py \
  --config configs/analysis/word2vec/train_material_affordance_ewok_4000.yaml
```

Evaluate the material-affordance model on full EWoK:

```bash
python scripts/analysis/word2vec/eval_ewok_word2vec.py \
  --config configs/analysis/word2vec/eval_material_affordance_ewok_full.yaml
```

For material-only slices, use:

```bash
python scripts/analysis/word2vec/eval_ewok_word2vec.py \
  --config configs/analysis/word2vec/eval_material_affordance_material_dynamics.yaml

python scripts/analysis/word2vec/eval_ewok_word2vec.py \
  --config configs/analysis/word2vec/eval_material_affordance_material_properties.yaml
```

For a future generated text corpus, train and evaluate in one timestamped run:

```bash
python scripts/analysis/word2vec/run_generated_text_ewok_probe.py \
  --train-text /path/to/train.txt \
  --name my_generated_strategy \
  --material-evals
```

Add custom EWoK slices with repeated `--eval label=/path/to/items.jsonl`.

To test generated text as an intervention on a BabyLM text background, first
build a reproducible mixed input. For example, to repeat physical-dynamics text
until it is roughly 1% of the BabyLM 10M clean train word count:

```bash
python scripts/analysis/word2vec/build_text_mix.py \
  --base-dir ${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/clean_text/babylm_10m/train \
  --base-glob "*.train" \
  --synthetic-text data/concept_curriculum/physical_dynamics_ewok_micro_1000_slip_bridge_qwen/train.txt \
  --synthetic-word-ratio 0.01 \
  --output-dir data/analysis/word2vec/text_mixes/babylm10m_plus_physical_1pct
```

The mixer writes `synthetic_repeated.txt`, `base_files.txt`, `manifest.json`,
and a ready-to-edit `w2v_command.sh`.

Evaluate Google News Word2Vec on EWoK:

```bash
GOOGLE_NEWS_WORD2VEC_PATH=/path/to/GoogleNews-vectors-negative300.bin.gz \
python scripts/analysis/word2vec/eval_ewok_word2vec.py \
  --config configs/analysis/word2vec/eval_google_news_ewok.yaml
```

The Google News config is case-preserving by default because those vectors are
case-sensitive. This is the default baseline setting used for the headline
Google News context-sensitivity score.

Evaluate a BabyLM-trained model:

```bash
BABYLM_WORD2VEC_MODEL_PATH=/path/to/model.model \
python scripts/analysis/word2vec/eval_ewok_word2vec.py \
  --config configs/analysis/word2vec/eval_babylm2026_strict_small_ewok.yaml
```

Compare two completed eval runs:

```bash
python scripts/analysis/word2vec/compare_ewok_word2vec.py \
  --left /path/to/google_news_eval \
  --right /path/to/babylm_eval \
  --left-label google_news \
  --right-label babylm \
  --output-dir /path/to/comparison
```

## Outputs

`item_scores.csv` contains raw `S11/S21/S22/S12`, `k1/k2`, half-credit fields,
context sensitivity, and retained-token counts for every span.

`domain_summary.csv` aggregates paper-style context sensitivity by EWoK domain.

`oov_summary.csv` lists frequent out-of-vocabulary tokens, which is important
for separating vector coverage failures from semantic failures.
