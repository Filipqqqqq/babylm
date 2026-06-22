# Scripts

Preferred entrypoints live in small topical folders:

```text
scripts/data/                                      dataset preparation and tokenization
scripts/train/                                     training entrypoints
scripts/eval/                                      evaluation launchers and wrappers
scripts/sweeps/                                    training sweep launchers
scripts/analysis/                                  plotting, result summaries, and lexical reports
scripts/hf/                                        Hugging Face download/upload helpers
```

The root-level Python scripts are compatibility shims for older commands. New
docs and new automation should prefer the organized paths above.

Lexical analysis helpers:

See `src/babysteps/analysis/lexical/README.md` for the conceptual guide and
output-file descriptions.

The lexical context summary writes raw neighbor columns plus
`top_left_content_neighbors` and `top_right_content_neighbors`, which filter
common English stopwords by default. Use `--include-context-stopwords` to make
those content-neighbor columns match the raw neighbor counts.

```bash
python scripts/analysis/report_lexical_analysis.py --max-docs 10000

python scripts/analysis/report_lexical_analysis.py \
  --config configs/analysis/lexical_babylm2026_strict_small.yaml

python scripts/analysis/export_ewok_lexical_targets.py

python scripts/analysis/report_lexical_analysis.py \
  --config configs/analysis/lexical_babylm2026_strict_small_ewok.yaml

python scripts/analysis/plot_domain_concept_frequencies.py \
  --target-coverage /path/to/target_coverage.csv \
  --concepts configs/analysis/targets/ewok_fast_domain_concepts.yaml

python scripts/analysis/benchmark_spacy_pipe.py \
  --max-docs 5000 \
  --batch-sizes 64,128,256,512 \
  --n-processes 1,2,4
```

EWoK LM context-sensitivity diagnostics:

```bash
python scripts/eval/eval_ewok_context_sensitivity.py \
  --run-dir /path/to/training/run \
  --batch-size 64

python scripts/analysis/plot_ewok_context_dynamics.py \
  --summary /path/to/training/run/benchmarks/ewok/context_summary.csv
```

The same evaluator can run during training by adding a top-level config block:

```yaml
ewok_context_sensitivity:
  enabled: true
  every_steps: 500
  batch_size: 64
  score_reduction: sum
  plot: true
  plot_every_evals: 1
```

By default this uses the full-eval EWoK folder:

```text
external/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered
```

If that folder is absent, EWoK evaluation fails loudly. For fast eval, override
`data_dir` explicitly:

```yaml
ewok_context_sensitivity:
  data_dir: external/babylm-eval/strict/evaluation_data/fast_eval/ewok_fast
```

During training, EWoK context-sensitivity outputs are written under:

```text
<run_dir>/benchmarks/ewok/context_items.jsonl
<run_dir>/benchmarks/ewok/context_summary.csv
<run_dir>/benchmarks/ewok/context_domain_dynamics.png
```

Benchmark plot x-axis:

Training writes exact `tokens_seen` into benchmark item and summary files. To
also write `words_seen`, provide a conversion factor:

```yaml
logging:
  benchmark_plot_x_axis: words_seen
  words_per_token: 0.57
```

or equivalently:

```yaml
logging:
  benchmark_plot_x_axis: words_seen
  tokens_per_word: 1.75
```

Or let training estimate the conversion once from a deterministic prefix of the
training tokens:

```yaml
logging:
  benchmark_plot_x_axis: words_seen
  estimate_words_per_token: true
  words_per_token_sample_tokens: 2048
```

If `benchmark_plot_x_axis: words_seen` is set without an explicit conversion,
training estimates `words_per_token` automatically. The estimate is approximate
and is written back into the saved run config.

`benchmark_plot_x_axis` can be `step`, `tokens_seen`, or `words_seen`. The
standalone plotting scripts accept the same choice with `--x-axis`.

BLiMP likelihood diagnostics:

```bash
python scripts/analysis/plot_blimp_likelihood_dynamics.py \
  --summary /path/to/training/run/benchmarks/blimp/likelihood_summary.csv \
  --label BLiMP \
  --summary /path/to/training/run/benchmarks/blimp_supplement/likelihood_summary.csv \
  --label "BLiMP Supplement"
```

The same scorer can run during training by adding a top-level config block:

```yaml
blimp_likelihood:
  enabled: true
  every_steps: 500
  batch_size: 64
  plot: true
  plot_every_evals: 1
```

By default this evaluates both local full-eval folders:

```text
external/babylm-eval/strict/evaluation_data/full_eval/blimp_filtered
external/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered
```

During training, BLiMP likelihood outputs are written under:

```text
<run_dir>/benchmarks/blimp/likelihood_items.jsonl
<run_dir>/benchmarks/blimp/likelihood_summary.csv
<run_dir>/benchmarks/blimp/overall_accuracy.png
<run_dir>/benchmarks/blimp/field_dynamics.png
<run_dir>/benchmarks/blimp/linguistics_term_dynamics.png
<run_dir>/benchmarks/blimp_supplement/likelihood_items.jsonl
<run_dir>/benchmarks/blimp_supplement/likelihood_summary.csv
<run_dir>/benchmarks/blimp_supplement/overall_accuracy.png
<run_dir>/benchmarks/blimp_supplement/uid_dynamics.png
<run_dir>/benchmarks/blimp_likelihood_overall_accuracy.png
```

Entity Tracking likelihood diagnostics:

```bash
python scripts/analysis/plot_entity_tracking_dynamics.py \
  --summary /path/to/training/run/benchmarks/entity_tracking/likelihood_summary.csv
```

The same scorer can run during training by adding a top-level config block:

```yaml
entity_tracking_likelihood:
  enabled: true
  every_steps: 500
  batch_size: 64
  plot: true
  plot_every_evals: 1
```

By default this evaluates the local full-eval folder:

```text
external/babylm-eval/strict/evaluation_data/full_eval/entity_tracking
```

For fast eval, override `data_dir`:

```yaml
entity_tracking_likelihood:
  enabled: true
  every_steps: 500
  data_dir: external/babylm-eval/strict/evaluation_data/fast_eval/entity_tracking_fast
  plot: true
```

During training, Entity Tracking likelihood outputs are written under:

```text
<run_dir>/benchmarks/entity_tracking/likelihood_items.jsonl
<run_dir>/benchmarks/entity_tracking/likelihood_summary.csv
<run_dir>/benchmarks/entity_tracking/overall_accuracy.png
<run_dir>/benchmarks/entity_tracking/family_dynamics.png
<run_dir>/benchmarks/entity_tracking/family_numops_dynamics.png
```

COMPS likelihood diagnostics:

```bash
python scripts/analysis/plot_comps_dynamics.py \
  --summary /path/to/training/run/benchmarks/comps/likelihood_summary.csv
```

The same scorer can run during training by adding a top-level config block:

```yaml
comps_likelihood:
  enabled: true
  every_steps: 500
  batch_size: 64
  plot: true
  plot_every_evals: 1
```

By default this evaluates the local full-eval folder:

```text
external/babylm-eval/strict/evaluation_data/full_eval/comps
```

The overall COMPS score is the official-style macro average over the four
COMPS subtasks. Item-level correctness is strict: the acceptable prefix must
assign higher continuation likelihood than the unacceptable prefix.

During training, COMPS likelihood outputs are written under:

```text
<run_dir>/benchmarks/comps/likelihood_items.jsonl
<run_dir>/benchmarks/comps/likelihood_summary.csv
<run_dir>/benchmarks/comps/overall_accuracy.png
<run_dir>/benchmarks/comps/subtask_dynamics.png
<run_dir>/benchmarks/comps/negative_sample_type_dynamics.png
```

Removed deprecated aliases:

```text
scripts/build_hf_token_shards.py  -> scripts/data/build_token_shards.py
scripts/train_gpt2_small.py       -> scripts/train/train_causal_lm.py
```
