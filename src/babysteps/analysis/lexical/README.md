# Lexical Analysis

The lexical analysis tools connect BabyLM evaluation targets back to the
training corpus and tokenizer. The goal is to ask whether evaluation-relevant
concepts were lexically available to the model, whether they appeared in useful
forms and contexts, and whether the tokenizer preserved or fragmented them.

This does not prove that a model learned a concept. It gives corpus and
tokenizer diagnostics that help interpret evaluation performance.

## What It Measures

The analysis currently focuses on four questions:

- **Concept availability:** how often target words appear in the corpus.
- **Concept form:** whether targets appear as exact surface forms, lemmas, or both.
- **Tokenizer accessibility:** how many subword pieces target words produce.
- **Context quality:** what words and example sentences surround each target.

## Where Things Live

```text
src/babysteps/analysis/lexical/       reusable lexical-analysis package
scripts/analysis/report_lexical_analysis.py
scripts/analysis/export_ewok_lexical_targets.py
scripts/analysis/plot_domain_concept_frequencies.py
scripts/analysis/benchmark_spacy_pipe.py
configs/analysis/                    dataset/report configs and target lists
```

The package files are organized by responsibility:

```text
pipeline.py        end-to-end report pipeline
targets.py         target YAML parsing and surface/lemma matching
text.py            regex and spaCy text processors
tokenization.py    tokenizer fertility and token-piece inspection
stats.py           in-memory target-level counters
reports.py         CSV/JSON report builders and writers
context.py         context-neighbor stopword filtering
sampling.py        bounded example sampling
domain_frequency.py
                   per-domain concept frequency tables and plots
ewok_targets.py    EWoK ConceptA/ConceptB export helpers
```

## Running

Install the analysis dependencies before running spaCy-backed reports:

```bash
pip install -e ".[analysis]"
python -m spacy download en_core_web_sm
```

Run the default strict-small report:

```bash
python scripts/analysis/report_lexical_analysis.py \
  --config configs/analysis/lexical_babylm2026_strict_small.yaml
```

Run with EWoK-derived lexical targets:

```bash
python scripts/analysis/export_ewok_lexical_targets.py

python scripts/analysis/report_lexical_analysis.py \
  --config configs/analysis/lexical_babylm2026_strict_small_ewok.yaml
```

For a quick smoke run, cap the number of documents:

```bash
python scripts/analysis/report_lexical_analysis.py \
  --config configs/analysis/lexical_babylm2026_strict_small.yaml \
  --max-docs 10000
```

Analysis outputs are written outside git by default, under:

```text
${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/analysis/lexical/
```

The older local-cleaned BabyLM 10M lexical config has been removed to avoid
confusing it with the current BabyLM-community Strict-Small corpus. Use
`lexical_babylm2026_strict_small.yaml` or
`lexical_babylm2026_strict_small_ewok.yaml` for community-dataset lexical
reports.

## Main Outputs

A lexical analysis run writes one report directory containing:

```text
target_coverage.csv
exposure_thresholds.csv
global_tokenizer_summary.json
global_tokenizer_words.csv
context_summary.csv
context_examples.jsonl
manifest.json
```

## Output Files

### `target_coverage.csv`

One row per lexical target.

This is the main target-level report. It includes total mentions, surface and
lemma match counts, document counts, exposure buckets, common observed forms,
common sources, and tokenizer fragmentation for matched target words.

Use this file to ask: did the model have many chances to see this concept?

### `exposure_thresholds.csv`

A compact summary of how many targets fall into each mention-count bucket.

Use this file to ask: how many evaluation targets are unseen, rare, moderately
common, or frequent?

### `global_tokenizer_summary.json`

Corpus-level tokenizer fertility statistics.

This summarizes average subwords per word type and per word occurrence, plus
frequent words that are split into many subwords.

Use this file to ask: is this tokenizer generally fragmenting the corpus?

### `global_tokenizer_words.csv`

One row per observed word type in the corpus.

It includes the word count, number of tokenizer pieces, and the actual token
pieces.

Use this file to inspect specific words or find highly fragmented frequent
words.

### `context_summary.csv`

One row per lexical target with common neighboring words.

It includes raw neighbor columns:

```text
top_left_neighbors
top_right_neighbors
```

and stopword-filtered content-neighbor columns:

```text
top_left_content_neighbors
top_right_content_neighbors
```

The content-neighbor columns remove common English stopwords by default while
preserving negation words such as `not`, `no`, and `never`.

Use this file to ask: does this concept appear near informative words?

### `context_examples.jsonl`

Sampled examples for each target.

Each row contains the sentence containing the target, optional previous and next
sentences, a local word window, tokenization pieces, and the number of times the
target appeared in that sentence.

Use this file to manually inspect whether target mentions are meaningful,
ambiguous, formulaic, or noisy.

### `manifest.json`

Metadata for the run.

It records the dataset, tokenizer, target file, config options, output paths,
document count, word count, and run timestamp.

Use this file to make reports reproducible.

## Targets

Lexical targets are YAML files under:

```text
configs/analysis/targets/
```

The current matcher supports single-token targets. Each target can define
surface forms and lemmas:

```yaml
version: 1
targets:
  - id: dog
    label: dog
    surfaces: [dog, dogs, puppy, puppies]
    lemmas: [dog, puppy]
```

Surface matches count exact observed forms. Lemma matches use the configured
text processor, usually spaCy, to group inflected forms when possible.

## EWoK Targets

EWoK concepts can be exported from the `ConceptA` and `ConceptB` columns:

```bash
python scripts/analysis/export_ewok_lexical_targets.py
```

This creates:

```text
configs/analysis/targets/ewok_fast_lexical_targets.yaml
configs/analysis/targets/ewok_fast_domain_concepts.yaml
configs/analysis/targets/ewok_fast_unsupported_targets.csv
```

Single-token concepts become lexical targets. Multiword concepts and
placeholders are written to the unsupported CSV instead of being silently
dropped. The domain concepts YAML can be used with the domain-frequency plotting
script.

## Domain Frequency Plots

After running a lexical report, plot concept frequencies by domain:

```bash
python scripts/analysis/plot_domain_concept_frequencies.py \
  --target-coverage /path/to/target_coverage.csv \
  --concepts configs/analysis/targets/ewok_fast_domain_concepts.yaml
```

Multiword concepts are currently marked as unsupported in the frequency table
and plotted as missing values.

## Important Config Options

```yaml
analysis:
  neighbor_window: 5
  context_word_radius: 20
  examples_per_target: 10
  filter_context_neighbor_stopwords: true
  tokenizer_mode: leading_space
  progress_logging: true
  progress_interval_docs: 10000
  progress_interval_seconds: 30
```

`neighbor_window` controls how many neighboring words are counted for
`context_summary.csv`.

`context_word_radius` controls how much row-local text is shown around sampled
examples in `context_examples.jsonl`.

`examples_per_target` controls the maximum number of sampled example sentences
per target.

`filter_context_neighbor_stopwords` controls whether the content-neighbor
columns in `context_summary.csv` remove common English stopwords. Use
`--include-context-stopwords` to disable this from the CLI.

`tokenizer_mode` controls how isolated words are passed to the tokenizer. The
default, `leading_space`, is useful for GPT-style byte-BPE tokenizers because
word-initial tokens may differ depending on whitespace.

`progress_logging`, `progress_interval_docs`, and
`progress_interval_seconds` control periodic CLI progress updates. Use
`--no-progress` to disable them for quiet runs.

## Current Limitations

- Multiword concepts are exported but not directly matched yet.
- Lexical exposure is evidence of availability, not evidence of learning.
- Context examples are sampled and row-local.
- Lemmatization quality depends on the configured text processor.
- Some evaluation datasets contain noisy or fuzzy concept-property pairs.
