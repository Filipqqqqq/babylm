# Scripts

Preferred entrypoints live in small topical folders:

```text
scripts/data/                                      dataset preparation and tokenization
scripts/train/                                     training entrypoints
scripts/eval/                                      evaluation launchers and wrappers
scripts/sweeps/                                    training sweep launchers
scripts/analysis/                                  plotting, result summaries, and lexical reports
scripts/hf/                                        Hugging Face download/upload helpers
scripts/synthetic_data_generation/material_world/  material-world synthetic generation, audit, and review utilities
scripts/material_world/                            older material-world compatibility entrypoints
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

Removed deprecated aliases:

```text
scripts/build_hf_token_shards.py  -> scripts/data/build_token_shards.py
scripts/train_gpt2_small.py       -> scripts/train/train_causal_lm.py
```
