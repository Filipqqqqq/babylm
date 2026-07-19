# Comparable 1% Linguistic-Repair Text Mix

This directory contains a deterministic 10,000,000-word UTF-8 text analogue of
the GPT-BERT 1% linguistic-repair intervention. Use `train.txt` when adapting
the data to a modified model or tokenizer.

This is not a text decoding of the exact 1% experiment. That experiment
replaced blocks after tokenization, so its authoritative input remains the
[exact token-shard package](../../token_shards/gpt_bert_linguistic_repair_1pct_v1/README.md).

The text mix uses 9,900,000 target base words and 100,000 target repair words.
The repair allocation is the standalone 1.5% recipe scaled proportionally to
1%: 33,333 congruence, 33,333 turn-taking, 10,000 destination, 6,667 local
preposition, and 16,667 who/name words. Whole-document sampling produced
100,070 repair words in the capped corpus, or 1.0007% of its 10,000,000 words.

## Fetch and verify

From the repository root:

```bash
git lfs pull --include="data/text_mixes/babylm2026_strict_small_linguistic_repair_1pct_seed42/train.txt"
cd data/text_mixes/babylm2026_strict_small_linguistic_repair_1pct_seed42
sha256sum -c SHA256SUMS
cd ../../..
```

`counts.json`, `mix_config.yaml`, and `mix_manifest.json` record the source
allocations, seeds, selection policy, and final counts. The much larger
document-level `mix_documents.jsonl` intermediate is intentionally omitted;
`train.txt` is sufficient for tokenization and training.

## Tokenize

For the original GPT-BERT mixed tokenizer:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_linguistic_repair_1pct_gpt_bert_mixed_shards.yaml
```

For a modified tokenizer, copy that configuration and change
`tokenizer.source`, `tokenizer.output_dir`, and `shards.output_dir`. A modified
GPT-BERT architecture that keeps the original vocabulary and special-token IDs
may instead reuse the exact pre-tokenized shards.

## Rebuild the text mix

```bash
python scripts/data/build_text_mix.py \
  --config configs/data_mixes/babylm2026_strict_small_linguistic_repair_1pct_seed42.yaml
```

Rebuilding requires the canonical BabyLM 2026 Strict-Small text at the base
source path recorded in `mix_config.yaml`. The checked-in `train.txt` avoids
that prerequisite for ordinary reuse.
