# GPT-BERT 1% linguistic-repair token shards (v1)

This directory is the authoritative, byte-identical training input used for the
`repair_1pct` arm of `gpt_bert_linguistic_repair_dose_response_v1` and the later
AdamMuon 1% curriculum runs. The intervention was applied after tokenization,
so these binary shards—not a reconstructed UTF-8 corpus—define the experiment.

The dataset contains all 18,024,113 token IDs: approximately 99% canonical
BabyLM 2026 Strict-Small content and 1% linguistic-repair content. Ten of 1,038
eligible loader blocks were replaced, for 163,840 replacement tokens and an
actual block fraction of 0.963391%. The validation shard is byte-identical to
the canonical baseline.

## Fetch and verify

From the repository root:

```bash
git lfs pull --include="data/token_shards/gpt_bert_linguistic_repair_1pct_v1/*.bin"
cd data/token_shards/gpt_bert_linguistic_repair_1pct_v1
sha256sum -c SHA256SUMS
cd ../../..
```

## Fetch the pinned GPT-BERT assets

The shards reuse the public GPT-BERT mixed tokenizer. Download the exact
tokenizer, configuration, and remote-code revision used by this experiment:

```bash
conda run --no-capture-output -n babylm hf download \
  BabyLM-community/babylm-baseline-10m-gpt-bert-mixed \
  config.json configuration_gpt_bert.py modeling_gpt_bert.py \
  tokenizer.json tokenizer_config.json special_tokens_map.json \
  --revision 4d2fa91ec77bd1fdc2203da23c21727c50798306 \
  --local-dir cache/gpt_bert_mixed_4d2fa91
```

`tokenizer_provenance.json` records both the pinned upstream file hashes and the
hashes of the locally materialized tokenizer used to build the shards. Their
vocabularies, special-token IDs, and test encodings were verified equivalent.

## Reproduce the AdamMuon curriculum run

The command below uses the existing tracked training recipe while overriding
only its data and pinned GPT-BERT asset locations:

```bash
conda run --no-capture-output -n babylm python scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_baseline_mixed_babylm2026_strict_small_adamuon_lr0p02_aux0p006_tailavg_seed42.yaml \
  --set data.data_dir=data/token_shards/gpt_bert_linguistic_repair_1pct_v1 \
  --set tokenizer.name=cache/gpt_bert_mixed_4d2fa91 \
  --set model.repo=cache/gpt_bert_mixed_4d2fa91 \
  --set run.name=adamuon-10m-mixed-linguistic-repair1pct-lr0.02-aux0.006-s42-curriculum-tailavg
```

The original run set `run.deterministic: false`. These files guarantee
byte-identical data, but retraining is not expected to produce byte-identical
weights across hardware and software stacks. Use the published checkpoint when
exact model weights are required.

## Provenance

- Base data: `BabyLM-community/BabyLM-2026-Strict-Small` (MIT)
- Tokenizer/model assets: `BabyLM-community/babylm-baseline-10m-gpt-bert-mixed`
  at revision `4d2fa91ec77bd1fdc2203da23c21727c50798306` (Apache-2.0)
- Selection seed: `20260714`
- Representation: raw little-endian `uint16` token IDs
- Document format: `[BOS/CLS] + tokenizer(text) + [EOS/SEP]`

The readable synthetic inputs remain under `data/linguistic_repair/`; their
paths, token allocations, and hashes are listed in `repair_sources.tsv`.
