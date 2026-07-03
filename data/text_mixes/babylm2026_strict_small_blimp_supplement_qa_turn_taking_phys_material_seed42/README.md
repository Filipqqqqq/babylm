# BLIMP Supplement QA Turn-Taking Physical Material Mix

This directory contains the 10,000,000-word strict-small text mix used for the
reader-facing GPT-BERT BLIMP Supplement repair path.

The checked-in artifact is:

```text
train.txt
```

Small provenance sidecars are also checked in:

```text
counts.json
mix_config.yaml
mix_manifest.json
```

The full `mix_documents.jsonl` file is intentionally not checked in. It expands
every selected line into JSON provenance and is much larger than the plain text.
The `train.txt` file is enough for tokenization and training.

## Tokenize

Build GPT-BERT mixed token shards with:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_gptbert_shards_seed42.yaml
```

This writes shards under:

```text
${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/token_shards/babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_gptbert
```

## Train

After tokenization, train with:

```bash
python scripts/train/train_hybrid_lm.py \
  --config configs/train/gptbert_babylm2026_strict_small_blimp_supplement_qa_turn_taking_phys_material_muon_seed42.yaml
```
