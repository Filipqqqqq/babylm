# Data and caches

`main` does not store generated datasets or token shards in Git. This folder
contains only this guide; `data/*` is ignored so an existing local cache is not
deleted when the repository layout changes.

The two public preparation recipes stream the official datasets from Hugging
Face:

| Model | Dataset | Config |
| --- | --- | --- |
| MurphysLaw | `BabyLM-community/BabyLM-2026-Strict-Small` | `configs/tokenization/submission/murphyslaw_strict_small_shards.yaml` |
| Chatty BERT | `BabyLM-community/BabyLM-2026-Strict` | `configs/tokenization/submission/chatty_bert_strict_shards.yaml` |

Run a recipe from the repository root:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/submission/murphyslaw_strict_small_shards.yaml
```

By default, reusable outputs go to:

```text
~/babylm-baby-steps-cache/tokenizers/
~/babylm-baby-steps-cache/token_shards/
```

Set `BABYSTEPS_CACHE_DIR` to move the complete cache. More specific environment
variables shown in each YAML can override one tokenizer or shard directory.
Training outputs use `BABYSTEPS_RUNS_DIR` and also remain outside Git.

Chatty BERT needs one post-preparation step. It preserves the trained vocabulary
and raw encodings while making ordinary encoding prepend `<s>`:

```bash
python scripts/tokenization/prepare_submission_bos_tokenizer.py \
  --source ~/babylm-baby-steps-cache/tokenizers/babylm2026_strict_byte_bpe_16384 \
  --output ~/babylm-baby-steps-cache/tokenizers/babylm2026_strict_byte_bpe_16384_bos_default
```

The original SidBert shard-preparation config will be added with its exact
training YAML. It is deliberately not reconstructed from another experiment.

Generated data, downloaded models, and evaluation results should stay local.
The historical datasets and generation tools remain on
[`jorge/research`](https://github.com/JorgeCMurillo/babylm-baby-steps/tree/jorge/research).
