# Tokenization Helpers

This folder is for reusable tokenizer code. Runnable commands should live in
`scripts/`, and tokenizer/chunking recipes should live in
`configs/tokenization/`.

Current useful check:

```bash
BABYSTEPS_FULL_GPT2_CHUNK_PARITY=1 pytest -q tests/test_gpt2_chunking_parity.py
```

That rebuilds plain GPT-2 BabyLM 10M chunks from the old cleaned CustomLlama
text and verifies they exactly match `gpt_chunked.pkl`. It takes about a minute
locally, so it is gated behind the environment variable.

Train a custom byte-BPE tokenizer:

```bash
python scripts/train_tokenizer.py --config configs/tokenization/byte_bpe_8192_10m_block512.yaml
python scripts/train_tokenizer.py --config configs/tokenization/byte_bpe_16384_10m_block512.yaml
```

Build token chunks from a tokenizer:

```bash
python scripts/build_token_chunks.py --config configs/tokenization/gpt2_pretrained_10m_block512.yaml
python scripts/build_token_chunks.py --config configs/tokenization/byte_bpe_8192_10m_block512.yaml
```

Current flow:

```text
clean text -> train/load tokenizer -> build token chunks -> inspect chunks -> train
```
