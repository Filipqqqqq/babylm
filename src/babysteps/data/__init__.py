"""Data loading and preprocessing helpers."""

from importlib import import_module

__all__ = [
    "TokenChunkDataset",
    "TokenBlockMixResult",
    "TokenShardDataset",
    "TokenShardDatasetConfig",
    "SyntheticSource",
    "build_token_chunk_loader",
    "build_token_shard_loader",
    "collate_token_chunks",
    "load_token_chunks",
    "mix_token_blocks",
    "read_synthetic_documents",
    "requested_tokens_to_blocks",
]


def __getattr__(name: str):
    if name in {
        "TokenChunkDataset",
        "build_token_chunk_loader",
        "collate_token_chunks",
        "load_token_chunks",
    }:
        token_chunks = import_module("babysteps.data.token_chunks")
        return getattr(token_chunks, name)
    if name in {
        "SyntheticSource",
        "TokenBlockMixResult",
        "mix_token_blocks",
        "read_synthetic_documents",
        "requested_tokens_to_blocks",
    }:
        token_block_mixer = import_module("babysteps.data.token_block_mixer")
        return getattr(token_block_mixer, name)
    if name in {
        "TokenShardDataset",
        "TokenShardDatasetConfig",
        "build_token_shard_loader",
    }:
        token_shards = import_module("babysteps.data.token_shards")
        return getattr(token_shards, name)
    raise AttributeError(name)
