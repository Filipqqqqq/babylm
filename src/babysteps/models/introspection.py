"""Small model-shape helpers shared by training entrypoints."""

from __future__ import annotations

import torch


def model_hidden_size(model: torch.nn.Module) -> int:
    """Infer the hidden size used by the model's final hidden states."""

    config = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    embeddings = model.get_input_embeddings()
    return int(embeddings.weight.shape[-1])


def input_embedding_size(model: torch.nn.Module) -> int:
    """Infer the width of the model's input token embeddings."""

    embeddings = model.get_input_embeddings()
    embedding_dim = getattr(embeddings, "embedding_dim", None)
    if embedding_dim is not None:
        return int(embedding_dim)
    return int(embeddings.weight.shape[-1])


__all__ = ["input_embedding_size", "model_hidden_size"]
