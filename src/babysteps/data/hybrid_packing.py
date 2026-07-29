"""Composable training-batch packing policies for hybrid GPT-BERT runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


PACKING_MODES = ("continuous", "sid_wrapped")


@dataclass(frozen=True)
class PackedTrainingBatch:
    """A model-ready batch plus provenance useful for experiment logging."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    source_tokens: int


class TrainingBatchPacker(Protocol):
    """Convert a fixed-token loader batch to one curriculum-length batch."""

    mode: str

    def pack(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        seq_len: int,
    ) -> PackedTrainingBatch: ...


def _validate_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    seq_len: int,
) -> int:
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Training input_ids and attention_mask must be matching 2D tensors.")
    if int(seq_len) <= 0:
        raise ValueError("seq_len must be positive.")
    token_count = int(input_ids.numel())
    if token_count % int(seq_len):
        raise ValueError(
            f"Training batch has {token_count} tokens, not divisible by seq_len={seq_len}."
        )
    return token_count


@dataclass(frozen=True)
class ContinuousBatchPacker:
    """Preserve the token stream and only reshape curriculum-length sequences."""

    mode: str = "continuous"

    def pack(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        seq_len: int,
    ) -> PackedTrainingBatch:
        token_count = _validate_batch(input_ids, attention_mask, seq_len=seq_len)
        batch_size = token_count // int(seq_len)
        return PackedTrainingBatch(
            input_ids=input_ids.reshape(batch_size, int(seq_len)),
            attention_mask=attention_mask.reshape(batch_size, int(seq_len)),
            source_tokens=token_count,
        )


@dataclass(frozen=True)
class SidWrappedBatchPacker:
    """Wrap every packed sequence with CLS/SEP, matching Sid's packing convention.

    Source shards for this policy should use ``document_format: eos_only`` so that
    documents contribute SEP boundaries while packed sequences contribute CLS/SEP.
    The loader still supplies a fixed model-token budget. A small suffix of source
    tokens is therefore unused in each batch to make room for the wrappers.
    """

    cls_token_id: int
    sep_token_id: int
    mode: str = "sid_wrapped"

    def pack(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        seq_len: int,
    ) -> PackedTrainingBatch:
        token_count = _validate_batch(input_ids, attention_mask, seq_len=seq_len)
        if int(seq_len) < 3:
            raise ValueError("sid_wrapped packing requires seq_len >= 3.")
        if not bool(attention_mask.all()):
            raise ValueError("sid_wrapped packing requires an unpadded token-stream batch.")

        batch_size = token_count // int(seq_len)
        content_len = int(seq_len) - 2
        source_tokens = batch_size * content_len
        content = input_ids.reshape(-1)[:source_tokens].reshape(batch_size, content_len)
        packed = torch.empty(
            (batch_size, int(seq_len)),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        packed[:, 0] = int(self.cls_token_id)
        packed[:, 1:-1] = content
        packed[:, -1] = int(self.sep_token_id)
        return PackedTrainingBatch(
            input_ids=packed,
            attention_mask=torch.ones_like(packed, dtype=attention_mask.dtype),
            source_tokens=source_tokens,
        )


def build_training_batch_packer(
    mode: str | None,
    *,
    cls_token_id: int | None = None,
    sep_token_id: int | None = None,
) -> TrainingBatchPacker:
    """Build a packing policy from a declarative config value."""

    normalized = str(mode or "continuous").strip().lower().replace("-", "_")
    aliases = {
        "reshape": "continuous",
        "local": "continuous",
        "sid": "sid_wrapped",
        "wrapped": "sid_wrapped",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "continuous":
        return ContinuousBatchPacker()
    if normalized == "sid_wrapped":
        if cls_token_id is None or sep_token_id is None:
            raise ValueError("sid_wrapped packing requires tokenizer CLS and SEP token ids.")
        return SidWrappedBatchPacker(
            cls_token_id=int(cls_token_id),
            sep_token_id=int(sep_token_id),
        )
    raise ValueError(
        f"Unsupported training packing mode {mode!r}; expected one of {PACKING_MODES}."
    )


__all__ = [
    "PACKING_MODES",
    "ContinuousBatchPacker",
    "PackedTrainingBatch",
    "SidWrappedBatchPacker",
    "TrainingBatchPacker",
    "build_training_batch_packer",
]
