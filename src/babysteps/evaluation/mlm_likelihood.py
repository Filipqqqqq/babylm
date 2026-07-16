"""Masked-language-model span likelihoods matching BabyLM zero-shot scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MlmTextSpanRequest:
    """One text whose suffix beginning at ``target_start_char`` should be scored."""

    text: str
    target_start_char: int


@dataclass(frozen=True)
class EncodedMlmTextSpan:
    """Tokenized text and the tokens belonging to the scored character span."""

    input_ids: tuple[int, ...]
    target_indices: tuple[int, ...]
    target_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class MlmScore:
    """Summed masked-token log probability for one request."""

    sum_logprob: float
    scored_token_count: int
    input_token_count: int


def resolve_mask_token_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "mask_token_id", None)
    if value is None:
        raise ValueError("MLM likelihood evaluation requires tokenizer.mask_token_id.")
    return int(value)


def resolve_padding_token_id(tokenizer: Any) -> int:
    for attr in ("pad_token_id", "cls_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _flatten_single_tokenizer_output(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        return 0
    for start in range(0, len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    raise ValueError("Could not locate non-special tokens inside special-token sequence.")


def _target_indices_without_offsets(
    tokenizer: Any,
    text: str,
    target_start_char: int,
) -> tuple[list[int], list[int]]:
    core_ids = list(tokenizer.encode(text, add_special_tokens=False))
    prefix_ids = list(tokenizer.encode(text[:target_start_char], add_special_tokens=False))
    token_ids = list(tokenizer.encode(text, add_special_tokens=True))
    first_core_index = _find_subsequence(token_ids, core_ids)
    target_indices = [
        first_core_index + index
        for index in range(len(prefix_ids), len(core_ids))
    ]
    return token_ids, target_indices


def encode_mlm_text_span(
    tokenizer: Any,
    request: MlmTextSpanRequest,
) -> EncodedMlmTextSpan:
    """Encode a span with the target convention used by BabyLM's MLM backend."""

    if request.target_start_char < 0 or request.target_start_char > len(request.text):
        raise ValueError(
            "target_start_char must be within the text, "
            f"got {request.target_start_char} for {request.text!r}."
        )

    try:
        output = tokenizer(request.text, return_offsets_mapping=True)
        token_ids = list(_flatten_single_tokenizer_output(output["input_ids"]))
        offsets = list(_flatten_single_tokenizer_output(output["offset_mapping"]))
        target_indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > request.target_start_char and end > start
        ]
    except (NotImplementedError, TypeError, KeyError, ValueError):
        token_ids, target_indices = _target_indices_without_offsets(
            tokenizer,
            request.text,
            request.target_start_char,
        )

    if not target_indices:
        raise ValueError(f"MLM request has no scoreable target tokens: {request.text!r}")

    return EncodedMlmTextSpan(
        input_ids=tuple(token_ids),
        target_indices=tuple(target_indices),
        target_token_ids=tuple(token_ids[index] for index in target_indices),
    )


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _logits_from_outputs(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs[0]


def score_mlm_text_spans(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    requests: list[MlmTextSpanRequest],
    batch_size: int = 64,
    device: torch.device | str | None = None,
    temperature: float = 1.0,
) -> list[MlmScore]:
    """Mask each target token in place and sum its original-token log probability."""

    if not requests:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    was_training = model.training
    model.eval()
    model_device_ = torch.device(device) if device is not None else model_device(model)
    padding_token_id = resolve_padding_token_id(tokenizer)
    mask_token_id = resolve_mask_token_id(tokenizer)
    encoded = [encode_mlm_text_span(tokenizer, request) for request in requests]

    flat_examples: list[tuple[int, tuple[int, ...], int, int]] = []
    for request_index, item in enumerate(encoded):
        for target_index, target_token_id in zip(
            item.target_indices,
            item.target_token_ids,
            strict=True,
        ):
            masked_ids = list(item.input_ids)
            masked_ids[target_index] = mask_token_id
            flat_examples.append(
                (request_index, tuple(masked_ids), target_index, target_token_id)
            )

    sums = [0.0 for _ in encoded]
    counts = [0 for _ in encoded]

    try:
        with torch.inference_mode():
            for batch in _batched(flat_examples, batch_size):
                max_length = max(len(input_ids) for _, input_ids, _, _ in batch)
                input_ids_tensor = torch.full(
                    (len(batch), max_length),
                    padding_token_id,
                    dtype=torch.long,
                    device=model_device_,
                )
                attention_mask = torch.zeros_like(input_ids_tensor)
                target_indices = torch.empty(
                    len(batch), dtype=torch.long, device=model_device_
                )
                target_ids = torch.empty(
                    len(batch), dtype=torch.long, device=model_device_
                )

                for row_index, (_, input_ids, target_index, target_id) in enumerate(batch):
                    row_ids = torch.tensor(
                        input_ids, dtype=torch.long, device=model_device_
                    )
                    input_ids_tensor[row_index, : len(input_ids)] = row_ids
                    attention_mask[row_index, : len(input_ids)] = 1
                    target_indices[row_index] = target_index
                    target_ids[row_index] = target_id

                outputs = model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask,
                )
                logits = _logits_from_outputs(outputs).float()
                row_indices = torch.arange(len(batch), device=model_device_)
                selected_logits = logits[row_indices, target_indices]
                log_probs = torch.log_softmax(selected_logits / temperature, dim=-1)
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)

                for row_index, (request_index, _, _, _) in enumerate(batch):
                    sums[request_index] += float(token_log_probs[row_index].item())
                    counts[request_index] += 1
    finally:
        if was_training:
            model.train()

    return [
        MlmScore(
            sum_logprob=sums[index],
            scored_token_count=counts[index],
            input_token_count=len(item.input_ids),
        )
        for index, item in enumerate(encoded)
    ]


__all__ = [
    "EncodedMlmTextSpan",
    "MlmScore",
    "MlmTextSpanRequest",
    "encode_mlm_text_span",
    "score_mlm_text_spans",
]
