from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from babysteps.training.branch_scoring import Branch, encode_branch, score_causal_branches


class WhitespaceTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "a": 3,
            "b": 4,
            "c": 5,
            "d": 6,
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self.vocab[token] for token in text.split()]


class ContextHostileModel(nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -5.0,
            device=input_ids.device,
        )
        for row in range(input_ids.size(0)):
            for pos in range(input_ids.size(1) - 1):
                next_id = int(input_ids[row, pos + 1].item())
                logits[row, pos, next_id] = -20.0 if pos < 2 else 20.0
        return SimpleNamespace(logits=logits + self.dummy * 0.0)


class TinyDifferentiableModel(nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 4)
        self.lm_head = nn.Linear(4, vocab_size)

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


def test_encode_branch_slices_target_tokens_after_context() -> None:
    tokenizer = WhitespaceTokenizer()

    encoded = encode_branch(
        tokenizer,
        Branch(context="a b", target="c d"),
        bos_like_token_id=tokenizer.bos_token_id,
    )

    assert encoded.input_ids == (1, 3, 4, 5, 6)
    assert encoded.target_start == 2
    assert encoded.target_end == 4


def test_score_causal_branches_masks_context_tokens() -> None:
    tokenizer = WhitespaceTokenizer()
    model = ContextHostileModel()

    output = score_causal_branches(
        model=model,
        tokenizer=tokenizer,
        branches=[Branch(context="a b", target="c d")],
        score_reduction="mean",
    )

    assert output.target_token_counts.tolist() == [2.0]
    assert output.scores.item() > -1e-3


def test_score_causal_branches_is_differentiable() -> None:
    tokenizer = WhitespaceTokenizer()
    model = TinyDifferentiableModel()

    output = score_causal_branches(
        model=model,
        tokenizer=tokenizer,
        branches=[Branch(context="a b", target="c d")],
        score_reduction="mean",
    )
    loss = -output.scores.mean()
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())
