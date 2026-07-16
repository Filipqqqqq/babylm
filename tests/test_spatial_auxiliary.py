from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from babysteps.spatial_synth.generate import generate_context_pairs, write_items
from babysteps.training.spatial_auxiliary import (
    ContextRankExample,
    CounterfactualRankExample,
    SpatialAuxiliarySampler,
    build_spatial_branches,
    compute_spatial_auxiliary_loss,
    evaluate_spatial_auxiliary,
    read_spatial_auxiliary_examples,
    split_spatial_auxiliary_examples,
)


class TinySpatialTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "c1": 3,
            "c2": 4,
            "t1": 5,
            "t2": 6,
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self.vocab[token] for token in text.split()]


class MatchedBranchModel(nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -4.0,
            device=input_ids.device,
        )
        for row in range(input_ids.size(0)):
            row_ids = set(int(token_id) for token_id in input_ids[row].tolist())
            matched = (3 in row_ids and 5 in row_ids) or (4 in row_ids and 6 in row_ids)
            boost = 12.0 if matched else -12.0
            for pos in range(input_ids.size(1) - 1):
                next_id = int(input_ids[row, pos + 1].item())
                logits[row, pos, next_id] = boost
        return SimpleNamespace(logits=logits + self.dummy * 0.0)


def test_read_and_split_spatial_auxiliary_context_rank_rows(tmp_path: Path) -> None:
    pairs = generate_context_pairs(6, seed=5)
    path = tmp_path / "spatial_context_rank.jsonl"
    write_items(pairs, path)

    examples = read_spatial_auxiliary_examples(path, objective="context_rank")
    train_examples, eval_examples = split_spatial_auxiliary_examples(
        examples,
        eval_items=2,
        seed=11,
    )

    assert len(examples) == 6
    assert len(train_examples) == 4
    assert len(eval_examples) == 2
    assert all(isinstance(example, ContextRankExample) for example in examples)


def test_spatial_auxiliary_sampler_wraps_and_reshuffles() -> None:
    examples = [
        ContextRankExample("1", "c1", "c2", "t1", "t2"),
        ContextRankExample("2", "c1", "c2", "t1", "t2"),
    ]
    sampler = SpatialAuxiliarySampler(examples, batch_size=3, seed=0)

    batch = sampler.next_batch()

    assert len(batch) == 3
    assert {example.item_id for example in batch} == {"1", "2"}


def test_context_rank_branches_follow_ewok_score_order() -> None:
    example = ContextRankExample("1", "c1", "c2", "t1", "t2")

    branches = build_spatial_branches([example])

    assert [(branch.context, branch.target) for branch in branches] == [
        ("c1", "t1"),
        ("c2", "t1"),
        ("c2", "t2"),
        ("c1", "t2"),
    ]


def test_context_rank_loss_and_eval_prefer_matched_contexts() -> None:
    tokenizer = TinySpatialTokenizer()
    model = MatchedBranchModel()
    examples = [ContextRankExample("1", "c1", "c2", "t1", "t2")]

    loss = compute_spatial_auxiliary_loss(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
    )
    metrics = evaluate_spatial_auxiliary(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
    )

    assert loss.item() < 1e-3
    assert metrics["objective"] == "context_rank"
    assert metrics["score"] == 1.0
    assert metrics["k1_score"] == 1.0
    assert metrics["k2_score"] == 1.0
    assert metrics["mean_margin"] > 0.0


def test_counterfactual_rank_loss_and_eval_prefer_full_diagonal() -> None:
    tokenizer = TinySpatialTokenizer()
    model = MatchedBranchModel()
    examples = [CounterfactualRankExample("1", "c1", "c2", "t1", "t2")]

    loss = compute_spatial_auxiliary_loss(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
    )
    metrics = evaluate_spatial_auxiliary(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
    )

    assert loss.item() < 1e-3
    assert metrics["objective"] == "counterfactual_rank"
    assert metrics["score"] == 1.0
    assert metrics["comparisons"] == 4


def test_read_spatial_auxiliary_rejects_objective_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "completion.jsonl"
    path.write_text(
        json.dumps(
            {
                "rank_objective": "completion_rank",
                "context": "c1",
                "completion": "t1",
                "hard_negative": "t2",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        read_spatial_auxiliary_examples(path, objective="context_rank")
    except ValueError as exc:
        assert "Expected spatial auxiliary objective" in str(exc)
    else:
        raise AssertionError("Expected objective mismatch to fail.")
