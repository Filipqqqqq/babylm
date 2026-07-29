from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from babysteps.data.hybrid_packing import build_training_batch_packer
from babysteps.models import build_causal_lm_model
from babysteps.training.hybrid_lm_loop import (
    build_special_mask,
    compute_gpt_bert_loss,
    make_causal_batch,
    make_masked_batch,
    set_gpt_bert_is_causal,
)
from babysteps.training.optimizers import build_optimizer
from babysteps.training.objective_batching import plan_objective_microbatches


class _TinyLlamaTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self) -> int:
        return 64


class _TinyGptBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=16)
        self.model = SimpleNamespace(is_causal=True)
        self.embedding = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 16)

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        return_dict=True,
        output_hidden_states=False,
    ):
        del attention_mask, return_dict, output_hidden_states
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


def test_chatty_bert_style_llama_completes_one_cpu_update() -> None:
    config = {
        "model": {
            "architecture": "llama",
            "vocab_size": 64,
            "context_length": 16,
            "n_layer": 1,
            "n_head": 4,
            "n_embd": 32,
            "llama_intermediate_size": 64,
            "llama_num_key_value_heads": 2,
            "llama_tie_word_embeddings": True,
        }
    }
    optimizer_config = {
        "optimizer": "muon_pe",
        "learning_rate": 0.0001,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "muon_lr": 0.01,
        "muon_weight_decay": 0.1,
        "muon_momentum": 0.95,
        "muon_ns_steps": 2,
        "muon_nesterov": True,
        "muon_split_qkv": True,
        "muon_batch_updates": True,
    }

    torch.manual_seed(42)
    model = build_causal_lm_model(config, _TinyLlamaTokenizer())
    optimizer, info = build_optimizer(model, optimizer_config)
    input_ids = torch.tensor(
        [[1, 4, 5, 6, 7, 2], [1, 8, 9, 10, 11, 2]],
        dtype=torch.long,
    )
    before = model.get_input_embeddings().weight.detach().clone()

    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
    )
    output.loss.backward()
    optimizer.step()

    assert info.name == "muon_pe"
    assert torch.isfinite(output.loss)
    assert not torch.equal(before, model.get_input_embeddings().weight.detach())


def test_murphyslaw_style_gpt_bert_completes_causal_and_masked_updates() -> None:
    torch.manual_seed(42)
    model = _TinyGptBert()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    input_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])
    attention_mask = torch.ones_like(input_ids)

    set_gpt_bert_is_causal(model, True)
    causal_input_ids, causal_labels = make_causal_batch(input_ids)
    causal_loss, _, causal_tokens = compute_gpt_bert_loss(
        model,
        input_ids=causal_input_ids,
        labels=causal_labels,
        attention_mask=attention_mask,
        z_loss=0.0001,
    )
    causal_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    set_gpt_bert_is_causal(model, False)
    special_mask = build_special_mask(input_ids, {0, 1, 2, 9})
    masked_input_ids, masked_labels = make_masked_batch(
        input_ids,
        special_mask,
        mask_ratio=1.0,
        vocab_size=16,
        mask_id=9,
        p_random=0.0,
        p_keep=0.0,
        gen=torch.Generator().manual_seed(42),
    )
    masked_loss, _, masked_tokens = compute_gpt_bert_loss(
        model,
        input_ids=masked_input_ids,
        labels=masked_labels,
        attention_mask=attention_mask,
        z_loss=0.0001,
    )
    masked_loss.backward()
    optimizer.step()

    assert causal_tokens.item() == 6
    assert masked_tokens.item() == 4
    assert torch.isfinite(causal_loss)
    assert torch.isfinite(masked_loss)
    assert model.model.is_causal is False


def test_training_batch_packers_report_consumed_source_tokens() -> None:
    input_ids = torch.arange(16).reshape(2, 8)
    attention_mask = torch.ones_like(input_ids)

    continuous = build_training_batch_packer("continuous").pack(
        input_ids,
        attention_mask,
        seq_len=4,
    )
    wrapped = build_training_batch_packer(
        "sid_wrapped",
        cls_token_id=101,
        sep_token_id=102,
    ).pack(input_ids, attention_mask, seq_len=8)

    assert continuous.input_ids.shape == (4, 4)
    assert continuous.source_tokens == 16
    assert wrapped.input_ids.tolist() == [
        [101, 0, 1, 2, 3, 4, 5, 102],
        [101, 6, 7, 8, 9, 10, 11, 102],
    ]
    assert wrapped.source_tokens == 12


def test_objectives_can_cycle_per_microbatch() -> None:
    plan = plan_objective_microbatches(
        batch_size=8,
        cursor=1,
        causal_steps=1,
        masked_steps=1,
        granularity="microbatch",
        micro_batch_size=2,
    )

    assert [(item.start, item.stop, item.objective) for item in plan] == [
        (0, 2, "masked"),
        (2, 4, "causal"),
        (4, 6, "masked"),
        (6, 8, "causal"),
    ]
