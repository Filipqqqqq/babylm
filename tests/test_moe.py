from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM

from babysteps.models import build_causal_lm_model
from babysteps.training.moe import compute_moe_router_losses, moe_settings_from_model
from babysteps.training.optimizers import build_optimizer


class _TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self) -> int:
        return 64


def _tiny_moe_config(**overrides) -> dict:
    model = {
        "architecture": "moe",
        "vocab_size": 64,
        "context_length": 16,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 32,
        "llama_intermediate_size": 96,
        "llama_num_key_value_heads": 4,
        "llama_tie_word_embeddings": True,
        "llama_dropout": 0.1,
        "moe_num_experts": 4,
        "moe_top_k": 2,
    }
    model.update(overrides)
    return {"model": model}


def _muon_config() -> dict:
    return {
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


def test_moe_config_matches_llama_attention_and_active_budget() -> None:
    model = build_causal_lm_model(_tiny_moe_config(), _TinyTokenizer())
    config = model.config

    assert config.model_type == "qwen2_moe"
    assert config.num_experts == 4
    assert config.num_experts_per_tok == 2
    # Default sizing: shared + top_k routed experts ~= one dense FFN (96 // 3).
    assert config.moe_intermediate_size == 32
    assert config.shared_expert_intermediate_size == 32
    assert config.output_router_logits is True
    assert config.babysteps_router_z_loss_coef == 0.001

    layer = model.model.layers[0]
    assert layer.self_attn.q_proj.bias is None
    assert len(layer.mlp.experts) == 4
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_router_losses_reward_balance_and_reach_the_router() -> None:
    settings = moe_settings_from_model(build_causal_lm_model(_tiny_moe_config(), _TinyTokenizer()))
    assert settings is not None

    balanced = torch.zeros(400, 4, requires_grad=True)
    collapsed = torch.tensor([[8.0, 7.0, -8.0, -8.0]]).repeat(400, 1)
    balanced_losses = compute_moe_router_losses(
        SimpleNamespace(router_logits=(balanced,)), settings
    )
    collapsed_losses = compute_moe_router_losses(
        SimpleNamespace(router_logits=(collapsed,)), settings
    )

    assert collapsed_losses.aux_loss > balanced_losses.aux_loss
    assert collapsed_losses.max_load_ratio.item() == 2.0
    assert collapsed_losses.z_loss > balanced_losses.z_loss
    balanced_losses.weighted_loss.backward()
    assert balanced.grad is not None


def test_dense_models_have_no_moe_settings() -> None:
    config = _tiny_moe_config()
    config["model"]["architecture"] = "llama"
    assert moe_settings_from_model(build_causal_lm_model(config, _TinyTokenizer())) is None


def test_moe_routers_use_adamw_and_one_update_trains_everything() -> None:
    torch.manual_seed(42)
    model = build_causal_lm_model(_tiny_moe_config(), _TinyTokenizer())
    optimizer, info = build_optimizer(model, _muon_config())

    router_ids = {
        id(param)
        for name, param in model.named_parameters()
        if name.endswith(("mlp.gate.weight", "mlp.shared_expert_gate.weight"))
    }
    assert len(router_ids) == 4
    for group in optimizer.param_groups:
        if group.get("use_muon"):
            assert not router_ids & {id(param) for param in group["params"]}

    settings = moe_settings_from_model(model)
    input_ids = torch.tensor([[1, 4, 5, 6, 7, 2], [1, 8, 9, 10, 11, 2]])
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    lm_loss = torch.nn.functional.cross_entropy(
        outputs.logits[:, :-1].reshape(-1, 64), input_ids[:, 1:].reshape(-1)
    )
    router = compute_moe_router_losses(outputs, settings)
    (lm_loss + router.weighted_loss).backward()

    gate = model.model.layers[0].mlp.gate.weight
    assert gate.grad is not None and gate.grad.abs().sum() > 0
    before = gate.detach().clone()
    optimizer.step()
    assert not torch.equal(before, gate.detach())


def test_moe_checkpoint_reloads_with_stock_transformers(tmp_path) -> None:
    torch.manual_seed(0)
    model = build_causal_lm_model(_tiny_moe_config(), _TinyTokenizer()).eval()
    model.save_pretrained(tmp_path)

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    input_ids = torch.tensor([[1, 4, 5, 6, 7, 2]])
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits
        actual = reloaded(input_ids=input_ids).logits
    assert torch.allclose(expected, actual, atol=1e-5)
