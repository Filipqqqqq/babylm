"""Hugging Face causal-LM config/model helpers.

These builders keep tokenizer/data handling stable while letting training
configs switch between GPT-2 and LLaMA-style decoder architectures.
"""

from __future__ import annotations

import inspect
from typing import Any

from torch import nn
from transformers import AutoModelForCausalLM, GPT2Config, LlamaConfig, PretrainedConfig


SUPPORTED_MODEL_ARCHES = ("gpt2", "llama")


def normalize_model_arch(model_arch: str | None) -> str:
    """Return the canonical architecture name used by BabySteps configs."""

    arch = str(model_arch or "gpt2").strip().lower().replace("_", "-")
    aliases = {
        "gpt2": "gpt2",
        "gpt-2": "gpt2",
        "llama": "llama",
        "llama2": "llama",
        "llama-2": "llama",
        "llama3": "llama",
        "llama-3": "llama",
    }
    if arch not in aliases:
        supported = ", ".join(SUPPORTED_MODEL_ARCHES)
        raise ValueError(f"Unsupported model architecture={model_arch!r}; expected one of: {supported}.")
    return aliases[arch]


def default_llama_intermediate_size(n_embd: int) -> int:
    """Default LLaMA MLP width for BabySteps architecture ablations."""

    return 4 * int(n_embd)


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _dropout_probability(value: Any, *, name: str) -> float:
    probability = float(value)
    if probability < 0.0 or probability >= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0), got {probability}.")
    return probability


def _token_id(tokenizer: Any, primary: str, fallback: str | None = None) -> int | None:
    value = getattr(tokenizer, primary, None)
    if value is None and fallback is not None:
        value = getattr(tokenizer, fallback, None)
    return None if value is None else int(value)


def _llama_rope_kwargs(rope_theta: float) -> dict[str, Any]:
    """Return RoPE kwargs for the installed Transformers LLaMA config surface."""

    llama_params = inspect.signature(LlamaConfig.__init__).parameters
    if "rope_parameters" in llama_params:
        return {
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": float(rope_theta),
            }
        }
    return {"rope_theta": float(rope_theta)}


def resolve_model_vocab_size(model_config: dict[str, Any], tokenizer: Any) -> int:
    """Resolve configured vocab size, defaulting to the tokenizer length."""

    vocab_size = model_config.get("vocab_size")
    if vocab_size is None or int(vocab_size) <= 0:
        vocab_size = len(tokenizer)
    return int(vocab_size)


def build_causal_lm_config(
    *,
    architecture: str,
    vocab_size: int,
    bos_token_id: int | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
    context_length: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
    resid_pdrop: float = 0.1,
    embd_pdrop: float = 0.1,
    attn_pdrop: float = 0.1,
    llama_intermediate_size: int | None = None,
    llama_num_key_value_heads: int | None = None,
    llama_tie_word_embeddings: bool = True,
    rope_theta: float = 10000.0,
    attention_dropout: float = 0.0,
    llama_embd_pdrop: float = 0.0,
    llama_resid_pdrop: float = 0.0,
) -> PretrainedConfig:
    """Build a from-scratch causal-LM config for a supported architecture."""

    arch = normalize_model_arch(architecture)
    vocab_size = int(vocab_size)
    context_length = int(context_length)
    n_embd = int(n_embd)
    n_head = int(n_head)
    n_layer = int(n_layer)

    if arch == "gpt2":
        config = GPT2Config(
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            n_ctx=context_length,
            n_positions=context_length,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            resid_pdrop=float(resid_pdrop),
            embd_pdrop=float(embd_pdrop),
            attn_pdrop=float(attn_pdrop),
        )
    else:
        if n_embd % n_head != 0:
            raise ValueError(
                f"LLaMA hidden size must be divisible by attention heads, got n_embd={n_embd} "
                f"and n_head={n_head}."
            )

        intermediate_size = (
            int(llama_intermediate_size)
            if llama_intermediate_size is not None and int(llama_intermediate_size) > 0
            else default_llama_intermediate_size(n_embd)
        )
        num_key_value_heads = (
            int(llama_num_key_value_heads)
            if llama_num_key_value_heads is not None and int(llama_num_key_value_heads) > 0
            else n_head
        )
        if n_head % num_key_value_heads != 0:
            raise ValueError(
                "LLaMA key/value heads must divide attention heads, "
                f"got n_head={n_head} and llama_num_key_value_heads={num_key_value_heads}."
            )

        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=n_embd,
            intermediate_size=intermediate_size,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=context_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            attention_dropout=_dropout_probability(
                attention_dropout,
                name="model.attention_dropout",
            ),
            tie_word_embeddings=bool(llama_tie_word_embeddings),
            **_llama_rope_kwargs(float(rope_theta)),
        )
        config.babysteps_llama_embd_pdrop = _dropout_probability(
            llama_embd_pdrop,
            name="model.llama_embd_pdrop",
        )
        config.babysteps_llama_resid_pdrop = _dropout_probability(
            llama_resid_pdrop,
            name="model.llama_resid_pdrop",
        )

    config.use_cache = False
    return config


def build_causal_lm_config_from_dict(
    config: dict[str, Any],
    tokenizer: Any,
) -> PretrainedConfig:
    """Build a causal-LM config from a BabySteps YAML-style config dict."""

    model_config = config["model"]
    architecture = normalize_model_arch(
        _first_present(model_config, ("architecture", "model_arch"), "gpt2")
    )
    context_length = int(_first_present(model_config, ("context_length", "seq_len"), 1024))
    n_embd = int(_first_present(model_config, ("n_embd", "hidden_size"), 768))
    n_head = int(_first_present(model_config, ("n_head", "num_attention_heads"), 12))
    n_layer = int(_first_present(model_config, ("n_layer", "num_hidden_layers"), 12))
    llama_dropout = _first_present(model_config, ("llama_dropout",), 0.0)

    return build_causal_lm_config(
        architecture=architecture,
        vocab_size=resolve_model_vocab_size(model_config, tokenizer),
        bos_token_id=_token_id(tokenizer, "bos_token_id", "eos_token_id"),
        eos_token_id=_token_id(tokenizer, "eos_token_id"),
        pad_token_id=_token_id(tokenizer, "pad_token_id"),
        context_length=context_length,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        resid_pdrop=float(model_config.get("resid_pdrop", 0.1)),
        embd_pdrop=float(model_config.get("embd_pdrop", 0.1)),
        attn_pdrop=float(model_config.get("attn_pdrop", 0.1)),
        llama_intermediate_size=_first_present(
            model_config,
            ("llama_intermediate_size", "intermediate_size"),
        ),
        llama_num_key_value_heads=_first_present(
            model_config,
            ("llama_num_key_value_heads", "num_key_value_heads"),
        ),
        llama_tie_word_embeddings=bool(
            _first_present(
                model_config,
                ("llama_tie_word_embeddings", "tie_word_embeddings"),
                True,
            )
        ),
        rope_theta=float(model_config.get("rope_theta", 10000.0)),
        attention_dropout=_first_present(
            model_config,
            ("llama_attention_dropout", "attention_dropout"),
            llama_dropout,
        ),
        llama_embd_pdrop=_first_present(
            model_config,
            ("llama_embd_pdrop", "llama_embedding_dropout"),
            llama_dropout,
        ),
        llama_resid_pdrop=_first_present(
            model_config,
            ("llama_resid_pdrop", "llama_hidden_dropout", "hidden_dropout"),
            llama_dropout,
        ),
    )


def apply_llama_training_dropout(model, hf_config: PretrainedConfig):
    """Attach optional BabySteps dropout hooks to a LLaMA model.

    Hugging Face LLaMA exposes attention dropout, but not GPT-2-style embedding
    or decoder-output dropout. These hooks add train-mode-only dropout at those
    two points without changing checkpoint evaluation behavior.
    """

    if getattr(hf_config, "model_type", None) != "llama":
        return model

    embed_dropout = float(getattr(hf_config, "babysteps_llama_embd_pdrop", 0.0))
    resid_dropout = float(getattr(hf_config, "babysteps_llama_resid_pdrop", 0.0))

    for handle in getattr(model, "_babysteps_dropout_hook_handles", ()):
        handle.remove()

    handles = []
    model_core = getattr(model, "model", None)
    if model_core is None:
        raise ValueError("Expected a LLaMA causal LM with a .model backbone.")

    if embed_dropout > 0.0:
        model_core.babysteps_embed_dropout = nn.Dropout(embed_dropout)

        def embed_hook(_module, _inputs, output):
            return model_core.babysteps_embed_dropout(output)

        handles.append(model_core.embed_tokens.register_forward_hook(embed_hook))

    if resid_dropout > 0.0:
        for layer in model_core.layers:
            layer.babysteps_resid_dropout = nn.Dropout(resid_dropout)

            def layer_hook(module, _inputs, output):
                dropout = module.babysteps_resid_dropout
                if isinstance(output, tuple):
                    return (dropout(output[0]),) + output[1:]
                return dropout(output)

            handles.append(layer.register_forward_hook(layer_hook))

    model._babysteps_dropout_hook_handles = handles
    return model


def build_causal_lm_model(config: dict[str, Any], tokenizer: Any):
    """Instantiate a Hugging Face causal LM from a BabySteps config dict."""

    hf_config = build_causal_lm_config_from_dict(config, tokenizer)
    model_kwargs: dict[str, Any] = {}
    attn_implementation = config["model"].get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = str(attn_implementation)
    model = AutoModelForCausalLM.from_config(hf_config, **model_kwargs)
    return apply_llama_training_dropout(model, hf_config)
