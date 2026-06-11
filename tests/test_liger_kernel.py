from __future__ import annotations

import sys
import types
import builtins

import pytest

from babysteps.training.liger import apply_liger_kernel_if_requested


def test_liger_kernel_noop_when_disabled() -> None:
    assert (
        apply_liger_kernel_if_requested(
            use_liger_kernel=False,
            model_arch="gpt2",
        )
        is False
    )


def test_liger_kernel_requires_llama_arch() -> None:
    with pytest.raises(ValueError, match="architecture=llama"):
        apply_liger_kernel_if_requested(
            use_liger_kernel=True,
            model_arch="gpt2",
        )


def test_liger_kernel_applies_safe_llama_patches(monkeypatch) -> None:
    calls = []

    def fake_apply_liger_kernel_to_llama(**kwargs):
        calls.append(kwargs)

    liger_pkg = types.ModuleType("liger_kernel")
    transformers_mod = types.ModuleType("liger_kernel.transformers")
    transformers_mod.apply_liger_kernel_to_llama = fake_apply_liger_kernel_to_llama
    liger_pkg.transformers = transformers_mod

    monkeypatch.setitem(sys.modules, "liger_kernel", liger_pkg)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", transformers_mod)

    assert (
        apply_liger_kernel_if_requested(
            use_liger_kernel=True,
            model_arch="llama",
        )
        is True
    )
    assert calls == [
        {
            "rope": True,
            "swiglu": True,
            "rms_norm": True,
            "cross_entropy": False,
            "fused_linear_cross_entropy": False,
        }
    ]


def test_liger_kernel_reports_missing_dependency(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "liger_kernel", raising=False)
    monkeypatch.delitem(sys.modules, "liger_kernel.transformers", raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("liger_kernel"):
            raise ImportError("no test liger")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="requires a working Liger Kernel install"):
        apply_liger_kernel_if_requested(
            use_liger_kernel=True,
            model_arch="llama",
        )
