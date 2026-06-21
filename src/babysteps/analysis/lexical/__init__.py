"""Lexical corpus analysis for target coverage and tokenizer fertility."""

from __future__ import annotations

from babysteps.analysis.lexical.targets import LexicalTarget, load_targets

__all__ = [
    "LexicalAnalysisResult",
    "LexicalTarget",
    "load_targets",
    "run_lexical_analysis",
]


def __getattr__(name: str):
    if name in {"LexicalAnalysisResult", "run_lexical_analysis"}:
        from babysteps.analysis.lexical.pipeline import (
            LexicalAnalysisResult,
            run_lexical_analysis,
        )

        values = {
            "LexicalAnalysisResult": LexicalAnalysisResult,
            "run_lexical_analysis": run_lexical_analysis,
        }
        return values[name]
    raise AttributeError(name)
