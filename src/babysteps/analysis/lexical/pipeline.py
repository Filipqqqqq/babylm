"""End-to-end lexical analysis pipeline."""

from __future__ import annotations

import os
import itertools
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from babysteps.analysis.lexical.context import context_neighbor_stopwords_from_config
from babysteps.analysis.lexical.reports import (
    CONTEXT_SUMMARY_FIELDS,
    GLOBAL_WORD_FIELDS,
    TARGET_COVERAGE_FIELDS,
    THRESHOLD_FIELDS,
    build_context_summary_rows,
    build_global_summary,
    build_global_word_rows,
    build_target_rows,
    build_threshold_rows,
    write_csv,
    write_json,
    write_jsonl,
)
from babysteps.analysis.lexical.sampling import ReservoirSampler, stable_seed
from babysteps.analysis.lexical.stats import TargetStats
from babysteps.analysis.lexical.targets import LexicalTarget, TargetMatcher, load_targets
from babysteps.analysis.lexical.text import SentenceRecord, TextProcessor, make_text_processor
from babysteps.analysis.lexical.tokenization import TokenizerAnalyzer
from babysteps.data.sources import dataset_kind, iter_dataset_rows
from babysteps.paths import (
    expand_env_vars,
    looks_like_local_path,
    resolve_path,
    validate_tokenizer_dir,
)


DEFAULT_RUNS_DIR = Path(
    os.environ.get("BABYSTEPS_RUNS_DIR", "~/babylm-baby-steps-runs")
).expanduser()


@dataclass(frozen=True)
class LexicalAnalysisResult:
    """Paths and summary metadata for a completed lexical analysis run."""

    output_dir: Path
    files: dict[str, Path]
    manifest: dict[str, Any]


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    """Load an analysis YAML config with environment expansion."""

    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _config_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}.")


def _log_progress(message: str) -> None:
    print(f"[lexical] {message}", file=sys.stderr, flush=True)


def _format_rate(count: int, elapsed_seconds: float, *, unit: str) -> str:
    if elapsed_seconds <= 0:
        return f"0 {unit}/sec"
    return f"{count / elapsed_seconds:,.0f} {unit}/sec"


def _nested_get(row: dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _source_for_row(row: dict[str, Any], source_field: str) -> str:
    if not source_field:
        return ""
    value = _nested_get(row, source_field)
    return "" if value is None else str(value)


def _resolve_optional_path(
    value: str | Path | None,
    *,
    root_dir: Path,
) -> Path | None:
    if value is None:
        return None
    return resolve_path(value, base_dir=root_dir)


def _resolve_output_dir(
    config: dict[str, Any],
    *,
    root_dir: Path,
    output_dir: str | Path | None,
    created_utc: datetime,
) -> Path:
    if output_dir is not None:
        return resolve_path(output_dir, base_dir=Path.cwd())

    raw_output_config = config.get("output", {})
    output_config = raw_output_config if isinstance(raw_output_config, dict) else {}
    configured = output_config.get("dir")
    if configured is None:
        configured = (
            DEFAULT_RUNS_DIR
            / "analysis"
            / "lexical"
            / str(config.get("name", "run"))
        )
    base_dir = resolve_path(configured, base_dir=root_dir)
    if not bool(output_config.get("timestamp", True)):
        return base_dir
    return base_dir / created_utc.strftime("%Y%m%d_%H%M%S")


def _load_tokenizer(config: dict[str, Any], *, root_dir: Path):
    tokenizer_config = config.get("tokenizer", {})
    tokenizer_path = tokenizer_config.get("path", tokenizer_config.get("output_dir"))
    if tokenizer_path is None:
        raise KeyError(
            "Lexical analysis config requires tokenizer.path or tokenizer.output_dir."
        )
    if looks_like_local_path(tokenizer_path):
        resolved_path = resolve_path(tokenizer_path, base_dir=root_dir)
        validate_tokenizer_dir(resolved_path)
        tokenizer_path = resolved_path

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


def _target_examples(
    targets: list[LexicalTarget],
    samplers: dict[str, ReservoirSampler],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        rows.extend(samplers[target.id].samples)
    return rows


def _sentence_text(sentences: list[SentenceRecord], index: int) -> str:
    if index < 0 or index >= len(sentences):
        return ""
    return sentences[index].text


def _context_window(
    sentence: SentenceRecord,
    token_index: int,
    *,
    neighbor_window: int,
) -> tuple[list[str], list[str]]:
    return _context_window_for_span(
        sentence,
        start_index=token_index,
        end_index=token_index + 1,
        neighbor_window=neighbor_window,
    )


def _context_window_for_span(
    sentence: SentenceRecord,
    *,
    start_index: int,
    end_index: int,
    neighbor_window: int,
) -> tuple[list[str], list[str]]:
    tokens = list(sentence.tokens)
    left = [
        token.normalized
        for token in tokens[max(0, start_index - neighbor_window) : start_index]
    ]
    right = [
        token.normalized
        for token in tokens[end_index : end_index + neighbor_window]
    ]
    return left, right


def _example_word_window(
    sentence: SentenceRecord,
    token_index: int,
    *,
    radius: int,
) -> tuple[str, str]:
    return _example_word_window_for_span(
        sentence,
        start_index=token_index,
        end_index=token_index + 1,
        radius=radius,
    )


def _example_word_window_for_span(
    sentence: SentenceRecord,
    *,
    start_index: int,
    end_index: int,
    radius: int,
) -> tuple[str, str]:
    if radius <= 0:
        return "", ""

    tokens = list(sentence.tokens)
    left_tokens = tokens[max(0, start_index - radius) : start_index]
    right_tokens = tokens[end_index : end_index + radius]
    return (
        " ".join(token.surface for token in left_tokens),
        " ".join(token.surface for token in right_tokens),
    )


def _span_surface(sentence: SentenceRecord, start_index: int, end_index: int) -> str:
    return " ".join(token.surface for token in sentence.tokens[start_index:end_index])


def _iter_document_items(
    dataset_config: dict[str, Any],
    *,
    text_field: str,
    source_field: str,
    max_docs: int | None,
):
    docs_seen = 0
    for doc_index, row in enumerate(iter_dataset_rows(dataset_config)):
        if max_docs is not None and docs_seen >= max_docs:
            break
        if text_field not in row:
            raise KeyError(
                f"Dataset row is missing text field {text_field!r}: {list(row.keys())}"
            )
        text = row[text_field]
        if text is None:
            continue
        yield {
            "doc_index": doc_index,
            "text": str(text),
            "source": _source_for_row(row, source_field),
        }
        docs_seen += 1


def run_lexical_analysis(
    config: dict[str, Any],
    *,
    root_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_docs: int | None = None,
    tokenizer: Any | None = None,
    text_processor: TextProcessor | None = None,
) -> LexicalAnalysisResult:
    """Run lexical analysis and write report artifacts."""

    root = Path(root_dir) if root_dir is not None else Path.cwd()
    created_utc = datetime.now(timezone.utc)
    analysis_config = config.get("analysis", {})
    dataset_config = config["dataset"]
    text_field = str(dataset_config.get("text_field", "text"))
    source_field = str(analysis_config.get("source_field", "source"))
    neighbor_window = int(analysis_config.get("neighbor_window", 5))
    context_word_radius = int(analysis_config.get("context_word_radius", 20))
    examples_per_target = int(analysis_config.get("examples_per_target", 10))
    seed = int(analysis_config.get("seed", 13))
    tokenizer_mode = str(analysis_config.get("tokenizer_mode", "leading_space"))
    context_neighbor_stopwords = context_neighbor_stopwords_from_config(analysis_config)
    min_frequency_for_fragmented = int(
        analysis_config.get("min_frequency_for_fragmented", 1)
    )
    progress_logging = _config_bool(
        analysis_config.get("progress_logging"),
        default=True,
    )
    progress_interval_docs = max(
        0,
        int(analysis_config.get("progress_interval_docs", 10000)),
    )
    progress_interval_seconds = max(
        0.0,
        float(analysis_config.get("progress_interval_seconds", 30)),
    )
    resolved_max_docs = max_docs
    if resolved_max_docs is None and analysis_config.get("max_docs") is not None:
        resolved_max_docs = int(analysis_config["max_docs"])

    target_config = config.get("targets", {})
    target_path = (
        _resolve_optional_path(target_config.get("path"), root_dir=root)
        if isinstance(target_config, dict)
        else None
    )
    targets = load_targets(target_path) if target_path is not None else []
    matcher = TargetMatcher(targets)
    stats_by_target = {target.id: TargetStats(target) for target in targets}
    samplers = {
        target.id: ReservoirSampler(
            limit=examples_per_target,
            seed=stable_seed(seed, target.id),
        )
        for target in targets
    }

    processor = text_processor or make_text_processor(config.get("lemmatizer", {}))
    active_tokenizer = tokenizer or _load_tokenizer(config, root_dir=root)
    tokenizer_analyzer = TokenizerAnalyzer(active_tokenizer, mode=tokenizer_mode)

    word_counts: Counter[str] = Counter()
    docs_processed = 0
    word_tokens = 0
    target_mentions = 0
    started_at = time.monotonic()
    next_progress_at = (
        started_at + progress_interval_seconds
        if progress_interval_seconds > 0
        else float("inf")
    )

    if progress_logging:
        max_docs_label = f"{resolved_max_docs:,}" if resolved_max_docs else "all"
        _log_progress(
            "Starting lexical analysis: "
            f"dataset_kind={dataset_kind(dataset_config)} "
            f"targets={len(targets):,} max_docs={max_docs_label}"
        )

    document_items = _iter_document_items(
        dataset_config,
        text_field=text_field,
        source_field=source_field,
        max_docs=resolved_max_docs,
    )
    metadata_items, text_items = itertools.tee(document_items)
    processed_texts = processor.pipe(item["text"] for item in text_items)

    for item, sentences in zip(metadata_items, processed_texts):
        doc_index = int(item["doc_index"])
        source = str(item["source"])
        for sentence_index, sentence in enumerate(sentences):
            sentence_examples: dict[str, dict[str, Any]] = {}
            for token in sentence.tokens:
                word_counts[token.normalized] += 1
                word_tokens += 1
                tokenizer_analyzer.analyze(token.normalized)

            for match in matcher.match_sentence(sentence.tokens):
                target_mentions += 1
                span_tokens = sentence.tokens[match.start_index : match.end_index]
                subword_count = 0
                token_pieces: list[str] = []
                for span_token in span_tokens:
                    tokenization = tokenizer_analyzer.analyze(span_token.normalized)
                    subword_count += tokenization.fertility
                    token_pieces.extend(tokenization.pieces)

                left_neighbors, right_neighbors = _context_window_for_span(
                    sentence,
                    start_index=match.start_index,
                    end_index=match.end_index,
                    neighbor_window=neighbor_window,
                )
                left_window_words, right_window_words = _example_word_window_for_span(
                    sentence,
                    start_index=match.start_index,
                    end_index=match.end_index,
                    radius=context_word_radius,
                )
                tokenization_text = " ".join(token_pieces)
                target_id = match.target_id
                stats = stats_by_target[target_id]
                stats.observe(
                    doc_index=doc_index,
                    source=source,
                    surface=match.surface,
                    lemma=match.lemma,
                    match_kinds=set(match.match_kinds),
                    tokenization=tokenization_text,
                    subword_count=subword_count,
                    left_neighbors=left_neighbors,
                    right_neighbors=right_neighbors,
                )
                if target_id not in sentence_examples:
                    matched_phrase = _span_surface(
                        sentence,
                        match.start_index,
                        match.end_index,
                    )
                    sentence_examples[target_id] = {
                        "target_id": target_id,
                        "label": stats.target.label,
                        "surface": match.surface,
                        "lemma": match.lemma,
                        "source": source,
                        "doc_index": doc_index,
                        "sentence_index": sentence_index,
                        "previous_sentence": _sentence_text(
                            sentences,
                            sentence_index - 1,
                        ),
                        "sentence": sentence.text,
                        "next_sentence": _sentence_text(
                            sentences,
                            sentence_index + 1,
                        ),
                        "context_word_radius": context_word_radius,
                        "left_window_words": left_window_words,
                        "matched_word": matched_phrase,
                        "matched_phrase": matched_phrase,
                        "right_window_words": right_window_words,
                        "tokenizer_mode": tokenizer_mode,
                        "token_pieces": token_pieces,
                        "subword_count": subword_count,
                        "example_sampling_unit": "sentence",
                        "sentence_target_occurrences": 0,
                        "sentence_target_surface_counts": {},
                    }
                example = sentence_examples[target_id]
                example["sentence_target_occurrences"] += 1
                surface_counts = example["sentence_target_surface_counts"]
                surface_counts[match.surface] = surface_counts.get(match.surface, 0) + 1
            for target_id, example in sentence_examples.items():
                samplers[target_id].add(example)
        docs_processed += 1
        now = time.monotonic()
        doc_due = (
            progress_interval_docs > 0
            and docs_processed % progress_interval_docs == 0
        )
        time_due = progress_interval_seconds > 0 and now >= next_progress_at
        if progress_logging and (doc_due or time_due):
            elapsed = now - started_at
            _log_progress(
                "Processed "
                f"{docs_processed:,} docs; "
                f"{word_tokens:,} word tokens; "
                f"{target_mentions:,} target mentions; "
                f"elapsed={elapsed:.1f}s; "
                f"{_format_rate(word_tokens, elapsed, unit='words')}"
            )
            while progress_interval_seconds > 0 and next_progress_at <= now:
                next_progress_at += progress_interval_seconds

    out_dir = _resolve_output_dir(
        config,
        root_dir=root,
        output_dir=output_dir,
        created_utc=created_utc,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "target_coverage": out_dir / "target_coverage.csv",
        "exposure_thresholds": out_dir / "exposure_thresholds.csv",
        "global_tokenizer_summary": out_dir / "global_tokenizer_summary.json",
        "global_tokenizer_words": out_dir / "global_tokenizer_words.csv",
        "context_summary": out_dir / "context_summary.csv",
        "context_examples": out_dir / "context_examples.jsonl",
        "manifest": out_dir / "manifest.json",
    }

    if progress_logging:
        _log_progress(f"Writing lexical reports to {out_dir}")

    target_rows = build_target_rows(targets, stats_by_target)
    threshold_rows = build_threshold_rows(target_rows)
    context_rows = build_context_summary_rows(
        targets,
        stats_by_target,
        context_neighbor_stopwords=context_neighbor_stopwords,
    )
    global_summary = build_global_summary(
        word_counts,
        tokenizer_analyzer,
        min_frequency_for_fragmented=min_frequency_for_fragmented,
    )
    global_word_rows = build_global_word_rows(word_counts, tokenizer_analyzer)
    example_rows = _target_examples(targets, samplers)

    write_csv(target_rows, files["target_coverage"], fieldnames=TARGET_COVERAGE_FIELDS)
    write_csv(threshold_rows, files["exposure_thresholds"], fieldnames=THRESHOLD_FIELDS)
    write_json(global_summary, files["global_tokenizer_summary"])
    write_csv(
        global_word_rows,
        files["global_tokenizer_words"],
        fieldnames=GLOBAL_WORD_FIELDS,
    )
    write_csv(context_rows, files["context_summary"], fieldnames=CONTEXT_SUMMARY_FIELDS)
    write_jsonl(example_rows, files["context_examples"])

    manifest = {
        "version": "lexical_analysis_v1",
        "created_utc": created_utc.isoformat(),
        "name": config.get("name"),
        "dataset_kind": dataset_kind(dataset_config),
        "dataset": dataset_config,
        "text_field": text_field,
        "source_field": source_field,
        "targets_path": str(target_path) if target_path is not None else None,
        "target_count": len(targets),
        "docs_processed": docs_processed,
        "word_tokens": word_tokens,
        "unique_word_types": len(word_counts),
        "lemmatizer": config.get("lemmatizer", {}),
        "tokenizer": config.get("tokenizer", {}),
        "analysis": {
            "neighbor_window": neighbor_window,
            "context_word_radius": context_word_radius,
            "examples_per_target": examples_per_target,
            "seed": seed,
            "tokenizer_mode": tokenizer_mode,
            "filter_context_neighbor_stopwords": bool(context_neighbor_stopwords),
            "context_neighbor_stopword_count": len(context_neighbor_stopwords),
            "min_frequency_for_fragmented": min_frequency_for_fragmented,
            "max_docs": resolved_max_docs,
            "example_sampling_unit": "sentence",
            "progress_logging": progress_logging,
            "progress_interval_docs": progress_interval_docs,
            "progress_interval_seconds": progress_interval_seconds,
        },
        "files": {name: str(path) for name, path in files.items()},
    }
    write_json(_jsonable(manifest), files["manifest"])
    if progress_logging:
        elapsed = time.monotonic() - started_at
        _log_progress(
            "Finished lexical analysis: "
            f"{docs_processed:,} docs; "
            f"{word_tokens:,} word tokens; "
            f"{target_mentions:,} target mentions; "
            f"elapsed={elapsed:.1f}s"
        )
    return LexicalAnalysisResult(output_dir=out_dir, files=files, manifest=manifest)


__all__ = ["LexicalAnalysisResult", "load_analysis_config", "run_lexical_analysis"]
