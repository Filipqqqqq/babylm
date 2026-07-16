"""Edit alias-binding corpus drafts with a constrained Qwen copy-edit pass."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from babysteps.concept_curriculum.alias_binding_corpus import (
    DEFAULT_OUTPUT_DIR as DEFAULT_ALIAS_INPUT_DIR,
)
from babysteps.concept_curriculum.alias_binding_corpus import (
    normalize_space,
    read_jsonl,
    word_count,
    write_json,
    write_jsonl,
)
from babysteps.material_world.fireworks_client import FireworksChatClient, load_api_key
from babysteps.paths import resolve_path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_ID = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_INPUT_FILENAME = "documents.jsonl"
DEFAULT_BATCH_SIZE = 10
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096
PROMPT_ONLY = "prompt_only"
DETERMINISTIC = "deterministic"
FIREWORKS = "fireworks"
RENDERER_TYPES = {PROMPT_ONLY, DETERMINISTIC, FIREWORKS}
REJECT_ACTIONS = {"keep_original", "drop"}

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
FORBIDDEN_PATTERN = re.compile(
    r"\{handle\}|\btherefore\b|\b(?:wug|fep|dax|blicket)\b|^\s*A\s+\w+\s+is\s+an?\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "has",
    "had",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


@dataclass(frozen=True)
class AliasBindingEditConfig:
    input_dir: Path
    output_dir: Path
    input_filename: str = DEFAULT_INPUT_FILENAME
    renderer: str = PROMPT_ONLY
    model_id: str = DEFAULT_MODEL_ID
    batch_size: int = DEFAULT_BATCH_SIZE
    document_limit: int | None = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning_effort: str = "none"
    reasoning_history: str = "disabled"
    timeout_seconds: int = 600
    max_request_retries: int = 4
    reject_action: str = "keep_original"


@dataclass(frozen=True)
class AliasBindingEditResult:
    outputs: dict[str, Path]
    manifest: dict[str, Any]
    audit_report: dict[str, Any]


class AliasBindingEditRenderer(Protocol):
    def __call__(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        response_format: dict[str, Any] | None = None,
    ) -> str: ...


def strip_response_fence(raw: str) -> str:
    text = str(raw).strip()
    if text.startswith("```"):
        text = FENCE_PATTERN.sub("", text).strip()
    return text


def parse_json_object(raw: str) -> dict[str, Any]:
    text = strip_response_fence(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("renderer response must be a JSON object")
    return payload


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "documents_raw": output_dir / "documents_raw.jsonl",
        "edit_prompts": output_dir / "edit_prompts.jsonl",
        "edited_documents_raw": output_dir / "edited_documents_raw.jsonl",
        "documents": output_dir / "documents.jsonl",
        "train": output_dir / "train.txt",
        "audit_report": output_dir / "edit_audit_report.json",
        "manifest": output_dir / "manifest.json",
    }


def default_output_dir(input_dir: Path) -> Path:
    return input_dir.parent / f"{input_dir.name}_qwen_edited"


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(size))
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _compact_binding(binding: dict[str, Any]) -> dict[str, str]:
    return {
        "handle": normalize_space(binding.get("handle") or ""),
        "concept": normalize_space(binding.get("concept") or ""),
        "fact": normalize_space(binding.get("fact") or ""),
        "template_type": normalize_space(binding.get("template_type") or ""),
        "rendered_template": normalize_space(binding.get("rendered_template") or ""),
    }


def prompt_document(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "text": normalize_space(row.get("text") or ""),
        "sequence_family": str(row.get("sequence_family") or ""),
        "carrier_type": str(row.get("carrier_type") or ""),
        "bindings": [_compact_binding(binding) for binding in row.get("bindings", [])],
    }


def build_edit_prompt(rows: list[dict[str, Any]], *, disable_thinking: bool) -> str:
    payload = {"items": [prompt_document(row) for row in rows]}
    prompt = f"""You are editing alias-binding training examples.

Goal:
- Make each example clear and grammatical.
- Preserve every handle, concept, mapping, and fact.
- Preserve unusual or surprising facts when they are correct.
- Keep the alias-binding task explicit: a handle maps to a concept, and later property sentences use the handle.

Rules:
- Return one item for every input id, in the same order.
- Use decision "keep" when the original is already clear.
- Use decision "rewrite" only for local clarity or grammar fixes.
- Use decision "reject" only if the example cannot be made clear without changing facts or mappings.
- Do not invent new facts.
- Do not remove any handle.
- Do not change any concept.
- Do not replace unusual facts with generic facts.
- Avoid the words "therefore", "wug", "fep", "dax", and "blicket".
- Do not use the format "A nonce is a concept. Therefore, ...".

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return only JSON with this schema:
{{
  "items": [
    {{
      "id": "input id",
      "decision": "keep|rewrite|reject",
      "edited_text": "final text when decision is keep or rewrite; empty string for reject",
      "reason": "short reason"
    }}
  ]
}}"""
    if disable_thinking:
        prompt = f"{prompt}\n\n/no_think"
    return prompt


def parse_edit_items(raw: str, *, expected_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    payload = parse_json_object(raw)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("edit response must include an items list")
    parsed: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "")
        if item_id:
            parsed[item_id] = item
    missing = [item_id for item_id in expected_ids if item_id not in parsed]
    if missing:
        raise ValueError(f"edit response missing ids: {missing[:5]}")
    return parsed


def phrase_in_text(text: str, phrase: str) -> bool:
    phrase = normalize_space(phrase)
    if not phrase:
        return True
    pattern = r"(?<![A-Za-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def content_tokens(text: str, *, blocked: set[str] | None = None) -> set[str]:
    blocked = blocked or set()
    tokens = {
        token.lower()
        for token in WORD_PATTERN.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS and token.lower() not in blocked
    }
    return tokens


def fact_token_requirement(tokens: set[str]) -> int:
    if not tokens:
        return 0
    if len(tokens) <= 2:
        return 1
    return 2


def validate_edited_text(row: dict[str, Any], edited_text: str) -> list[str]:
    text = normalize_space(edited_text)
    reasons: list[str] = []
    if not text:
        return ["empty_text"]
    if FORBIDDEN_PATTERN.search(text):
        reasons.append("forbidden_wugs_or_therefore_pattern")

    all_text_tokens = content_tokens(text)
    for binding in row.get("bindings", []):
        handle = normalize_space(binding.get("handle") or "")
        concept = normalize_space(binding.get("concept") or "")
        fact = normalize_space(binding.get("fact") or "")
        rendered_template = normalize_space(binding.get("rendered_template") or "")
        if not phrase_in_text(text, handle):
            reasons.append(f"missing_handle:{handle}")
        if not phrase_in_text(text, concept):
            reasons.append(f"missing_concept:{concept}")

        blocked = content_tokens(f"{handle} {concept}")
        fact_tokens = content_tokens(f"{fact} {rendered_template}", blocked=blocked)
        required = fact_token_requirement(fact_tokens)
        overlap = fact_tokens & all_text_tokens
        if required and len(overlap) < required:
            reasons.append(f"missing_fact_content:{handle}")
    return reasons


def _final_row(
    row: dict[str, Any],
    *,
    text: str,
    status: str,
    decision: str,
    reason: str,
    validation_reasons: list[str] | None = None,
) -> dict[str, Any]:
    final = dict(row)
    original_text = normalize_space(row.get("text") or "")
    final["original_text"] = original_text
    final["text"] = normalize_space(text)
    final["word_count"] = word_count(final["text"])
    final["edit_status"] = status
    final["edit_decision"] = decision
    final["edit_reason"] = reason
    final["edit_validation_reasons"] = validation_reasons or []
    return final


def apply_edit_item(
    row: dict[str, Any],
    item: dict[str, Any] | None,
    *,
    reject_action: str,
) -> dict[str, Any] | None:
    original = normalize_space(row.get("text") or "")
    if item is None:
        return _final_row(
            row,
            text=original,
            status="kept_original_missing_response",
            decision="keep",
            reason="missing renderer item",
        )

    decision = normalize_space(item.get("decision") or "keep").lower()
    reason = normalize_space(item.get("reason") or "")
    if decision not in {"keep", "rewrite", "reject"}:
        return _final_row(
            row,
            text=original,
            status="kept_original_invalid_decision",
            decision=decision,
            reason=reason,
        )
    if decision == "keep":
        return _final_row(
            row,
            text=original,
            status="kept_original_qwen_keep",
            decision="keep",
            reason=reason,
        )
    if decision == "reject":
        if reject_action == "drop":
            return None
        return _final_row(
            row,
            text=original,
            status="kept_original_qwen_reject",
            decision="reject",
            reason=reason,
        )

    edited_text = normalize_space(item.get("edited_text") or "")
    validation_reasons = validate_edited_text(row, edited_text)
    if validation_reasons:
        return _final_row(
            row,
            text=original,
            status="kept_original_invalid_rewrite",
            decision="rewrite",
            reason=reason,
            validation_reasons=validation_reasons,
        )
    return _final_row(
        row,
        text=edited_text,
        status="accepted_rewrite",
        decision="rewrite",
        reason=reason,
    )


class DeterministicAliasBindingEditRenderer:
    """Local renderer for smoke tests; it keeps clear deterministic drafts unchanged."""

    def __call__(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        del system_prompt, temperature, max_tokens, stage, response_format
        marker = "Input JSON:"
        schema_marker = "Return only JSON"
        start = prompt.index(marker) + len(marker)
        end = prompt.index(schema_marker, start)
        payload = json.loads(prompt[start:end].strip())
        items = [
            {
                "id": str(item["id"]),
                "decision": "keep",
                "edited_text": str(item["text"]),
                "reason": "deterministic draft kept",
            }
            for item in payload["items"]
        ]
        return json.dumps({"items": items}, ensure_ascii=False)


def create_renderer(
    config: AliasBindingEditConfig,
    *,
    api_key: str = "",
) -> AliasBindingEditRenderer | None:
    if config.renderer == PROMPT_ONLY:
        return None
    if config.renderer == DETERMINISTIC:
        return DeterministicAliasBindingEditRenderer()
    if config.renderer == FIREWORKS:
        return FireworksChatClient(
            model=config.model_id,
            api_key=api_key,
            timeout_seconds=config.timeout_seconds,
            max_request_retries=config.max_request_retries,
            reasoning_effort=config.reasoning_effort,
            reasoning_history=config.reasoning_history,
        )
    raise ValueError(f"renderer must be one of {sorted(RENDERER_TYPES)}")


def edit_alias_binding_corpus(
    config: AliasBindingEditConfig,
    *,
    renderer: AliasBindingEditRenderer | None = None,
) -> AliasBindingEditResult:
    if config.renderer not in RENDERER_TYPES:
        raise ValueError(f"renderer must be one of {sorted(RENDERER_TYPES)}")
    if config.reject_action not in REJECT_ACTIONS:
        raise ValueError(f"reject_action must be one of {sorted(REJECT_ACTIONS)}")

    input_path = config.input_dir / config.input_filename
    rows = read_jsonl(input_path)
    if config.document_limit is not None:
        rows = rows[: max(0, int(config.document_limit))]

    paths = output_paths(config.output_dir)
    write_jsonl(paths["documents_raw"], rows)

    disable_thinking = str(config.reasoning_effort).lower() in {"none", "false", "0", "disabled"}
    renderer = renderer if renderer is not None else create_renderer(config)
    prompt_only = renderer is None
    batches = chunked(rows, int(config.batch_size))

    prompt_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()

    progress = tqdm(
        enumerate(batches),
        total=len(batches),
        desc="Alias edits",
        unit="batch",
        dynamic_ncols=True,
        disable=prompt_only,
    )
    for batch_index, batch in progress:
        batch_id = f"edit_batch_{batch_index:06d}"
        doc_ids = tuple(str(row.get("id") or "") for row in batch)
        prompt = build_edit_prompt(batch, disable_thinking=disable_thinking)
        prompt_rows.append({"id": batch_id, "document_ids": doc_ids, "prompt": prompt})

        parsed_items: dict[str, dict[str, Any]] = {}
        raw = ""
        parse_error = ""
        if not prompt_only:
            assert renderer is not None
            try:
                raw = renderer(
                    prompt=prompt,
                    system_prompt=(
                        "You are a careful dataset copy editor. Return only JSON and preserve "
                        "all alias bindings, concepts, and facts."
                    ),
                    temperature=float(config.temperature),
                    max_tokens=int(config.max_tokens),
                    stage="alias_binding_edit",
                    response_format={"type": "json_object"},
                )
                parsed_items = parse_edit_items(raw, expected_ids=doc_ids)
            except Exception as exc:  # noqa: BLE001 - recorded for audit and safe fallback.
                parse_error = str(exc)
                parse_errors.append({"batch_id": batch_id, "document_ids": doc_ids, "error": parse_error})
        raw_rows.append(
            {
                "id": batch_id,
                "document_ids": doc_ids,
                "renderer": config.renderer,
                "raw_response": raw,
                "parse_error": parse_error,
            }
        )

        for row in batch:
            if prompt_only:
                final = _final_row(
                    row,
                    text=normalize_space(row.get("text") or ""),
                    status="kept_original_prompt_only",
                    decision="keep",
                    reason="prompt_only renderer",
                )
            elif parse_error:
                final = _final_row(
                    row,
                    text=normalize_space(row.get("text") or ""),
                    status="kept_original_batch_parse_error",
                    decision="keep",
                    reason=parse_error,
                )
            else:
                final = apply_edit_item(
                    row,
                    parsed_items.get(str(row.get("id") or "")),
                    reject_action=config.reject_action,
                )
            if final is None:
                status_counts["dropped_qwen_reject"] += 1
                continue
            status_counts[str(final["edit_status"])] += 1
            validation_counts.update(final.get("edit_validation_reasons") or [])
            final_rows.append(final)

    write_jsonl(paths["edit_prompts"], prompt_rows)
    write_jsonl(paths["edited_documents_raw"], raw_rows)
    write_jsonl(paths["documents"], final_rows)
    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    paths["train"].write_text("".join(f"{row['text']}\n" for row in final_rows), encoding="utf-8")

    counts = {
        "input_documents": len(rows),
        "final_documents": len(final_rows),
        "batches": len(batches),
        "words": sum(int(row.get("word_count") or word_count(row.get("text") or "")) for row in final_rows),
        "parse_errors": len(parse_errors),
    }
    audit_report = {
        "counts": counts,
        "status_counts": dict(sorted(status_counts.items())),
        "validation_rejection_counts": dict(sorted(validation_counts.items())),
        "parse_errors": parse_errors,
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "comps_alias_binding_qwen_edit",
        "renderer": config.renderer,
        "model_id": config.model_id if config.renderer == FIREWORKS else None,
        "temperature": float(config.temperature),
        "max_tokens": int(config.max_tokens),
        "reasoning_effort": config.reasoning_effort if config.renderer == FIREWORKS else None,
        "reasoning_history": config.reasoning_history if config.renderer == FIREWORKS else None,
        "batch_size": int(config.batch_size),
        "reject_action": config.reject_action,
        "source_files": {"input_documents": str(input_path)},
        "outputs": {key: str(path) for key, path in paths.items()},
        "counts": counts,
    }
    write_json(paths["audit_report"], audit_report)
    write_json(paths["manifest"], manifest)
    return AliasBindingEditResult(outputs=paths, manifest=manifest, audit_report=audit_report)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_ALIAS_INPUT_DIR)
    parser.add_argument("--output-dir")
    parser.add_argument("--input-filename", default=DEFAULT_INPUT_FILENAME)
    parser.add_argument("--renderer", choices=sorted(RENDERER_TYPES), default=PROMPT_ONLY)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--document-limit", type=int)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--reasoning-history", default="disabled")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-request-retries", type=int, default=4)
    parser.add_argument("--reject-action", choices=sorted(REJECT_ACTIONS), default="keep_original")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-file")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> tuple[AliasBindingEditConfig, str]:
    input_dir = resolve_path(args.input_dir, base_dir=ROOT)
    output_dir = (
        resolve_path(args.output_dir, base_dir=ROOT)
        if args.output_dir
        else default_output_dir(input_dir)
    )
    api_key = ""
    if args.renderer == FIREWORKS:
        api_key_file = (
            str(resolve_path(args.api_key_file, base_dir=ROOT))
            if args.api_key_file
            else None
        )
        api_key = load_api_key(api_key=args.api_key, api_key_file=api_key_file)
    config = AliasBindingEditConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        input_filename=str(args.input_filename),
        renderer=str(args.renderer),
        model_id=str(args.model_id),
        batch_size=int(args.batch_size),
        document_limit=args.document_limit,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        reasoning_effort=str(args.reasoning_effort),
        reasoning_history=str(args.reasoning_history),
        timeout_seconds=int(args.timeout_seconds),
        max_request_retries=int(args.max_request_retries),
        reject_action=str(args.reject_action),
    )
    return config, api_key


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, api_key = config_from_args(args)
    renderer = create_renderer(config, api_key=api_key)
    result = edit_alias_binding_corpus(config, renderer=renderer)
    print(
        "COMPS alias-binding edits: "
        f"documents={result.manifest['counts']['final_documents']} "
        f"words={result.manifest['counts']['words']} "
        f"renderer={config.renderer} "
        f"output_dir={config.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
