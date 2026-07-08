#!/usr/bin/env python3
"""Postprocess generated COMPS concept-property paragraphs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.material_world.fireworks_client import chat_json_or_text, load_api_key


DEFAULT_INPUT_DIR = (
    "data/concept_curriculum/comps_concept_properties_clean_no_neighbors_stratified50_qwen"
)
DEFAULT_MODEL_ID = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_METHOD = "less_repetitive_paragraphs"
DEFAULT_MAX_TOKENS = 3072
DEFAULT_RETRY_MAX_TOKENS = 4096
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MIN_CONCEPT_MENTIONS = 3
PROMPT_ONLY = "prompt_only"
TEMPLATE = "template"
FIREWORKS = "fireworks"
RENDERER_TYPES = {PROMPT_ONLY, TEMPLATE, FIREWORKS}
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
LEAK_PATTERN = re.compile(
    r"COMPS|benchmark|evaluation|minimal pair|property_phrase|"
    r"acceptable_concept|unacceptable_concept",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PostprocessResult:
    prompts: tuple[dict[str, Any], ...]
    documents: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    outputs: dict[str, Path]


class PostprocessError(ValueError):
    """A retryable or terminal validation failure for one concept postprocess."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        raw_text: str = "",
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.raw_text = raw_text


def word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(row)
    return tuple(rows)


def write_jsonl(path: Path, rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def group_documents_by_concept(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    """Group document rows by concept_id while preserving file order."""

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for index, row in enumerate(rows):
        concept_id = str(row.get("concept_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not concept_id:
            raise ValueError(f"documents[{index}] is missing concept_id")
        if not text:
            raise ValueError(f"documents[{index}] is missing text")
        groups.setdefault(concept_id, []).append(dict(row))
    return groups


def concept_display_name(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        concept = str(row.get("concept", "")).strip()
        if concept:
            return concept
    return str(rows[0].get("concept_id", "concept")).replace("_", " ")


def build_less_repetitive_prompt(
    *,
    concept_id: str,
    concept: str,
    rows: list[dict[str, Any]],
    min_concept_mentions: int = DEFAULT_MIN_CONCEPT_MENTIONS,
    disable_thinking: bool = True,
) -> str:
    """Build the per-concept cleanup prompt."""

    sense_hint = str(rows[0].get("sense_hint", "")).strip()
    sense_text = ""
    if sense_hint:
        sense_text = (
            f'\nIntended sense of "{concept}": {sense_hint}. '
            "Stay with this sense and do not discuss alternate word meanings.\n"
        )
    paragraph_blocks = []
    for paragraph_index, row in enumerate(rows, 1):
        style_id = str(row.get("style_id", "")).strip() or "paragraph"
        text = re.sub(r"\s+", " ", str(row["text"]).strip())
        paragraph_blocks.append(
            f"Paragraph {paragraph_index} ({style_id}):\n{text}"
        )
    joined_paragraphs = "\n\n".join(paragraph_blocks)
    prompt = f"""You are revising short educational corpus paragraphs about the concept: **{concept}**.
{sense_text}
The input paragraphs were generated separately, so their openings, category phrases, and definitions may be repetitive. Revise them into a less repetitive set of paragraphs while preserving the information.

The final text will be used as ordinary pretraining corpus text. The goal is for the concept name "{concept}" to co-occur repeatedly with varied characteristic properties, parts, behaviors, contexts, functions, state changes, and relations. Keep "{concept}" frequent, but make the surrounding information diverse.

Rules:
- Keep exactly {len(rows)} paragraphs, in the same order.
- Each output paragraph must correspond to the same-numbered input paragraph.
- Do not merge, split, summarize, or reorder paragraphs.
- Make minimal edits: vary repeated openings, fix typos, improve awkward wording, and keep prose natural.
- Do not shorten a paragraph just to make it smoother. Prefer rewriting repeated wording over deleting factual content.
- Preserve every distinct factual detail from each input paragraph unless it is plainly duplicated in the same paragraph or plainly incorrect.
- Keep each revised paragraph close to the input paragraph's information density and length. It may be a little shorter, but it should not become a compressed summary.
- Do not add new factual claims.
- Every output paragraph must include the exact concept name "{concept}" at least {int(min_concept_mentions)} times.
- Plural or inflected forms may appear naturally, but they do not replace the required exact "{concept}" mentions.
- Keep the concept name present naturally, but avoid starting every paragraph with the same definition.
- Prefer varied starts like a property, location, action, contrast, material, or use, as long as "{concept}" still appears soon after.
- Avoid repeating the same category phrase or definition across paragraphs; keep the clearest category statement once, then use later paragraphs for more specific facts.
- If two paragraphs repeat the same broad statement, preserve the more concrete details from both while reducing the repeated wording.
- When reducing a repeated category phrase, keep any meaningful modifiers inside it, such as size, shape, material, body covering, instrument type, food form, habitat, purpose, or mechanism.
- After the broad category has been stated once, later paragraphs should keep saying "{concept}" near concrete details from the input instead of repeating the same category phrase.
- When a paragraph currently repeats a broad phrase like "{concept} is a kind of thing," rewrite that space around more specific input facts: parts, materials, mechanisms, behaviors, uses, settings, comparisons, ingredients, preparation, habitats, sounds, motions, body features, or state changes.
- Do not drop concrete details such as parts, colors, textures, body coverings, ingredients, tools, mechanisms, locations, uses, functions, behaviors, sounds, movements, life-cycle facts, injuries, risks, maintenance, packaging, preparation, contrasts, or human interactions.
- Do not replace the concept name entirely with pronouns or synonyms such as "this animal," "the object," or "the vessel."
- Do not use headings, bullets, numbering, labels, markdown, or meta-commentary.
- Return only the revised paragraphs, separated by blank lines.

Input paragraphs for concept_id "{concept_id}":

{joined_paragraphs}"""
    if disable_thinking:
        prompt = f"{prompt}\n\n/no_think"
    return prompt


def build_retry_prompt(
    *,
    prompt: str,
    concept: str,
    error: BaseException,
    min_concept_mentions: int,
    disable_thinking: bool = True,
) -> str:
    """Append validation feedback for a retry completion."""

    retry_prompt = (
        f"{prompt}\n\n"
        "The previous revision failed validation and must be corrected.\n"
        f"Validation error: {error}\n"
        "Return the full revised paragraph set again, not only the failed paragraph.\n"
        f"Before returning, silently verify that every paragraph contains the exact string "
        f"\"{concept}\" at least {int(min_concept_mentions)} times.\n"
        "Return only the corrected paragraphs separated by blank lines."
    )
    if disable_thinking:
        retry_prompt = f"{retry_prompt}\n/no_think"
    return retry_prompt


def strip_response_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def parse_revised_paragraphs(raw: str, *, expected_count: int) -> tuple[str, ...]:
    """Parse and validate revised paragraphs from a model response."""

    text = strip_response_fence(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("paragraphs"), list):
        raw_paragraphs = payload["paragraphs"]
        paragraphs = tuple(
            re.sub(r"\s+", " ", str(paragraph).strip())
            for paragraph in raw_paragraphs
            if str(paragraph).strip()
        )
    else:
        paragraphs = tuple(
            re.sub(r"\s+", " ", paragraph.strip())
            for paragraph in re.split(r"\n\s*\n+", text)
            if paragraph.strip()
        )
    paragraphs = tuple(
        re.sub(r"^(?:paragraph\s*)?\d+[\).:\-]\s*", "", paragraph, flags=re.IGNORECASE)
        for paragraph in paragraphs
    )
    if len(paragraphs) != int(expected_count):
        raise PostprocessError(
            f"expected {expected_count} paragraphs, got {len(paragraphs)}",
            raw_text=raw,
        )
    if any(not paragraph for paragraph in paragraphs):
        raise PostprocessError("postprocess response contained an empty paragraph", raw_text=raw)
    lowered_start = text[:400].lower()
    if "thinking process" in lowered_start or "analysis:" in lowered_start:
        raise PostprocessError("postprocess response included reasoning/meta text", raw_text=raw)
    return paragraphs


def concept_mention_pattern(concept: str) -> re.Pattern[str]:
    """Build a simple exact concept-name matcher with a plural fallback."""

    words = [re.escape(word) for word in str(concept).strip().lower().split()]
    if not words:
        words = [r"concept"]
    words[-1] = words[-1] + r"s?"
    return re.compile(r"\b" + r"\s+".join(words) + r"\b", re.IGNORECASE)


def validate_paragraph_concept_mentions(
    paragraphs: tuple[str, ...],
    *,
    concept: str,
    min_mentions: int = DEFAULT_MIN_CONCEPT_MENTIONS,
) -> None:
    pattern = concept_mention_pattern(concept)
    low_count = [
        {
            "paragraph": index + 1,
            "count": len(pattern.findall(paragraph)),
        }
        for index, paragraph in enumerate(paragraphs)
        if len(pattern.findall(paragraph)) < int(min_mentions)
    ]
    if low_count:
        raise PostprocessError(
            "revised paragraphs do not mention concept name "
            f"{concept!r} at least {int(min_mentions)} times: {low_count}"
        )


def rewrite_rows_with_paragraphs(
    *,
    rows: list[dict[str, Any]],
    paragraphs: tuple[str, ...],
    method: str = DEFAULT_METHOD,
) -> tuple[dict[str, Any], ...]:
    if len(rows) != len(paragraphs):
        raise ValueError("row and paragraph counts must match")
    rewritten = []
    for row, text in zip(rows, paragraphs):
        source_text = str(row.get("text", ""))
        new_row = dict(row)
        new_row["source_text"] = source_text
        new_row["text"] = text
        new_row["word_count"] = word_count(text)
        new_row["postprocess"] = {
            "method": str(method),
            "source_id": str(row.get("id", "")),
            "source_word_count": int(row.get("word_count", word_count(source_text))),
        }
        rewritten.append(new_row)
    return tuple(rewritten)


def template_revise_paragraphs(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Deterministic local renderer for smoke tests; real cleanup uses Qwen."""

    return tuple(re.sub(r"\s+", " ", str(row["text"]).strip()) for row in rows)


def call_fireworks_postprocess(
    *,
    prompt: str,
    api_key: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | bool | None,
    reasoning_history: str | None,
    timeout_seconds: int,
    max_request_retries: int,
) -> str:
    return chat_json_or_text(
        model=str(model_id),
        api_key=api_key,
        messages=[
            {
                "role": "system",
                "content": (
                    "You lightly revise educational corpus paragraphs. "
                    "Return only the final revised paragraphs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        reasoning_effort=reasoning_effort,
        reasoning_history=reasoning_history,
        timeout_seconds=int(timeout_seconds),
        max_request_retries=int(max_request_retries),
    )


def postprocess_documents(
    *,
    input_documents: str | Path,
    output_documents: str | Path,
    output_train: str | Path,
    output_prompts: str | Path,
    output_manifest: str | Path,
    raw_output_dir: str | Path,
    renderer: str,
    api_key: str = "",
    model_id: str = DEFAULT_MODEL_ID,
    method: str = DEFAULT_METHOD,
    concept_ids: tuple[str, ...] = (),
    concept_limit: int | None = None,
    force: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retry_max_tokens: int = DEFAULT_RETRY_MAX_TOKENS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    temperature: float = DEFAULT_TEMPERATURE,
    reasoning_effort: str = "none",
    reasoning_history: str = "disabled",
    min_concept_mentions: int = DEFAULT_MIN_CONCEPT_MENTIONS,
    timeout_seconds: int = 600,
    max_request_retries: int = 4,
) -> PostprocessResult:
    """Run prompt-only, template, or Fireworks postprocessing over documents."""

    if renderer not in RENDERER_TYPES:
        raise ValueError(f"renderer must be one of {sorted(RENDERER_TYPES)}")
    reasoning_effort = str(reasoning_effort).strip() or "none"
    reasoning_history = str(reasoning_history).strip() or "disabled"
    disable_thinking = reasoning_effort.lower() in {"none", "false", "0", "disabled"}
    rows = read_jsonl(input_documents)
    groups = group_documents_by_concept(rows)
    selected_ids = tuple(concept_ids) if concept_ids else tuple(groups)
    missing = [concept_id for concept_id in selected_ids if concept_id not in groups]
    if missing:
        raise ValueError(f"requested concept_id not found in documents: {missing}")
    if concept_limit is not None:
        selected_ids = selected_ids[: max(0, int(concept_limit))]

    prompt_rows = []
    rewritten_rows = []
    raw_dir = Path(raw_output_dir)
    if renderer != PROMPT_ONLY:
        raw_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(
        total=len(selected_ids),
        desc="COMPS postprocess",
        unit="concept",
        dynamic_ncols=True,
        disable=renderer == PROMPT_ONLY,
    )
    try:
        for concept_id in selected_ids:
            concept_rows = groups[concept_id]
            concept = concept_display_name(concept_rows)
            prompt = build_less_repetitive_prompt(
                concept_id=concept_id,
                concept=concept,
                rows=concept_rows,
                min_concept_mentions=int(min_concept_mentions),
                disable_thinking=disable_thinking,
            )
            prompt_row = {
                "id": f"{concept_id}__{method}",
                "concept_id": concept_id,
                "concept": concept,
                "method": method,
                "paragraph_count": len(concept_rows),
                "prompt": prompt,
            }
            prompt_rows.append(prompt_row)
            if renderer == PROMPT_ONLY:
                continue
            status = "template"
            if renderer == TEMPLATE:
                paragraphs = template_revise_paragraphs(concept_rows)
            else:
                last_error: BaseException | None = None
                paragraphs = ()
                status = "rendered"
                for attempt_index in range(max(1, int(max_attempts))):
                    token_budget = int(max_tokens) if attempt_index == 0 else int(retry_max_tokens)
                    raw_path = raw_dir / f"{concept_id}__attempt{attempt_index}.txt"
                    attempt_prompt = prompt
                    if last_error is not None:
                        attempt_prompt = build_retry_prompt(
                            prompt=prompt,
                            concept=concept,
                            error=last_error,
                            min_concept_mentions=int(min_concept_mentions),
                            disable_thinking=disable_thinking,
                        )
                    if raw_path.exists() and not force:
                        raw = raw_path.read_text(encoding="utf-8")
                        status = f"reused attempt{attempt_index}"
                    else:
                        raw = call_fireworks_postprocess(
                            prompt=attempt_prompt,
                            api_key=api_key,
                            model_id=model_id,
                            max_tokens=token_budget,
                            temperature=temperature,
                            reasoning_effort=reasoning_effort,
                            reasoning_history=reasoning_history,
                            timeout_seconds=timeout_seconds,
                            max_request_retries=max_request_retries,
                        )
                        raw_path.write_text(raw, encoding="utf-8")
                        status = f"written attempt{attempt_index}"
                    try:
                        paragraphs = parse_revised_paragraphs(
                            raw,
                            expected_count=len(concept_rows),
                        )
                        validate_paragraph_concept_mentions(
                            paragraphs,
                            concept=concept,
                            min_mentions=int(min_concept_mentions),
                        )
                        last_error = None
                        break
                    except PostprocessError as exc:
                        last_error = exc
                        status = f"retry after attempt{attempt_index}"
                        if not exc.retryable:
                            break
                if last_error is not None:
                    raise PostprocessError(
                        f"{concept_id}: {last_error}",
                        retryable=getattr(last_error, "retryable", True),
                        raw_text=getattr(last_error, "raw_text", ""),
                    ) from last_error
            validate_paragraph_concept_mentions(
                paragraphs,
                concept=concept,
                min_mentions=int(min_concept_mentions),
            )
            rewritten_rows.extend(
                rewrite_rows_with_paragraphs(
                    rows=concept_rows,
                    paragraphs=paragraphs,
                    method=method,
                )
            )
            progress.set_postfix_str(f"{status} {concept_id[:48]}", refresh=False)
            progress.update(1)
    finally:
        progress.close()

    output_paths = {
        "prompts": Path(output_prompts),
        "documents": Path(output_documents),
        "train": Path(output_train),
        "manifest": Path(output_manifest),
    }
    write_jsonl(output_paths["prompts"], prompt_rows)
    documents_tuple = tuple(rewritten_rows)
    if renderer != PROMPT_ONLY:
        write_jsonl(output_paths["documents"], documents_tuple)
        output_paths["train"].write_text(
            "".join(f"{row['text'].strip()}\n" for row in documents_tuple),
            encoding="utf-8",
        )

    leak_docs = [
        str(row.get("id", ""))
        for row in documents_tuple
        if LEAK_PATTERN.search(str(row.get("text", "")))
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "comps_concept_properties_postprocess",
        "method": method,
        "renderer": renderer,
        "model_id": model_id if renderer == FIREWORKS else None,
        "reasoning_effort": reasoning_effort if renderer == FIREWORKS else None,
        "reasoning_history": reasoning_history if renderer == FIREWORKS else None,
        "max_tokens": int(max_tokens),
        "retry_max_tokens": int(retry_max_tokens),
        "max_attempts": int(max_attempts),
        "min_concept_mentions": int(min_concept_mentions),
        "input_documents": str(input_documents),
        "counts": {
            "concepts": len(selected_ids),
            "prompts": len(prompt_rows),
            "documents": len(documents_tuple),
            "train_words": sum(int(row.get("word_count", 0)) for row in documents_tuple),
            "leaky_documents": len(leak_docs),
        },
        "concept_ids": list(selected_ids),
        "leaky_document_ids": leak_docs,
        "outputs": {
            key: str(path)
            for key, path in output_paths.items()
            if renderer != PROMPT_ONLY or key in {"prompts", "manifest"}
        },
    }
    write_json(output_paths["manifest"], manifest)
    return PostprocessResult(
        prompts=tuple(prompt_rows),
        documents=documents_tuple,
        manifest=manifest,
        outputs=output_paths,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-documents")
    parser.add_argument("--output-documents")
    parser.add_argument("--output-train")
    parser.add_argument("--output-prompts")
    parser.add_argument("--output-manifest")
    parser.add_argument("--raw-output-dir")
    parser.add_argument("--renderer", default=PROMPT_ONLY, choices=sorted(RENDERER_TYPES))
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--concept-id", action="append", default=[])
    parser.add_argument("--concept-limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-file", default="fireworks_api.txt")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--retry-max-tokens", type=int, default=DEFAULT_RETRY_MAX_TOKENS)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum validation attempts per concept. Attempts after the first use --retry-max-tokens.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        help='Fireworks reasoning effort. Use "none" for no-thinking mode, or values like low/medium/high.',
    )
    parser.add_argument(
        "--reasoning-history",
        default="disabled",
        help='Fireworks reasoning history setting. Default "disabled" keeps reasoning out of outputs.',
    )
    parser.add_argument("--min-concept-mentions", type=int, default=DEFAULT_MIN_CONCEPT_MENTIONS)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-request-retries", type=int, default=4)
    return parser


def default_output_paths(input_dir: Path, method: str) -> dict[str, Path]:
    suffix = "less_repetitive" if str(method) == DEFAULT_METHOD else str(method)
    return {
        "documents": input_dir / f"documents_{suffix}.jsonl",
        "train": input_dir / f"train_{suffix}.txt",
        "prompts": input_dir / f"postprocess_prompts_{suffix}.jsonl",
        "manifest": input_dir / f"postprocess_manifest_{suffix}.json",
        "raw_dir": input_dir / f"postprocess_raw_{suffix}",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_dir = resolve_repo_path(args.input_dir)
    defaults = default_output_paths(input_dir, args.method)
    input_documents = (
        resolve_repo_path(args.input_documents)
        if args.input_documents
        else input_dir / "documents.jsonl"
    )
    output_documents = (
        resolve_repo_path(args.output_documents)
        if args.output_documents
        else defaults["documents"]
    )
    output_train = (
        resolve_repo_path(args.output_train)
        if args.output_train
        else defaults["train"]
    )
    output_prompts = (
        resolve_repo_path(args.output_prompts)
        if args.output_prompts
        else defaults["prompts"]
    )
    output_manifest = (
        resolve_repo_path(args.output_manifest)
        if args.output_manifest
        else defaults["manifest"]
    )
    raw_output_dir = (
        resolve_repo_path(args.raw_output_dir)
        if args.raw_output_dir
        else defaults["raw_dir"]
    )
    if args.renderer == FIREWORKS:
        api_key_file = str(resolve_repo_path(args.api_key_file)) if args.api_key_file else None
        api_key = load_api_key(api_key=args.api_key, api_key_file=api_key_file)
    else:
        api_key = ""

    result = postprocess_documents(
        input_documents=input_documents,
        output_documents=output_documents,
        output_train=output_train,
        output_prompts=output_prompts,
        output_manifest=output_manifest,
        raw_output_dir=raw_output_dir,
        renderer=str(args.renderer),
        api_key=api_key,
        model_id=str(args.model_id),
        method=str(args.method),
        concept_ids=tuple(str(item).strip() for item in args.concept_id if str(item).strip()),
        concept_limit=args.concept_limit,
        force=bool(args.force),
        max_tokens=int(args.max_tokens),
        retry_max_tokens=int(args.retry_max_tokens),
        max_attempts=int(args.max_attempts),
        temperature=float(args.temperature),
        reasoning_effort=str(args.reasoning_effort),
        reasoning_history=str(args.reasoning_history),
        min_concept_mentions=int(args.min_concept_mentions),
        timeout_seconds=int(args.timeout_seconds),
        max_request_retries=int(args.max_request_retries),
    )
    print(
        "COMPS concept properties postprocess: "
        f"method={result.manifest['method']} "
        f"renderer={result.manifest['renderer']} "
        f"reasoning_effort={result.manifest['reasoning_effort']} "
        f"prompts={len(result.prompts)} "
        f"documents={len(result.documents)} "
        f"train_words={result.manifest['counts']['train_words']} "
        f"output_documents={result.outputs['documents']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
