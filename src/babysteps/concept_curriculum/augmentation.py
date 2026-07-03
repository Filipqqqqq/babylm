"""Deterministic BabyLM-style augmentation for concept-curriculum documents."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from babysteps.material_world.fireworks_client import (
    DEFAULT_FIREWORKS_CHAT_URL,
    FireworksChatClient,
    load_api_key,
)
from babysteps.paths import expand_env_vars, resolve_path


DEFAULT_STYLES = (
    "caregiver_qa",
    "prediction_qa",
    "what_object_qa",
    "who_agent_qa",
    "speaker_perspective_who_qa",
    "possessive_deixis_who_qa",
    "where_location_qa",
    "fragment_answer",
    "state_tracking_dialogue",
)
DEFAULT_POSITIVE_ONLY_MARKERS = (
    "no",
    "not",
    "never",
    "wrong",
    "instead",
    "rather than",
    "as opposed to",
    "unlike",
    "avoid",
    "prevent",
    "keep from",
)
DEFAULT_WORD_PATTERN = r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*"
DEFAULT_FIREWORKS_MOT_CHI_MODEL = "accounts/fireworks/models/qwen3p6-plus"
NO_THINK_SUFFIX = "/no_think"
SPEAKER_TAG_PATTERN = re.compile(r"\*[A-Z]{2,4}:")
STYLE_ASSIGNMENT_ALL_PER_SOURCE = "all_per_source"
STYLE_ASSIGNMENT_BALANCED_QUOTA = "balanced_quota"
CHILD_ANSWER_MODE_NONE = "none"
CHILD_ANSWER_MODE_BALANCED_QUOTA = "balanced_quota"
DEFAULT_CHILD_ANSWER_MODES = (
    "fragment",
    "short_clause",
    "explanatory_clause",
)
DEFAULT_AGENT_POOL = (
    "Mommy",
    "Daddy",
    "Nana",
    "Grandpa",
    "Mia",
    "Jared",
    "Sarah",
    "David",
    "the teacher",
    "the child",
    "the helper",
    "the gardener",
    "the mailman",
)
SPEAKER_PERSPECTIVE_WHO_STYLE = "speaker_perspective_who_qa"
SPEAKER_PERSPECTIVE_ROLES = ("child", "caregiver")
POSSESSIVE_DEIXIS_WHO_STYLE = "possessive_deixis_who_qa"
DEFAULT_DEICTIC_AGENT_POOL = (
    {
        "child_who_answer": "my friend",
        "caregiver_followup_subject": "your friend",
    },
    {
        "child_who_answer": "my brother",
        "caregiver_followup_subject": "your brother",
    },
    {
        "child_who_answer": "my sister",
        "caregiver_followup_subject": "your sister",
    },
)
PHYSICAL_TARGET_FRAME_PRESENT_PROCESS = "present_process_observation"
PHYSICAL_TARGET_FRAME_PAST_EVENT = "past_event_result"
PHYSICAL_TARGET_FRAME_GENERIC = "generic_process"
PHYSICAL_BASE_PROCESS_TARGETS = frozenset(
    {
        "fall",
        "float",
        "grow",
        "rise",
        "roll",
        "shrink",
        "sink",
        "slide",
        "slip",
    }
)
PHYSICAL_PRESENT_FINITE_PROCESS_TARGETS = frozenset(
    {
        "falls",
        "floats",
        "grows",
        "rises",
        "rolls",
        "shrinks",
        "sinks",
        "slides",
        "slips",
    }
)
PHYSICAL_PRESENT_PROGRESSIVE_PHRASE_TARGETS = frozenset(
    {
        "decreasing speed",
        "increasing speed",
        "losing speed",
        "slowing down",
        "speeding up",
    }
)
PHYSICAL_PAST_EVENT_TARGETS = frozenset(
    {
        "accelerated",
        "decelerated",
        "fell",
        "floated",
        "grew",
        "moved faster",
        "moved slower",
        "rolled",
        "rose",
        "sank",
        "shrank",
        "shrunk",
        "slid",
        "slipped",
        "slowed down",
        "sped up",
        "went faster",
        "went slower",
    }
)
MATERIAL_AGENT_ACTIONS = frozenset(
    {
        "bend",
        "crumple",
        "fold",
        "grip",
        "hang",
        "pour",
        "press",
        "rip",
        "squeeze",
        "stir",
        "tap",
        "tear",
    }
)
MOT_CHI_STYLE_GUIDANCE = {
    "caregiver_qa": (
        "Caregiver asks what happened or what can happen; child gives a short "
        "answer; caregiver confirms with the rule evidence."
    ),
    "prediction_qa": (
        "Caregiver asks what will happen next; child predicts with a short "
        "answer; caregiver confirms with visible evidence."
    ),
    "what_object_qa": (
        "Caregiver asks what object or material they used or watched; child "
        "answers with a noun phrase; caregiver links it to the event."
    ),
    "who_agent_qa": (
        "Caregiver asks who used or watched the object; child answers with a "
        "simple third-person person name or noun phrase; caregiver repeats "
        "that exact person in the next caregiver turn and links the person to "
        "the event."
    ),
    "speaker_perspective_who_qa": (
        "Caregiver asks who used or watched the object; child answers with "
        "speaker-relative I/you; caregiver maps the pronoun to the other "
        "speaker's perspective in the next turn and links that speaker to the "
        "event."
    ),
    "possessive_deixis_who_qa": (
        "Caregiver asks who used or watched the object; child answers with a "
        "child-relative possessive noun phrase such as my brother; caregiver "
        "maps it to the caregiver's perspective, such as your brother, in the "
        "next turn and links that same person to the event."
    ),
    "where_location_qa": (
        "Caregiver asks where the object or material is; child gives a short "
        "location; caregiver links the location to the event."
    ),
    "fragment_answer": (
        "Caregiver uses a very short prompt; child answers with a fragment; "
        "caregiver gives a short evidence sentence."
    ),
    "state_tracking_dialogue": (
        "Caregiver sets up the scene, asks what happened, child answers, and "
        "caregiver closes with a state/process tracking sentence."
    ),
}
MOT_CHI_CHILD_ANSWER_MODE_GUIDANCE = {
    "fragment": "Child answer is a fragment or bare phrase, about 1 to 4 words.",
    "short_clause": "Child answer is one short child-like clause.",
    "explanatory_clause": (
        "Child answer includes one simple reason or result clause, still child-like "
        "and under about 12 words."
    ),
}


@dataclass(frozen=True)
class StyleAssignmentConfig:
    strategy: str


@dataclass(frozen=True)
class ChildAnswerModeConfig:
    strategy: str
    modes: tuple[str, ...]


@dataclass(frozen=True)
class RepairConfig:
    enabled: bool
    max_rounds: int


@dataclass(frozen=True)
class RendererConfig:
    kind: str
    model_id: str | None
    url: str
    api_key: str | None
    api_key_file: str | None
    documents_per_request: int
    temperature: float
    max_tokens: int
    timeout_seconds: int
    max_request_retries: int
    initial_backoff_seconds: float
    max_backoff_seconds: float
    invalid_json_retries: int
    reasoning_effort: str | bool | None
    reasoning_history: str | None
    resume_from_checkpoints: bool


@dataclass(frozen=True)
class AugmentationConfig:
    name: str
    input_dir: Path
    output_dir: Path
    seed: int
    scenario_plans: str
    source_documents: str
    include_audit_status: tuple[str, ...]
    styles: tuple[str, ...]
    renderer: RendererConfig
    max_source_documents: int | None
    max_augmented_documents: int | None
    word_pattern: str
    min_words: int
    max_words: int
    positive_only: bool
    positive_only_markers: tuple[str, ...]
    require_question_mark: bool
    required_speaker_tags: tuple[str, ...]
    max_turns: int | None
    agent_pool: tuple[str, ...]
    deictic_agent_pool: tuple[dict[str, str], ...]
    style_assignment: StyleAssignmentConfig
    child_answer_modes: ChildAnswerModeConfig
    repair: RepairConfig


@dataclass(frozen=True)
class AugmentationResult:
    config: AugmentationConfig
    documents: tuple[dict[str, Any], ...]
    audit_report: dict[str, Any]
    manifest: dict[str, Any]
    outputs: dict[str, Path]


class RegexTermMatcher:
    """Small term matcher with basic English inflection support."""

    def __init__(self, word_pattern: str = DEFAULT_WORD_PATTERN) -> None:
        self.word_pattern = word_pattern
        self._word_regex = re.compile(word_pattern)

    def word_count(self, text: str) -> int:
        return len(self._word_regex.findall(text))

    def contains(self, text: str, term: str) -> bool:
        term = term.strip()
        if not term:
            return True
        return any(self._contains_exact(text, form) for form in self.inflected_forms(term))

    def _contains_exact(self, text: str, term: str) -> bool:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        return re.search(rf"(?<!\w){escaped}(?!\w)", text, flags=re.IGNORECASE) is not None

    def inflected_forms(self, term: str) -> tuple[str, ...]:
        if not re.fullmatch(r"[A-Za-z]+", term):
            return (term,)
        forms = {term}
        lower = term.lower()
        if lower.endswith("y") and len(term) > 1 and lower[-2] not in "aeiou":
            forms.add(term[:-1] + "ies")
        else:
            forms.add(term + "s")
        if lower.endswith("e"):
            forms.add(term + "d")
            forms.add(term[:-1] + "ing")
        else:
            forms.add(term + "ed")
            forms.add(term + "ing")
        if len(term) >= 3 and lower[-1] not in "aeiouwxy" and lower[-2] in "aeiou":
            forms.add(term + term[-1] + "ed")
            forms.add(term + term[-1] + "ing")
        return tuple(sorted(forms, key=lambda value: (value != term, value)))


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def augmentation_config_from_mapping(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
) -> AugmentationConfig:
    if not isinstance(mapping, dict):
        raise TypeError("Augmentation config must be a mapping.")

    input_dir = resolve_path(mapping["input_dir"], base_dir=base_dir)
    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    source = mapping.get("source") or {}
    selection = mapping.get("selection") or {}
    audit = mapping.get("audit") or {}
    word_count = mapping.get("word_count") or {}
    renderer = _renderer_config_from_mapping(mapping.get("renderer") or {})
    agent_pool = tuple(str(value) for value in mapping.get("agent_pool", DEFAULT_AGENT_POOL))
    if not agent_pool:
        raise ValueError("agent_pool must contain at least one entry.")
    deictic_agent_pool = _deictic_agent_pool_from_mapping(mapping.get("deictic_agent_pool"))
    style_assignment = _style_assignment_config_from_mapping(mapping.get("style_assignment"))
    child_answer_modes = _child_answer_mode_config_from_mapping(mapping.get("child_answer_modes"))
    repair = _repair_config_from_mapping(mapping.get("repair"))

    styles = tuple(_style_id(style) for style in mapping.get("styles", DEFAULT_STYLES))
    if not styles:
        raise ValueError("Augmentation config requires at least one style.")

    include_status = tuple(str(value) for value in selection.get("include_audit_status", ("pass", "warn")))
    max_source_documents = selection.get("max_source_documents")
    max_augmented_documents = selection.get("max_augmented_documents")

    return AugmentationConfig(
        name=str(mapping["name"]),
        input_dir=input_dir,
        output_dir=output_dir,
        seed=int(mapping.get("seed", 42)),
        scenario_plans=str(source.get("scenario_plans", "scenario_plans.jsonl")),
        source_documents=str(source.get("documents", "documents_audited.jsonl")),
        include_audit_status=include_status,
        styles=styles,
        renderer=renderer,
        max_source_documents=(
            None if max_source_documents is None else int(max_source_documents)
        ),
        max_augmented_documents=(
            None if max_augmented_documents is None else int(max_augmented_documents)
        ),
        word_pattern=str(word_count.get("pattern", DEFAULT_WORD_PATTERN)),
        min_words=int(audit.get("min_words", 8)),
        max_words=int(audit.get("max_words", 90)),
        positive_only=bool(audit.get("positive_only", True)),
        positive_only_markers=tuple(
            str(value)
            for value in audit.get("positive_only_markers", DEFAULT_POSITIVE_ONLY_MARKERS)
        ),
        require_question_mark=bool(audit.get("require_question_mark", True)),
        required_speaker_tags=tuple(
            str(value) for value in audit.get("required_speaker_tags", ("*MOT:", "*CHI:"))
        ),
        max_turns=(None if audit.get("max_turns") is None else int(audit["max_turns"])),
        agent_pool=agent_pool,
        deictic_agent_pool=deictic_agent_pool,
        style_assignment=style_assignment,
        child_answer_modes=child_answer_modes,
        repair=repair,
    )


def _deictic_agent_pool_from_mapping(value: Any) -> tuple[dict[str, str], ...]:
    raw_items = DEFAULT_DEICTIC_AGENT_POOL if value is None else value
    pairs: list[dict[str, str]] = []
    for item in raw_items:
        if isinstance(item, str):
            child = item.strip()
            caregiver = _caregiver_possessive_surface(child)
        elif isinstance(item, dict):
            child = str(
                item.get("child_who_answer")
                or item.get("child_agent")
                or item.get("child_surface")
                or ""
            ).strip()
            caregiver = str(
                item.get("caregiver_followup_subject")
                or item.get("caregiver_agent")
                or item.get("caregiver_surface")
                or ""
            ).strip()
        else:
            raise TypeError("deictic_agent_pool entries must be strings or mappings.")
        if not child or not caregiver:
            raise ValueError(
                "deictic_agent_pool entries require child_who_answer and "
                "caregiver_followup_subject."
            )
        pairs.append(
            {
                "child_who_answer": child,
                "caregiver_followup_subject": caregiver,
            }
        )
    if not pairs:
        raise ValueError("deictic_agent_pool must contain at least one entry.")
    return tuple(pairs)


def _caregiver_possessive_surface(child_surface: str) -> str:
    clean = child_surface.strip()
    lower = clean.lower()
    if lower.startswith("my "):
        return "your " + clean[3:]
    if lower.startswith("our "):
        return "your " + clean[4:]
    return clean


def _style_assignment_config_from_mapping(value: Any) -> StyleAssignmentConfig:
    if value is None:
        return StyleAssignmentConfig(strategy=STYLE_ASSIGNMENT_ALL_PER_SOURCE)
    if isinstance(value, str):
        mapping = {"strategy": value}
    elif isinstance(value, dict):
        mapping = value
    else:
        raise TypeError("style_assignment must be a mapping or string.")

    strategy = str(
        mapping.get("strategy") or mapping.get("type") or STYLE_ASSIGNMENT_ALL_PER_SOURCE
    ).strip().lower()
    if strategy not in {STYLE_ASSIGNMENT_ALL_PER_SOURCE, STYLE_ASSIGNMENT_BALANCED_QUOTA}:
        raise ValueError(f"Unsupported style_assignment strategy: {strategy!r}")
    return StyleAssignmentConfig(strategy=strategy)


def _child_answer_mode_config_from_mapping(value: Any) -> ChildAnswerModeConfig:
    if value is None:
        return ChildAnswerModeConfig(strategy=CHILD_ANSWER_MODE_NONE, modes=())
    if isinstance(value, str):
        mapping = {"strategy": value}
    elif isinstance(value, dict):
        mapping = value
    else:
        raise TypeError("child_answer_modes must be a mapping or string.")

    strategy = str(
        mapping.get("strategy") or mapping.get("type") or CHILD_ANSWER_MODE_BALANCED_QUOTA
    ).strip().lower()
    if strategy not in {CHILD_ANSWER_MODE_NONE, CHILD_ANSWER_MODE_BALANCED_QUOTA}:
        raise ValueError(f"Unsupported child_answer_modes strategy: {strategy!r}")

    raw_modes = mapping.get("modes", DEFAULT_CHILD_ANSWER_MODES if strategy != CHILD_ANSWER_MODE_NONE else ())
    modes = tuple(str(value).strip() for value in raw_modes if str(value).strip())
    if strategy != CHILD_ANSWER_MODE_NONE and not modes:
        raise ValueError("child_answer_modes requires at least one mode.")
    return ChildAnswerModeConfig(strategy=strategy, modes=modes)


def _repair_config_from_mapping(value: Any) -> RepairConfig:
    if value is None:
        return RepairConfig(enabled=False, max_rounds=1)
    if isinstance(value, bool):
        return RepairConfig(enabled=value, max_rounds=1)
    if isinstance(value, dict):
        enabled = bool(value.get("enabled", False))
        max_rounds = max(0, int(value.get("max_rounds", 1)))
        return RepairConfig(enabled=enabled, max_rounds=max_rounds)
    raise TypeError("repair must be a mapping or boolean.")


def _renderer_config_from_mapping(value: Any) -> RendererConfig:
    if isinstance(value, str):
        mapping = {"type": value}
    elif isinstance(value, dict):
        mapping = value
    else:
        raise TypeError("renderer must be a mapping or string.")

    kind = str(mapping.get("type") or mapping.get("kind") or "deterministic").strip().lower()
    if kind in {"qwen", "qwen_fireworks", "fireworks_mot_chi"}:
        kind = "fireworks"
    if kind not in {"deterministic", "fireworks"}:
        raise ValueError(f"Unsupported augmentation renderer type: {kind!r}")

    model_id = mapping.get("model_id") or mapping.get("model")
    if model_id is None and kind == "fireworks":
        model_id = DEFAULT_FIREWORKS_MOT_CHI_MODEL

    return RendererConfig(
        kind=kind,
        model_id=None if model_id is None else str(model_id),
        url=str(mapping.get("url") or DEFAULT_FIREWORKS_CHAT_URL),
        api_key=None if mapping.get("api_key") is None else str(mapping["api_key"]),
        api_key_file=(
            None if mapping.get("api_key_file") is None else str(mapping["api_key_file"])
        ),
        documents_per_request=max(1, int(mapping.get("documents_per_request", 8))),
        temperature=float(mapping.get("temperature", 0.35)),
        max_tokens=int(mapping.get("max_tokens", 8192)),
        timeout_seconds=int(mapping.get("timeout_seconds", 600)),
        max_request_retries=int(mapping.get("max_request_retries", 4)),
        initial_backoff_seconds=float(mapping.get("initial_backoff_seconds", 1.0)),
        max_backoff_seconds=float(mapping.get("max_backoff_seconds", 16.0)),
        invalid_json_retries=max(0, int(mapping.get("invalid_json_retries", 1))),
        reasoning_effort=mapping.get("reasoning_effort", "none"),
        reasoning_history=(
            None
            if mapping.get("reasoning_history") is None
            else str(mapping.get("reasoning_history"))
        ),
        resume_from_checkpoints=bool(mapping.get("resume_from_checkpoints", True)),
    )


def _style_id(style: Any) -> str:
    if isinstance(style, str):
        return style
    if isinstance(style, dict):
        return str(style["id"])
    raise TypeError("Style entries must be strings or mappings with an id.")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected JSON object in {path}.")
                rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def build_document_augmentations(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
    generator: Any | None = None,
) -> AugmentationResult:
    config = augmentation_config_from_mapping(mapping, base_dir=base_dir)
    builder = DocumentAugmenter(
        matcher=RegexTermMatcher(config.word_pattern),
        renderer=create_augmentation_renderer(config.renderer, generator=generator),
    )
    return builder.build(config, config_path=config_path)


class DeterministicAugmentationRenderer:
    """Local rule-based renderer for deterministic augmentation runs."""

    def render(
        self,
        specs: list[dict[str, Any]],
        config: AugmentationConfig,
    ) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for request in build_augmentation_render_requests(specs, config):
            spec = request["spec"]
            style = str(request["augmentation_style"])
            assigned_agent = request.get("assigned_agent")
            child_answer_mode = request.get("child_answer_mode")
            speaker_perspective = request.get("speaker_perspective")
            deictic_agent = request.get("deictic_agent")
            documents.append(
                _document_from_rendered_text(
                    spec,
                    style,
                    render_augmented_text(
                        spec,
                        style,
                        assigned_agent=None if assigned_agent is None else str(assigned_agent),
                        child_answer_mode=(
                            None if child_answer_mode is None else str(child_answer_mode)
                        ),
                        speaker_perspective=(
                            None
                            if speaker_perspective is None
                            else dict(speaker_perspective)
                        ),
                        deictic_agent=(
                            None if deictic_agent is None else dict(deictic_agent)
                        ),
                    ),
                    assigned_agent=None if assigned_agent is None else str(assigned_agent),
                    child_answer_mode=(
                        None if child_answer_mode is None else str(child_answer_mode)
                    ),
                    speaker_perspective=(
                        None if speaker_perspective is None else dict(speaker_perspective)
                    ),
                    deictic_agent=(
                        None if deictic_agent is None else dict(deictic_agent)
                    ),
                )
            )
        return documents

    def repair(
        self,
        documents: list[dict[str, Any]],
        *,
        specs_by_id: dict[str, dict[str, Any]],
        config: AugmentationConfig,
        repair_round: int,
    ) -> list[dict[str, Any]]:
        repaired = []
        for document in documents:
            spec = specs_by_id[str(document["source_id"])]
            style = str(document["augmentation_style"])
            repaired.append(
                _document_from_repaired_text(
                    document,
                    render_augmented_text(
                        spec,
                        style,
                        assigned_agent=(
                            None
                            if document.get("assigned_agent") is None
                            else str(document["assigned_agent"])
                        ),
                        child_answer_mode=(
                            None
                            if document.get("child_answer_mode") is None
                            else str(document["child_answer_mode"])
                        ),
                        speaker_perspective=(
                            None
                            if document.get("speaker_perspective") is None
                            else dict(document["speaker_perspective"])
                        ),
                        deictic_agent=(
                            None
                            if document.get("deictic_agent") is None
                            else dict(document["deictic_agent"])
                        ),
                    ),
                    repair_round=repair_round,
                    repair_request_fingerprint=_deterministic_repair_fingerprint(
                        document,
                        repair_round=repair_round,
                    ),
                )
            )
        return repaired


def create_augmentation_renderer(
    renderer_config: RendererConfig,
    *,
    generator: Any | None = None,
) -> Any:
    if renderer_config.kind == "deterministic":
        return DeterministicAugmentationRenderer()
    if renderer_config.kind == "fireworks":
        return FireworksMotChiAugmentationRenderer(renderer_config, generator=generator)
    raise ValueError(f"Unsupported augmentation renderer type: {renderer_config.kind!r}")


class DocumentAugmenter:
    """Render positive BabyLM-style variants and audit required source rules."""

    def __init__(self, *, matcher: RegexTermMatcher, renderer: Any | None = None) -> None:
        self.matcher = matcher
        self.renderer = renderer or DeterministicAugmentationRenderer()

    def build(
        self,
        config: AugmentationConfig,
        *,
        config_path: Path | None = None,
    ) -> AugmentationResult:
        specs = read_jsonl(config.input_dir / config.scenario_plans)
        source_documents = read_jsonl(config.input_dir / config.source_documents)
        allowed_ids = self._allowed_source_ids(source_documents, config)
        selected_specs = [spec for spec in specs if str(spec.get("id")) in allowed_ids]
        if config.max_source_documents is not None:
            selected_specs = selected_specs[: config.max_source_documents]

        augmented: list[dict[str, Any]] = []
        specs_by_id = {str(spec.get("id")): spec for spec in selected_specs}
        for document in self.renderer.render(selected_specs, config):
            source_id = str(document.get("source_id") or "")
            if source_id not in specs_by_id:
                raise ValueError(
                    f"Rendered document {document.get('id')!r} has unknown source_id {source_id!r}."
                )
            document.update(self.audit_document(document, spec=specs_by_id[source_id], config=config))
            augmented.append(document)

        if config.repair.enabled and config.repair.max_rounds > 0:
            augmented = self._repair_failed_documents(
                augmented,
                specs_by_id=specs_by_id,
                config=config,
            )

        outputs = {
            "train": config.output_dir / "train.txt",
            "augmented_documents": config.output_dir / "augmented_documents.jsonl",
            "audit_report": config.output_dir / "audit_report.json",
            "manifest": config.output_dir / "augmentation_manifest.json",
            "config": config.output_dir / "augmentation_config.yaml",
            "render_checkpoints": config.output_dir / "render_checkpoints",
            "render_failures": config.output_dir / "render_failures",
            "repair_checkpoints": config.output_dir / "repair_checkpoints",
            "repair_failures": config.output_dir / "repair_failures",
        }
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if config_path is not None:
            shutil.copyfile(config_path, outputs["config"])
        else:
            write_json(_config_record(config), outputs["config"])

        audit_report = build_audit_report(augmented)
        train_count = write_train_text(augmented, outputs["train"])
        manifest = build_manifest(
            config=config,
            source_specs=len(specs),
            source_documents=len(source_documents),
            selected_specs=len(selected_specs),
            augmented_documents=len(augmented),
            train_documents=train_count,
            audit_report=audit_report,
            outputs=outputs,
            config_path=config_path,
        )

        write_jsonl(augmented, outputs["augmented_documents"])
        write_json(audit_report, outputs["audit_report"])
        write_json(manifest, outputs["manifest"])

        return AugmentationResult(
            config=config,
            documents=tuple(augmented),
            audit_report=audit_report,
            manifest=manifest,
            outputs=outputs,
        )

    def _repair_failed_documents(
        self,
        documents: list[dict[str, Any]],
        *,
        specs_by_id: dict[str, dict[str, Any]],
        config: AugmentationConfig,
    ) -> list[dict[str, Any]]:
        if not hasattr(self.renderer, "repair"):
            raise ValueError("Configured renderer does not support repair.")

        repaired_documents = list(documents)
        for repair_round in range(1, config.repair.max_rounds + 1):
            failed = [
                document
                for document in repaired_documents
                if str(document.get("audit_status")) == "fail"
            ]
            if not failed:
                break

            repairs = self.renderer.repair(
                failed,
                specs_by_id=specs_by_id,
                config=config,
                repair_round=repair_round,
            )
            repairs_by_id = {str(document.get("id")): document for document in repairs}
            next_documents = []
            for document in repaired_documents:
                repaired = repairs_by_id.get(str(document.get("id")))
                if repaired is None:
                    next_documents.append(document)
                    continue

                source_id = str(repaired.get("source_id") or "")
                if source_id not in specs_by_id:
                    raise ValueError(
                        f"Repaired document {repaired.get('id')!r} has unknown "
                        f"source_id {source_id!r}."
                    )
                repaired.update(
                    self.audit_document(
                        repaired,
                        spec=specs_by_id[source_id],
                        config=config,
                    )
                )
                next_documents.append(repaired)
            repaired_documents = next_documents
        return repaired_documents

    def _allowed_source_ids(
        self,
        source_documents: list[dict[str, Any]],
        config: AugmentationConfig,
    ) -> set[str]:
        allowed = set(config.include_audit_status)
        ids = {
            str(row.get("id") or row.get("scenario_id"))
            for row in source_documents
            if str(row.get("audit_status", "pass")) in allowed
        }
        ids.discard("")
        if not ids:
            raise ValueError(
                "No source documents matched include_audit_status="
                f"{sorted(config.include_audit_status)}."
            )
        return ids

    def audit_document(
        self,
        document: dict[str, Any],
        *,
        spec: dict[str, Any],
        config: AugmentationConfig,
    ) -> dict[str, Any]:
        text = str(document.get("text", "")).strip()
        flags: list[dict[str, str]] = []
        if not text:
            flags.append(_flag("error", "empty_text", "Augmented document text is empty."))

        words = self.matcher.word_count(text)
        if words < config.min_words:
            flags.append(
                _flag(
                    "warning",
                    "too_short",
                    f"Document has {words} words; minimum is {config.min_words}.",
                )
            )
        if words > config.max_words:
            flags.append(
                _flag(
                    "warning",
                    "too_long",
                    f"Document has {words} words; maximum is {config.max_words}.",
                )
            )

        if config.positive_only:
            marker_hits = [
                marker
                for marker in config.positive_only_markers
                if self.matcher.contains(text, marker)
            ]
            if marker_hits:
                flags.append(
                    _flag(
                        "error",
                        "positive_only_marker",
                        "Positive-only document includes marker(s): "
                        + ", ".join(sorted(marker_hits)),
                    )
                )

        if config.require_question_mark and "?" not in text:
            flags.append(_flag("error", "missing_question", "QA-style document has no question mark."))

        for tag in config.required_speaker_tags:
            if tag not in text:
                flags.append(
                    _flag("error", "missing_speaker_tag", f"Missing required speaker tag {tag!r}.")
                )

        if config.max_turns is not None:
            turns = len(SPEAKER_TAG_PATTERN.findall(text))
            if turns > config.max_turns:
                flags.append(
                    _flag(
                        "warning",
                        "too_many_turns",
                        f"Document has {turns} speaker turns; maximum is {config.max_turns}.",
                    )
                )

        flags.extend(self._audit_who_agent_binding(text, document))
        flags.extend(self._audit_speaker_perspective_binding(text, document))
        flags.extend(self._audit_deictic_agent_binding(text, document))
        flags.extend(self._audit_physical_who_target_frame(text, document, spec))
        flags.extend(self._audit_source_rule_terms(text, spec))

        if any(flag["severity"] == "error" for flag in flags):
            status = "fail"
        elif flags:
            status = "warn"
        else:
            status = "pass"
        return {
            "audit_status": status,
            "audit_flags": flags,
            "word_count": words,
        }

    def _audit_source_rule_terms(self, text: str, spec: dict[str, Any]) -> list[dict[str, str]]:
        if _is_material_spec(spec):
            return self._audit_material_terms(text, spec)
        return self._audit_physical_terms(text, spec)

    def _audit_who_agent_binding(
        self,
        text: str,
        document: dict[str, Any],
    ) -> list[dict[str, str]]:
        if str(document.get("augmentation_style") or "") != "who_agent_qa":
            return []
        assigned_agent = str(document.get("assigned_agent") or "").strip()
        if not assigned_agent:
            return []

        post_child = text.split("*CHI:", 1)
        if len(post_child) < 2 or "*MOT:" not in post_child[1]:
            return [
                _flag(
                    "error",
                    "missing_who_agent_followup",
                    "Who-agent QA has no caregiver follow-up after the child answer.",
                )
            ]

        caregiver_followup = post_child[1].split("*MOT:", 1)[1]
        if self.matcher.contains(caregiver_followup, assigned_agent):
            return []
        return [
            _flag(
                "error",
                "missing_who_agent_binding",
                "Caregiver follow-up does not repeat the assigned who-agent: "
                f"{assigned_agent}",
            )
        ]

    def _audit_speaker_perspective_binding(
        self,
        text: str,
        document: dict[str, Any],
    ) -> list[dict[str, str]]:
        if str(document.get("augmentation_style") or "") != SPEAKER_PERSPECTIVE_WHO_STYLE:
            return []
        perspective = document.get("speaker_perspective")
        if not isinstance(perspective, dict):
            return [
                _flag(
                    "error",
                    "missing_speaker_perspective",
                    "Speaker-perspective who QA is missing speaker_perspective metadata.",
                )
            ]

        child_answer = str(perspective.get("child_who_answer") or "").strip()
        caregiver_subject = str(perspective.get("caregiver_followup_subject") or "").strip()
        if not child_answer or not caregiver_subject:
            return [
                _flag(
                    "error",
                    "missing_speaker_perspective_pronoun",
                    "Speaker-perspective metadata is missing required pronouns.",
                )
            ]

        segments = text.split("*CHI:", 1)
        if len(segments) < 2 or "*MOT:" not in segments[1]:
            return [
                _flag(
                    "error",
                    "missing_speaker_perspective_followup",
                    "Speaker-perspective QA has no caregiver follow-up after the child answer.",
                )
            ]

        child_segment, caregiver_followup = segments[1].split("*MOT:", 1)
        flags = []
        if not self.matcher.contains(child_segment, child_answer):
            flags.append(
                _flag(
                    "error",
                    "missing_child_perspective_pronoun",
                    "Child answer does not use expected speaker-relative pronoun: "
                    f"{child_answer}",
                )
            )
        if not self.matcher.contains(caregiver_followup, caregiver_subject):
            flags.append(
                _flag(
                    "error",
                    "missing_caregiver_perspective_pronoun",
                    "Caregiver follow-up does not use mapped speaker-relative pronoun: "
                    f"{caregiver_subject}",
                )
            )
        return flags

    def _audit_deictic_agent_binding(
        self,
        text: str,
        document: dict[str, Any],
    ) -> list[dict[str, str]]:
        if str(document.get("augmentation_style") or "") != POSSESSIVE_DEIXIS_WHO_STYLE:
            return []
        deictic_agent = document.get("deictic_agent")
        if not isinstance(deictic_agent, dict):
            return [
                _flag(
                    "error",
                    "missing_deictic_agent",
                    "Possessive deixis QA is missing deictic_agent metadata.",
                )
            ]

        child_answer = str(deictic_agent.get("child_who_answer") or "").strip()
        caregiver_subject = str(
            deictic_agent.get("caregiver_followup_subject") or ""
        ).strip()
        if not child_answer or not caregiver_subject:
            return [
                _flag(
                    "error",
                    "missing_deictic_agent_mapping",
                    "Deictic-agent metadata is missing child/caregiver surfaces.",
                )
            ]

        segments = text.split("*CHI:", 1)
        if len(segments) < 2 or "*MOT:" not in segments[1]:
            return [
                _flag(
                    "error",
                    "missing_deictic_agent_followup",
                    "Possessive deixis QA has no caregiver follow-up after the child answer.",
                )
            ]

        child_segment, caregiver_followup = segments[1].split("*MOT:", 1)
        flags = []
        if not self.matcher.contains(child_segment, child_answer):
            flags.append(
                _flag(
                    "error",
                    "missing_child_deictic_agent",
                    "Child answer does not use expected child-relative agent: "
                    f"{child_answer}",
                )
            )
        if not self.matcher.contains(caregiver_followup, caregiver_subject):
            flags.append(
                _flag(
                    "error",
                    "missing_caregiver_deictic_agent",
                    "Caregiver follow-up does not use mapped caregiver-relative agent: "
                    f"{caregiver_subject}",
                )
            )
        if (
            child_answer.casefold() != caregiver_subject.casefold()
            and self.matcher.contains(caregiver_followup, child_answer)
        ):
            flags.append(
                _flag(
                    "error",
                    "caregiver_repeats_child_deictic_agent",
                    "Caregiver follow-up repeats the child-relative agent instead of "
                    f"the mapped caregiver-relative agent: {caregiver_subject}",
                )
            )
        return flags

    def _audit_physical_who_target_frame(
        self,
        text: str,
        document: dict[str, Any],
        spec: dict[str, Any],
    ) -> list[dict[str, str]]:
        if _is_material_spec(spec):
            return []
        style = str(document.get("augmentation_style") or "")
        if style not in {
            "who_agent_qa",
            SPEAKER_PERSPECTIVE_WHO_STYLE,
            POSSESSIVE_DEIXIS_WHO_STYLE,
        }:
            return []
        target = str(spec.get("lexical_target") or "").strip()
        if target.casefold() not in PHYSICAL_PRESENT_FINITE_PROCESS_TARGETS:
            return []

        segments = text.split("*CHI:", 1)
        if len(segments) < 2 or "*MOT:" not in segments[1]:
            return []
        caregiver_followup = segments[1].split("*MOT:", 1)[1]
        target_pattern = re.escape(target).replace(r"\ ", r"\s+")
        if re.search(
            rf"\b(?:watched|saw|sees|used|made)\b[^.?!;]*"
            rf"(?<!\w){target_pattern}(?!\w)",
            caregiver_followup,
            flags=re.IGNORECASE,
        ):
            return [
                _flag(
                    "error",
                    "present_target_embedded_in_agent_clause",
                    "Present-process lexical target should be in a separate "
                    f"object-subject observation sentence: {target}",
                )
            ]
        return []

    def _audit_material_terms(self, text: str, spec: dict[str, Any]) -> list[dict[str, str]]:
        flags: list[dict[str, str]] = []
        materials = _string_list(spec.get("required_materials")) or _string_list(spec.get("material"))
        missing_materials = [term for term in materials if not self.matcher.contains(text, term)]
        if missing_materials:
            flags.append(
                _flag(
                    "error",
                    "missing_required_material",
                    "Missing required material(s): " + ", ".join(sorted(missing_materials)),
                )
            )

        affordances = _string_list(spec.get("required_affordances"))
        missing_affordances = [
            term for term in affordances if not self.matcher.contains(text, term)
        ]
        if missing_affordances:
            flags.append(
                _flag(
                    "error",
                    "missing_required_affordance",
                    "Missing required affordance(s): "
                    + ", ".join(sorted(missing_affordances)),
                )
            )
        flags.extend(self._audit_material_affordance_frames(text, spec))
        return flags

    def _audit_material_affordance_frames(
        self,
        text: str,
        spec: dict[str, Any],
    ) -> list[dict[str, str]]:
        bad_frames = []
        for verb in _material_affordance_verbs(spec):
            escaped = re.escape(verb)
            if re.search(rf"(?<!\w)who\s+{escaped}(?!\w)", text, flags=re.IGNORECASE):
                bad_frames.append(f"who {verb}")
            pronoun_pattern = (
                rf"(?<!\w)(?:he|she|it|this|that)\s+{escaped}(?!\w)"
            )
            for match in re.finditer(pronoun_pattern, text, flags=re.IGNORECASE):
                prefix = text[max(0, match.start() - 24) : match.start()].casefold()
                if re.search(
                    r"\b(?:saw|see|sees|watch|watched|watching|made|make|"
                    r"makes|let|lets|help|helps|helped)\s+$",
                    prefix,
                ):
                    continue
                bad_frames.append(f"third-person subject + {verb}")

        if not bad_frames:
            return []
        return [
            _flag(
                "error",
                "bad_material_affordance_verb_frame",
                "Bare material affordance verbs need a grammatical frame "
                "(can/will/did/does/to + verb, or I/you/we/they + verb): "
                + ", ".join(sorted(set(bad_frames))),
            )
        ]

    def _audit_physical_terms(self, text: str, spec: dict[str, Any]) -> list[dict[str, str]]:
        flags: list[dict[str, str]] = []
        object_terms = _string_list(spec.get("object"))
        missing_objects = [term for term in object_terms if not self.matcher.contains(text, term)]
        if missing_objects:
            flags.append(
                _flag(
                    "error",
                    "missing_required_object",
                    "Missing required object(s): " + ", ".join(sorted(missing_objects)),
                )
            )

        lexical_target = str(spec.get("lexical_target") or "").strip()
        if lexical_target and not self.matcher.contains(text, lexical_target):
            flags.append(
                _flag(
                    "error",
                    "missing_lexical_target",
                    f"Missing assigned lexical target: {lexical_target}",
                )
            )

        for key, code in (
            ("required_cause_terms", "missing_required_cause"),
            ("required_process_terms", "missing_required_process"),
            ("required_outcome_terms", "missing_required_outcome"),
        ):
            required_terms = _string_list(spec.get(key))
            missing = [term for term in required_terms if not self.matcher.contains(text, term)]
            if missing:
                flags.append(
                    _flag("error", code, f"Missing {key}: " + ", ".join(sorted(missing)))
                )
        return flags


class FireworksMotChiAugmentationRenderer:
    """Qwen/Fireworks renderer for child-directed MOT/CHI dialogue."""

    def __init__(self, renderer_config: RendererConfig, *, generator: Any | None = None) -> None:
        self.renderer_config = renderer_config
        self.generator = generator or self._build_fireworks_generator(renderer_config)

    def _build_fireworks_generator(self, renderer_config: RendererConfig) -> FireworksChatClient:
        if not renderer_config.model_id:
            raise ValueError("Fireworks augmentation renderer requires renderer.model_id.")
        api_key = load_api_key(
            api_key=renderer_config.api_key,
            api_key_file=renderer_config.api_key_file,
        )
        return FireworksChatClient(
            model=renderer_config.model_id,
            api_key=api_key,
            url=renderer_config.url,
            timeout_seconds=renderer_config.timeout_seconds,
            max_request_retries=renderer_config.max_request_retries,
            initial_backoff_seconds=renderer_config.initial_backoff_seconds,
            max_backoff_seconds=renderer_config.max_backoff_seconds,
            reasoning_effort=renderer_config.reasoning_effort,
            reasoning_history=renderer_config.reasoning_history,
        )

    def render(
        self,
        specs: list[dict[str, Any]],
        config: AugmentationConfig,
    ) -> list[dict[str, Any]]:
        requests = build_mot_chi_render_requests(specs, config)
        documents: list[dict[str, Any]] = []
        batch_size = self.renderer_config.documents_per_request
        checkpoint_dir = (
            config.output_dir / "render_checkpoints"
            if self.renderer_config.resume_from_checkpoints
            else None
        )
        failure_dir = config.output_dir / "render_failures"
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            documents.extend(
                self._render_batch(
                    batch=batch,
                    config=config,
                    start=start,
                    checkpoint_dir=checkpoint_dir,
                    failure_dir=failure_dir,
                )
            )
        return documents

    def repair(
        self,
        documents: list[dict[str, Any]],
        *,
        specs_by_id: dict[str, dict[str, Any]],
        config: AugmentationConfig,
        repair_round: int,
    ) -> list[dict[str, Any]]:
        requests = [
            _mot_chi_repair_request_for_document(
                document,
                spec=specs_by_id[str(document["source_id"])],
                config=config,
                repair_round=repair_round,
            )
            for document in documents
        ]
        repaired: list[dict[str, Any]] = []
        batch_size = self.renderer_config.documents_per_request
        checkpoint_dir = (
            config.output_dir / "repair_checkpoints"
            if self.renderer_config.resume_from_checkpoints
            else None
        )
        failure_dir = config.output_dir / "repair_failures"
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            repaired.extend(
                self._repair_batch(
                    batch=batch,
                    config=config,
                    repair_round=repair_round,
                    start=start,
                    checkpoint_dir=checkpoint_dir,
                    failure_dir=failure_dir,
                )
            )
        return repaired

    def _repair_batch(
        self,
        *,
        batch: list[dict[str, Any]],
        config: AugmentationConfig,
        repair_round: int,
        start: int,
        checkpoint_dir: Path | None,
        failure_dir: Path,
    ) -> list[dict[str, Any]]:
        checkpointed = _load_mot_chi_repair_checkpoint(
            checkpoint_dir,
            repair_round=repair_round,
            start=start,
            batch=batch,
        )
        if checkpointed is not None:
            return checkpointed

        prompt = build_mot_chi_repair_prompt(batch, config=config)
        last_error: Exception | None = None
        for attempt in range(1, self.renderer_config.invalid_json_retries + 2):
            raw = _call_mot_chi_generator(
                self.generator,
                prompt=prompt,
                renderer_config=self.renderer_config,
                stage="augmentation_mot_chi_repair",
            )
            try:
                repaired = _normalize_mot_chi_repair_documents(
                    _json_items_from_text(raw),
                    batch=batch,
                    renderer_config=self.renderer_config,
                )
            except ValueError as exc:
                last_error = exc
                _write_mot_chi_failure(
                    failure_dir=failure_dir,
                    start=start,
                    batch=batch,
                    attempt=attempt,
                    error=exc,
                    raw=raw,
                )
                continue

            _write_mot_chi_repair_checkpoint(
                checkpoint_dir,
                repair_round=repair_round,
                start=start,
                batch=batch,
                documents=repaired,
            )
            return repaired

        raise ValueError(
            f"MOT/CHI repair failed for round {repair_round} batch starting at "
            f"{start}: {last_error}"
        )

    def _render_batch(
        self,
        *,
        batch: list[dict[str, Any]],
        config: AugmentationConfig,
        start: int,
        checkpoint_dir: Path | None,
        failure_dir: Path,
    ) -> list[dict[str, Any]]:
        checkpointed = _load_mot_chi_checkpoint(checkpoint_dir, start=start, batch=batch)
        if checkpointed is not None:
            return checkpointed

        prompt = build_mot_chi_render_prompt(batch, config=config)
        last_error: Exception | None = None
        for attempt in range(1, self.renderer_config.invalid_json_retries + 2):
            raw = _call_mot_chi_generator(
                self.generator,
                prompt=prompt,
                renderer_config=self.renderer_config,
                stage="augmentation_mot_chi_render",
            )
            try:
                documents = _normalize_mot_chi_documents(
                    _json_items_from_text(raw),
                    batch=batch,
                    renderer_config=self.renderer_config,
                )
            except ValueError as exc:
                last_error = exc
                _write_mot_chi_failure(
                    failure_dir=failure_dir,
                    start=start,
                    batch=batch,
                    attempt=attempt,
                    error=exc,
                    raw=raw,
                )
                continue

            _write_mot_chi_checkpoint(
                checkpoint_dir,
                start=start,
                batch=batch,
                documents=documents,
            )
            return documents

        raise ValueError(
            f"MOT/CHI render failed for batch starting at {start}: {last_error}"
        )


class DeterministicMotChiMockGenerator:
    """Local Fireworks-shaped generator for dry runs and tests."""

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
        marker = (
            "REPAIR_ITEMS_JSON:"
            if "REPAIR_ITEMS_JSON:" in prompt
            else "SCENARIO_SPECS_JSON:"
        )
        payload = _json_from_text_after_marker(prompt, marker=marker)
        items = []
        for row in payload.get("items", []):
            spec = dict(row.get("source_spec") or {})
            style = str(row.get("augmentation_style") or "caregiver_qa")
            assigned_agent = row.get("assigned_agent")
            child_answer_mode = row.get("child_answer_mode")
            speaker_perspective = row.get("speaker_perspective")
            deictic_agent = row.get("deictic_agent")
            items.append(
                {
                    "id": str(row["id"]),
                    "text": render_augmented_text(
                        spec,
                        style,
                        assigned_agent=None if assigned_agent is None else str(assigned_agent),
                        child_answer_mode=(
                            None if child_answer_mode is None else str(child_answer_mode)
                        ),
                        speaker_perspective=(
                            None
                            if speaker_perspective is None
                            else dict(speaker_perspective)
                        ),
                        deictic_agent=(
                            None if deictic_agent is None else dict(deictic_agent)
                        ),
                    ),
                }
            )
        return json.dumps({"items": items})


def build_mot_chi_render_requests(
    specs: list[dict[str, Any]],
    config: AugmentationConfig,
) -> list[dict[str, Any]]:
    return [
        _mot_chi_request_for_spec(
            request["spec"],
            str(request["augmentation_style"]),
            config,
            child_answer_mode=(
                None
                if request.get("child_answer_mode") is None
                else str(request["child_answer_mode"])
            ),
            speaker_perspective=(
                None
                if request.get("speaker_perspective") is None
                else dict(request["speaker_perspective"])
            ),
            deictic_agent=(
                None
                if request.get("deictic_agent") is None
                else dict(request["deictic_agent"])
            ),
        )
        for request in build_augmentation_render_requests(specs, config)
    ]


def build_augmentation_render_requests(
    specs: list[dict[str, Any]],
    config: AugmentationConfig,
) -> list[dict[str, Any]]:
    requests = _style_assigned_requests(specs, config)
    requests = _assign_child_answer_modes(requests, config)
    requests = _assign_speaker_perspectives(requests, config)
    requests = _assign_deictic_agents(requests, config)
    for request in requests:
        assigned_agent = assigned_agent_for_spec(
            request["spec"],
            str(request["augmentation_style"]),
            config,
        )
        if assigned_agent is not None:
            request["assigned_agent"] = assigned_agent
        if request.get("speaker_perspective") is None:
            speaker_perspective = assigned_speaker_perspective_for_spec(
                request["spec"],
                str(request["augmentation_style"]),
                config,
            )
            if speaker_perspective is not None:
                request["speaker_perspective"] = speaker_perspective
        if request.get("deictic_agent") is None:
            deictic_agent = assigned_deictic_agent_for_spec(
                request["spec"],
                str(request["augmentation_style"]),
                config,
            )
            if deictic_agent is not None:
                request["deictic_agent"] = deictic_agent
    return requests


def _style_assigned_requests(
    specs: list[dict[str, Any]],
    config: AugmentationConfig,
) -> list[dict[str, Any]]:
    if config.style_assignment.strategy == STYLE_ASSIGNMENT_ALL_PER_SOURCE:
        return _all_styles_per_source_requests(specs, config)
    if config.style_assignment.strategy == STYLE_ASSIGNMENT_BALANCED_QUOTA:
        return _balanced_style_quota_requests(specs, config)
    raise ValueError(
        f"Unsupported style assignment strategy: {config.style_assignment.strategy!r}"
    )


def _all_styles_per_source_requests(
    specs: list[dict[str, Any]],
    config: AugmentationConfig,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for spec in specs:
        for style in config.styles:
            requests.append({"spec": spec, "augmentation_style": style})
            if (
                config.max_augmented_documents is not None
                and len(requests) >= config.max_augmented_documents
            ):
                return requests
    return requests


def _balanced_style_quota_requests(
    specs: list[dict[str, Any]],
    config: AugmentationConfig,
) -> list[dict[str, Any]]:
    if not specs:
        return []
    request_count = len(specs)
    if config.max_augmented_documents is not None:
        request_count = min(request_count, config.max_augmented_documents)
    selected_specs = specs[:request_count]
    styles = _balanced_label_sequence(
        config.styles,
        len(selected_specs),
        seed=config.seed,
        salt="style_assignment",
    )
    return [
        {"spec": spec, "augmentation_style": style}
        for spec, style in zip(selected_specs, styles, strict=True)
    ]


def _assign_child_answer_modes(
    requests: list[dict[str, Any]],
    config: AugmentationConfig,
) -> list[dict[str, Any]]:
    if config.child_answer_modes.strategy == CHILD_ANSWER_MODE_NONE:
        return requests
    if config.child_answer_modes.strategy != CHILD_ANSWER_MODE_BALANCED_QUOTA:
        raise ValueError(
            "Unsupported child answer mode strategy: "
            f"{config.child_answer_modes.strategy!r}"
        )

    assigned = [dict(request) for request in requests]
    indices_by_style: dict[str, list[int]] = {}
    for index, request in enumerate(assigned):
        style = str(request["augmentation_style"])
        indices_by_style.setdefault(style, []).append(index)

    mode_labels_by_style = _balanced_child_answer_modes_by_style(
        indices_by_style,
        modes=config.child_answer_modes.modes,
        seed=config.seed,
    )
    for style, labels in mode_labels_by_style.items():
        indices = indices_by_style[style]
        labels = _seeded_shuffle(
            tuple(labels),
            seed=config.seed,
            salt=f"child_answer_mode:{style}:order",
        )
        for index, mode in zip(indices, labels, strict=True):
            assigned[index]["child_answer_mode"] = mode
    return assigned


def _assign_speaker_perspectives(
    requests: list[dict[str, Any]],
    config: AugmentationConfig,
) -> list[dict[str, Any]]:
    assigned = [dict(request) for request in requests]
    indices = [
        index
        for index, request in enumerate(assigned)
        if str(request["augmentation_style"]) == SPEAKER_PERSPECTIVE_WHO_STYLE
    ]
    roles = _balanced_label_sequence(
        SPEAKER_PERSPECTIVE_ROLES,
        len(indices),
        seed=config.seed,
        salt="speaker_perspective_role",
    )
    for index, role in zip(indices, roles, strict=True):
        assigned[index]["speaker_perspective"] = _speaker_perspective_for_role(role)
    return assigned


def _assign_deictic_agents(
    requests: list[dict[str, Any]],
    config: AugmentationConfig,
) -> list[dict[str, Any]]:
    assigned = [dict(request) for request in requests]
    indices = [
        index
        for index, request in enumerate(assigned)
        if str(request["augmentation_style"]) == POSSESSIVE_DEIXIS_WHO_STYLE
    ]
    labels = _balanced_label_sequence(
        tuple(str(index) for index in range(len(config.deictic_agent_pool))),
        len(indices),
        seed=config.seed,
        salt="deictic_agent",
    )
    for index, label in zip(indices, labels, strict=True):
        assigned[index]["deictic_agent"] = dict(config.deictic_agent_pool[int(label)])
    return assigned


def _balanced_child_answer_modes_by_style(
    indices_by_style: dict[str, list[int]],
    *,
    modes: tuple[str, ...],
    seed: int,
) -> dict[str, list[str]]:
    total = sum(len(indices) for indices in indices_by_style.values())
    remaining = _balanced_label_counts(
        modes,
        total,
        seed=seed,
        salt="child_answer_mode:global",
    )
    result: dict[str, list[str]] = {}
    style_order = _seeded_shuffle(
        tuple(sorted(indices_by_style)),
        seed=seed,
        salt="child_answer_mode:style_order",
    )
    for style in style_order:
        local_counts = {mode: 0 for mode in modes}
        labels = []
        for step in range(len(indices_by_style[style])):
            candidates = tuple(mode for mode in modes if remaining[mode] > 0)
            if not candidates:
                raise ValueError("Child answer mode quota was exhausted unexpectedly.")
            tie_order = {
                mode: index
                for index, mode in enumerate(
                    _seeded_shuffle(
                        candidates,
                        seed=seed,
                        salt=f"child_answer_mode:{style}:{step}",
                    )
                )
            }
            mode = min(
                candidates,
                key=lambda value: (local_counts[value], -remaining[value], tie_order[value]),
            )
            labels.append(mode)
            local_counts[mode] += 1
            remaining[mode] -= 1
        result[style] = labels
    return result


def _balanced_label_sequence(
    labels: tuple[str, ...],
    count: int,
    *,
    seed: int,
    salt: str,
) -> tuple[str, ...]:
    if count <= 0:
        return ()
    if not labels:
        raise ValueError("Cannot build a balanced quota from an empty label list.")

    counts = _balanced_label_counts(labels, count, seed=seed, salt=salt)

    sequence: list[str] = []
    for label in labels:
        sequence.extend([label] * counts[label])
    return tuple(_seeded_shuffle(tuple(sequence), seed=seed, salt=salt))


def _balanced_label_counts(
    labels: tuple[str, ...],
    count: int,
    *,
    seed: int,
    salt: str,
) -> dict[str, int]:
    if not labels:
        raise ValueError("Cannot build balanced counts from an empty label list.")
    base_count = count // len(labels)
    remainder = count % len(labels)
    counts = {label: base_count for label in labels}
    for label in _seeded_shuffle(labels, seed=seed, salt=f"{salt}:remainder")[:remainder]:
        counts[label] += 1
    return counts


def _seeded_shuffle(values: tuple[str, ...], *, seed: int, salt: str) -> list[str]:
    shuffled = list(values)
    random.Random(_seeded_int(seed, salt)).shuffle(shuffled)
    return shuffled


def _seeded_int(seed: int, salt: str) -> int:
    payload = f"{seed}\0{salt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_mot_chi_render_prompt(
    requests: list[dict[str, Any]],
    *,
    config: AugmentationConfig,
) -> str:
    payload = {
        "task": "render_child_directed_mot_chi_dialogue",
        "constraints": {
            "word_count_range": [config.min_words, config.max_words],
            "positive_only": config.positive_only,
            "avoid_negative_or_correction_markers": list(config.positive_only_markers),
            "required_speaker_tags": list(config.required_speaker_tags),
            "max_turns": config.max_turns,
            "one_document_per_item": True,
            "plain_text_only": True,
        },
        "style_guidance": {
            style: MOT_CHI_STYLE_GUIDANCE.get(style, "Short child-directed dialogue.")
            for style in config.styles
        },
        "child_answer_mode_guidance": {
            mode: MOT_CHI_CHILD_ANSWER_MODE_GUIDANCE.get(
                mode,
                "Child answer follows the named mode while staying short and natural.",
            )
            for mode in config.child_answer_modes.modes
        },
        "items": [
            {
                "id": request["id"],
                "source_id": request["source_id"],
                "augmentation_style": request["augmentation_style"],
                "source_domain": request["source_domain"],
                "source_document_type": request["source_document_type"],
                **(
                    {"child_answer_mode": request["child_answer_mode"]}
                    if request.get("child_answer_mode") is not None
                    else {}
                ),
                **(
                    {"assigned_agent": request["assigned_agent"]}
                    if request.get("assigned_agent") is not None
                    else {}
                ),
                **(
                    {"speaker_perspective": request["speaker_perspective"]}
                    if request.get("speaker_perspective") is not None
                    else {}
                ),
                **(
                    {"deictic_agent": request["deictic_agent"]}
                    if request.get("deictic_agent") is not None
                    else {}
                ),
                "must_include_exact_terms": request["must_include_exact_terms"],
                "source_spec": request["source_spec"],
            }
            for request in requests
        ],
    }
    return (
        "Rewrite each item as one short child-directed dialogue document.\n"
        "Return only valid JSON with shape {\"items\": [{\"id\": string, \"text\": string}]}.\n"
        "Hard rules:\n"
        "- Every text must contain *MOT: and *CHI: speaker tags.\n"
        "- Every text must include at least one question mark.\n"
        "- Use 2 to 5 speaker turns.\n"
        "- If child_answer_mode is present, follow it exactly: fragment means a 1-to-4-word fragment or bare phrase; short_clause means one short child-like clause; explanatory_clause means one simple reason/result clause under about 12 words.\n"
        "- If child_answer_mode is absent, keep the child's answer short, often a fragment.\n"
        "- Keep all dialogue on one line; no metadata, source tags, bullets, or explanations.\n"
        "- Do not mention benchmark, test, score, metadata, target, scenario, source_id, or hidden labels.\n"
        "- Do not write negative correction or contrast wording, including avoid, prevent, or keep from.\n"
        "- If assigned_agent is present, use that exact third-person string as the who-question answer. The next caregiver *MOT: turn after the child answer must repeat the exact assigned_agent and connect that person to the watched/used event. Do not replace it, rename it, gender it, switch it to I/you, or invent another person.\n"
        "- If speaker_perspective is present, use speaker-relative pronouns exactly: the *CHI: who-answer must use speaker_perspective.child_who_answer, and the next caregiver *MOT: turn must use speaker_perspective.caregiver_followup_subject while connecting that same speaker to the watched/used event. This mapping is intentional: child 'I' maps to caregiver 'you', and child 'you' maps to caregiver 'I'.\n"
        "- If deictic_agent is present, use child-relative possessive wording exactly: the *CHI: who-answer must use deictic_agent.child_who_answer, and the next caregiver *MOT: turn must use deictic_agent.caregiver_followup_subject while connecting that same person to the watched/used event. This mapping is intentional: child 'my brother' maps to caregiver 'your brother'; do not make the caregiver repeat child-relative 'my'.\n"
        "- For physical who-question styles with source_spec.lexical_target_frame='present_process_observation', use a present-time question such as 'Who is watching the X now?' or 'Who sees the X now?'. In the caregiver follow-up, bind the person first, for example 'Yes, your brother is watching.' Then put the exact lexical_target and required process terms in a separate object-subject observation sentence, for example 'The X rolls...' or 'The X is rolling...'. Do not write ungrammatical past-frame clauses like 'watched the X rolls' or 'saw the X grows'.\n"
        "- For material affordance terms that are bare verbs, use grammatical base-verb frames such as can/will/did/does/to + verb, or I/you/we/they + verb. Do not write ungrammatical clauses like 'Who bend...', 'He press...', or 'It spill...'.\n"
        "- Include every string in must_include_exact_terms exactly as written. Do not substitute synonyms, tense changes, plural changes, or other inflections. If both \"floats\" and \"floating\" appear in must_include_exact_terms, include both exact strings.\n"
        "- The final *MOT: turn must include every string in must_include_exact_terms exactly as written. Use that final caregiver turn as the compact evidence sentence if needed.\n"
        "- Include every required material, affordance, object, lexical target, cause term, process term, and outcome term from source_spec.\n"
        "- When lexical_target is present, use that exact lexical_target naturally.\n"
        "- Keep the physical/material rule true; do not invent a different event.\n\n"
        "SCENARIO_SPECS_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_mot_chi_repair_prompt(
    requests: list[dict[str, Any]],
    *,
    config: AugmentationConfig,
) -> str:
    payload = {
        "task": "repair_child_directed_mot_chi_dialogue",
        "constraints": {
            "word_count_range": [config.min_words, config.max_words],
            "positive_only": config.positive_only,
            "avoid_negative_or_correction_markers": list(config.positive_only_markers),
            "required_speaker_tags": list(config.required_speaker_tags),
            "max_turns": config.max_turns,
            "one_document_per_item": True,
            "plain_text_only": True,
        },
        "items": [
            {
                "id": request["id"],
                "source_id": request["source_id"],
                "augmentation_style": request["augmentation_style"],
                "child_answer_mode": request.get("child_answer_mode"),
                "assigned_agent": request.get("assigned_agent"),
                "speaker_perspective": request.get("speaker_perspective"),
                "deictic_agent": request.get("deictic_agent"),
                "existing_text": request["existing_text"],
                "audit_flags": request["audit_flags"],
                "missing_terms": request["missing_terms"],
                "must_include_exact_terms": request["must_include_exact_terms"],
                "source_spec": request["source_spec"],
            }
            for request in requests
        ],
    }
    return (
        "Repair each failed child-directed dialogue with the smallest possible edit.\n"
        "Return only valid JSON with shape {\"items\": [{\"id\": string, \"text\": string}]}.\n"
        "Hard rules:\n"
        "- Keep the same id, style, event, source object/material, speaker tags, and one-line format.\n"
        "- Preserve *MOT: and *CHI: tags, keep at least one question mark, and use 2 to 5 speaker turns.\n"
        "- Fix only the listed audit_flags; do not rewrite unrelated content.\n"
        "- Do not add negative correction or contrast wording, including avoid, prevent, or keep from.\n"
        "- Include every string in missing_terms exactly as written.\n"
        "- Include every string in must_include_exact_terms exactly as written. Do not substitute synonyms, tense changes, plural changes, or other inflections.\n"
        "- The final *MOT: turn must include every string in must_include_exact_terms exactly as written. It may be a compact evidence sentence.\n"
        "- If assigned_agent is present, keep that exact third-person string as the who-answer and repeat it in the next caregiver *MOT: event sentence.\n"
        "- If speaker_perspective is present, keep the *CHI: who-answer as speaker_perspective.child_who_answer and the next caregiver *MOT: subject as speaker_perspective.caregiver_followup_subject.\n"
        "- If deictic_agent is present, keep the *CHI: who-answer as deictic_agent.child_who_answer and the next caregiver *MOT: subject as deictic_agent.caregiver_followup_subject; do not make the caregiver repeat child-relative 'my'.\n"
        "- For physical who-question repairs with source_spec.lexical_target_frame='present_process_observation', keep the who/person binding in a present-time clause, then put the exact lexical_target in a separate object-subject observation sentence. Do not repair into clauses like 'watched the X rolls' or 'saw the X grows'.\n"
        "- For material affordance repairs, keep exact bare affordance verbs in grammatical frames such as can/will/did/does/to + verb, or I/you/we/they + verb. Do not repair into clauses like 'Who bend...', 'He press...', or 'It spill...'.\n"
        "- Keep the physical/material rule true; do not invent a different event.\n\n"
        "REPAIR_ITEMS_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _mot_chi_request_for_spec(
    spec: dict[str, Any],
    style: str,
    config: AugmentationConfig,
    *,
    child_answer_mode: str | None = None,
    speaker_perspective: dict[str, str] | None = None,
    deictic_agent: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_id = str(spec.get("id"))
    request = {
        "id": _augmentation_document_id(source_id, style, child_answer_mode),
        "source_id": source_id,
        "source_domain": str(spec.get("domain", "")),
        "source_document_type": str(spec.get("document_type", "")),
        "augmentation_style": style,
        "source_spec": _mot_chi_prompt_spec(spec),
        "must_include_exact_terms": _must_include_exact_terms(spec),
    }
    if child_answer_mode is not None:
        request["child_answer_mode"] = child_answer_mode
    if speaker_perspective is not None:
        request["speaker_perspective"] = speaker_perspective
    if deictic_agent is not None:
        request["deictic_agent"] = deictic_agent
    assigned_agent = assigned_agent_for_spec(spec, style, config)
    if assigned_agent is not None:
        request["assigned_agent"] = assigned_agent
    if request.get("deictic_agent") is None:
        assigned_deictic_agent = assigned_deictic_agent_for_spec(spec, style, config)
        if assigned_deictic_agent is not None:
            request["deictic_agent"] = assigned_deictic_agent
    return request


def _mot_chi_repair_request_for_document(
    document: dict[str, Any],
    *,
    spec: dict[str, Any],
    config: AugmentationConfig,
    repair_round: int,
) -> dict[str, Any]:
    style = str(document["augmentation_style"])
    request = _mot_chi_request_for_spec(
        spec,
        style,
        config,
        child_answer_mode=(
            None
            if document.get("child_answer_mode") is None
            else str(document["child_answer_mode"])
        ),
        speaker_perspective=(
            None
            if document.get("speaker_perspective") is None
            else dict(document["speaker_perspective"])
        ),
        deictic_agent=(
            None if document.get("deictic_agent") is None else dict(document["deictic_agent"])
        ),
    )
    if document.get("assigned_agent") is not None:
        request["assigned_agent"] = str(document["assigned_agent"])
    if document.get("deictic_agent") is not None:
        request["deictic_agent"] = dict(document["deictic_agent"])
    request.update(
        {
            "existing_text": str(document.get("text") or ""),
            "audit_flags": list(document.get("audit_flags") or []),
            "missing_terms": _missing_terms_from_audit_flags(document.get("audit_flags") or []),
            "repair_round": repair_round,
            "original_document": document,
        }
    )
    return request


def _missing_terms_from_audit_flags(flags: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for flag in flags:
        message = str(flag.get("message") or "")
        if ":" not in message:
            continue
        _, value = message.split(":", 1)
        terms.extend(term.strip() for term in value.split(",") if term.strip())
    return _unique_nonempty_terms(terms)


def _mot_chi_prompt_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if _is_material_spec(spec):
        keys = (
            "id",
            "domain",
            "document_type",
            "material",
            "required_materials",
            "required_affordances",
            "target_affordance",
            "material_class",
            "action",
            "actions",
        )
    else:
        keys = (
            "id",
            "domain",
            "document_type",
            "object",
            "object_type",
            "material_or_surface",
            "contact_surface",
            "geometry_or_medium",
            "bridge_clause",
            "evidence_clause",
            "same_sentence_evidence_terms",
            "lexical_target",
            "expected_process",
            "cause_evidence",
            "outcome_evidence",
            "required_cause_terms",
            "required_process_terms",
            "required_outcome_terms",
        )
    record = {
        key: spec[key]
        for key in keys
        if key in spec and spec[key] not in (None, "", [], {})
    }
    if not _is_material_spec(spec):
        record["lexical_target_frame"] = _physical_target_frame(spec)
    return record


def _physical_target_frame(spec: dict[str, Any]) -> str:
    target = str(spec.get("lexical_target") or "").strip().casefold()
    if not target:
        return PHYSICAL_TARGET_FRAME_GENERIC
    if (
        target in PHYSICAL_BASE_PROCESS_TARGETS
        or target in PHYSICAL_PRESENT_FINITE_PROCESS_TARGETS
        or target in PHYSICAL_PRESENT_PROGRESSIVE_PHRASE_TARGETS
        or target.endswith("ing")
    ):
        return PHYSICAL_TARGET_FRAME_PRESENT_PROCESS
    if target in PHYSICAL_PAST_EVENT_TARGETS:
        return PHYSICAL_TARGET_FRAME_PAST_EVENT
    return PHYSICAL_TARGET_FRAME_GENERIC


def _must_include_exact_terms(spec: dict[str, Any]) -> list[str]:
    if _is_material_spec(spec):
        terms = [
            *_string_list(spec.get("required_materials")),
            *_string_list(spec.get("material")),
            *_string_list(spec.get("target_affordance")),
            *_string_list(spec.get("required_affordances")),
        ]
    else:
        terms = [
            *_string_list(spec.get("object")),
            *_string_list(spec.get("lexical_target")),
            *_string_list(spec.get("required_cause_terms")),
            *_string_list(spec.get("required_process_terms")),
            *_string_list(spec.get("required_outcome_terms")),
        ]
    return _unique_nonempty_terms(terms)


def _material_affordance_verbs(spec: dict[str, Any]) -> list[str]:
    terms = [
        *_string_list(spec.get("required_affordances")),
        *_string_list(spec.get("target_affordance")),
    ]
    return [
        term
        for term in _unique_nonempty_terms(terms)
        if re.fullmatch(r"[A-Za-z]+", term)
    ]


def _unique_nonempty_terms(terms: list[str]) -> list[str]:
    seen = set()
    result = []
    for term in terms:
        clean = str(term).strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def assigned_agent_for_spec(
    spec: dict[str, Any],
    style: str,
    config: AugmentationConfig,
) -> str | None:
    if style != "who_agent_qa":
        return None
    payload = f"{config.seed}\0{spec.get('id')}\0{style}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(config.agent_pool)
    return config.agent_pool[index]


def assigned_deictic_agent_for_spec(
    spec: dict[str, Any],
    style: str,
    config: AugmentationConfig,
) -> dict[str, str] | None:
    if style != POSSESSIVE_DEIXIS_WHO_STYLE:
        return None
    payload = f"{config.seed}\0{spec.get('id')}\0{style}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(
        config.deictic_agent_pool
    )
    return dict(config.deictic_agent_pool[index])


def assigned_speaker_perspective_for_spec(
    spec: dict[str, Any],
    style: str,
    config: AugmentationConfig,
) -> dict[str, str] | None:
    if style != SPEAKER_PERSPECTIVE_WHO_STYLE:
        return None
    payload = f"{config.seed}\0{spec.get('id')}\0{style}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(
        SPEAKER_PERSPECTIVE_ROLES
    )
    role = SPEAKER_PERSPECTIVE_ROLES[index]
    if role == "child":
        return {
            "role": "child",
            "child_who_answer": "I",
            "caregiver_followup_subject": "you",
        }
    return {
        "role": "caregiver",
        "child_who_answer": "you",
        "caregiver_followup_subject": "I",
    }


def _document_from_rendered_text(
    spec: dict[str, Any],
    style: str,
    text: str,
    *,
    assigned_agent: str | None = None,
    child_answer_mode: str | None = None,
    speaker_perspective: dict[str, str] | None = None,
    deictic_agent: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_id = str(spec.get("id"))
    row = {
        "id": _augmentation_document_id(source_id, style, child_answer_mode),
        "source_id": source_id,
        "source_domain": str(spec.get("domain", "")),
        "source_document_type": str(spec.get("document_type", "")),
        "augmentation_style": style,
        "renderer": "deterministic",
        "text": _one_line_text(text),
    }
    if child_answer_mode is not None:
        row["child_answer_mode"] = child_answer_mode
    if assigned_agent is not None:
        row["assigned_agent"] = assigned_agent
    if speaker_perspective is not None:
        row["speaker_perspective"] = speaker_perspective
    if deictic_agent is not None:
        row["deictic_agent"] = deictic_agent
    return row


def _augmentation_document_id(
    source_id: str,
    style: str,
    child_answer_mode: str | None,
) -> str:
    if child_answer_mode is None:
        return f"{source_id}__{style}"
    return f"{source_id}__{style}__{child_answer_mode}"


def _call_mot_chi_generator(
    generator: Any,
    *,
    prompt: str,
    renderer_config: RendererConfig,
    stage: str,
) -> str:
    if NO_THINK_SUFFIX not in prompt:
        prompt = f"{prompt}\n\n{NO_THINK_SUFFIX}"
    return generator(
        prompt=prompt,
        system_prompt=_mot_chi_system_prompt(),
        temperature=renderer_config.temperature,
        max_tokens=renderer_config.max_tokens,
        stage=stage,
        response_format={"type": "json_object"},
    )


def _mot_chi_system_prompt() -> str:
    return (
        "You write short CHILDES-like *MOT:/*CHI: synthetic training dialogue. "
        "Return only valid JSON. Preserve required source terms exactly."
    )


def _normalize_mot_chi_documents(
    items: list[dict[str, Any]],
    *,
    batch: list[dict[str, Any]],
    renderer_config: RendererConfig,
) -> list[dict[str, Any]]:
    if len(items) != len(batch):
        raise ValueError(f"MOT/CHI renderer returned {len(items)} items; expected {len(batch)}.")

    expected_ids = [str(request["id"]) for request in batch]
    item_ids = [str(item.get("id") or "") for item in items]
    if all(item_ids) and set(item_ids) == set(expected_ids):
        items_by_id = {str(item["id"]): item for item in items}
        ordered_items = [items_by_id[expected_id] for expected_id in expected_ids]
    else:
        ordered_items = items

    documents = []
    for request, item in zip(batch, ordered_items, strict=True):
        text = _one_line_text(str(item.get("text") or item.get("document") or ""))
        if not text:
            raise ValueError(f"MOT/CHI item {request['id']!r} has empty text.")
        row = {
            "id": request["id"],
            "source_id": request["source_id"],
            "source_domain": request["source_domain"],
            "source_document_type": request["source_document_type"],
            "augmentation_style": request["augmentation_style"],
            "renderer": renderer_config.kind,
            "render_model_id": renderer_config.model_id,
            "render_request_fingerprint": _mot_chi_request_fingerprint(request),
            "text": text,
        }
        if request.get("child_answer_mode") is not None:
            row["child_answer_mode"] = request["child_answer_mode"]
        if request.get("assigned_agent") is not None:
            row["assigned_agent"] = request["assigned_agent"]
        if request.get("speaker_perspective") is not None:
            row["speaker_perspective"] = request["speaker_perspective"]
        if request.get("deictic_agent") is not None:
            row["deictic_agent"] = request["deictic_agent"]
        documents.append(row)
    return documents


def _normalize_mot_chi_repair_documents(
    items: list[dict[str, Any]],
    *,
    batch: list[dict[str, Any]],
    renderer_config: RendererConfig,
) -> list[dict[str, Any]]:
    if len(items) != len(batch):
        raise ValueError(f"MOT/CHI repair returned {len(items)} items; expected {len(batch)}.")

    expected_ids = [str(request["id"]) for request in batch]
    item_ids = [str(item.get("id") or "") for item in items]
    if all(item_ids) and set(item_ids) == set(expected_ids):
        items_by_id = {str(item["id"]): item for item in items}
        ordered_items = [items_by_id[expected_id] for expected_id in expected_ids]
    else:
        ordered_items = items

    repaired = []
    for request, item in zip(batch, ordered_items, strict=True):
        text = _one_line_text(str(item.get("text") or item.get("document") or ""))
        if not text:
            raise ValueError(f"MOT/CHI repair item {request['id']!r} has empty text.")
        repaired.append(
            _document_from_repaired_text(
                request["original_document"],
                text,
                repair_round=int(request["repair_round"]),
                repair_request_fingerprint=_mot_chi_repair_request_fingerprint(request),
                render_model_id=renderer_config.model_id,
            )
        )
    return repaired


def _document_from_repaired_text(
    original_document: dict[str, Any],
    text: str,
    *,
    repair_round: int,
    repair_request_fingerprint: str,
    render_model_id: str | None = None,
) -> dict[str, Any]:
    stale_keys = {"audit_status", "audit_flags", "word_count"}
    row = {
        key: value
        for key, value in original_document.items()
        if key not in stale_keys and not key.startswith("repair_original_")
    }
    row["text"] = _one_line_text(text)
    row["repair_attempted"] = True
    row["repair_round"] = repair_round
    row["repair_request_fingerprint"] = repair_request_fingerprint
    if render_model_id is not None:
        row["repair_model_id"] = render_model_id
    row["repair_original_text"] = str(original_document.get("text") or "")
    row["repair_original_audit_status"] = str(original_document.get("audit_status") or "")
    row["repair_original_audit_flags"] = list(original_document.get("audit_flags") or [])
    row["repair_original_word_count"] = original_document.get("word_count")
    return row


def _one_line_text(text: str) -> str:
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _mot_chi_request_fingerprint(request: dict[str, Any]) -> str:
    payload = {
        key: request.get(key)
        for key in (
            "id",
            "source_id",
            "augmentation_style",
            "child_answer_mode",
            "source_domain",
            "source_document_type",
            "assigned_agent",
            "speaker_perspective",
            "deictic_agent",
            "must_include_exact_terms",
            "source_spec",
        )
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _mot_chi_repair_request_fingerprint(request: dict[str, Any]) -> str:
    payload = {
        key: request.get(key)
        for key in (
            "id",
            "source_id",
            "augmentation_style",
            "child_answer_mode",
            "assigned_agent",
            "speaker_perspective",
            "deictic_agent",
            "existing_text",
            "audit_flags",
            "missing_terms",
            "must_include_exact_terms",
            "source_spec",
            "repair_round",
        )
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _deterministic_repair_fingerprint(
    document: dict[str, Any],
    *,
    repair_round: int,
) -> str:
    payload = {
        "id": document.get("id"),
        "text": document.get("text"),
        "audit_flags": document.get("audit_flags"),
        "repair_round": repair_round,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _mot_chi_checkpoint_path(
    checkpoint_dir: Path | None,
    *,
    start: int,
    batch: list[dict[str, Any]],
) -> Path | None:
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / f"batch_{start:06d}_n{len(batch):03d}.jsonl"


def _load_mot_chi_checkpoint(
    checkpoint_dir: Path | None,
    *,
    start: int,
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    path = _mot_chi_checkpoint_path(checkpoint_dir, start=start, batch=batch)
    if path is None or not path.exists():
        return None
    rows = read_jsonl(path)
    expected_ids = [str(request["id"]) for request in batch]
    expected_fingerprints = [
        _mot_chi_request_fingerprint(request)
        for request in batch
    ]
    actual_ids = [str(row.get("id") or "") for row in rows]
    actual_fingerprints = [
        str(row.get("render_request_fingerprint") or "")
        for row in rows
    ]
    if actual_ids != expected_ids or actual_fingerprints != expected_fingerprints:
        return None
    return rows


def _write_mot_chi_checkpoint(
    checkpoint_dir: Path | None,
    *,
    start: int,
    batch: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> None:
    path = _mot_chi_checkpoint_path(checkpoint_dir, start=start, batch=batch)
    if path is not None:
        write_jsonl(documents, path)


def _mot_chi_repair_checkpoint_path(
    checkpoint_dir: Path | None,
    *,
    repair_round: int,
    start: int,
    batch: list[dict[str, Any]],
) -> Path | None:
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / f"round_{repair_round:02d}_batch_{start:06d}_n{len(batch):03d}.jsonl"


def _load_mot_chi_repair_checkpoint(
    checkpoint_dir: Path | None,
    *,
    repair_round: int,
    start: int,
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    path = _mot_chi_repair_checkpoint_path(
        checkpoint_dir,
        repair_round=repair_round,
        start=start,
        batch=batch,
    )
    if path is None or not path.exists():
        return None
    rows = read_jsonl(path)
    expected_ids = [str(request["id"]) for request in batch]
    expected_fingerprints = [
        _mot_chi_repair_request_fingerprint(request)
        for request in batch
    ]
    actual_ids = [str(row.get("id") or "") for row in rows]
    actual_fingerprints = [
        str(row.get("repair_request_fingerprint") or "")
        for row in rows
    ]
    if actual_ids != expected_ids or actual_fingerprints != expected_fingerprints:
        return None
    return rows


def _write_mot_chi_repair_checkpoint(
    checkpoint_dir: Path | None,
    *,
    repair_round: int,
    start: int,
    batch: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> None:
    path = _mot_chi_repair_checkpoint_path(
        checkpoint_dir,
        repair_round=repair_round,
        start=start,
        batch=batch,
    )
    if path is not None:
        write_jsonl(documents, path)


def _write_mot_chi_failure(
    *,
    failure_dir: Path,
    start: int,
    batch: list[dict[str, Any]],
    attempt: int,
    error: Exception,
    raw: str,
) -> None:
    failure_dir.mkdir(parents=True, exist_ok=True)
    stem = f"batch_{start:06d}_n{len(batch):03d}_attempt{attempt:02d}"
    raw_path = failure_dir / f"{stem}.raw.txt"
    meta_path = failure_dir / f"{stem}.json"
    raw_path.write_text(raw, encoding="utf-8")
    write_json(
        {
            "stage": "augmentation_mot_chi_render",
            "start": start,
            "batch_size": len(batch),
            "attempt": attempt,
            "ids": [str(request["id"]) for request in batch],
            "error": str(error),
            "raw_output_path": str(raw_path),
        },
        meta_path,
    )


def _json_items_from_text(text: str) -> list[dict[str, Any]]:
    return _json_items(_json_from_text(text))


def _json_from_text_after_marker(text: str, *, marker: str) -> dict[str, Any]:
    if marker not in text:
        raise ValueError(f"Prompt did not contain marker {marker!r}.")
    value = _json_from_text(text.split(marker, 1)[1].replace(NO_THINK_SUFFIX, "").strip())
    if not isinstance(value, dict):
        raise ValueError("Marked JSON payload must be an object.")
    return value


def _json_from_text(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    starts = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
    if not starts:
        raise ValueError(f"Model output did not contain JSON: {stripped[:400]}")
    start = min(starts)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end <= start:
        raise ValueError(f"Model output contained incomplete JSON: {stripped[:400]}")
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON from model output: {stripped[:800]}") from exc


def _json_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        for key in ("items", "documents", "rows", "data"):
            if isinstance(value.get(key), list):
                items = value[key]
                break
        else:
            raise ValueError("JSON object must include an items/documents/rows/data list.")
    else:
        raise ValueError(f"Expected JSON list or object, got {type(value).__name__}.")

    rows: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            rows.append(item)
        elif isinstance(item, str):
            rows.append({"text": item})
        else:
            raise ValueError(f"JSON items must be objects or strings, got {type(item).__name__}.")
    return rows


def render_augmented_text(
    spec: dict[str, Any],
    style: str,
    *,
    assigned_agent: str | None = None,
    child_answer_mode: str | None = None,
    speaker_perspective: dict[str, str] | None = None,
    deictic_agent: dict[str, str] | None = None,
) -> str:
    if style not in DEFAULT_STYLES:
        raise ValueError(f"Unsupported augmentation style: {style!r}")
    if _is_material_spec(spec):
        return _render_material(
            spec,
            style,
            assigned_agent=assigned_agent,
            child_answer_mode=child_answer_mode,
            speaker_perspective=speaker_perspective,
            deictic_agent=deictic_agent,
        )
    return _render_physical(
        spec,
        style,
        assigned_agent=assigned_agent,
        child_answer_mode=child_answer_mode,
        speaker_perspective=speaker_perspective,
        deictic_agent=deictic_agent,
    )


def _render_material(
    spec: dict[str, Any],
    style: str,
    *,
    assigned_agent: str | None = None,
    child_answer_mode: str | None = None,
    speaker_perspective: dict[str, str] | None = None,
    deictic_agent: dict[str, str] | None = None,
) -> str:
    material = _first(spec.get("required_materials")) or str(spec.get("material") or "thing")
    affordances = _string_list(spec.get("required_affordances"))
    target = str(spec.get("target_affordance") or (affordances[0] if affordances else "change"))
    affordance_phrase = _list_phrase(affordances or [target])
    rule_sentence = _material_rule_sentence(
        material=material,
        target=target,
        affordances=affordances or [target],
    )
    event_answer = _material_event_child_answer(
        material=material,
        target=target,
        affordances=affordances or [target],
        mode=child_answer_mode,
    )

    if style == "caregiver_qa":
        return (
            f"*MOT:\twhat can happen with the {material}? "
            f"*CHI:\t{event_answer}. "
            f"*MOT:\tyes, {rule_sentence}"
        )
    if style == "prediction_qa":
        return (
            f"*MOT:\twhat will happen when we use the {material}? "
            f"*CHI:\t{event_answer}. "
            f"*MOT:\tright, {rule_sentence}"
        )
    if style == "what_object_qa":
        return (
            f"*MOT:\twhat did we use in the test? "
            f"*CHI:\t{_object_child_answer(f'the {material}', event_answer, child_answer_mode)}. "
            f"*MOT:\tyes, the {material}; {rule_sentence}"
        )
    if style == "who_agent_qa":
        agent = assigned_agent or "Mia"
        return (
            f"*MOT:\twho used the {material}? "
            f"*CHI:\t{_agent_child_answer(agent, event_answer, child_answer_mode)}. "
            f"*MOT:\tyes, {_agent_action_phrase(agent, 'used')} the {material}; {rule_sentence}"
        )
    if style == SPEAKER_PERSPECTIVE_WHO_STYLE:
        perspective = speaker_perspective or _speaker_perspective_for_role("child")
        child_subject = perspective["child_who_answer"]
        caregiver_subject = perspective["caregiver_followup_subject"]
        return (
            f"*MOT:\twho used the {material}? "
            f"*CHI:\t{_speaker_perspective_child_answer(child_subject, 'used', child_answer_mode)}. "
            f"*MOT:\tyes, {_agent_action_phrase(caregiver_subject, 'used')} the {material}; {rule_sentence}"
        )
    if style == POSSESSIVE_DEIXIS_WHO_STYLE:
        deixis = deictic_agent or _default_deictic_agent()
        child_subject = deixis["child_who_answer"]
        caregiver_subject = deixis["caregiver_followup_subject"]
        return (
            f"*MOT:\twho used the {material}? "
            f"*CHI:\t{_agent_child_answer(child_subject, event_answer, child_answer_mode)}. "
            f"*MOT:\tyes, {_agent_action_phrase(caregiver_subject, 'used')} the {material}; {rule_sentence}"
        )
    if style == "where_location_qa":
        return (
            f"*MOT:\twhere is the {material}? "
            f"*CHI:\t{_location_child_answer('on the table', event_answer, child_answer_mode)}. "
            f"*MOT:\tyes, the {material} is on the table; {rule_sentence}"
        )
    if style == "fragment_answer":
        return (
            f"*MOT:\t{material}? "
            f"*CHI:\t{event_answer}. "
            f"*MOT:\t{affordance_phrase} with the {material}."
        )
    return (
        f"*MOT:\tthe {material} is on the table. "
        f"*MOT:\tnow we use it in a small test. what happens with the {material}? "
        f"*CHI:\t{event_answer}. "
        f"*MOT:\t{rule_sentence}"
    )


def _render_physical(
    spec: dict[str, Any],
    style: str,
    *,
    assigned_agent: str | None = None,
    child_answer_mode: str | None = None,
    speaker_perspective: dict[str, str] | None = None,
    deictic_agent: dict[str, str] | None = None,
) -> str:
    obj = str(spec.get("object") or spec.get("object_type") or "thing")
    target = str(
        spec.get("lexical_target")
        or _first(spec.get("required_process_terms"))
        or spec.get("expected_process")
        or "moves"
    )
    process = _first(spec.get("required_process_terms")) or target
    cause = _evidence_phrase(
        _string_list(spec.get("required_cause_terms")) or _string_list(spec.get("cause_evidence"))
    )
    outcome = _evidence_phrase(
        _string_list(spec.get("required_outcome_terms")) or _string_list(spec.get("outcome_evidence"))
    )
    answer = _process_answer(target)
    evidence = _physical_evidence_sentence(cause=cause, outcome=outcome, process=process)
    target_frame = _physical_target_frame(spec)
    object_evidence = _physical_object_evidence_sentence(
        obj=obj,
        target=target,
        evidence=evidence,
        frame=target_frame,
    )
    location = _physical_location_phrase(spec)
    event_answer = _physical_event_child_answer(
        target=target,
        answer=answer,
        cause=cause,
        outcome=outcome,
        mode=child_answer_mode,
    )

    if style == "caregiver_qa":
        return (
            f"*MOT:\twhat happened to the {obj}? "
            f"*CHI:\t{event_answer}. "
            f"*MOT:\tyes, {answer}; {evidence}"
        )
    if style == "prediction_qa":
        return (
            f"*MOT:\twhat will happen to the {obj}? "
            f"*CHI:\t{event_answer}. "
            f"*MOT:\tyes, watch the {obj}; {evidence}"
        )
    if style == "what_object_qa":
        return (
            f"*MOT:\twhat did we watch? "
            f"*CHI:\t{_object_child_answer(f'the {obj}', event_answer, child_answer_mode)}. "
            f"*MOT:\tyes, the {obj}; {answer}; {evidence}"
        )
    if style == "who_agent_qa":
        agent = assigned_agent or "Mia"
        if target_frame == PHYSICAL_TARGET_FRAME_PRESENT_PROCESS:
            return (
                f"*MOT:\twho is watching {_definite_np(obj)} now? "
                f"*CHI:\t{_watcher_child_answer(agent, child_answer_mode)}. "
                f"*MOT:\tyes, {_watching_phrase(agent)}. {object_evidence}"
            )
        return (
            f"*MOT:\twho watched the {obj}? "
            f"*CHI:\t{_agent_child_answer(agent, event_answer, child_answer_mode)}. "
            f"*MOT:\tyes, {_agent_action_phrase(agent, 'watched')} the {obj}; {answer}; {evidence}"
        )
    if style == SPEAKER_PERSPECTIVE_WHO_STYLE:
        perspective = speaker_perspective or _speaker_perspective_for_role("child")
        child_subject = perspective["child_who_answer"]
        caregiver_subject = perspective["caregiver_followup_subject"]
        if target_frame == PHYSICAL_TARGET_FRAME_PRESENT_PROCESS:
            return (
                f"*MOT:\twho is watching {_definite_np(obj)} now? "
                f"*CHI:\t{_watcher_child_answer(child_subject, child_answer_mode)}. "
                f"*MOT:\tyes, {_watching_phrase(caregiver_subject)}. {object_evidence}"
            )
        return (
            f"*MOT:\twho watched the {obj}? "
            f"*CHI:\t{_speaker_perspective_child_answer(child_subject, 'watched', child_answer_mode)}. "
            f"*MOT:\tyes, {_agent_action_phrase(caregiver_subject, 'watched')} the {obj}; {answer}; {evidence}"
        )
    if style == POSSESSIVE_DEIXIS_WHO_STYLE:
        deixis = deictic_agent or _default_deictic_agent()
        child_subject = deixis["child_who_answer"]
        caregiver_subject = deixis["caregiver_followup_subject"]
        if target_frame == PHYSICAL_TARGET_FRAME_PRESENT_PROCESS:
            return (
                f"*MOT:\twho is watching {_definite_np(obj)} now? "
                f"*CHI:\t{_watcher_child_answer(child_subject, child_answer_mode)}. "
                f"*MOT:\tyes, {_watching_phrase(caregiver_subject)}. {object_evidence}"
            )
        return (
            f"*MOT:\twho watched the {obj}? "
            f"*CHI:\t{_agent_child_answer(child_subject, event_answer, child_answer_mode)}. "
            f"*MOT:\tyes, {_agent_action_phrase(caregiver_subject, 'watched')} the {obj}; {answer}; {evidence}"
        )
    if style == "where_location_qa":
        return (
            f"*MOT:\twhere is the {obj}? "
            f"*CHI:\t{_location_child_answer(location, event_answer, child_answer_mode)}. "
            f"*MOT:\tyes, the {obj} is {location}; {answer}; {evidence}"
        )
    if style == "fragment_answer":
        return (
            f"*MOT:\t{obj}? "
            f"*CHI:\t{event_answer}. "
            f"*MOT:\t{evidence}"
        )
    return (
        f"*MOT:\tthe {obj} is here. "
        f"*MOT:\tnow watch the {obj}. what happens? "
        f"*CHI:\t{event_answer}. "
        f"*MOT:\t{evidence}"
    )


def _physical_evidence_sentence(*, cause: str, outcome: str, process: str) -> str:
    parts = []
    if cause:
        parts.append(cause)
    if outcome:
        parts.append(outcome)
    if process:
        parts.append(f"we call that {process}")
    return ", ".join(parts) + "."


def _physical_object_evidence_sentence(
    *,
    obj: str,
    target: str,
    evidence: str,
    frame: str,
) -> str:
    target_clause = _physical_target_clause(_definite_np(obj), target, frame)
    if evidence:
        return f"{target_clause}; {evidence}"
    return f"{target_clause}."


def _physical_target_clause(subject: str, target: str, frame: str) -> str:
    clean_target = target.strip()
    lower = clean_target.casefold()
    if frame == PHYSICAL_TARGET_FRAME_PRESENT_PROCESS:
        if lower in PHYSICAL_BASE_PROCESS_TARGETS:
            return f"{subject} can {clean_target}"
        if lower.endswith("ing") or lower in PHYSICAL_PRESENT_PROGRESSIVE_PHRASE_TARGETS:
            return f"{subject} is {clean_target}"
    return f"{subject} {clean_target}"


def _material_rule_sentence(*, material: str, target: str, affordances: list[str]) -> str:
    companions = [value for value in affordances if value != target]
    if target in MATERIAL_AGENT_ACTIONS:
        sentence = f"someone can {target} the {material}"
        if companions:
            sentence += f", and the {material} can {_list_phrase(companions)}"
        return sentence + "."
    if companions:
        return (
            f"after {_action_event_phrase(companions)}, "
            f"the {material} can {target}."
        )
    return f"the {material} can {target}."


def _agent_action_phrase(agent: str, verb_past: str) -> str:
    lower = agent.strip().lower()
    if lower == "i":
        return f"I {verb_past}"
    if lower == "you":
        return f"you {verb_past}"
    return f"{agent} {verb_past}"


def _watching_phrase(agent: str) -> str:
    lower = agent.strip().lower()
    if lower == "i":
        return "I am watching"
    if lower == "you":
        return "you are watching"
    return f"{agent} is watching"


def _speaker_perspective_for_role(role: str) -> dict[str, str]:
    if role == "caregiver":
        return {
            "role": "caregiver",
            "child_who_answer": "you",
            "caregiver_followup_subject": "I",
        }
    return {
        "role": "child",
        "child_who_answer": "I",
        "caregiver_followup_subject": "you",
    }


def _default_deictic_agent() -> dict[str, str]:
    return dict(DEFAULT_DEICTIC_AGENT_POOL[0])


def _speaker_perspective_child_answer(
    subject: str,
    verb_past: str,
    mode: str | None,
) -> str:
    subject = subject.strip()
    if mode is None or mode == "fragment":
        return f"{subject} did"
    if mode == "short_clause":
        return f"{subject} {verb_past} it"
    if mode == "explanatory_clause":
        return f"{subject} {verb_past} it because {subject} looked"
    return f"{subject} did"


def _action_event_phrase(actions: list[str]) -> str:
    if not actions:
        return "a small action"
    if len(actions) == 1:
        return f"a {actions[0]}"
    return _list_phrase([f"a {actions[0]}", *actions[1:]], conjunction="or")


def _physical_location_phrase(spec: dict[str, Any]) -> str:
    for location in (
        _prepositional_location(str(spec.get("contact_surface") or "").strip()),
        _location_from_material_or_surface(str(spec.get("material_or_surface") or "").strip()),
        _location_from_bridge_clause(str(spec.get("bridge_clause") or "").strip()),
        _location_from_geometry(str(spec.get("geometry_or_medium") or "").strip()),
    ):
        if location:
            return location
    return "right here"


def _location_from_material_or_surface(value: str) -> str:
    match = re.search(
        r"\b(?P<prep>at|by|in|inside|near|on|onto|under|underwater|with)\s+"
        r"(?P<place>.+)$",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return _join_preposition(match.group("prep"), match.group("place"))


def _location_from_bridge_clause(value: str) -> str:
    if not value:
        return ""
    if re.match(r"^(?:after|as|with)\b", value, flags=re.IGNORECASE):
        return ""
    if re.match(r"^past\s+the\s+table\s+edge\b", value, flags=re.IGNORECASE):
        return "near the table edge"
    return _prepositional_location(value)


def _location_from_geometry(value: str) -> str:
    if not value:
        return ""
    base = re.split(r"\s+(?:above|with)\s+", value, maxsplit=1, flags=re.IGNORECASE)[0]
    lower = base.lower()
    if "bowl" in lower or "jar" in lower or "tank" in lower or "room" in lower:
        return _join_preposition("in", base)
    if "wall" in lower or "edge" in lower:
        return _join_preposition("near", base)
    return _prepositional_location(base)


def _prepositional_location(location: str) -> str:
    if not location:
        return ""
    if re.match(
        r"^(?:at|by|in|inside|near|on|onto|under|underwater|with)\b",
        location,
        flags=re.IGNORECASE,
    ):
        return location
    if any(word in location.lower().split() for word in ("air", "room", "tank", "water")):
        return f"in the {location}"
    return f"on the {location}"


def _join_preposition(preposition: str, place: str) -> str:
    place = place.strip()
    if not place:
        return ""
    preposition = preposition.lower()
    if preposition == "with":
        preposition = "near"
    if re.match(r"^(?:a|an|the|this|that|these|those|my|your|our)\b", place, flags=re.IGNORECASE):
        return f"{preposition} {place}"
    return f"{preposition} the {place}"


def _definite_np(label: str) -> str:
    clean = label.strip()
    if not clean:
        return "the thing"
    if re.match(r"^(?:a|an|the|this|that|these|those|my|your|our)\b", clean, flags=re.IGNORECASE):
        return clean
    return f"the {clean}"


def _process_answer(target: str) -> str:
    lower = target.lower()
    if lower.endswith("ing"):
        return f"it is {target}"
    if lower in {"float", "sink", "fall", "rise", "roll", "slide", "grow", "shrink"}:
        return f"it can {target}"
    if " " in target and not lower.startswith("it "):
        return f"it is {target}"
    return f"it {target}"


def _material_event_child_answer(
    *,
    material: str,
    target: str,
    affordances: list[str],
    mode: str | None,
) -> str:
    if mode is None or mode == "fragment":
        return target
    short = _material_short_clause(target)
    if mode == "short_clause":
        return short
    if mode == "explanatory_clause":
        return f"{short} because the {material} can {_list_phrase(affordances)}"
    return short


def _material_short_clause(target: str) -> str:
    if target in MATERIAL_AGENT_ACTIONS:
        return f"it can {target}"
    return f"it can {target}"


def _physical_event_child_answer(
    *,
    target: str,
    answer: str,
    cause: str,
    outcome: str,
    mode: str | None,
) -> str:
    if mode is None:
        return answer
    if mode == "fragment":
        return target
    if mode == "short_clause":
        return answer
    if mode == "explanatory_clause":
        reason = cause or outcome
        return f"{answer} because {reason}" if reason else answer
    return answer


def _object_child_answer(
    label: str,
    event_answer: str,
    mode: str | None,
) -> str:
    if mode is None or mode == "fragment":
        return label
    if mode == "short_clause":
        return f"{label} did"
    if mode == "explanatory_clause":
        return f"{label}, {event_answer}"
    return label


def _agent_child_answer(
    agent: str,
    event_answer: str,
    mode: str | None,
) -> str:
    if mode is None or mode == "fragment":
        return agent
    if mode == "short_clause":
        return _agent_action_phrase(agent, "did")
    if mode == "explanatory_clause":
        return f"{agent}, {event_answer}"
    return agent


def _watcher_child_answer(
    agent: str,
    mode: str | None,
) -> str:
    if mode is None or mode == "fragment":
        return agent
    if mode == "short_clause":
        return _watching_phrase(agent)
    if mode == "explanatory_clause":
        lower = agent.strip().lower()
        if lower == "i":
            return "I see it"
        if lower == "you":
            return "you see it"
        return f"{agent} sees it"
    return agent


def _location_child_answer(
    location: str,
    event_answer: str,
    mode: str | None,
) -> str:
    if mode is None or mode == "fragment":
        return location
    if mode == "short_clause":
        return f"it is {location}"
    if mode == "explanatory_clause":
        return f"it is {location}, {event_answer}"
    return location


def _is_material_spec(spec: dict[str, Any]) -> bool:
    domain = str(spec.get("domain", "")).lower()
    return (
        "material" in domain
        or bool(spec.get("required_affordances"))
        or bool(spec.get("required_materials"))
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _first(value: Any) -> str:
    values = _string_list(value)
    return values[0] if values else ""


def _list_phrase(values: list[str], *, conjunction: str = "and") -> str:
    values = [value for value in values if value]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} {conjunction} {values[1]}"
    return ", ".join(values[:-1]) + f", {conjunction} {values[-1]}"


def _evidence_phrase(values: list[str]) -> str:
    if len(values) == 2 and values == ["farther", "apart"]:
        return "farther apart"
    return _list_phrase(values)


def _flag(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def build_audit_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    flag_counts: dict[str, int] = {}
    by_style: dict[str, dict[str, int]] = {}
    by_child_answer_mode: dict[str, dict[str, int]] = {}
    by_style_and_child_answer_mode: dict[str, dict[str, dict[str, int]]] = {}
    repair_counts = {
        "attempted": 0,
        "pass": 0,
        "warn": 0,
        "fail": 0,
    }
    for row in rows:
        status = str(row.get("audit_status", "pass"))
        counts[status] = counts.get(status, 0) + 1
        if row.get("repair_attempted"):
            repair_counts["attempted"] += 1
            repair_counts[status] = repair_counts.get(status, 0) + 1
        style = str(row.get("augmentation_style", "unknown"))
        by_style.setdefault(style, {"pass": 0, "warn": 0, "fail": 0})
        by_style[style][status] = by_style[style].get(status, 0) + 1
        child_answer_mode = str(row.get("child_answer_mode", "none"))
        by_child_answer_mode.setdefault(
            child_answer_mode,
            {"pass": 0, "warn": 0, "fail": 0},
        )
        by_child_answer_mode[child_answer_mode][status] = (
            by_child_answer_mode[child_answer_mode].get(status, 0) + 1
        )
        by_style_and_child_answer_mode.setdefault(style, {})
        by_style_and_child_answer_mode[style].setdefault(
            child_answer_mode,
            {"pass": 0, "warn": 0, "fail": 0},
        )
        by_style_and_child_answer_mode[style][child_answer_mode][status] = (
            by_style_and_child_answer_mode[style][child_answer_mode].get(status, 0) + 1
        )
        for flag in row.get("audit_flags", []):
            code = str(flag.get("code", "unknown"))
            flag_counts[code] = flag_counts.get(code, 0) + 1
    return {
        "counts": counts,
        "counts_by_style": dict(sorted(by_style.items())),
        "counts_by_child_answer_mode": dict(sorted(by_child_answer_mode.items())),
        "counts_by_style_and_child_answer_mode": {
            style: dict(sorted(mode_counts.items()))
            for style, mode_counts in sorted(by_style_and_child_answer_mode.items())
        },
        "repair_counts": repair_counts,
        "flag_counts": dict(sorted(flag_counts.items())),
    }


def write_train_text(rows: list[dict[str, Any]], path: str | Path) -> int:
    kept = [
        str(row.get("text", "")).strip()
        for row in rows
        if str(row.get("audit_status", "pass")) in {"pass", "warn"}
        and str(row.get("text", "")).strip()
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept)


def build_manifest(
    *,
    config: AugmentationConfig,
    source_specs: int,
    source_documents: int,
    selected_specs: int,
    augmented_documents: int,
    train_documents: int,
    audit_report: dict[str, Any],
    outputs: dict[str, Path],
    config_path: Path | None,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": config.name,
        "seed": config.seed,
        "config_path": str(config_path) if config_path is not None else None,
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "source": {
            "scenario_plans": config.scenario_plans,
            "documents": config.source_documents,
            "include_audit_status": list(config.include_audit_status),
        },
        "styles": list(config.styles),
        "style_assignment": _style_assignment_record(config.style_assignment),
        "child_answer_modes": _child_answer_mode_record(config.child_answer_modes),
        "repair": {
            **_repair_record(config.repair),
            "counts": audit_report["repair_counts"],
        },
        "agent_pool": list(config.agent_pool),
        "deictic_agent_pool": [dict(pair) for pair in config.deictic_agent_pool],
        "renderer": _renderer_record(config.renderer),
        "word_count": {
            "kind": "regex",
            "pattern": config.word_pattern,
        },
        "counts": {
            "source_scenario_plans": source_specs,
            "source_documents": source_documents,
            "selected_source_scenarios": selected_specs,
            "augmented_documents": augmented_documents,
            "train_documents": train_documents,
            **audit_report["counts"],
        },
        "audit": {
            "min_words": config.min_words,
            "max_words": config.max_words,
            "positive_only": config.positive_only,
            "positive_only_markers": list(config.positive_only_markers),
            "require_question_mark": config.require_question_mark,
            "required_speaker_tags": list(config.required_speaker_tags),
            "max_turns": config.max_turns,
            "flag_counts": audit_report["flag_counts"],
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def _config_record(config: AugmentationConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "seed": config.seed,
        "source": {
            "scenario_plans": config.scenario_plans,
            "documents": config.source_documents,
        },
        "selection": {
            "include_audit_status": list(config.include_audit_status),
            "max_source_documents": config.max_source_documents,
            "max_augmented_documents": config.max_augmented_documents,
        },
        "styles": list(config.styles),
        "style_assignment": _style_assignment_record(config.style_assignment),
        "child_answer_modes": _child_answer_mode_record(config.child_answer_modes),
        "repair": _repair_record(config.repair),
        "agent_pool": list(config.agent_pool),
        "deictic_agent_pool": [dict(pair) for pair in config.deictic_agent_pool],
        "renderer": _renderer_record(config.renderer),
        "word_count": {"pattern": config.word_pattern},
        "audit": {
            "min_words": config.min_words,
            "max_words": config.max_words,
            "positive_only": config.positive_only,
            "positive_only_markers": list(config.positive_only_markers),
            "require_question_mark": config.require_question_mark,
            "required_speaker_tags": list(config.required_speaker_tags),
            "max_turns": config.max_turns,
        },
    }


def _renderer_record(renderer: RendererConfig) -> dict[str, Any]:
    record = {
        "type": renderer.kind,
        "model_id": renderer.model_id,
        "url": renderer.url,
        "documents_per_request": renderer.documents_per_request,
        "temperature": renderer.temperature,
        "max_tokens": renderer.max_tokens,
        "timeout_seconds": renderer.timeout_seconds,
        "max_request_retries": renderer.max_request_retries,
        "initial_backoff_seconds": renderer.initial_backoff_seconds,
        "max_backoff_seconds": renderer.max_backoff_seconds,
        "invalid_json_retries": renderer.invalid_json_retries,
        "reasoning_effort": renderer.reasoning_effort,
        "reasoning_history": renderer.reasoning_history,
        "resume_from_checkpoints": renderer.resume_from_checkpoints,
    }
    if renderer.api_key_file:
        record["api_key_file"] = renderer.api_key_file
    return record


def _style_assignment_record(config: StyleAssignmentConfig) -> dict[str, Any]:
    return {"strategy": config.strategy}


def _child_answer_mode_record(config: ChildAnswerModeConfig) -> dict[str, Any]:
    return {
        "strategy": config.strategy,
        "modes": list(config.modes),
    }


def _repair_record(config: RepairConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "max_rounds": config.max_rounds,
    }


__all__ = [
    "AugmentationConfig",
    "AugmentationResult",
    "ChildAnswerModeConfig",
    "DeterministicAugmentationRenderer",
    "DeterministicMotChiMockGenerator",
    "DocumentAugmenter",
    "FireworksMotChiAugmentationRenderer",
    "RegexTermMatcher",
    "RepairConfig",
    "RendererConfig",
    "StyleAssignmentConfig",
    "augmentation_config_from_mapping",
    "build_audit_report",
    "build_augmentation_render_requests",
    "build_document_augmentations",
    "build_mot_chi_render_prompt",
    "build_mot_chi_render_requests",
    "build_mot_chi_repair_prompt",
    "create_augmentation_renderer",
    "assigned_deictic_agent_for_spec",
    "assigned_speaker_perspective_for_spec",
    "load_yaml_config",
    "read_jsonl",
    "render_augmented_text",
    "write_train_text",
]
