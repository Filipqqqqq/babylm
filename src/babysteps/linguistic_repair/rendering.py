"""Renderers for QA-congruence repair documents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from babysteps.linguistic_repair.planning import SCENELESS_STYLES, ScenePlan
from babysteps.material_world.fireworks_client import (
    DEFAULT_FIREWORKS_CHAT_URL,
    FireworksChatClient,
    load_api_key,
)


DEFAULT_FIREWORKS_QA_MODEL = "accounts/fireworks/models/qwen3p7-plus"
NO_THINK_SUFFIX = "/no_think"
LOCAL_TEMPLATE_STYLES = frozenset({"simple_prose", *SCENELESS_STYLES})
SPEAKER_STYLE_PAIRS = {
    "classroom_tutor": ("Teacher", "Student"),
    "parent_child_dialogue": ("Parent", "Child"),
    "friend_dialogue": ("Friend 1", "Friend 2"),
    "coach_player_dialogue": ("Coach", "Player"),
    "nurse_child_dialogue": ("Nurse", "Child"),
    "clerk_customer_dialogue": ("Clerk", "Customer"),
}


class QaRenderer(Protocol):
    def render_batch(self, plans: list[ScenePlan]) -> list[dict[str, Any]]:
        """Render one batch of scene plans into document rows."""


@dataclass(frozen=True)
class RendererConfig:
    kind: str
    model_id: str | None = None
    url: str = DEFAULT_FIREWORKS_CHAT_URL
    api_key: str | None = None
    api_key_file: str | None = None
    documents_per_request: int = 8
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: int = 600
    max_request_retries: int = 4
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 16.0
    invalid_json_retries: int = 1
    reasoning_effort: str | bool | None = "none"
    reasoning_history: str | None = "disabled"
    resume_from_checkpoints: bool = True


def renderer_config_from_mapping(mapping: dict[str, Any] | None) -> RendererConfig:
    mapping = {} if mapping is None else dict(mapping)
    kind = str(mapping.get("type") or mapping.get("kind") or "template")
    model_id = mapping.get("model_id") or mapping.get("model")
    if kind == "fireworks" and model_id is None:
        model_id = DEFAULT_FIREWORKS_QA_MODEL
    return RendererConfig(
        kind=kind,
        model_id=None if model_id is None else str(model_id),
        url=str(mapping.get("url", DEFAULT_FIREWORKS_CHAT_URL)),
        api_key=None if mapping.get("api_key") is None else str(mapping["api_key"]),
        api_key_file=(
            None if mapping.get("api_key_file") is None else str(mapping["api_key_file"])
        ),
        documents_per_request=max(1, int(mapping.get("documents_per_request", 8))),
        temperature=float(mapping.get("temperature", 0.2)),
        max_tokens=int(mapping.get("max_tokens", 4096)),
        timeout_seconds=int(mapping.get("timeout_seconds", 600)),
        max_request_retries=int(mapping.get("max_request_retries", 4)),
        initial_backoff_seconds=float(mapping.get("initial_backoff_seconds", 1.0)),
        max_backoff_seconds=float(mapping.get("max_backoff_seconds", 16.0)),
        invalid_json_retries=max(0, int(mapping.get("invalid_json_retries", 1))),
        reasoning_effort=mapping.get("reasoning_effort", "none"),
        reasoning_history=mapping.get("reasoning_history", "disabled"),
        resume_from_checkpoints=bool(mapping.get("resume_from_checkpoints", True)),
    )


def render_template_text(plan: ScenePlan) -> str:
    """Render a plan locally with exact planned questions and answers."""

    if plan.style == "minimal_qa_pair":
        return " ".join(
            f"{question.question} {question.answer}" for question in plan.questions
        )

    if plan.style == "subtitle_clean":
        lines = [plan.scene_sentence]
        for question in plan.questions:
            lines.append(question.question)
            lines.append(question.answer)
        return " ".join(lines)

    if plan.style in SPEAKER_STYLE_PAIRS:
        question_speaker, answer_speaker = SPEAKER_STYLE_PAIRS[plan.style]
        lines = [plan.scene_sentence]
        for question in plan.questions:
            lines.append(f"{question_speaker}: {question.question}")
            lines.append(f"{answer_speaker}: {question.answer}")
        return " ".join(lines)

    if plan.style == "simple_prose":
        lines = [plan.scene_sentence]
        for question in plan.questions:
            lines.append(f"The helper asked, \"{question.question}\"")
            lines.append(f"The answer was, \"{question.answer}\"")
        return " ".join(lines)

    lines = [plan.scene_sentence]
    for question in plan.questions:
        lines.append(f"A: {question.question}")
        lines.append(f"B: {question.answer}")
    return " ".join(lines)


class TemplateQaRenderer:
    """Deterministic renderer used for local pilots and tests."""

    def render_batch(self, plans: list[ScenePlan]) -> list[dict[str, Any]]:
        return _template_rows(plans)


class FireworksQaRenderer:
    """Fireworks/Qwen renderer that receives fixed plans and returns JSON."""

    def __init__(
        self,
        renderer_config: RendererConfig,
        *,
        generator: Any | None = None,
    ) -> None:
        self.renderer_config = renderer_config
        self.generator = generator or self._build_fireworks_generator(renderer_config)

    def _build_fireworks_generator(self, renderer_config: RendererConfig) -> FireworksChatClient:
        if not renderer_config.model_id:
            raise ValueError("Fireworks QA renderer requires renderer.model_id.")
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

    def render_batch(self, plans: list[ScenePlan]) -> list[dict[str, Any]]:
        local_plans = [plan for plan in plans if plan.style in LOCAL_TEMPLATE_STYLES]
        fireworks_plans = [plan for plan in plans if plan.style not in LOCAL_TEMPLATE_STYLES]
        local_rows = _template_rows(local_plans)
        fireworks_rows = self._render_fireworks_batch(fireworks_plans) if fireworks_plans else []
        by_id = {str(row["id"]): row for row in [*local_rows, *fireworks_rows]}
        return [by_id[plan.id] for plan in plans]

    def _render_fireworks_batch(self, plans: list[ScenePlan]) -> list[dict[str, Any]]:
        prompt = build_fireworks_prompt(plans)
        last_error: Exception | None = None
        for _attempt in range(self.renderer_config.invalid_json_retries + 1):
            raw = self.generator(
                prompt=f"{prompt}\n\n{NO_THINK_SUFFIX}",
                system_prompt=system_prompt(),
                temperature=self.renderer_config.temperature,
                max_tokens=self.renderer_config.max_tokens,
                stage="qa_congruence_render",
                response_format={"type": "json_object"},
            )
            try:
                return normalize_rendered_documents(
                    _json_from_text(raw),
                    plans=plans,
                    renderer="fireworks",
                    render_model_id=self.renderer_config.model_id,
                )
            except ValueError as exc:
                last_error = exc
        raise ValueError(f"QA render failed: {last_error}") from last_error


def _template_rows(plans: list[ScenePlan]) -> list[dict[str, Any]]:
    return [
        {
            "id": plan.id,
            "renderer": "template",
            "render_model_id": None,
            "text": _one_line(render_template_text(plan)),
        }
        for plan in plans
    ]


class DeterministicQaMockGenerator:
    """Fireworks-shaped deterministic generator for tests and dry runs."""

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
        payload = _json_after_marker(prompt, "QA_PLANS_JSON:")
        items = []
        for row in payload["items"]:
            plan = _plan_from_prompt_item(row)
            items.append({"id": plan.id, "text": render_template_text(plan)})
        return json.dumps({"items": items})


def build_renderer(
    renderer_config: RendererConfig,
    *,
    generator: Any | None = None,
) -> QaRenderer:
    if renderer_config.kind == "template":
        return TemplateQaRenderer()
    if renderer_config.kind == "fireworks":
        return FireworksQaRenderer(renderer_config, generator=generator)
    raise ValueError(f"Unsupported QA renderer type: {renderer_config.kind!r}")


def system_prompt() -> str:
    return (
        "You render short fluent English QA training documents. Return only valid JSON. "
        "Do not choose questions, change their order, change answers, or add contrastive bad examples."
    )


def build_fireworks_prompt(plans: list[ScenePlan]) -> str:
    items = []
    for plan in plans:
        items.append(
            {
                "id": plan.id,
                "style": plan.style,
                "style_instruction": style_instruction(plan.style),
                "scene_sentence": plan.scene_sentence,
                "questions": [
                    {
                        "question_type": question.question_type,
                        "question": question.question,
                        "answer": question.answer,
                    }
                    for question in plan.questions
                ],
            }
        )
    return (
        "Render each item as one short document for a child language model.\n"
        "Rules:\n"
        "- Preserve the scene sentence exactly.\n"
        "- Start the document with the scene sentence as plain narration, with no speaker label before it.\n"
        "- Preserve every question string exactly and in the listed order.\n"
        "- Preserve every answer string exactly.\n"
        "- Follow each item's style_instruction for labels and document shape.\n"
        "- Do not add or remove questions.\n"
        "- Do not include labels like Good, Bad, wrong, or incorrect.\n"
        "- Return JSON as {\"items\": [{\"id\": \"...\", \"text\": \"...\"}]}.\n\n"
        "QA_PLANS_JSON:\n"
        + json.dumps({"items": items}, ensure_ascii=False, indent=2)
    )


def style_instruction(style: str) -> str:
    if style in SPEAKER_STYLE_PAIRS:
        question_speaker, answer_speaker = SPEAKER_STYLE_PAIRS[style]
        return (
            "Write the scene sentence as plain narration without a speaker label. "
            f"Use {question_speaker}: before every question. "
            f"Use {answer_speaker}: before every answer."
        )
    if style == "simple_prose":
        return (
            "Write the scene sentence as plain narration without a speaker label, "
            "then render each QA pair as simple prose using 'asked' and 'answer' language."
        )
    if style == "subtitle_clean":
        return (
            "Write the scene sentence as plain narration without a speaker label, "
            "then write subtitle-like question and answer lines without speaker names."
        )
    return (
        "Write the scene sentence as plain narration without a speaker label. "
        "Use compact A: question and B: answer dialogue labels after the scene."
    )


def normalize_rendered_documents(
    payload: Any,
    *,
    plans: list[ScenePlan],
    renderer: str,
    render_model_id: str | None,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Renderer response must be a JSON object with an items list.")
    by_id = {plan.id: plan for plan in plans}
    rows = []
    seen: set[str] = set()
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise ValueError("Rendered item must be an object.")
        item_id = str(item.get("id", ""))
        if item_id not in by_id:
            raise ValueError(f"Unexpected rendered item id: {item_id!r}")
        if item_id in seen:
            raise ValueError(f"Duplicate rendered item id: {item_id!r}")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Rendered item {item_id!r} has empty text.")
        seen.add(item_id)
        rows.append(
            {
                "id": item_id,
                "renderer": renderer,
                "render_model_id": render_model_id,
                "text": _one_line(text),
            }
        )
    missing = [plan.id for plan in plans if plan.id not in seen]
    if missing:
        raise ValueError("Renderer response missed item ids: " + ", ".join(missing))
    rows.sort(key=lambda row: [plan.id for plan in plans].index(str(row["id"])))
    return rows


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _json_from_text(text: str) -> Any:
    cleaned = text.strip().replace(NO_THINK_SUFFIX, "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Renderer response was not valid JSON: {cleaned[:500]}") from exc


def _json_after_marker(text: str, marker: str) -> Any:
    if marker not in text:
        raise ValueError(f"Prompt did not include marker {marker!r}.")
    return _json_from_text(text.split(marker, 1)[1])


def _plan_from_prompt_item(row: dict[str, Any]) -> ScenePlan:
    from babysteps.linguistic_repair.planning import QuestionPlan

    questions = tuple(
        QuestionPlan(
            question_type=str(question["question_type"]),
            question=str(question["question"]),
            answer=str(question["answer"]),
            expected_slot=str(question["answer"]).rstrip("."),
        )
        for question in row["questions"]
    )
    return ScenePlan(
        id=str(row["id"]),
        style=str(row.get("style", "clean_dialogue")),
        topic="mock",
        agent="",
        action_base="",
        action_past="",
        object_singular="",
        object_plural="",
        object_reference="",
        quantity_value=0,
        quantity_mode="word",
        quantity_surface="zero",
        object_phrase="",
        location="",
        time="",
        reason="",
        scene_sentence=str(row["scene_sentence"]),
        questions=questions,
    )
