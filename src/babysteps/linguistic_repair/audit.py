"""Auditing for QA-congruence repair documents."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from babysteps.linguistic_repair.planning import SCENELESS_STYLES, ScenePlan


BANNED_CONTRAST_MARKERS = (
    "bad:",
    "wrong:",
    "incorrect:",
    "bad answer",
    "wrong answer",
    "not the answer",
)


@dataclass(frozen=True)
class AuditFlag:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class AuditResult:
    status: str
    flags: tuple[AuditFlag, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "flags": [asdict(flag) for flag in self.flags],
        }


def audit_document(text: str, plan: ScenePlan) -> AuditResult:
    flags: list[AuditFlag] = []
    normalized = _one_line(text)
    lower = normalized.lower()

    if plan.style in SCENELESS_STYLES:
        if plan.scene_sentence in normalized:
            flags.append(
                AuditFlag(
                    "error",
                    "unexpected_scene",
                    "Scene-less style must not include the scene sentence.",
                )
            )
    elif plan.scene_sentence not in normalized:
        flags.append(
            AuditFlag("error", "missing_scene", "Rendered text is missing the exact scene sentence.")
        )
    elif not normalized.startswith(plan.scene_sentence):
        flags.append(
            AuditFlag(
                "error",
                "scene_prefix",
                "Rendered text must start with the exact scene sentence as plain narration.",
            )
        )

    for marker in BANNED_CONTRAST_MARKERS:
        if marker in lower:
            flags.append(
                AuditFlag("error", "contrast_marker", f"Rendered text includes {marker!r}.")
            )

    question_count = normalized.count("?")
    if question_count != len(plan.questions):
        flags.append(
            AuditFlag(
                "error",
                "question_count",
                f"Expected {len(plan.questions)} questions, found {question_count}.",
            )
        )

    last_index = -1
    for question in plan.questions:
        index = normalized.find(question.question)
        if index < 0:
            flags.append(
                AuditFlag(
                    "error",
                    "missing_question",
                    f"Missing exact question: {question.question}",
                )
            )
            continue
        if index <= last_index:
            flags.append(
                AuditFlag(
                    "error",
                    "question_order",
                    f"Question appears out of order: {question.question}",
                )
            )
        last_index = index

        next_index = _next_question_index(normalized, start=index + len(question.question))
        segment = normalized[index:next_index] if next_index >= 0 else normalized[index:]
        if question.answer not in segment:
            flags.append(
                AuditFlag(
                    "error",
                    "missing_answer",
                    f"Missing exact answer after question: {question.answer}",
                )
            )
        if not _answer_type_is_valid(question.question_type, question.answer, question.expected_slot):
            flags.append(
                AuditFlag(
                    "error",
                    "answer_type",
                    f"Answer failed type check for {question.question_type}: {question.answer}",
                )
            )

    status = "fail" if any(flag.severity == "error" for flag in flags) else "pass"
    return AuditResult(status=status, flags=tuple(flags))


def audit_documents(documents: list[dict[str, Any]], plans: list[ScenePlan]) -> dict[str, Any]:
    by_id = {plan.id: plan for plan in plans}
    counts: Counter[str] = Counter()
    by_question_type: Counter[str] = Counter()
    audited = []
    for document in documents:
        plan = by_id[str(document["id"])]
        result = audit_document(str(document["text"]), plan)
        row = dict(document)
        row["audit_status"] = result.status
        row["audit_flags"] = [asdict(flag) for flag in result.flags]
        audited.append(row)
        counts[result.status] += 1
        for question in plan.questions:
            by_question_type[question.question_type] += 1
    return {
        "counts": {
            "pass": counts.get("pass", 0),
            "warn": counts.get("warn", 0),
            "fail": counts.get("fail", 0),
        },
        "question_type_counts": dict(sorted(by_question_type.items())),
        "documents": audited,
    }


def _answer_type_is_valid(question_type: str, answer: str, expected_slot: str) -> bool:
    answer_lower = answer.lower()
    expected_lower = expected_slot.lower()
    if question_type == "why":
        return answer_lower.startswith(("because ", "so ", "for ", "to "))
    if question_type == "how_many":
        return bool(re.search(r"\b(?:none|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
                              r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
                              r"eighteen|nineteen|twenty|thirty|forty|fifty|seventy-five|"
                              r"one hundred|\d+)\b", answer_lower))
    return expected_lower in answer_lower


def _next_question_index(text: str, *, start: int) -> int:
    match = re.search(r"[^?.!]*\?", text[start:])
    if match is None:
        return -1
    return start + match.start()


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
