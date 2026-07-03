"""Deterministic who-question person-frame repair data."""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from babysteps.paths import resolve_path


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
DIRECT_DIALOGUE = "direct_dialogue"
SCENE_QA = "scene_qa"
QUESTION_TYPE = "who"
BANNED_QUESTION_WORDS = ("where", "when", "why", "how many", "how_many")
BANNED_CONTRAST_MARKERS = ("bad:", "wrong:", "incorrect:", "bad answer", "wrong answer")


@dataclass(frozen=True)
class WhoFrame:
    family: str
    frame_id: str
    question_template: str
    scene_template: str
    answer_pool: str
    detail_pool: tuple[str, ...] = ()


@dataclass(frozen=True)
class WhoPersonFramePlan:
    id: str
    index: int
    style: str
    frame_family: str
    frame_id: str
    agent: str | None
    answer_person: str
    answer_pool: str
    detail: str | None
    question: str
    answer: str
    scene_sentence: str | None
    text: str
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QaWhoPersonFrameResult:
    name: str
    output_dir: Path
    documents: tuple[dict[str, Any], ...]
    audit_report: dict[str, Any]
    outputs: dict[str, Path]
    manifest: dict[str, Any]


AGENTS = (
    "Noor",
    "Mia",
    "Jamal",
    "Luis",
    "Priya",
    "Omar",
    "Aisha",
    "Kenji",
    "Fatima",
    "Hana",
    "Mateo",
    "Yuna",
    "Ben",
    "Nia",
    "Sofia",
    "Diego",
)

NAMES = (
    "Noor",
    "Mia",
    "Jamal",
    "Luis",
    "Priya",
    "Omar",
    "Aisha",
    "Kenji",
    "Fatima",
    "Hana",
    "Mateo",
    "Yuna",
)

COMMON_ROLES = (
    "a teacher",
    "a doctor",
    "a nurse",
    "Mom",
    "Dad",
    "Grandma",
    "Grandpa",
    "a parent",
    "a babysitter",
    "a neighbor",
    "a friend",
    "a classmate",
    "a child",
    "a student",
    "a coach",
    "a librarian",
    "a clerk",
    "a cashier",
    "a cook",
    "a driver",
    "a mail carrier",
    "a police officer",
    "a firefighter",
    "a dentist",
    "a janitor",
    "the principal",
    "an aunt",
    "an uncle",
    "a cousin",
)

ANSWER_POOLS: dict[str, tuple[str, ...]] = {
    "contact_people": COMMON_ROLES + NAMES,
    "helpers": (
        "a teacher",
        "a nurse",
        "Mom",
        "Dad",
        "Grandma",
        "Grandpa",
        "a parent",
        "a babysitter",
        "a neighbor",
        "a friend",
        "a coach",
        "a doctor",
        "a police officer",
        "a firefighter",
        "an aunt",
        "an uncle",
    )
    + NAMES,
    "general_subjects": (
        "a teacher",
        "a nurse",
        "Mom",
        "Dad",
        "Grandma",
        "Grandpa",
        "a parent",
        "a neighbor",
        "a friend",
        "a classmate",
        "a child",
        "a student",
        "a coach",
        "a librarian",
        "a cook",
        "a driver",
        "a police officer",
        "a firefighter",
        "a janitor",
        "the principal",
    )
    + NAMES,
    "readers": ("a teacher", "a librarian", "a student", "Mom", "Dad") + NAMES,
    "cooks": ("a cook", "Mom", "Dad", "Grandma", "Grandpa", "a babysitter") + NAMES,
    "drivers": ("a driver", "Mom", "Dad", "a police officer", "a firefighter", "a parent") + NAMES,
    "task_doers": (
        "a teacher",
        "a nurse",
        "Mom",
        "Dad",
        "Grandma",
        "Grandpa",
        "a parent",
        "a friend",
        "a classmate",
        "a student",
        "a coach",
        "a librarian",
        "a clerk",
        "a cashier",
        "a cook",
        "a janitor",
        "the principal",
        "a cousin",
    )
    + NAMES,
}

WHO_FRAME_FAMILIES: dict[str, tuple[WhoFrame, ...]] = {
    "contact_visit_question": (
        WhoFrame(
            "contact_visit_question",
            "call_person",
            "Who did {agent} call?",
            "{agent} called {answer_obj} {time}.",
            "contact_people",
        ),
        WhoFrame(
            "contact_visit_question",
            "visit_person",
            "Who did {agent} visit?",
            "{agent} visited {answer_obj} {time}.",
            "contact_people",
        ),
        WhoFrame(
            "contact_visit_question",
            "greet_person",
            "Who did {agent} greet at the door?",
            "{agent} greeted {answer_obj} at the door {time}.",
            "contact_people",
        ),
        WhoFrame(
            "contact_visit_question",
            "thank_person",
            "Who did {agent} thank for {detail}?",
            "{agent} thanked {answer_obj} for {detail} {time}.",
            "contact_people",
            ("the snack", "the ride", "the card", "the note", "the warm soup"),
        ),
        WhoFrame(
            "contact_visit_question",
            "invite_person",
            "Who did {agent} invite to {detail}?",
            "{agent} invited {answer_obj} to {detail} {time}.",
            "contact_people",
            ("the table", "the game", "the porch", "the classroom", "the picnic", "the party"),
        ),
        WhoFrame(
            "contact_visit_question",
            "ask_person",
            "Who did {agent} ask about {detail}?",
            "{agent} asked {answer_obj} about {detail} {time}.",
            "contact_people",
            ("the form", "the recipe", "the missing shoe", "the schedule", "the bus"),
        ),
    ),
    "assist_care_help": (
        WhoFrame(
            "assist_care_help",
            "help_with_item",
            "Who helped {agent} with {detail}?",
            "{answer_subj} helped {agent} with {detail} {time}.",
            "helpers",
            ("the bag", "the puzzle", "the zipper", "the tray", "the homework"),
        ),
        WhoFrame(
            "assist_care_help",
            "guide_to_place",
            "Who guided {agent} to {detail}?",
            "{answer_subj} guided {agent} to {detail} {time}.",
            "helpers",
            ("the room", "the office", "the bus", "the bench", "the library"),
        ),
        WhoFrame(
            "assist_care_help",
            "check_on_person",
            "Who checked on {agent}?",
            "{answer_subj} checked on {agent} {time}.",
            "helpers",
        ),
        WhoFrame(
            "assist_care_help",
            "comfort_person",
            "Who comforted {agent} after {detail}?",
            "{answer_subj} comforted {agent} after {detail} {time}.",
            "helpers",
            ("the loud noise", "the fall", "the storm", "the long wait", "the hard game"),
        ),
        WhoFrame(
            "assist_care_help",
            "remind_person",
            "Who reminded {agent} about {detail}?",
            "{answer_subj} reminded {agent} about {detail} {time}.",
            "helpers",
            ("the form", "the lunch box", "the coat", "the appointment", "the library book"),
        ),
        WhoFrame(
            "assist_care_help",
            "watch_person",
            "Who watched {agent} during {detail}?",
            "{answer_subj} watched {agent} during {detail} {time}.",
            "helpers",
            ("recess", "practice", "the visit", "the game", "the lesson"),
        ),
    ),
    "role_action_subject": (
        WhoFrame(
            "role_action_subject",
            "waiting_subject",
            "Who was waiting by the door?",
            "{answer_subj} was waiting by the door {time}.",
            "general_subjects",
        ),
        WhoFrame(
            "role_action_subject",
            "laughing_subject",
            "Who was laughing near the table?",
            "{answer_subj} was laughing near the table {time}.",
            "general_subjects",
        ),
        WhoFrame(
            "role_action_subject",
            "reading_subject",
            "Who was reading in the corner?",
            "{answer_subj} was reading in the corner {time}.",
            "readers",
        ),
        WhoFrame(
            "role_action_subject",
            "singing_subject",
            "Who was singing on the porch?",
            "{answer_subj} was singing on the porch {time}.",
            "general_subjects",
        ),
        WhoFrame(
            "role_action_subject",
            "cooking_subject",
            "Who was cooking lunch?",
            "{answer_subj} was cooking lunch {time}.",
            "cooks",
        ),
        WhoFrame(
            "role_action_subject",
            "driving_subject",
            "Who was driving the bus?",
            "{answer_subj} was driving the bus {time}.",
            "drivers",
        ),
        WhoFrame(
            "role_action_subject",
            "knocking_subject",
            "Who was knocking at the door?",
            "{answer_subj} was knocking at the door {time}.",
            "general_subjects",
        ),
        WhoFrame(
            "role_action_subject",
            "waving_subject",
            "Who was waving from the steps?",
            "{answer_subj} was waving from the steps {time}.",
            "general_subjects",
        ),
        WhoFrame(
            "role_action_subject",
            "using_restroom_subject",
            "Who was using the restroom?",
            "{answer_subj} was using the restroom {time}.",
            "general_subjects",
        ),
        WhoFrame(
            "role_action_subject",
            "using_charger_subject",
            "Who was using the charger?",
            "{answer_subj} was using the charger {time}.",
            "general_subjects",
        ),
    ),
    "creation_task_doer": (
        WhoFrame(
            "creation_task_doer",
            "build_tower",
            "Who built the block tower?",
            "{answer_subj} built the block tower {time}.",
            "task_doers",
        ),
        WhoFrame(
            "creation_task_doer",
            "fix_zipper",
            "Who fixed the zipper?",
            "{answer_subj} fixed the zipper {time}.",
            "task_doers",
        ),
        WhoFrame(
            "creation_task_doer",
            "fold_card",
            "Who folded the card?",
            "{answer_subj} folded the card {time}.",
            "task_doers",
        ),
        WhoFrame(
            "creation_task_doer",
            "decorate_sign",
            "Who decorated the sign?",
            "{answer_subj} decorated the sign {time}.",
            "task_doers",
        ),
        WhoFrame(
            "creation_task_doer",
            "wash_cup",
            "Who washed the cup?",
            "{answer_subj} washed the cup {time}.",
            "task_doers",
        ),
        WhoFrame(
            "creation_task_doer",
            "clean_table",
            "Who cleaned the table?",
            "{answer_subj} cleaned the table {time}.",
            "task_doers",
        ),
        WhoFrame(
            "creation_task_doer",
            "pack_bag",
            "Who packed the lunch bag?",
            "{answer_subj} packed the lunch bag {time}.",
            "task_doers",
        ),
        WhoFrame(
            "creation_task_doer",
            "arrange_books",
            "Who arranged the books?",
            "{answer_subj} arranged the books {time}.",
            "task_doers",
        ),
    ),
}

TIMES = (
    "after lunch",
    "before dinner",
    "after school",
    "this morning",
    "this afternoon",
    "before bedtime",
    "on Monday",
    "on Tuesday",
    "on Wednesday",
    "on Thursday",
    "on Friday",
)


def build_qa_who_person_frames(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> QaWhoPersonFrameResult:
    """Build deterministic QA who-person repair documents."""

    if not isinstance(mapping, dict):
        raise TypeError("QA who-person frame config must be a mapping.")

    name = str(mapping["name"])
    seed = int(mapping.get("seed", 42))
    document_count = int(mapping.get("document_count", 500))
    if document_count <= 0:
        raise ValueError("document_count must be positive.")

    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    direct_ratio = float(mapping.get("direct_dialogue_ratio", 0.5))
    family_names = tuple(str(name) for name in mapping.get("frame_families", WHO_FRAME_FAMILIES))
    unknown = sorted(set(family_names) - set(WHO_FRAME_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown who frame families: {unknown}")
    if not family_names:
        raise ValueError("frame_families must not be empty.")

    rng = random.Random(seed)
    agents = tuple(str(agent) for agent in mapping.get("agents", AGENTS))
    plans = _build_plans(
        name=name,
        document_count=document_count,
        family_names=family_names,
        direct_ratio=direct_ratio,
        agents=agents,
        rng=rng,
    )
    documents = tuple(_document_from_plan(plan) for plan in plans)
    audit_report = _audit_documents(name, documents, family_names=family_names)

    paths = {
        "documents": output_dir / "documents.jsonl",
        "train": output_dir / "train.txt",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
        "config": output_dir / "qa_who_person_frames_config.yaml",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copyfile(config_path, paths["config"])
    _write_jsonl(paths["documents"], list(documents))
    _write_train(paths["train"], list(documents))
    _write_json(paths["audit_report"], audit_report)

    manifest = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "document_count": document_count,
        "generator": "deterministic_template",
        "question_type": QUESTION_TYPE,
        "direct_dialogue_ratio": direct_ratio,
        "frame_families": list(family_names),
        "counts": {
            "documents": len(documents),
            "train_documents": sum(1 for row in documents if row["audit_status"] == "pass"),
            **audit_report["counts"],
            "frame_family_counts": audit_report["frame_family_counts"],
            "style_counts": audit_report["style_counts"],
            "answer_pool_counts": audit_report["answer_pool_counts"],
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    return QaWhoPersonFrameResult(
        name=name,
        output_dir=output_dir,
        documents=documents,
        audit_report=audit_report,
        outputs=paths,
        manifest=manifest,
    )


def _build_plans(
    *,
    name: str,
    document_count: int,
    family_names: tuple[str, ...],
    direct_ratio: float,
    agents: tuple[str, ...],
    rng: random.Random,
) -> list[WhoPersonFramePlan]:
    family_sequence = _balanced_sequence(family_names, document_count, rng=rng)
    styles = _balanced_style_sequence_by_family(
        family_sequence,
        direct_ratio=direct_ratio,
        family_names=family_names,
        rng=rng,
    )
    family_offsets: Counter[str] = Counter()
    plans: list[WhoPersonFramePlan] = []
    seen_texts: set[str] = set()
    for index, family_name in enumerate(family_sequence):
        frames = WHO_FRAME_FAMILIES[family_name]
        frame = frames[family_offsets[family_name] % len(frames)]
        family_offsets[family_name] += 1
        agent = rng.choice(agents) if "{agent}" in frame.question_template else None
        for attempt in range(100):
            answer_person = _choose_answer(frame.answer_pool, agent=agent, rng=rng)
            detail = rng.choice(frame.detail_pool) if frame.detail_pool else None
            time = rng.choice(TIMES)
            fields = {
                "agent": agent or "",
                "answer_obj": _object_person(answer_person),
                "answer_subj": _sentence_initial(answer_person),
                "detail": detail or "",
                "time": time,
            }
            question = _normalize_spaces(frame.question_template.format(**fields))
            answer = _answer_sentence(answer_person)
            scene_sentence = (
                _normalize_spaces(frame.scene_template.format(**fields))
                if styles[index] == SCENE_QA
                else None
            )
            text = _render_text(
                question=question,
                answer=answer,
                scene_sentence=scene_sentence,
                style=styles[index],
            )
            if text not in seen_texts or attempt == 99:
                break
        seen_texts.add(text)
        plans.append(
            WhoPersonFramePlan(
                id=f"{name}_{index:05d}",
                index=index,
                style=styles[index],
                frame_family=family_name,
                frame_id=frame.frame_id,
                agent=agent,
                answer_person=answer_person,
                answer_pool=frame.answer_pool,
                detail=detail,
                question=question,
                answer=answer,
                scene_sentence=scene_sentence,
                text=text,
                template_id=styles[index],
            )
        )
    return plans


def _balanced_sequence(values: tuple[str, ...], count: int, *, rng: random.Random) -> list[str]:
    base, remainder = divmod(count, len(values))
    sequence: list[str] = []
    for index, value in enumerate(values):
        sequence.extend([value] * (base + (1 if index < remainder else 0)))
    rng.shuffle(sequence)
    return sequence


def _balanced_style_sequence_by_family(
    family_sequence: list[str],
    *,
    direct_ratio: float,
    family_names: tuple[str, ...],
    rng: random.Random,
) -> list[str]:
    family_counts = Counter(family_sequence)
    target_direct_total = round(len(family_sequence) * direct_ratio)
    target_direct_total = min(max(target_direct_total, 0), len(family_sequence))
    family_direct_counts: dict[str, int] = {}
    running_total = 0
    fractional: list[tuple[float, str]] = []
    for family_name in family_names:
        exact = family_counts[family_name] * direct_ratio
        direct_count = int(exact)
        family_direct_counts[family_name] = direct_count
        running_total += direct_count
        fractional.append((exact - direct_count, family_name))
    for _, family_name in sorted(fractional, reverse=True)[: target_direct_total - running_total]:
        family_direct_counts[family_name] += 1

    family_styles: dict[str, list[str]] = {}
    for family_name in family_names:
        direct_count = family_direct_counts[family_name]
        scene_count = family_counts[family_name] - direct_count
        styles = [DIRECT_DIALOGUE] * direct_count + [SCENE_QA] * scene_count
        rng.shuffle(styles)
        family_styles[family_name] = styles

    family_offsets: Counter[str] = Counter()
    styles_for_sequence: list[str] = []
    for family_name in family_sequence:
        offset = family_offsets[family_name]
        styles_for_sequence.append(family_styles[family_name][offset])
        family_offsets[family_name] += 1
    return styles_for_sequence


def _choose_answer(answer_pool: str, *, agent: str | None, rng: random.Random) -> str:
    pool = ANSWER_POOLS[answer_pool]
    candidates = [answer for answer in pool if answer != agent]
    if not candidates:
        raise ValueError(f"answer pool {answer_pool!r} has no valid candidates.")
    return rng.choice(candidates)


def _object_person(answer_person: str) -> str:
    if answer_person[:1].isupper():
        return answer_person
    return answer_person


def _sentence_initial(answer_person: str) -> str:
    return f"{answer_person[:1].upper()}{answer_person[1:]}"


def _answer_sentence(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}."


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _render_text(
    *,
    question: str,
    answer: str,
    scene_sentence: str | None,
    style: str,
) -> str:
    qa = f"A: {question} B: {answer}"
    if style == DIRECT_DIALOGUE:
        return qa
    if style == SCENE_QA:
        if scene_sentence is None:
            raise ValueError("scene_qa style requires a scene sentence.")
        return f"{scene_sentence} {qa}"
    raise ValueError(f"Unsupported style: {style!r}")


def _document_from_plan(plan: WhoPersonFramePlan) -> dict[str, Any]:
    row = plan.to_dict()
    row["question_type"] = QUESTION_TYPE
    row["word_count"] = _word_count(plan.text)
    audit = _audit_document(row)
    row["audit_status"] = audit["status"]
    row["audit_flags"] = audit["flags"]
    return row


def _audit_documents(
    name: str,
    documents: tuple[dict[str, Any], ...],
    *,
    family_names: tuple[str, ...],
) -> dict[str, Any]:
    counts = Counter(row["audit_status"] for row in documents)
    family_counts = Counter(str(row["frame_family"]) for row in documents)
    style_counts = Counter(str(row["style"]) for row in documents)
    answer_pool_counts = Counter(str(row["answer_pool"]) for row in documents)
    family_values = [family_counts[name] for name in family_names]
    balance_delta = max(family_values) - min(family_values) if family_values else 0
    flags = []
    if balance_delta > 1:
        flags.append(
            {
                "severity": "warn",
                "code": "frame_family_balance",
                "message": "Frame family counts differ by more than one.",
            }
        )
    return {
        "name": name,
        "counts": {
            "pass": counts.get("pass", 0),
            "warn": counts.get("warn", 0),
            "fail": counts.get("fail", 0),
        },
        "total_documents": len(documents),
        "train_documents": counts.get("pass", 0),
        "question_type_counts": {QUESTION_TYPE: len(documents)},
        "frame_family_counts": dict(sorted(family_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
        "answer_pool_counts": dict(sorted(answer_pool_counts.items())),
        "balance": {
            "frame_family_max_minus_min": balance_delta,
        },
        "flags": flags,
    }


def _audit_document(row: dict[str, Any]) -> dict[str, Any]:
    flags = []
    text = str(row["text"])
    lower = text.lower()
    question = str(row["question"])
    answer = str(row["answer"])
    if text.count("?") != 1:
        flags.append(_flag("error", "question_count", "Expected exactly one question."))
    if not question.startswith("Who "):
        flags.append(_flag("error", "question_shape", "Question must start with 'Who'."))
    for word in BANNED_QUESTION_WORDS:
        if word in lower:
            flags.append(_flag("error", "wrong_question_type", f"Found banned question word: {word}"))
    for marker in BANNED_CONTRAST_MARKERS:
        if marker in lower:
            flags.append(_flag("error", "contrast_marker", f"Found contrast marker: {marker}"))
    if question not in text:
        flags.append(_flag("error", "missing_question", "Text is missing the planned question."))
    if answer not in text:
        flags.append(_flag("error", "missing_answer", "Text is missing the planned answer."))
    if "\n" in text:
        flags.append(_flag("error", "multiline_text", "Train text must stay one line per example."))
    status = "fail" if any(flag["severity"] == "error" for flag in flags) else "pass"
    return {"status": status, "flags": flags}


def _flag(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def _word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def _write_train(path: Path, documents: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for document in documents:
            if document["audit_status"] == "pass":
                f.write(str(document["text"]).strip())
                f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
