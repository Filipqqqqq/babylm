"""Deterministic common-name who-question repair data."""

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
BARE_NAME = "bare_name"
BARE_ROLE = "bare_role"
FULL_SENTENCE_NAME = "full_sentence_name"
KINSHIP_TITLE = "kinship_title"
ANSWER_STYLE_ORDER = (BARE_NAME, BARE_ROLE, FULL_SENTENCE_NAME, KINSHIP_TITLE)
BANNED_QUESTION_WORDS = ("where", "when", "why", "how many", "how_many")
BANNED_CONTRAST_MARKERS = ("bad:", "wrong:", "incorrect:", "bad answer", "wrong answer")
BANNED_NAMES = frozenset({"Sarah", "David", "Davis"})
BANNED_BENCHMARK_PHRASES = (
    "who did you see",
    "who did you meet",
    "who helped you",
    "who kissed",
    "who shook your hand",
    "who is sleeping",
    "who is drawing",
    "who painted that picture",
    "who is playing the guitar",
)


@dataclass(frozen=True)
class CommonNameWhoFrame:
    family: str
    frame_id: str
    question: str
    scene_template: str
    full_answer_template: str


@dataclass(frozen=True)
class CommonNameWhoPlan:
    id: str
    index: int
    style: str
    frame_family: str
    frame_id: str
    answer_style: str
    answer_entity: str
    answer_entity_type: str
    question: str
    answer: str
    scene_sentence: str | None
    text: str
    time: str
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QaWhoCommonNameFrameResult:
    name: str
    output_dir: Path
    documents: tuple[dict[str, Any], ...]
    audit_report: dict[str, Any]
    outputs: dict[str, Path]
    manifest: dict[str, Any]


COMMON_NAMES = (
    "Ross",
    "Mark",
    "John",
    "Adam",
    "Paul",
    "Marky",
    "Ben",
    "Nina",
    "George",
    "Mary",
    "Edna",
    "Phil",
    "Eve",
    "Bessie",
    "Peter",
    "Jack",
    "Richard",
    "Dorothy",
)

PERSON_ROLES = (
    "a nurse",
    "a clerk",
    "a driver",
    "a cook",
    "a parent",
    "a neighbor",
    "a student",
    "a librarian",
    "a coach",
    "a cashier",
    "a mail carrier",
    "a babysitter",
)

KINSHIP_TITLES = (
    "Mom",
    "Dad",
    "Grandma",
    "Grandpa",
    "an aunt",
    "an uncle",
)

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

WHO_COMMON_NAME_FRAMES: dict[str, tuple[CommonNameWhoFrame, ...]] = {
    "simple_agent_action": (
        CommonNameWhoFrame(
            "simple_agent_action",
            "open_folder",
            "Who opened the folder?",
            "{answer_subj} opened the folder {time}.",
            "{answer_subj} opened it.",
        ),
        CommonNameWhoFrame(
            "simple_agent_action",
            "close_drawer",
            "Who closed the drawer?",
            "{answer_subj} closed the drawer {time}.",
            "{answer_subj} closed it.",
        ),
        CommonNameWhoFrame(
            "simple_agent_action",
            "move_basket",
            "Who moved the basket?",
            "{answer_subj} moved the basket {time}.",
            "{answer_subj} moved it.",
        ),
    ),
    "household_task": (
        CommonNameWhoFrame(
            "household_task",
            "fold_towel",
            "Who folded the towel?",
            "{answer_subj} folded the towel {time}.",
            "{answer_subj} folded it.",
        ),
        CommonNameWhoFrame(
            "household_task",
            "wipe_counter",
            "Who wiped the counter?",
            "{answer_subj} wiped the counter {time}.",
            "{answer_subj} wiped it.",
        ),
        CommonNameWhoFrame(
            "household_task",
            "pack_bag",
            "Who packed the bag?",
            "{answer_subj} packed the bag {time}.",
            "{answer_subj} packed it.",
        ),
    ),
    "school_classroom": (
        CommonNameWhoFrame(
            "school_classroom",
            "pass_out_paper",
            "Who passed out the paper?",
            "{answer_subj} passed out the paper {time}.",
            "{answer_subj} passed it out.",
        ),
        CommonNameWhoFrame(
            "school_classroom",
            "erase_board",
            "Who erased the board?",
            "{answer_subj} erased the board {time}.",
            "{answer_subj} erased it.",
        ),
        CommonNameWhoFrame(
            "school_classroom",
            "line_up_group",
            "Who lined up the group?",
            "{answer_subj} lined up the group {time}.",
            "{answer_subj} lined them up.",
        ),
    ),
    "helping_care": (
        CommonNameWhoFrame(
            "helping_care",
            "help_bag",
            "Who helped with the bag?",
            "{answer_subj} helped with the bag {time}.",
            "{answer_subj} helped with it.",
        ),
        CommonNameWhoFrame(
            "helping_care",
            "check_bandage",
            "Who checked the bandage?",
            "{answer_subj} checked the bandage {time}.",
            "{answer_subj} checked it.",
        ),
        CommonNameWhoFrame(
            "helping_care",
            "hold_door",
            "Who held the door?",
            "{answer_subj} held the door {time}.",
            "{answer_subj} held it.",
        ),
    ),
    "communication": (
        CommonNameWhoFrame(
            "communication",
            "answer_phone",
            "Who answered the phone?",
            "{answer_subj} answered the phone {time}.",
            "{answer_subj} answered it.",
        ),
        CommonNameWhoFrame(
            "communication",
            "read_note",
            "Who read the note?",
            "{answer_subj} read the note {time}.",
            "{answer_subj} read it.",
        ),
        CommonNameWhoFrame(
            "communication",
            "call_office",
            "Who called the office?",
            "{answer_subj} called the office {time}.",
            "{answer_subj} called the office.",
        ),
    ),
    "finding_bringing": (
        CommonNameWhoFrame(
            "finding_bringing",
            "bring_keys",
            "Who brought the keys?",
            "{answer_subj} brought the keys {time}.",
            "{answer_subj} brought them.",
        ),
        CommonNameWhoFrame(
            "finding_bringing",
            "find_ticket",
            "Who found the ticket?",
            "{answer_subj} found the ticket {time}.",
            "{answer_subj} found it.",
        ),
        CommonNameWhoFrame(
            "finding_bringing",
            "carry_tray",
            "Who carried the tray?",
            "{answer_subj} carried the tray {time}.",
            "{answer_subj} carried it.",
        ),
    ),
    "checking_reminding": (
        CommonNameWhoFrame(
            "checking_reminding",
            "check_list",
            "Who checked the list?",
            "{answer_subj} checked the list {time}.",
            "{answer_subj} checked it.",
        ),
        CommonNameWhoFrame(
            "checking_reminding",
            "count_cups",
            "Who counted the cups?",
            "{answer_subj} counted the cups {time}.",
            "{answer_subj} counted them.",
        ),
        CommonNameWhoFrame(
            "checking_reminding",
            "remind_group",
            "Who reminded the group?",
            "{answer_subj} reminded the group {time}.",
            "{answer_subj} reminded them.",
        ),
    ),
    "movement_arrival": (
        CommonNameWhoFrame(
            "movement_arrival",
            "come_to_door",
            "Who came to the door?",
            "{answer_subj} came to the door {time}.",
            "{answer_subj} came to the door.",
        ),
        CommonNameWhoFrame(
            "movement_arrival",
            "walk_into_room",
            "Who walked into the room?",
            "{answer_subj} walked into the room {time}.",
            "{answer_subj} walked into the room.",
        ),
        CommonNameWhoFrame(
            "movement_arrival",
            "wait_outside",
            "Who waited outside?",
            "{answer_subj} waited outside {time}.",
            "{answer_subj} waited outside.",
        ),
    ),
    "role_answer_control": (
        CommonNameWhoFrame(
            "role_answer_control",
            "serve_lunch",
            "Who served lunch?",
            "{answer_subj} served lunch {time}.",
            "{answer_subj} served lunch.",
        ),
        CommonNameWhoFrame(
            "role_answer_control",
            "drive_van",
            "Who drove the van?",
            "{answer_subj} drove the van {time}.",
            "{answer_subj} drove it.",
        ),
        CommonNameWhoFrame(
            "role_answer_control",
            "fix_sink",
            "Who fixed the sink?",
            "{answer_subj} fixed the sink {time}.",
            "{answer_subj} fixed it.",
        ),
    ),
    "short_sentence_ready": (
        CommonNameWhoFrame(
            "short_sentence_ready",
            "open_mailbox",
            "Who opened the mailbox?",
            "{answer_subj} opened the mailbox {time}.",
            "{answer_subj} opened it.",
        ),
        CommonNameWhoFrame(
            "short_sentence_ready",
            "bring_towel",
            "Who brought the towel?",
            "{answer_subj} brought the towel {time}.",
            "{answer_subj} brought it.",
        ),
        CommonNameWhoFrame(
            "short_sentence_ready",
            "answer_bell",
            "Who answered the bell?",
            "{answer_subj} answered the bell {time}.",
            "{answer_subj} answered it.",
        ),
    ),
}


def build_qa_who_common_name_frames(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> QaWhoCommonNameFrameResult:
    """Build deterministic QA who/common-name repair documents."""

    if not isinstance(mapping, dict):
        raise TypeError("QA who common-name frame config must be a mapping.")

    name = str(mapping["name"])
    seed = int(mapping.get("seed", 42))
    document_count = int(mapping.get("document_count", 500))
    if document_count <= 0:
        raise ValueError("document_count must be positive.")

    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    direct_ratio = float(mapping.get("direct_dialogue_ratio", 0.5))
    family_names = tuple(str(name) for name in mapping.get("frame_families", WHO_COMMON_NAME_FRAMES))
    unknown = sorted(set(family_names) - set(WHO_COMMON_NAME_FRAMES))
    if unknown:
        raise ValueError(f"Unknown who common-name frame families: {unknown}")
    if not family_names:
        raise ValueError("frame_families must not be empty.")

    excluded_names = frozenset(str(value) for value in mapping.get("excluded_names", BANNED_NAMES))
    names = tuple(str(value) for value in mapping.get("names", COMMON_NAMES))
    roles = tuple(str(value) for value in mapping.get("roles", PERSON_ROLES))
    kinship_titles = tuple(str(value) for value in mapping.get("kinship_titles", KINSHIP_TITLES))
    _validate_entities(names, roles, kinship_titles, excluded_names=excluded_names)
    answer_style_weights = {
        style: float(weight)
        for style, weight in mapping.get(
            "answer_style_weights",
            {BARE_NAME: 0.5, BARE_ROLE: 0.25, FULL_SENTENCE_NAME: 0.20, KINSHIP_TITLE: 0.05},
        ).items()
    }

    rng = random.Random(seed)
    plans = _build_plans(
        name=name,
        document_count=document_count,
        family_names=family_names,
        direct_ratio=direct_ratio,
        answer_style_weights=answer_style_weights,
        names=names,
        roles=roles,
        kinship_titles=kinship_titles,
        rng=rng,
    )
    documents = tuple(_document_from_plan(plan) for plan in plans)
    audit_report = _audit_documents(
        name,
        documents,
        family_names=family_names,
        answer_style_weights=answer_style_weights,
    )

    paths = {
        "documents": output_dir / "documents.jsonl",
        "train": output_dir / "train.txt",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
        "config": output_dir / "qa_who_common_name_frames_config.yaml",
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
        "answer_style_weights": answer_style_weights,
        "names": list(names),
        "excluded_names": sorted(excluded_names),
        "counts": {
            "documents": len(documents),
            "train_documents": sum(1 for row in documents if row["audit_status"] == "pass"),
            **audit_report["counts"],
            "frame_family_counts": audit_report["frame_family_counts"],
            "frame_id_counts": audit_report["frame_id_counts"],
            "style_counts": audit_report["style_counts"],
            "answer_style_counts": audit_report["answer_style_counts"],
            "answer_entity_type_counts": audit_report["answer_entity_type_counts"],
            "answer_entity_counts": audit_report["answer_entity_counts"],
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    return QaWhoCommonNameFrameResult(
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
    answer_style_weights: dict[str, float],
    names: tuple[str, ...],
    roles: tuple[str, ...],
    kinship_titles: tuple[str, ...],
    rng: random.Random,
) -> list[CommonNameWhoPlan]:
    family_sequence = _balanced_sequence(family_names, document_count, rng=rng)
    styles = _balanced_style_sequence_by_family(
        family_sequence,
        direct_ratio=direct_ratio,
        family_names=family_names,
        rng=rng,
    )
    answer_styles = _weighted_sequence(
        ANSWER_STYLE_ORDER,
        answer_style_weights,
        document_count,
        rng=rng,
    )
    family_offsets: Counter[str] = Counter()
    plans: list[CommonNameWhoPlan] = []
    seen_texts: set[str] = set()
    for index, family_name in enumerate(family_sequence):
        frames = WHO_COMMON_NAME_FRAMES[family_name]
        frame = frames[family_offsets[family_name] % len(frames)]
        family_offsets[family_name] += 1
        answer_style = answer_styles[index]
        for attempt in range(100):
            answer_entity, answer_entity_type = _choose_answer_entity(
                answer_style,
                names=names,
                roles=roles,
                kinship_titles=kinship_titles,
                rng=rng,
            )
            time = rng.choice(TIMES)
            answer_subj = _sentence_initial(answer_entity)
            question = frame.question
            scene_sentence = (
                _normalize_spaces(frame.scene_template.format(answer_subj=answer_subj, time=time))
                if styles[index] == SCENE_QA
                else None
            )
            answer = _render_answer(
                frame,
                answer_style=answer_style,
                answer_entity=answer_entity,
                answer_subj=answer_subj,
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
            CommonNameWhoPlan(
                id=f"{name}_{index:05d}",
                index=index,
                style=styles[index],
                frame_family=family_name,
                frame_id=frame.frame_id,
                answer_style=answer_style,
                answer_entity=answer_entity,
                answer_entity_type=answer_entity_type,
                question=question,
                answer=answer,
                scene_sentence=scene_sentence,
                text=text,
                time=time,
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


def _weighted_sequence(
    values: tuple[str, ...],
    weights: dict[str, float],
    count: int,
    *,
    rng: random.Random,
) -> list[str]:
    unknown = sorted(set(weights) - set(values))
    if unknown:
        raise ValueError(f"Unknown answer styles: {unknown}")
    total_weight = sum(weights.get(value, 0.0) for value in values)
    if total_weight <= 0:
        raise ValueError("answer_style_weights must have a positive total.")
    sequence: list[str] = []
    floors: dict[str, int] = {}
    fractional: list[tuple[float, str]] = []
    running_total = 0
    for value in values:
        exact = count * weights.get(value, 0.0) / total_weight
        floor = int(exact)
        floors[value] = floor
        running_total += floor
        fractional.append((exact - floor, value))
    for _, value in sorted(fractional, reverse=True)[: count - running_total]:
        floors[value] += 1
    for value in values:
        sequence.extend([value] * floors[value])
    rng.shuffle(sequence)
    return sequence


def _choose_answer_entity(
    answer_style: str,
    *,
    names: tuple[str, ...],
    roles: tuple[str, ...],
    kinship_titles: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, str]:
    if answer_style == BARE_NAME or answer_style == FULL_SENTENCE_NAME:
        return rng.choice(names), "common_name"
    if answer_style == BARE_ROLE:
        return rng.choice(roles), "person_role"
    if answer_style == KINSHIP_TITLE:
        return rng.choice(kinship_titles), "kinship_title"
    raise ValueError(f"Unsupported answer style: {answer_style!r}")


def _render_answer(
    frame: CommonNameWhoFrame,
    *,
    answer_style: str,
    answer_entity: str,
    answer_subj: str,
) -> str:
    if answer_style == FULL_SENTENCE_NAME:
        return _normalize_spaces(frame.full_answer_template.format(answer_subj=answer_subj))
    if answer_style in {BARE_NAME, BARE_ROLE, KINSHIP_TITLE}:
        return _answer_sentence(answer_entity)
    raise ValueError(f"Unsupported answer style: {answer_style!r}")


def _validate_entities(
    names: tuple[str, ...],
    roles: tuple[str, ...],
    kinship_titles: tuple[str, ...],
    *,
    excluded_names: frozenset[str],
) -> None:
    if not names:
        raise ValueError("names must not be empty.")
    if not roles:
        raise ValueError("roles must not be empty.")
    if not kinship_titles:
        raise ValueError("kinship_titles must not be empty.")
    overlap = sorted(set(names) & excluded_names)
    if overlap:
        raise ValueError(f"names include excluded names: {overlap}")


def _sentence_initial(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}"


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


def _document_from_plan(plan: CommonNameWhoPlan) -> dict[str, Any]:
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
    answer_style_weights: dict[str, float],
) -> dict[str, Any]:
    counts = Counter(row["audit_status"] for row in documents)
    family_counts = Counter(str(row["frame_family"]) for row in documents)
    frame_id_counts = Counter(str(row["frame_id"]) for row in documents)
    style_counts = Counter(str(row["style"]) for row in documents)
    answer_style_counts = Counter(str(row["answer_style"]) for row in documents)
    answer_entity_type_counts = Counter(str(row["answer_entity_type"]) for row in documents)
    answer_entity_counts = Counter(str(row["answer_entity"]) for row in documents)
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
        "frame_id_counts": dict(sorted(frame_id_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
        "answer_style_counts": dict(sorted(answer_style_counts.items())),
        "answer_entity_type_counts": dict(sorted(answer_entity_type_counts.items())),
        "answer_entity_counts": dict(sorted(answer_entity_counts.items())),
        "balance": {
            "frame_family_max_minus_min": balance_delta,
            "answer_style_weights": answer_style_weights,
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
    for phrase in BANNED_BENCHMARK_PHRASES:
        if phrase in lower:
            flags.append(_flag("error", "benchmark_phrase", f"Found benchmark-like phrase: {phrase}"))
    for name in BANNED_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", text):
            flags.append(_flag("error", "excluded_name", f"Found excluded name: {name}"))
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
