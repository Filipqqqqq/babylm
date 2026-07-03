"""Deterministic what-question object-frame repair data."""

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
THIRD_PERSON = "third_person"
SECOND_PERSON = "second_person"
FRAGMENT_ANSWER = "fragment_answer"
ECHO_ANSWER = "first_person_echo_answer"
QUESTION_TYPE = "what"
BANNED_QUESTION_WORDS = ("who", "where", "when", "why", "how many", "how_many")
BANNED_CONTRAST_MARKERS = ("bad:", "wrong:", "incorrect:", "bad answer", "wrong answer")
BANNED_EXACT_PROMPTS = (
    "what did you get",
    "what did you sell",
    "what did you make",
    "what did you close",
    "what did you break",
)
BANNED_EXACT_QUESTIONS = (
    "what did you get?",
    "what did you sell?",
    "what did you break?",
    "what did you make?",
    "what did you cook?",
    "what did you eat?",
    "what did you close?",
    "what did you open?",
    "what did you lock?",
)
PERSON_ANSWER_WORDS = (
    "teacher",
    "doctor",
    "nurse",
    "mom",
    "dad",
    "grandma",
    "grandpa",
    "parent",
    "babysitter",
    "neighbor",
    "friend",
    "classmate",
    "child",
    "student",
    "coach",
    "librarian",
    "clerk",
    "cashier",
    "cook",
    "driver",
    "carrier",
    "officer",
    "firefighter",
    "dentist",
    "janitor",
    "principal",
    "aunt",
    "uncle",
    "cousin",
)


@dataclass(frozen=True)
class WhatFrame:
    family: str
    frame_id: str
    question_template: str
    scene_template: str
    answer_pool: str
    detail_pool: tuple[str, ...] = ()


@dataclass(frozen=True)
class WhatObjectFramePlan:
    id: str
    index: int
    style: str
    speaker_mode: str
    answer_style: str
    frame_family: str
    frame_id: str
    agent: str
    answer_object: str
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
class QaWhatObjectFrameResult:
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

ANSWER_POOLS: dict[str, tuple[str, ...]] = {
    "school_home_objects": (
        "a ruler",
        "the folder",
        "a blanket",
        "the lunch box",
        "the notebook",
        "a worksheet",
        "the marker",
        "a recipe card",
        "the backpack",
        "a scarf",
        "the water bottle",
        "a library card",
        "the paper bag",
        "a small towel",
    ),
    "openable_objects": (
        "the cabinet",
        "the drawer",
        "the lunch box",
        "the backpack",
        "the mailbox",
        "the closet door",
        "the supply bin",
        "the pencil case",
        "the toy chest",
        "the cookie tin",
    ),
    "repair_clean_objects": (
        "the zipper",
        "a button",
        "the backpack",
        "the pencil case",
        "the toy truck",
        "the toy basket",
        "a loose strap",
        "the lunch box",
    ),
    "wash_clean_objects": (
        "the backpack",
        "the scarf",
        "the water bottle",
        "the lunch tray",
        "the table",
        "the window",
        "the sink",
        "the paint brush",
        "a small cup",
        "the toy truck",
        "the puzzle box",
    ),
    "drawn_objects": (
        "a circle",
        "a star",
        "a flower",
        "a map",
        "a small house",
        "a rainbow",
        "a face",
        "a tree",
        "a boat",
        "a sun",
    ),
    "craft_objects": (
        "the card",
        "the paper star",
        "a ribbon",
        "the sign",
        "the sticker sheet",
        "a paper crown",
        "the worksheet",
        "the calendar page",
        "a name tag",
        "the paper boat",
        "the fabric square",
        "a gift tag",
    ),
    "fold_objects": (
        "a cloth",
        "a shirt",
        "a small towel",
        "a napkin",
        "a scarf",
        "a blanket",
        "the fabric square",
        "the paper boat",
        "the card",
        "a paper crown",
    ),
    "transport_objects": (
        "the lunch box",
        "a blanket",
        "the backpack",
        "the notebook",
        "a worksheet",
        "the marker",
        "a recipe card",
        "the paper bag",
        "a small towel",
        "the puzzle box",
        "the toy basket",
        "a raincoat",
    ),
}

WHAT_FRAME_FAMILIES: dict[str, tuple[WhatFrame, ...]] = {
    "borrow_return_choose": (
        WhatFrame(
            "borrow_return_choose",
            "borrow_object",
            "What did {agent} borrow?",
            "{agent} borrowed {answer_obj} {time}.",
            "school_home_objects",
        ),
        WhatFrame(
            "borrow_return_choose",
            "return_object",
            "What did {agent} return?",
            "{agent} returned {answer_obj} {time}.",
            "school_home_objects",
        ),
        WhatFrame(
            "borrow_return_choose",
            "choose_object",
            "What did {agent} choose for {detail}?",
            "{agent} chose {answer_obj} for {detail} {time}.",
            "school_home_objects",
            ("the lesson", "the game", "the project", "the shelf", "the picnic"),
        ),
        WhatFrame(
            "borrow_return_choose",
            "pick_object",
            "What did {agent} pick from the shelf?",
            "{agent} picked {answer_obj} from the shelf {time}.",
            "school_home_objects",
        ),
    ),
    "open_lock_unlock": (
        WhatFrame(
            "open_lock_unlock",
            "open_object",
            "What did {agent} open?",
            "{agent} opened {answer_obj} {time}.",
            "openable_objects",
        ),
        WhatFrame(
            "open_lock_unlock",
            "lock_object",
            "What did {agent} lock?",
            "{agent} locked {answer_obj} {time}.",
            "openable_objects",
        ),
        WhatFrame(
            "open_lock_unlock",
            "unlock_object",
            "What did {agent} unlock?",
            "{agent} unlocked {answer_obj} {time}.",
            "openable_objects",
        ),
        WhatFrame(
            "open_lock_unlock",
            "cover_object",
            "What did {agent} cover with a cloth?",
            "{agent} covered {answer_obj} with a cloth {time}.",
            "openable_objects",
        ),
    ),
    "repair_wash_clean": (
        WhatFrame(
            "repair_wash_clean",
            "repair_object",
            "What did {agent} repair?",
            "{agent} repaired {answer_obj} {time}.",
            "repair_clean_objects",
        ),
        WhatFrame(
            "repair_wash_clean",
            "fix_object",
            "What did {agent} fix?",
            "{agent} fixed {answer_obj} {time}.",
            "repair_clean_objects",
        ),
        WhatFrame(
            "repair_wash_clean",
            "wash_object",
            "What did {agent} wash?",
            "{agent} washed {answer_obj} {time}.",
            "wash_clean_objects",
        ),
        WhatFrame(
            "repair_wash_clean",
            "clean_object",
            "What did {agent} clean?",
            "{agent} cleaned {answer_obj} {time}.",
            "wash_clean_objects",
        ),
        WhatFrame(
            "repair_wash_clean",
            "wipe_object",
            "What did {agent} wipe?",
            "{agent} wiped {answer_obj} {time}.",
            "wash_clean_objects",
        ),
    ),
    "pack_carry_bring": (
        WhatFrame(
            "pack_carry_bring",
            "pack_object",
            "What did {agent} pack?",
            "{agent} packed {answer_obj} {time}.",
            "transport_objects",
        ),
        WhatFrame(
            "pack_carry_bring",
            "carry_object",
            "What did {agent} carry to {detail}?",
            "{agent} carried {answer_obj} to {detail} {time}.",
            "transport_objects",
            ("the table", "the office", "the classroom", "the porch", "the car"),
        ),
        WhatFrame(
            "pack_carry_bring",
            "bring_object",
            "What did {agent} bring to {detail}?",
            "{agent} brought {answer_obj} to {detail} {time}.",
            "transport_objects",
            ("the table", "the office", "the classroom", "the porch", "the picnic"),
        ),
        WhatFrame(
            "pack_carry_bring",
            "hang_object",
            "What did {agent} hang by the door?",
            "{agent} hung {answer_obj} by the door {time}.",
            "transport_objects",
        ),
    ),
    "craft_fold_decorate": (
        WhatFrame(
            "craft_fold_decorate",
            "draw_object",
            "What did {agent} draw on the paper?",
            "{agent} drew {answer_obj} on the paper {time}.",
            "drawn_objects",
        ),
        WhatFrame(
            "craft_fold_decorate",
            "cut_object",
            "What did {agent} cut carefully?",
            "{agent} cut {answer_obj} carefully {time}.",
            "craft_objects",
        ),
        WhatFrame(
            "craft_fold_decorate",
            "fold_object",
            "What did {agent} fold?",
            "{agent} folded {answer_obj} {time}.",
            "fold_objects",
        ),
        WhatFrame(
            "craft_fold_decorate",
            "decorate_object",
            "What did {agent} decorate?",
            "{agent} decorated {answer_obj} {time}.",
            "craft_objects",
        ),
        WhatFrame(
            "craft_fold_decorate",
            "label_object",
            "What did {agent} label?",
            "{agent} labeled {answer_obj} {time}.",
            "craft_objects",
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


def build_qa_what_object_frames(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> QaWhatObjectFrameResult:
    """Build deterministic QA what-object repair documents."""

    if not isinstance(mapping, dict):
        raise TypeError("QA what-object frame config must be a mapping.")

    name = str(mapping["name"])
    seed = int(mapping.get("seed", 42))
    document_count = int(mapping.get("document_count", 500))
    if document_count <= 0:
        raise ValueError("document_count must be positive.")

    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    direct_ratio = float(mapping.get("direct_dialogue_ratio", 0.5))
    second_person_ratio = float(mapping.get("second_person_ratio", 0.5))
    first_person_echo_answer_ratio = float(mapping.get("first_person_echo_answer_ratio", 0.5))
    family_names = tuple(str(name) for name in mapping.get("frame_families", WHAT_FRAME_FAMILIES))
    unknown = sorted(set(family_names) - set(WHAT_FRAME_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown what frame families: {unknown}")
    if not family_names:
        raise ValueError("frame_families must not be empty.")

    rng = random.Random(seed)
    agents = tuple(str(agent) for agent in mapping.get("agents", AGENTS))
    plans = _build_plans(
        name=name,
        document_count=document_count,
        family_names=family_names,
        direct_ratio=direct_ratio,
        second_person_ratio=second_person_ratio,
        first_person_echo_answer_ratio=first_person_echo_answer_ratio,
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
        "config": output_dir / "qa_what_object_frames_config.yaml",
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
        "second_person_ratio": second_person_ratio,
        "first_person_echo_answer_ratio": first_person_echo_answer_ratio,
        "frame_families": list(family_names),
        "counts": {
            "documents": len(documents),
            "train_documents": sum(1 for row in documents if row["audit_status"] == "pass"),
            **audit_report["counts"],
            "frame_family_counts": audit_report["frame_family_counts"],
            "style_counts": audit_report["style_counts"],
            "speaker_mode_counts": audit_report["speaker_mode_counts"],
            "answer_style_counts": audit_report["answer_style_counts"],
            "answer_pool_counts": audit_report["answer_pool_counts"],
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    return QaWhatObjectFrameResult(
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
    second_person_ratio: float,
    first_person_echo_answer_ratio: float,
    agents: tuple[str, ...],
    rng: random.Random,
) -> list[WhatObjectFramePlan]:
    family_sequence = _balanced_sequence(family_names, document_count, rng=rng)
    styles = _balanced_style_sequence_by_family(
        family_sequence,
        direct_ratio=direct_ratio,
        family_names=family_names,
        rng=rng,
    )
    speaker_modes = _balanced_label_sequence_by_family(
        family_sequence,
        positive_label=SECOND_PERSON,
        negative_label=THIRD_PERSON,
        positive_ratio=second_person_ratio,
        family_names=family_names,
        rng=rng,
    )
    answer_styles = _balanced_answer_style_sequence_by_family(
        family_sequence,
        speaker_modes,
        echo_ratio=first_person_echo_answer_ratio,
        family_names=family_names,
        rng=rng,
    )
    family_offsets: Counter[str] = Counter()
    plans: list[WhatObjectFramePlan] = []
    seen_texts: set[str] = set()
    for index, family_name in enumerate(family_sequence):
        frames = WHAT_FRAME_FAMILIES[family_name]
        frame = frames[family_offsets[family_name] % len(frames)]
        family_offsets[family_name] += 1
        speaker_mode = speaker_modes[index]
        answer_style = answer_styles[index]
        agent = rng.choice(agents)
        question_agent = "you" if speaker_mode == SECOND_PERSON else agent
        scene_agent = "You" if speaker_mode == SECOND_PERSON else agent
        for attempt in range(100):
            answer_object = rng.choice(ANSWER_POOLS[frame.answer_pool])
            detail = rng.choice(frame.detail_pool) if frame.detail_pool else None
            time = rng.choice(TIMES)
            question_fields = {
                "agent": question_agent,
                "answer_obj": answer_object,
                "detail": detail or "",
                "time": time,
            }
            scene_fields = {
                **question_fields,
                "agent": scene_agent,
            }
            question = _avoid_eval_like_question(
                _normalize_spaces(frame.question_template.format(**question_fields))
            )
            answer = _answer_for_style(
                frame=frame,
                answer_object=answer_object,
                detail=detail,
                answer_style=answer_style,
            )
            scene_sentence = (
                _normalize_spaces(frame.scene_template.format(**scene_fields))
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
            WhatObjectFramePlan(
                id=f"{name}_{index:05d}",
                index=index,
                style=styles[index],
                speaker_mode=speaker_mode,
                answer_style=answer_style,
                frame_family=family_name,
                frame_id=frame.frame_id,
                agent=question_agent,
                answer_object=answer_object,
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
    return _balanced_label_sequence_by_family(
        family_sequence,
        positive_label=DIRECT_DIALOGUE,
        negative_label=SCENE_QA,
        positive_ratio=direct_ratio,
        family_names=family_names,
        rng=rng,
    )


def _balanced_label_sequence_by_family(
    family_sequence: list[str],
    *,
    positive_label: str,
    negative_label: str,
    positive_ratio: float,
    family_names: tuple[str, ...],
    rng: random.Random,
) -> list[str]:
    family_counts = Counter(family_sequence)
    target_positive_total = round(len(family_sequence) * positive_ratio)
    target_positive_total = min(max(target_positive_total, 0), len(family_sequence))
    family_positive_counts: dict[str, int] = {}
    running_total = 0
    fractional: list[tuple[float, str]] = []
    for family_name in family_names:
        exact = family_counts[family_name] * positive_ratio
        positive_count = int(exact)
        family_positive_counts[family_name] = positive_count
        running_total += positive_count
        fractional.append((exact - positive_count, family_name))
    for _, family_name in sorted(fractional, reverse=True)[: target_positive_total - running_total]:
        family_positive_counts[family_name] += 1

    family_styles: dict[str, list[str]] = {}
    for family_name in family_names:
        positive_count = family_positive_counts[family_name]
        negative_count = family_counts[family_name] - positive_count
        labels = [positive_label] * positive_count + [negative_label] * negative_count
        rng.shuffle(labels)
        family_styles[family_name] = labels

    family_offsets: Counter[str] = Counter()
    labels_for_sequence: list[str] = []
    for family_name in family_sequence:
        offset = family_offsets[family_name]
        labels_for_sequence.append(family_styles[family_name][offset])
        family_offsets[family_name] += 1
    return labels_for_sequence


def _balanced_answer_style_sequence_by_family(
    family_sequence: list[str],
    speaker_modes: list[str],
    *,
    echo_ratio: float,
    family_names: tuple[str, ...],
    rng: random.Random,
) -> list[str]:
    second_person_counts = Counter(
        family_name
        for family_name, speaker_mode in zip(family_sequence, speaker_modes)
        if speaker_mode == SECOND_PERSON
    )
    family_echo_counts: dict[str, int] = {}
    for family_name in family_names:
        family_echo_counts[family_name] = round(second_person_counts[family_name] * echo_ratio)

    family_answer_styles: dict[str, list[str]] = {}
    for family_name in family_names:
        echo_count = family_echo_counts[family_name]
        fragment_count = second_person_counts[family_name] - echo_count
        styles = [ECHO_ANSWER] * echo_count + [FRAGMENT_ANSWER] * fragment_count
        rng.shuffle(styles)
        family_answer_styles[family_name] = styles

    family_offsets: Counter[str] = Counter()
    answer_styles: list[str] = []
    for family_name, speaker_mode in zip(family_sequence, speaker_modes):
        if speaker_mode != SECOND_PERSON:
            answer_styles.append(FRAGMENT_ANSWER)
            continue
        offset = family_offsets[family_name]
        answer_styles.append(family_answer_styles[family_name][offset])
        family_offsets[family_name] += 1
    return answer_styles


def _answer_for_style(
    *,
    frame: WhatFrame,
    answer_object: str,
    detail: str | None,
    answer_style: str,
) -> str:
    if answer_style == FRAGMENT_ANSWER:
        return _answer_sentence(answer_object)
    if answer_style != ECHO_ANSWER:
        raise ValueError(f"Unsupported answer style: {answer_style!r}")
    fields = {
        "agent": "I",
        "answer_obj": answer_object,
        "detail": detail or "",
        "time": "",
    }
    return _normalize_spaces(frame.scene_template.format(**fields))


def _avoid_eval_like_question(question: str) -> str:
    if question.lower() in {"what did you open?", "what did you lock?"}:
        return f"{question[:-1]} today?"
    return question


def _answer_sentence(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}."


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([?.!,])", r"\1", text)


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


def _document_from_plan(plan: WhatObjectFramePlan) -> dict[str, Any]:
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
    speaker_mode_counts = Counter(str(row["speaker_mode"]) for row in documents)
    answer_style_counts = Counter(str(row["answer_style"]) for row in documents)
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
        "speaker_mode_counts": dict(sorted(speaker_mode_counts.items())),
        "answer_style_counts": dict(sorted(answer_style_counts.items())),
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
    answer_lower = answer.lower()
    if text.count("?") != 1:
        flags.append(_flag("error", "question_count", "Expected exactly one question."))
    if not question.startswith("What "):
        flags.append(_flag("error", "question_shape", "Question must start with 'What'."))
    for word in BANNED_QUESTION_WORDS:
        if word in lower:
            flags.append(_flag("error", "wrong_question_type", f"Found banned question word: {word}"))
    for prompt in BANNED_EXACT_PROMPTS:
        if prompt in lower:
            flags.append(_flag("error", "eval_like_prompt", f"Found banned prompt fragment: {prompt}"))
    if question.lower() in BANNED_EXACT_QUESTIONS:
        flags.append(_flag("error", "eval_like_question", f"Found banned exact question: {question}"))
    for marker in BANNED_CONTRAST_MARKERS:
        if marker in lower:
            flags.append(_flag("error", "contrast_marker", f"Found contrast marker: {marker}"))
    for word in PERSON_ANSWER_WORDS:
        if re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", answer_lower):
            flags.append(_flag("error", "person_answer", f"Answer appears animate/person-like: {word}"))
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
