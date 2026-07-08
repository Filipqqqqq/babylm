"""Deterministic local-preposition where-question repair data."""

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
PAIRED_WHAT_WHERE = "paired_what_where"
STYLE_ORDER = (DIRECT_DIALOGUE, SCENE_QA, PAIRED_WHAT_WHERE)
QUESTION_TYPE = "where"
WHERE_DIRECT_ACTION = "direct_action"
WHERE_NOW = "where_now"
WHERE_END_UP = "end_up"
WHERE_INDIRECT = "indirect_where"
WHERE_QUESTION_STYLE_ORDER = (
    WHERE_DIRECT_ACTION,
    WHERE_NOW,
    WHERE_END_UP,
    WHERE_INDIRECT,
)
BANNED_CONTRAST_MARKERS = ("bad:", "wrong:", "incorrect:", "bad answer", "wrong answer")
BANNED_BENCHMARK_WORDS = frozenset(
    {
        "package",
        "papers",
        "table",
        "sofa",
        "bush",
        "computer",
        "eggs",
        "book",
        "piano",
        "chair",
        "car",
        "milk",
        "teacher",
        "doctor",
        "sarah",
        "david",
        "lamp",
        "bowl",
        "gate",
        "window",
        "chest",
        "box",
        "statue",
    }
)


@dataclass(frozen=True)
class LocalPrepVerbFrame:
    family: str
    verb: str
    past: str


@dataclass(frozen=True)
class LocalPrepPlan:
    id: str
    index: int
    style: str
    verb_family: str
    verb: str
    verb_past: str
    agent: str
    object: str
    object_reference: str
    question_object_reference: str
    landmark: str
    location: str
    answer_preposition: str
    where_question_style: str
    question: str
    answer: str
    answer_style: str
    scene_sentence: str | None
    text: str
    second_person: bool
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QaWhereLocalPrepFrameResult:
    name: str
    output_dir: Path
    documents: tuple[dict[str, Any], ...]
    audit_report: dict[str, Any]
    outputs: dict[str, Path]
    manifest: dict[str, Any]


AGENTS = (
    "Aisha",
    "Amara",
    "Ben",
    "Carlos",
    "Diego",
    "Fatima",
    "Hana",
    "Jamal",
    "Kenji",
    "Leila",
    "Luis",
    "Maya",
    "Noor",
    "Omar",
    "Priya",
    "Ravi",
    "Sofia",
    "Yuna",
)

OBJECTS = (
    "notebook",
    "key",
    "scarf",
    "glove",
    "mitten",
    "backpack",
    "lunchbox",
    "pencil",
    "crayon",
    "marker",
    "envelope",
    "ticket",
    "wallet",
    "cup",
    "spoon",
    "brush",
    "towel",
    "sock",
    "toy truck",
    "button",
)

LOCATIONS_BY_PREPOSITION: dict[str, tuple[str, ...]] = {
    "behind": (
        "behind the curtain",
        "behind the cabinet",
        "behind the dresser",
        "behind the coat rack",
        "behind the laundry basket",
    ),
    "next to": (
        "next to the plant pot",
        "next to the shelf",
        "next to the toy bin",
        "next to the coat rack",
        "next to the stroller",
    ),
    "under": (
        "under the rug",
        "under the pillow",
        "under the blanket",
        "under the cabinet",
        "under the laundry basket",
    ),
    "on": (
        "on the shelf",
        "on the counter",
        "on the dresser",
        "on the shoe rack",
        "on the cabinet",
    ),
    "in": (
        "in the closet",
        "in the cabinet",
        "in the toy bin",
        "in the laundry basket",
        "in the mailbox",
    ),
    "near": (
        "near the doorway",
        "near the sink",
        "near the fence",
        "near the coat rack",
        "near the plant pot",
    ),
    "beside": (
        "beside the cabinet",
        "beside the closet",
        "beside the stroller",
        "beside the mailbox",
        "beside the sink",
    ),
}

VERB_FRAMES: dict[str, tuple[LocalPrepVerbFrame, ...]] = {
    "put_place_set": (
        LocalPrepVerbFrame("put_place_set", "put", "put"),
        LocalPrepVerbFrame("put_place_set", "place", "placed"),
        LocalPrepVerbFrame("put_place_set", "set", "set"),
    ),
    "leave_drop_hide_store": (
        LocalPrepVerbFrame("leave_drop_hide_store", "leave", "left"),
        LocalPrepVerbFrame("leave_drop_hide_store", "drop", "dropped"),
        LocalPrepVerbFrame("leave_drop_hide_store", "hide", "hid"),
        LocalPrepVerbFrame("leave_drop_hide_store", "store", "stored"),
    ),
    "find_discover_spot": (
        LocalPrepVerbFrame("find_discover_spot", "find", "found"),
        LocalPrepVerbFrame("find_discover_spot", "discover", "discovered"),
        LocalPrepVerbFrame("find_discover_spot", "spot", "spotted"),
    ),
}


def build_qa_where_local_prep_frames(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> QaWhereLocalPrepFrameResult:
    """Build deterministic where local-preposition repair documents."""

    if not isinstance(mapping, dict):
        raise TypeError("QA where local-prep config must be a mapping.")

    name = str(mapping["name"])
    seed = int(mapping.get("seed", 42))
    document_count = int(mapping.get("document_count", 500))
    if document_count <= 0:
        raise ValueError("document_count must be positive.")

    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    second_person_ratio = _ratio(mapping, "second_person_ratio", 0.65)
    pronoun_question_ratio = _ratio(mapping, "pronoun_question_ratio", 0.45)
    full_answer_ratio = _ratio(mapping, "full_answer_ratio", 0.45)
    style_weights = _style_weights(mapping)
    where_question_style_weights = _where_question_style_weights(mapping)
    prepositions = tuple(str(item) for item in mapping.get("prepositions", LOCATIONS_BY_PREPOSITION))
    unknown_prepositions = sorted(set(prepositions) - set(LOCATIONS_BY_PREPOSITION))
    if unknown_prepositions:
        raise ValueError(f"Unknown prepositions: {unknown_prepositions}")
    family_names = tuple(str(item) for item in mapping.get("verb_families", VERB_FRAMES))
    unknown_families = sorted(set(family_names) - set(VERB_FRAMES))
    if unknown_families:
        raise ValueError(f"Unknown verb families: {unknown_families}")
    if not prepositions:
        raise ValueError("prepositions must not be empty.")
    if not family_names:
        raise ValueError("verb_families must not be empty.")

    rng = random.Random(seed)
    plans = _build_plans(
        name=name,
        document_count=document_count,
        family_names=family_names,
        prepositions=prepositions,
        style_weights=style_weights,
        second_person_ratio=second_person_ratio,
        pronoun_question_ratio=pronoun_question_ratio,
        full_answer_ratio=full_answer_ratio,
        where_question_style_weights=where_question_style_weights,
        rng=rng,
    )
    documents = tuple(_document_from_plan(plan) for plan in plans)
    audit_report = _audit_documents(
        name,
        documents,
        prepositions=prepositions,
        family_names=family_names,
    )

    paths = {
        "documents": output_dir / "documents.jsonl",
        "train": output_dir / "train.txt",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
        "config": output_dir / "qa_where_local_prep_frames_config.yaml",
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
        "prepositions": list(prepositions),
        "verb_families": list(family_names),
        "style_weights": style_weights,
        "second_person_ratio": second_person_ratio,
        "pronoun_question_ratio": pronoun_question_ratio,
        "full_answer_ratio": full_answer_ratio,
        "where_question_style_weights": where_question_style_weights,
        "excluded_benchmark_words": sorted(BANNED_BENCHMARK_WORDS),
        "counts": {
            "documents": len(documents),
            "train_documents": sum(1 for row in documents if row["audit_status"] == "pass"),
            **audit_report["counts"],
            "style_counts": audit_report["style_counts"],
            "verb_family_counts": audit_report["verb_family_counts"],
            "verb_counts": audit_report["verb_counts"],
            "answer_preposition_counts": audit_report["answer_preposition_counts"],
            "answer_style_counts": audit_report["answer_style_counts"],
            "where_question_style_counts": audit_report["where_question_style_counts"],
            "second_person_counts": audit_report["second_person_counts"],
            "question_count_counts": audit_report["question_count_counts"],
            "question_object_reference_counts": audit_report[
                "question_object_reference_counts"
            ],
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    return QaWhereLocalPrepFrameResult(
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
    prepositions: tuple[str, ...],
    style_weights: dict[str, float],
    second_person_ratio: float,
    pronoun_question_ratio: float,
    full_answer_ratio: float,
    where_question_style_weights: dict[str, float],
    rng: random.Random,
) -> list[LocalPrepPlan]:
    family_sequence = _balanced_sequence(family_names, document_count, rng=rng)
    preposition_sequence = _balanced_sequence(prepositions, document_count, rng=rng)
    styles = _weighted_style_sequence(document_count, style_weights=style_weights, rng=rng)
    where_question_styles = _weighted_where_question_style_sequence(
        document_count,
        where_question_style_weights=where_question_style_weights,
        rng=rng,
    )
    family_offsets: Counter[str] = Counter()
    preposition_offsets: Counter[str] = Counter()
    object_sequence = _balanced_sequence(OBJECTS, document_count, rng=rng)
    plans: list[LocalPrepPlan] = []
    for index, family_name in enumerate(family_sequence):
        frames = VERB_FRAMES[family_name]
        frame = frames[family_offsets[family_name] % len(frames)]
        family_offsets[family_name] += 1
        preposition = preposition_sequence[index]
        locations = LOCATIONS_BY_PREPOSITION[preposition]
        location = locations[
            (preposition_offsets[preposition] + rng.randrange(len(locations))) % len(locations)
        ]
        preposition_offsets[preposition] += 1
        obj = object_sequence[index]
        object_reference = f"the {obj}"
        agent = rng.choice(AGENTS)
        second_person = rng.random() < second_person_ratio
        question_agent = "you" if second_person else agent
        where_question_style = where_question_styles[index]
        question_object_reference = (
            "it"
            if (
                where_question_style == WHERE_DIRECT_ACTION
                and styles[index] != DIRECT_DIALOGUE
                and second_person
                and rng.random() < pronoun_question_ratio
            )
            else object_reference
        )
        question = _render_where_question(
            style=where_question_style,
            agent=question_agent,
            frame=frame,
            object_reference=question_object_reference,
        )
        full_answer = second_person and rng.random() < full_answer_ratio
        answer = (
            _sentence(f"I {frame.past} {question_object_reference} {location}")
            if full_answer
            else _sentence(location)
        )
        answer_style = "first_person_location_sentence" if full_answer else "location_fragment"
        scene_agent = "I" if second_person else agent
        scene_sentence = _sentence(f"{scene_agent} {frame.past} {object_reference} {location}")
        text = _render_text(
            style=styles[index],
            scene_sentence=scene_sentence,
            question=question,
            answer=answer,
            agent=question_agent,
            frame=frame,
            object_reference=object_reference,
            question_object_reference=question_object_reference,
            location=location,
        )
        plans.append(
            LocalPrepPlan(
                id=f"{name}_{index:05d}",
                index=index,
                style=styles[index],
                verb_family=frame.family,
                verb=frame.verb,
                verb_past=frame.past,
                agent=agent,
                object=obj,
                object_reference=object_reference,
                question_object_reference=question_object_reference,
                landmark=_landmark_from_location(location, preposition),
                location=location,
                answer_preposition=preposition,
                where_question_style=where_question_style,
                question=question,
                answer=answer,
                answer_style=answer_style,
                scene_sentence=scene_sentence if styles[index] != DIRECT_DIALOGUE else None,
                text=text,
                second_person=second_person,
                template_id=styles[index],
            )
        )
    return plans


def _render_text(
    *,
    style: str,
    scene_sentence: str,
    question: str,
    answer: str,
    agent: str,
    frame: LocalPrepVerbFrame,
    object_reference: str,
    question_object_reference: str,
    location: str,
) -> str:
    if style == DIRECT_DIALOGUE:
        return f"A: {question} B: {answer}"
    if style == SCENE_QA:
        return f"{scene_sentence} A: {question} B: {answer}"
    if style == PAIRED_WHAT_WHERE:
        what_question = f"What did {agent} {frame.verb} {location}?"
        what_answer = _sentence(object_reference)
        where_answer = answer
        return (
            f"{scene_sentence} A: {what_question} B: {what_answer} "
            f"A: {question} B: {where_answer}"
        )
    raise ValueError(f"Unsupported style: {style!r}")


def _document_from_plan(plan: LocalPrepPlan) -> dict[str, Any]:
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
    prepositions: tuple[str, ...],
    family_names: tuple[str, ...],
) -> dict[str, Any]:
    counts = Counter(row["audit_status"] for row in documents)
    style_counts = Counter(str(row["style"]) for row in documents)
    family_counts = Counter(str(row["verb_family"]) for row in documents)
    verb_counts = Counter(str(row["verb"]) for row in documents)
    preposition_counts = Counter(str(row["answer_preposition"]) for row in documents)
    answer_style_counts = Counter(str(row["answer_style"]) for row in documents)
    where_question_style_counts = Counter(str(row["where_question_style"]) for row in documents)
    second_person_counts = Counter(
        "second_person" if row["second_person"] else "third_person" for row in documents
    )
    question_count_counts = Counter(str(row["text"].count("?")) for row in documents)
    question_object_reference_counts = Counter(
        str(row["question_object_reference"]) for row in documents
    )
    preposition_values = [preposition_counts[name] for name in prepositions]
    family_values = [family_counts[name] for name in family_names]
    flags = []
    if max(preposition_values) - min(preposition_values) > 1:
        flags.append(
            {
                "severity": "warn",
                "code": "preposition_balance",
                "message": "Preposition counts differ by more than one.",
            }
        )
    if max(family_values) - min(family_values) > 1:
        flags.append(
            {
                "severity": "warn",
                "code": "verb_family_balance",
                "message": "Verb family counts differ by more than one.",
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
        "style_counts": dict(sorted(style_counts.items())),
        "verb_family_counts": dict(sorted(family_counts.items())),
        "verb_counts": dict(sorted(verb_counts.items())),
        "answer_preposition_counts": dict(sorted(preposition_counts.items())),
        "answer_style_counts": dict(sorted(answer_style_counts.items())),
        "where_question_style_counts": dict(sorted(where_question_style_counts.items())),
        "second_person_counts": dict(sorted(second_person_counts.items())),
        "question_count_counts": dict(sorted(question_count_counts.items())),
        "question_object_reference_counts": dict(
            sorted(question_object_reference_counts.items())
        ),
        "flags": flags,
    }


def _audit_document(row: dict[str, Any]) -> dict[str, Any]:
    flags = []
    text = str(row["text"])
    lower = text.lower()
    question = str(row["question"])
    answer = str(row["answer"])
    question_count = text.count("?")
    expected_questions = 2 if row["style"] == PAIRED_WHAT_WHERE else 1
    if question_count != expected_questions:
        flags.append(_flag("error", "question_count", "Unexpected question count."))
    if not _valid_where_question_shape(
        where_question_style=str(row["where_question_style"]),
        question=question,
        object_reference=str(row["question_object_reference"]),
    ):
        flags.append(_flag("error", "question_shape", "Unexpected where-question shape."))
    if row["style"] == PAIRED_WHAT_WHERE and "What did " not in text:
        flags.append(_flag("error", "missing_what_question", "Paired style requires a what question."))
    for marker in BANNED_CONTRAST_MARKERS:
        if marker in lower:
            flags.append(_flag("error", "contrast_marker", f"Found contrast marker: {marker}"))
    if question not in text:
        flags.append(_flag("error", "missing_question", "Text is missing the planned where question."))
    if answer not in text:
        flags.append(_flag("error", "missing_answer", "Text is missing the planned answer."))
    if "\n" in text:
        flags.append(_flag("error", "multiline_text", "Train text must stay one line per example."))
    banned_words = _banned_words_in_text(
        " ".join(
            [
                str(row["object"]),
                str(row["landmark"]),
                str(row["location"]),
            ]
        )
    )
    if banned_words:
        flags.append(
            _flag(
                "error",
                "benchmark_word_overlap",
                f"Found benchmark words in object/location fields: {sorted(banned_words)}",
            )
        )
    status = "fail" if any(flag["severity"] == "error" for flag in flags) else "pass"
    return {"status": status, "flags": flags}


def _ratio(mapping: dict[str, Any], key: str, default: float) -> float:
    value = float(mapping.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1.")
    return value


def _style_weights(mapping: dict[str, Any]) -> dict[str, float]:
    raw = mapping.get(
        "style_weights",
        {
            DIRECT_DIALOGUE: 0.40,
            SCENE_QA: 0.35,
            PAIRED_WHAT_WHERE: 0.25,
        },
    )
    weights = {str(key): float(value) for key, value in dict(raw).items()}
    unknown = sorted(set(weights) - set(STYLE_ORDER))
    if unknown:
        raise ValueError(f"Unknown styles: {unknown}")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("style_weights must have positive total weight.")
    return {style: weights.get(style, 0.0) / total for style in STYLE_ORDER}


def _where_question_style_weights(mapping: dict[str, Any]) -> dict[str, float]:
    raw = mapping.get(
        "where_question_style_weights",
        {
            WHERE_DIRECT_ACTION: 0.70,
            WHERE_NOW: 0.15,
            WHERE_END_UP: 0.10,
            WHERE_INDIRECT: 0.05,
        },
    )
    weights = {str(key): float(value) for key, value in dict(raw).items()}
    unknown = sorted(set(weights) - set(WHERE_QUESTION_STYLE_ORDER))
    if unknown:
        raise ValueError(f"Unknown where_question_style_weights: {unknown}")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("where_question_style_weights must have positive total weight.")
    return {
        style: weights.get(style, 0.0) / total
        for style in WHERE_QUESTION_STYLE_ORDER
    }


def _weighted_style_sequence(
    count: int,
    *,
    style_weights: dict[str, float],
    rng: random.Random,
) -> list[str]:
    raw_counts = {style: count * style_weights[style] for style in STYLE_ORDER}
    style_counts = {style: int(raw_counts[style]) for style in STYLE_ORDER}
    assigned = sum(style_counts.values())
    remainders = sorted(
        STYLE_ORDER,
        key=lambda style: (raw_counts[style] - style_counts[style], style),
        reverse=True,
    )
    for style in remainders[: count - assigned]:
        style_counts[style] += 1
    sequence: list[str] = []
    for style in STYLE_ORDER:
        sequence.extend([style] * style_counts[style])
    rng.shuffle(sequence)
    return sequence


def _weighted_where_question_style_sequence(
    count: int,
    *,
    where_question_style_weights: dict[str, float],
    rng: random.Random,
) -> list[str]:
    raw_counts = {
        style: count * where_question_style_weights[style]
        for style in WHERE_QUESTION_STYLE_ORDER
    }
    style_counts = {style: int(raw_counts[style]) for style in WHERE_QUESTION_STYLE_ORDER}
    assigned = sum(style_counts.values())
    remainders = sorted(
        WHERE_QUESTION_STYLE_ORDER,
        key=lambda style: (raw_counts[style] - style_counts[style], style),
        reverse=True,
    )
    for style in remainders[: count - assigned]:
        style_counts[style] += 1
    sequence: list[str] = []
    for style in WHERE_QUESTION_STYLE_ORDER:
        sequence.extend([style] * style_counts[style])
    rng.shuffle(sequence)
    return sequence


def _balanced_sequence(values: tuple[str, ...], count: int, *, rng: random.Random) -> list[str]:
    base, remainder = divmod(count, len(values))
    sequence: list[str] = []
    for index, value in enumerate(values):
        sequence.extend([value] * (base + (1 if index < remainder else 0)))
    rng.shuffle(sequence)
    return sequence


def _landmark_from_location(location: str, preposition: str) -> str:
    text = location[len(preposition):].strip()
    return text[4:] if text.startswith("the ") else text


def _render_where_question(
    *,
    style: str,
    agent: str,
    frame: LocalPrepVerbFrame,
    object_reference: str,
) -> str:
    if style == WHERE_DIRECT_ACTION:
        return f"Where did {agent} {frame.verb} {object_reference}?"
    if style == WHERE_NOW:
        return f"Where is {object_reference} now?"
    if style == WHERE_END_UP:
        return f"Where did {object_reference} end up?"
    if style == WHERE_INDIRECT:
        return f"Can you tell me where {object_reference} is?"
    raise ValueError(f"Unsupported where question style: {style!r}")


def _valid_where_question_shape(
    *,
    where_question_style: str,
    question: str,
    object_reference: str,
) -> bool:
    if where_question_style == WHERE_DIRECT_ACTION:
        return question.startswith("Where did ") and question.endswith("?")
    if where_question_style == WHERE_NOW:
        return question == f"Where is {object_reference} now?"
    if where_question_style == WHERE_END_UP:
        return question == f"Where did {object_reference} end up?"
    if where_question_style == WHERE_INDIRECT:
        return question == f"Can you tell me where {object_reference} is?"
    return False


def _banned_words_in_text(text: str) -> set[str]:
    words = {word.lower() for word in WORD_PATTERN.findall(text)}
    return words & BANNED_BENCHMARK_WORDS


def _sentence(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}."


def _word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def _flag(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


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
