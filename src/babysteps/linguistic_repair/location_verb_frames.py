"""Deterministic where-question location verb-frame repair data."""

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

from babysteps.linguistic_repair.planning import TOPICS
from babysteps.paths import resolve_path


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
DIRECT_DIALOGUE = "direct_dialogue"
SCENE_QA = "scene_qa"
STYLE_ORDER = (DIRECT_DIALOGUE, SCENE_QA)
QUESTION_TYPE = "where"
BANNED_QUESTION_WORDS = ("when", "why", "how many", "how_many")
BANNED_CONTRAST_MARKERS = ("bad:", "wrong:", "incorrect:", "bad answer", "wrong answer")


@dataclass(frozen=True)
class VerbFrame:
    family: str
    verb: str
    past: str
    object_pool: tuple[str, ...]
    location_mode: str


@dataclass(frozen=True)
class LocationCandidate:
    answer: str
    scene_location: str
    source_location: str
    source_location_type: str
    answer_preposition: str
    derived: bool


@dataclass(frozen=True)
class LocationVerbFramePlan:
    id: str
    index: int
    style: str
    verb_family: str
    verb: str
    verb_past: str
    agent: str
    object: str | None
    object_reference: str | None
    question: str
    answer: str
    scene_sentence: str | None
    text: str
    source_location: str
    answer_location: str
    source_location_type: str
    answer_preposition: str
    derived_location: bool
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QaLocationVerbFrameResult:
    name: str
    output_dir: Path
    documents: tuple[dict[str, Any], ...]
    audit_report: dict[str, Any]
    outputs: dict[str, Path]
    manifest: dict[str, Any]


VERB_FAMILIES: dict[str, tuple[VerbFrame, ...]] = {
    "send_mail_ship_deliver": (
        VerbFrame("send_mail_ship_deliver", "send", "sent", ("letter", "package", "postcard"), "send_destination"),
        VerbFrame("send_mail_ship_deliver", "mail", "mailed", ("letter", "card", "postcard"), "broad_address_destination"),
        VerbFrame("send_mail_ship_deliver", "ship", "shipped", ("package", "box", "gift"), "broad_address_destination"),
        VerbFrame("send_mail_ship_deliver", "deliver", "delivered", ("letter", "package", "receipt"), "deliver_destination"),
    ),
    "go_walk_head_return": (
        VerbFrame("go_walk_head_return", "go", "went", (), "motion_destination"),
        VerbFrame("go_walk_head_return", "walk", "walked", (), "walk_destination"),
        VerbFrame("go_walk_head_return", "head", "headed", (), "motion_destination"),
        VerbFrame("go_walk_head_return", "return", "returned", (), "walk_destination"),
    ),
    "travel_ride_fly": (
        VerbFrame("travel_ride_fly", "travel", "traveled", (), "travel_destination"),
        VerbFrame("travel_ride_fly", "ride", "rode", (), "ride_destination"),
        VerbFrame("travel_ride_fly", "fly", "flew", (), "fly_destination"),
    ),
    "drive_park": (
        VerbFrame("drive_park", "drive", "drove", ("van", "cart", "car"), "drive_destination"),
        VerbFrame("drive_park", "park", "parked", ("cart", "bike", "stroller"), "park_location"),
    ),
    "find_discover_spot_notice": (
        VerbFrame("find_discover_spot_notice", "find", "found", ("ticket", "key", "note"), "local_any"),
        VerbFrame("find_discover_spot_notice", "discover", "discovered", ("ticket", "key", "glove"), "local_any"),
        VerbFrame("find_discover_spot_notice", "spot", "spotted", ("sticker", "shell", "card"), "local_any"),
        VerbFrame("find_discover_spot_notice", "notice", "noticed", ("ticket", "note", "receipt"), "local_any"),
    ),
    "leave_drop_hide_store": (
        VerbFrame("leave_drop_hide_store", "leave", "left", ("notebook", "ticket", "bag"), "local_any"),
        VerbFrame("leave_drop_hide_store", "drop", "dropped", ("key", "card", "glove"), "local_any"),
        VerbFrame("leave_drop_hide_store", "hide", "hid", ("note", "toy car", "box"), "local_any"),
        VerbFrame("leave_drop_hide_store", "store", "stored", ("notebook", "package", "bag"), "local_any"),
    ),
    "carry_bring_take": (
        VerbFrame("carry_bring_take", "carry", "carried", ("basket", "box", "bag"), "carry_destination"),
        VerbFrame("carry_bring_take", "bring", "brought", ("notebook", "package", "cup"), "carry_destination"),
        VerbFrame("carry_bring_take", "take", "took", ("letter", "ticket", "bag"), "carry_destination"),
    ),
    "put_place_set": (
        VerbFrame("put_place_set", "put", "put", ("notebook", "cup", "receipt"), "local_surface_container"),
        VerbFrame("put_place_set", "place", "placed", ("card", "ticket", "box"), "local_surface_container"),
        VerbFrame("put_place_set", "set", "set", ("notebook", "cup", "bag"), "local_surface_container"),
    ),
}

MOTION_TARGET_KEYWORDS = (
    "city",
    "country",
    "town",
    "state",
    "center",
    "college",
    "preschool",
    "room",
    "classroom",
    "office",
    "trees",
    "park",
    "workplace",
    "pond",
    "hallway",
    "door",
    "board",
    "bench",
    "goal",
    "sideline",
)
TRAVEL_TARGET_KEYWORDS = (
    "city",
    "country",
    "town",
    "state",
    "college",
    "workplace",
    "park",
)
WALK_TARGET_KEYWORDS = (
    "center",
    "college",
    "preschool",
    "room",
    "classroom",
    "office",
    "trees",
    "park",
    "workplace",
    "pond",
    "hallway",
    "door",
    "board",
    "bench",
    "goal",
    "sideline",
    "wall",
)
RIDE_TARGET_KEYWORDS = (
    "center",
    "college",
    "preschool",
    "park",
    "workplace",
    "hallway",
    "bench",
    "goal",
    "sideline",
)
FLY_TARGET_KEYWORDS = (
    "city",
    "country",
    "town",
    "state",
)
DRIVE_TARGET_KEYWORDS = (
    "city",
    "town",
    "state",
    "center",
    "college",
    "preschool",
    "park",
    "workplace",
)
SEND_TARGET_KEYWORDS = (
    "city",
    "country",
    "town",
    "state",
    "desk",
    "register",
    "counter",
    "door",
    "case",
    "office",
    "room",
    "preschool",
    "center",
    "college",
    "workplace",
)
BROAD_ADDRESS_TARGET_KEYWORDS = (
    "city",
    "country",
    "town",
    "state",
)
DELIVER_TARGET_KEYWORDS = (
    "desk",
    "register",
    "counter",
    "door",
    "case",
    "office",
    "room",
    "classroom",
    "preschool",
    "center",
    "college",
    "workplace",
)
CARRY_TARGET_KEYWORDS = (
    "desk",
    "register",
    "counter",
    "door",
    "case",
    "office",
    "room",
    "preschool",
    "center",
    "college",
    "workplace",
    "bag",
    "box",
    "bin",
    "tray",
    "basket",
    "closet",
    "shelf",
    "table",
)
PARK_TARGET_KEYWORDS = (
    "park",
    "workplace",
    "preschool",
    "center",
    "college",
    "sideline",
    "goal",
    "bench",
    "door",
    "register",
    "counter",
    "case",
    "wall",
    "pond",
    "trees",
)


def build_qa_location_verb_frames(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> QaLocationVerbFrameResult:
    """Build deterministic QA location verb-frame repair documents."""

    if not isinstance(mapping, dict):
        raise TypeError("QA location verb-frame config must be a mapping.")

    name = str(mapping["name"])
    seed = int(mapping.get("seed", 42))
    document_count = int(mapping.get("document_count", 500))
    if document_count <= 0:
        raise ValueError("document_count must be positive.")

    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    direct_ratio = float(mapping.get("direct_dialogue_ratio", 0.7))
    family_names = tuple(str(name) for name in mapping.get("verb_families", VERB_FAMILIES))
    unknown = sorted(set(family_names) - set(VERB_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown verb families: {unknown}")
    if not family_names:
        raise ValueError("verb_families must not be empty.")

    rng = random.Random(seed)
    agents = _agent_pool(mapping)
    locations = _build_location_pools()
    plans = _build_plans(
        name=name,
        document_count=document_count,
        family_names=family_names,
        direct_ratio=direct_ratio,
        agents=agents,
        locations=locations,
        rng=rng,
    )
    documents = tuple(_document_from_plan(plan) for plan in plans)
    audit_report = _audit_documents(name, documents, family_names=family_names)

    paths = {
        "documents": output_dir / "documents.jsonl",
        "train": output_dir / "train.txt",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
        "config": output_dir / "qa_location_verb_frames_config.yaml",
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
        "verb_families": list(family_names),
        "counts": {
            "documents": len(documents),
            "train_documents": sum(1 for row in documents if row["audit_status"] == "pass"),
            **audit_report["counts"],
            "verb_family_counts": audit_report["verb_family_counts"],
            "style_counts": audit_report["style_counts"],
            "source_location_type_counts": audit_report["source_location_type_counts"],
            "answer_preposition_counts": audit_report["answer_preposition_counts"],
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    return QaLocationVerbFrameResult(
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
    locations: dict[str, tuple[LocationCandidate, ...]],
    rng: random.Random,
) -> list[LocationVerbFramePlan]:
    family_sequence = _balanced_sequence(family_names, document_count, rng=rng)
    styles = _style_sequence(document_count, direct_ratio=direct_ratio, rng=rng)
    family_offsets: Counter[str] = Counter()
    plans: list[LocationVerbFramePlan] = []
    for index, family_name in enumerate(family_sequence):
        frames = VERB_FAMILIES[family_name]
        frame = frames[family_offsets[family_name] % len(frames)]
        family_offsets[family_name] += 1
        location_pool = locations[frame.location_mode]
        location = location_pool[(index + rng.randrange(len(location_pool))) % len(location_pool)]
        agent = rng.choice(agents)
        obj = rng.choice(frame.object_pool) if frame.object_pool else None
        object_reference = f"the {obj}" if obj else None
        question = _where_question(agent=agent, frame=frame, object_reference=object_reference)
        answer = _answer_sentence(location.answer)
        scene_sentence = (
            _scene_sentence(
                agent=agent,
                frame=frame,
                object_reference=object_reference,
                location=location.scene_location,
            )
            if styles[index] == SCENE_QA
            else None
        )
        text = _render_text(
            question=question,
            answer=answer,
            scene_sentence=scene_sentence,
            style=styles[index],
        )
        plans.append(
            LocationVerbFramePlan(
                id=f"{name}_{index:05d}",
                index=index,
                style=styles[index],
                verb_family=family_name,
                verb=frame.verb,
                verb_past=frame.past,
                agent=agent,
                object=obj,
                object_reference=object_reference,
                question=question,
                answer=answer,
                scene_sentence=scene_sentence,
                text=text,
                source_location=location.source_location,
                answer_location=location.answer,
                source_location_type=location.source_location_type,
                answer_preposition=location.answer_preposition,
                derived_location=location.derived,
                template_id=styles[index],
            )
        )
    return plans


def _agent_pool(mapping: dict[str, Any]) -> tuple[str, ...]:
    configured = mapping.get("agents")
    if configured:
        return tuple(str(agent) for agent in configured)
    agents = sorted(
        {
            agent
            for topic in TOPICS
            for agent in topic.agents
            if not agent.lower().startswith("the ")
        }
    )
    if not agents:
        raise ValueError("agent pool must not be empty.")
    return tuple(agents)


def _build_location_pools() -> dict[str, tuple[LocationCandidate, ...]]:
    unique_locations = sorted({location for topic in TOPICS for location in topic.locations})
    local_any = tuple(
        LocationCandidate(
            answer=location,
            scene_location=location,
            source_location=location,
            source_location_type=_source_location_type(location),
            answer_preposition=_first_word(location),
            derived=False,
        )
        for location in unique_locations
        if not _is_destination(location)
    )
    destination_candidates: list[LocationCandidate] = []
    for location in unique_locations:
        if _is_destination(location):
            candidate = LocationCandidate(
                answer=location,
                scene_location=location,
                source_location=location,
                source_location_type=_source_location_type(location),
                answer_preposition=_first_word(location),
                derived=False,
            )
        else:
            answer = _derive_destination(location)
            candidate = LocationCandidate(
                answer=answer,
                scene_location=answer,
                source_location=location,
                source_location_type=_source_location_type(location),
                answer_preposition=_first_word(answer),
                derived=True,
            )
        destination_candidates.append(candidate)

    destination_by_answer = {candidate.answer: candidate for candidate in destination_candidates}
    all_destinations = tuple(destination_by_answer[key] for key in sorted(destination_by_answer))
    pools = {
        "local_any": local_any,
        "local_surface_container": tuple(
            candidate
            for candidate in local_any
            if candidate.source_location_type in {"container_interior", "local_surface"}
        ),
        "park_location": tuple(
            candidate
            for candidate in local_any
            if candidate.answer_preposition in {"at", "beside", "near", "by", "under"}
            and any(keyword in _target_text(candidate.source_location) for keyword in PARK_TARGET_KEYWORDS)
        ),
        "motion_destination": _filter_destinations(
            all_destinations,
            keywords=MOTION_TARGET_KEYWORDS,
        ),
        "walk_destination": _filter_destinations(
            all_destinations,
            keywords=WALK_TARGET_KEYWORDS,
            include_existing_destinations=False,
        ),
        "travel_destination": _filter_destinations(
            all_destinations,
            keywords=TRAVEL_TARGET_KEYWORDS,
        ),
        "ride_destination": _filter_destinations(
            all_destinations,
            keywords=RIDE_TARGET_KEYWORDS,
            include_existing_destinations=False,
        ),
        "fly_destination": _filter_destinations(
            all_destinations,
            keywords=FLY_TARGET_KEYWORDS,
        ),
        "drive_destination": _filter_destinations(
            all_destinations,
            keywords=DRIVE_TARGET_KEYWORDS,
            include_existing_destinations=False,
        ),
        "send_destination": _filter_destinations(
            all_destinations,
            keywords=SEND_TARGET_KEYWORDS,
        ),
        "broad_address_destination": _filter_destinations(
            all_destinations,
            keywords=BROAD_ADDRESS_TARGET_KEYWORDS,
        ),
        "deliver_destination": _filter_destinations(
            all_destinations,
            keywords=DELIVER_TARGET_KEYWORDS,
            include_existing_destinations=False,
        ),
        "carry_destination": _filter_destinations(
            all_destinations,
            keywords=CARRY_TARGET_KEYWORDS,
            include_existing_destinations=False,
        ),
    }
    if any(not pool for pool in pools.values()):
        raise ValueError("location pools must not be empty.")
    return pools


def _filter_destinations(
    candidates: tuple[LocationCandidate, ...],
    *,
    keywords: tuple[str, ...],
    include_existing_destinations: bool = True,
) -> tuple[LocationCandidate, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if (include_existing_destinations and _is_destination(candidate.source_location))
        or any(keyword in _target_text(candidate.source_location) for keyword in keywords)
    )


def _balanced_sequence(values: tuple[str, ...], count: int, *, rng: random.Random) -> list[str]:
    base, remainder = divmod(count, len(values))
    sequence: list[str] = []
    for index, value in enumerate(values):
        sequence.extend([value] * (base + (1 if index < remainder else 0)))
    rng.shuffle(sequence)
    return sequence


def _style_sequence(count: int, *, direct_ratio: float, rng: random.Random) -> list[str]:
    direct_count = round(count * direct_ratio)
    direct_count = min(max(direct_count, 0), count)
    styles = [DIRECT_DIALOGUE] * direct_count + [SCENE_QA] * (count - direct_count)
    rng.shuffle(styles)
    return styles


def _where_question(*, agent: str, frame: VerbFrame, object_reference: str | None) -> str:
    if object_reference is None:
        return f"Where did {agent} {frame.verb}?"
    return f"Where did {agent} {frame.verb} {object_reference}?"


def _scene_sentence(
    *,
    agent: str,
    frame: VerbFrame,
    object_reference: str | None,
    location: str,
) -> str:
    object_part = f" {object_reference}" if object_reference else ""
    return f"{agent} {frame.past}{object_part} {location} after lunch."


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


def _document_from_plan(plan: LocationVerbFramePlan) -> dict[str, Any]:
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
    family_counts = Counter(str(row["verb_family"]) for row in documents)
    style_counts = Counter(str(row["style"]) for row in documents)
    location_counts = Counter(str(row["source_location_type"]) for row in documents)
    preposition_counts = Counter(str(row["answer_preposition"]) for row in documents)
    family_values = [family_counts[name] for name in family_names]
    balance_delta = max(family_values) - min(family_values) if family_values else 0
    flags = []
    if balance_delta > 1:
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
        "question_type_counts": {QUESTION_TYPE: len(documents)},
        "verb_family_counts": dict(sorted(family_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
        "source_location_type_counts": dict(sorted(location_counts.items())),
        "answer_preposition_counts": dict(sorted(preposition_counts.items())),
        "balance": {
            "verb_family_max_minus_min": balance_delta,
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
    if not question.startswith("Where did "):
        flags.append(_flag("error", "question_shape", "Question must start with 'Where did'."))
    for word in BANNED_QUESTION_WORDS:
        if word in lower and word != QUESTION_TYPE:
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


def _derive_destination(location: str) -> str:
    lower = location.lower()
    for prefix in ("at ", "in ", "on ", "inside ", "into "):
        if lower.startswith(prefix):
            return f"to {location[len(prefix):]}"
    for prefix in ("beside ", "near ", "by "):
        if lower.startswith(prefix):
            return f"to {location[len(prefix):]}"
    if lower.startswith("under "):
        return f"to {location[len('under '):]}"
    return f"to {location}"


def _target_text(location: str) -> str:
    lower = location.lower()
    for prefix in ("across ", "inside ", "beside ", "near ", "under ", "into "):
        if lower.startswith(prefix):
            lower = lower[len(prefix):]
            break
    for prefix in ("to ", "at ", "in ", "on ", "by "):
        if lower.startswith(prefix):
            lower = lower[len(prefix):]
            break
    for article in ("the ", "a ", "an "):
        if lower.startswith(article):
            lower = lower[len(article):]
            break
    return lower


def _source_location_type(location: str) -> str:
    lower = location.lower()
    if lower.startswith(("to ", "across ")):
        return "destination_path"
    if lower.startswith("at "):
        return "activity_place"
    if lower.startswith(("in ", "inside ", "into ")):
        return "container_interior"
    return "local_surface"


def _is_destination(location: str) -> bool:
    return location.lower().startswith(("to ", "across "))


def _first_word(text: str) -> str:
    return text.split(maxsplit=1)[0].lower()


def _answer_sentence(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}."


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
