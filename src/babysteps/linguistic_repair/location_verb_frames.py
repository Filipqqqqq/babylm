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
DEFAULT_EXCLUDED_NAMED_LOCATIONS = ("Canada", "Europe", "California")
NAMED_POOL_PREFIX = "named:"
NAMED_DESTINATION_MODES = frozenset(
    {
        "motion_destination",
        "travel_destination",
        "ride_destination",
        "fly_destination",
        "drive_destination",
        "send_destination",
        "broad_address_destination",
    }
)
NAMED_DESTINATION_KINDS_BY_MODE = {
    "motion_destination": frozenset({"city", "country", "state"}),
    "travel_destination": frozenset({"city", "country", "state"}),
    "ride_destination": frozenset({"city", "state"}),
    "fly_destination": frozenset({"city", "country", "state"}),
    "drive_destination": frozenset({"city", "state"}),
    "send_destination": frozenset({"city", "country", "state"}),
    "broad_address_destination": frozenset({"city", "country", "state"}),
}
DEFAULT_SECOND_PERSON_PRONOUN_WEIGHTS = {
    "find": 1.0,
    "discover": 1.0,
    "spot": 1.0,
    "notice": 1.0,
    "leave": 0.65,
    "drop": 0.8,
    "hide": 0.8,
    "store": 0.8,
    "put": 0.8,
    "place": 0.8,
    "set": 0.8,
}


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
    named_location: bool


@dataclass(frozen=True)
class NamedLocation:
    name: str
    kind: str


DEFAULT_NAMED_LOCATIONS = (
    NamedLocation("France", "country"),
    NamedLocation("Germany", "country"),
    NamedLocation("Spain", "country"),
    NamedLocation("India", "country"),
    NamedLocation("China", "country"),
    NamedLocation("Japan", "country"),
    NamedLocation("Mexico", "country"),
    NamedLocation("Brazil", "country"),
    NamedLocation("Chicago", "city"),
    NamedLocation("New York", "city"),
    NamedLocation("Paris", "city"),
    NamedLocation("Boston", "city"),
    NamedLocation("Seattle", "city"),
    NamedLocation("Denver", "city"),
    NamedLocation("London", "city"),
    NamedLocation("Arizona", "state"),
    NamedLocation("Texas", "state"),
    NamedLocation("Oregon", "state"),
    NamedLocation("Florida", "state"),
)


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
    answer_style: str
    scene_sentence: str | None
    text: str
    source_location: str
    answer_location: str
    source_location_type: str
    answer_preposition: str
    derived_location: bool
    named_location: bool
    second_person: bool
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
    named_location_ratio = float(mapping.get("named_location_ratio", 0.0))
    if not 0.0 <= named_location_ratio <= 1.0:
        raise ValueError("named_location_ratio must be between 0 and 1.")
    second_person_ratio = float(mapping.get("second_person_ratio", 0.0))
    if not 0.0 <= second_person_ratio <= 1.0:
        raise ValueError("second_person_ratio must be between 0 and 1.")
    second_person_full_answer_ratio = float(
        mapping.get("second_person_full_answer_ratio", 0.5)
    )
    if not 0.0 <= second_person_full_answer_ratio <= 1.0:
        raise ValueError("second_person_full_answer_ratio must be between 0 and 1.")
    second_person_pronoun_weights = _second_person_pronoun_weights(mapping)
    family_names = tuple(str(name) for name in mapping.get("verb_families", VERB_FAMILIES))
    unknown = sorted(set(family_names) - set(VERB_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown verb families: {unknown}")
    if not family_names:
        raise ValueError("verb_families must not be empty.")
    allowed_verbs = _allowed_verbs(mapping)
    verb_frames = _verb_frames_for_config(
        family_names=family_names,
        allowed_verbs=allowed_verbs,
    )

    rng = random.Random(seed)
    agents = _agent_pool(mapping)
    named_locations = _named_location_pool(mapping)
    excluded_location_strings = _excluded_location_strings(mapping)
    locations = _build_location_pools(
        named_locations=named_locations,
        excluded_location_strings=excluded_location_strings,
    )
    plans = _build_plans(
        name=name,
        document_count=document_count,
        family_names=family_names,
        verb_frames=verb_frames,
        direct_ratio=direct_ratio,
        named_location_ratio=named_location_ratio,
        second_person_ratio=second_person_ratio,
        second_person_full_answer_ratio=second_person_full_answer_ratio,
        second_person_pronoun_weights=second_person_pronoun_weights,
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
        "named_location_ratio": named_location_ratio,
        "second_person_ratio": second_person_ratio,
        "second_person_full_answer_ratio": second_person_full_answer_ratio,
        "second_person_pronoun_weights": dict(sorted(second_person_pronoun_weights.items())),
        "excluded_named_locations": list(
            mapping.get("excluded_named_locations", DEFAULT_EXCLUDED_NAMED_LOCATIONS)
        ),
        "excluded_location_strings": sorted(excluded_location_strings),
        "allowed_verbs": sorted(allowed_verbs) if allowed_verbs is not None else None,
        "verb_families": list(family_names),
        "counts": {
            "documents": len(documents),
            "train_documents": sum(1 for row in documents if row["audit_status"] == "pass"),
            **audit_report["counts"],
            "verb_family_counts": audit_report["verb_family_counts"],
            "style_counts": audit_report["style_counts"],
            "source_location_type_counts": audit_report["source_location_type_counts"],
            "answer_preposition_counts": audit_report["answer_preposition_counts"],
            "named_location_counts": audit_report["named_location_counts"],
            "second_person_counts": audit_report["second_person_counts"],
            "answer_style_counts": audit_report["answer_style_counts"],
            "question_object_reference_counts": audit_report[
                "question_object_reference_counts"
            ],
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
    verb_frames: dict[str, tuple[VerbFrame, ...]],
    direct_ratio: float,
    named_location_ratio: float,
    second_person_ratio: float,
    second_person_full_answer_ratio: float,
    second_person_pronoun_weights: dict[str, float],
    agents: tuple[str, ...],
    locations: dict[str, tuple[LocationCandidate, ...]],
    rng: random.Random,
) -> list[LocationVerbFramePlan]:
    family_sequence = _balanced_sequence(family_names, document_count, rng=rng)
    styles = _style_sequence(document_count, direct_ratio=direct_ratio, rng=rng)
    family_offsets: Counter[str] = Counter()
    plans: list[LocationVerbFramePlan] = []
    for index, family_name in enumerate(family_sequence):
        frames = verb_frames[family_name]
        frame = frames[family_offsets[family_name] % len(frames)]
        family_offsets[family_name] += 1
        location_pool = locations[frame.location_mode]
        named_location_pool = locations.get(f"{NAMED_POOL_PREFIX}{frame.location_mode}", ())
        if named_location_pool and rng.random() < named_location_ratio:
            location_pool = named_location_pool
        location = location_pool[(index + rng.randrange(len(location_pool))) % len(location_pool)]
        agent = rng.choice(agents)
        object_pool = _object_pool_for_location(frame, location)
        obj = rng.choice(object_pool) if object_pool else None
        object_reference = f"the {obj}" if obj else None
        second_person = second_person_ratio > 0.0 and rng.random() < second_person_ratio
        question_object_reference = _question_object_reference(
            frame=frame,
            object_reference=object_reference,
            second_person=second_person,
            pronoun_weights=second_person_pronoun_weights,
            rng=rng,
        )
        question = _where_question(
            agent="you" if second_person else agent,
            frame=frame,
            object_reference=question_object_reference,
        )
        full_second_person_answer = (
            second_person
            and rng.random() < second_person_full_answer_ratio
        )
        answer, answer_style = _answer_for_location(
            frame=frame,
            location=location,
            object_reference=question_object_reference,
            second_person=second_person,
            full_second_person_answer=full_second_person_answer,
        )
        scene_sentence = (
            _scene_sentence(
                agent="I" if second_person else agent,
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
                answer_style=answer_style,
                scene_sentence=scene_sentence,
                text=text,
                source_location=location.source_location,
                answer_location=location.answer,
                source_location_type=location.source_location_type,
                answer_preposition=location.answer_preposition,
                derived_location=location.derived,
                named_location=location.named_location,
                second_person=second_person,
                template_id=styles[index],
            )
        )
    return plans


def _object_pool_for_location(
    frame: VerbFrame,
    location: LocationCandidate,
) -> tuple[str, ...]:
    if location.named_location and frame.verb == "drive":
        return tuple(obj for obj in frame.object_pool if obj != "cart")
    return frame.object_pool


def _second_person_pronoun_weights(mapping: dict[str, Any]) -> dict[str, float]:
    weights = dict(DEFAULT_SECOND_PERSON_PRONOUN_WEIGHTS)
    for verb, raw_weight in mapping.get("second_person_pronoun_weights", {}).items():
        weight = float(raw_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("second_person_pronoun_weights must be between 0 and 1.")
        weights[str(verb)] = weight
    return weights


def _allowed_verbs(mapping: dict[str, Any]) -> frozenset[str] | None:
    raw = mapping.get("allowed_verbs")
    if raw is None:
        return None
    verbs = frozenset(str(verb) for verb in raw)
    if not verbs:
        raise ValueError("allowed_verbs must not be empty when provided.")
    known_verbs = {
        frame.verb
        for frames in VERB_FAMILIES.values()
        for frame in frames
    }
    unknown = sorted(verbs - known_verbs)
    if unknown:
        raise ValueError(f"Unknown allowed_verbs: {unknown}")
    return verbs


def _verb_frames_for_config(
    *,
    family_names: tuple[str, ...],
    allowed_verbs: frozenset[str] | None,
) -> dict[str, tuple[VerbFrame, ...]]:
    frames_by_family: dict[str, tuple[VerbFrame, ...]] = {}
    for family_name in family_names:
        frames = VERB_FAMILIES[family_name]
        if allowed_verbs is not None:
            frames = tuple(frame for frame in frames if frame.verb in allowed_verbs)
        if not frames:
            raise ValueError(
                f"verb family {family_name!r} has no frames after allowed_verbs filtering."
            )
        frames_by_family[family_name] = frames
    return frames_by_family


def _question_object_reference(
    *,
    frame: VerbFrame,
    object_reference: str | None,
    second_person: bool,
    pronoun_weights: dict[str, float],
    rng: random.Random,
) -> str | None:
    if not second_person or object_reference is None:
        return object_reference
    pronoun_weight = pronoun_weights.get(frame.verb, 0.0)
    if pronoun_weight > 0.0 and rng.random() < pronoun_weight:
        return "it"
    return object_reference


def _answer_for_location(
    *,
    frame: VerbFrame,
    location: LocationCandidate,
    object_reference: str | None,
    second_person: bool,
    full_second_person_answer: bool,
) -> tuple[str, str]:
    if not second_person or not full_second_person_answer:
        return _answer_sentence(location.answer), "location_fragment"
    object_part = f" {object_reference}" if object_reference else ""
    return (
        _answer_sentence(f"I {frame.past}{object_part} {location.answer}"),
        "first_person_location_sentence",
    )


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


def _named_location_pool(mapping: dict[str, Any]) -> tuple[NamedLocation, ...]:
    raw_locations = mapping.get("named_locations", DEFAULT_NAMED_LOCATIONS)
    raw_excluded = mapping.get("excluded_named_locations", DEFAULT_EXCLUDED_NAMED_LOCATIONS)
    excluded = {_normalize_place_name(str(location)) for location in raw_excluded}
    named_locations: list[NamedLocation] = []
    for item in raw_locations:
        if isinstance(item, NamedLocation):
            location = item
        elif isinstance(item, dict):
            location = NamedLocation(name=str(item["name"]), kind=str(item.get("kind", "place")))
        else:
            name = str(item)
            location = NamedLocation(name=name, kind=_infer_named_location_kind(name))
        if _normalize_place_name(location.name) not in excluded:
            named_locations.append(location)
    return tuple(named_locations)


def _build_location_pools(
    *,
    named_locations: tuple[NamedLocation, ...],
    excluded_location_strings: frozenset[str],
) -> dict[str, tuple[LocationCandidate, ...]]:
    unique_locations = sorted({location for topic in TOPICS for location in topic.locations})
    local_any = tuple(
        LocationCandidate(
            answer=location,
            scene_location=location,
            source_location=location,
            source_location_type=_source_location_type(location),
            answer_preposition=_first_word(location),
            derived=False,
            named_location=False,
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
                named_location=False,
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
                named_location=False,
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
    for mode in NAMED_DESTINATION_MODES:
        named_destinations = _named_destination_candidates(
            named_locations,
            allowed_kinds=NAMED_DESTINATION_KINDS_BY_MODE[mode],
        )
        if named_destinations and mode in pools:
            pools[f"{NAMED_POOL_PREFIX}{mode}"] = named_destinations
    if excluded_location_strings:
        pools = {
            name: _exclude_locations(pool, excluded_location_strings)
            for name, pool in pools.items()
        }
    if any(not pool for pool in pools.values()):
        raise ValueError("location pools must not be empty.")
    return pools


def _excluded_location_strings(mapping: dict[str, Any]) -> frozenset[str]:
    raw = mapping.get("excluded_location_strings", ())
    return frozenset(_normalize_location_string(str(item)) for item in raw)


def _exclude_locations(
    candidates: tuple[LocationCandidate, ...],
    excluded_location_strings: frozenset[str],
) -> tuple[LocationCandidate, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if _normalize_location_string(candidate.answer) not in excluded_location_strings
        and _normalize_location_string(candidate.scene_location) not in excluded_location_strings
        and _normalize_location_string(candidate.source_location) not in excluded_location_strings
    )


def _named_destination_candidates(
    named_locations: tuple[NamedLocation, ...],
    *,
    allowed_kinds: frozenset[str],
) -> tuple[LocationCandidate, ...]:
    # Experimental repair signal: use named places to test whether richer
    # location nouns help the model learn where-answer typing.
    candidates = [
        LocationCandidate(
            answer=f"to {location.name}",
            scene_location=f"to {location.name}",
            source_location=location.name,
            source_location_type=f"named_{location.kind}_destination",
            answer_preposition="to",
            derived=False,
            named_location=True,
        )
        for location in named_locations
        if location.kind in allowed_kinds
    ]
    return tuple(sorted(candidates, key=lambda candidate: candidate.answer.lower()))


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
    named_counts = Counter("named" if row.get("named_location") else "generic" for row in documents)
    second_person_counts = Counter(
        "second_person" if row.get("second_person") else "third_person"
        for row in documents
    )
    answer_style_counts = Counter(str(row["answer_style"]) for row in documents)
    question_object_reference_counts = Counter(
        _question_object_reference_surface(str(row["question"])) for row in documents
    )
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
        "named_location_counts": dict(sorted(named_counts.items())),
        "second_person_counts": dict(sorted(second_person_counts.items())),
        "answer_style_counts": dict(sorted(answer_style_counts.items())),
        "question_object_reference_counts": dict(
            sorted(question_object_reference_counts.items())
        ),
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


def _normalize_place_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_location_string(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _infer_named_location_kind(name: str) -> str:
    normalized = _normalize_place_name(name)
    if normalized in {"arizona", "texas", "oregon", "florida", "california"}:
        return "state"
    if normalized in {"chicago", "new york", "paris", "boston", "seattle", "denver", "london"}:
        return "city"
    return "country"


def _question_object_reference_surface(question: str) -> str:
    prefix = "Where did "
    if not question.startswith(prefix):
        return "unknown"
    body = question[len(prefix):].rstrip("?")
    parts = body.split(maxsplit=2)
    if len(parts) < 3:
        return "none"
    return parts[2]


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
