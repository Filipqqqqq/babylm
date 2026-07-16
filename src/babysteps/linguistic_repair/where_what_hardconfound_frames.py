"""Deterministic where/what hard-confound repair data."""

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

DESTINATION_PATH_NAMED = "destination_path_named"
DESTINATION_PATH_COMMON = "destination_path_common"
OBJECT_ANCHORED_LOCAL = "object_anchored_local"
SCENARIO_GROUP_ORDER = (
    DESTINATION_PATH_NAMED,
    DESTINATION_PATH_COMMON,
    OBJECT_ANCHORED_LOCAL,
)

BARE_FRAGMENT = "bare_fragment"
DIALOGUE_FRAGMENT = "dialogue_fragment"
BARE_FULL_SENTENCE = "bare_full_sentence"
DIALOGUE_FULL_SENTENCE = "dialogue_full_sentence"
STYLE_ORDER = (
    BARE_FRAGMENT,
    DIALOGUE_FRAGMENT,
    BARE_FULL_SENTENCE,
    DIALOGUE_FULL_SENTENCE,
)

WHERE_FIRST = "where_first"
WHAT_FIRST = "what_first"

SECOND_PERSON = "second_person"
THIRD_PERSON = "third_person"

QUESTION_TYPE = "where_what_pair"
BANNED_CONTRAST_MARKERS = ("bad:", "wrong:", "incorrect:", "bad answer", "wrong answer")
BANNED_EXACT_PHRASES = (
    "to california",
    "to europe",
    "across the country",
    "to the next town over",
    "to the other side of town",
    "under the papers",
    "on the table",
    "behind the sofa",
    "next to the bush",
    "where did you send the package?",
)


@dataclass(frozen=True)
class HardConfoundVerbFrame:
    family: str
    verb: str
    past: str
    theme_pool: tuple[str, ...]


@dataclass(frozen=True)
class WhereWhatHardConfoundPlan:
    id: str
    index: int
    scenario_group: str
    style: str
    order: str
    person: str
    speaker_pair: tuple[str, str] | None
    verb_family: str
    verb: str
    verb_past: str
    agent: str
    question_agent: str
    answer_agent: str
    theme: str
    location: str
    location_type: str
    answer_preposition: str
    named_location: bool
    where_question: str
    where_answer: str
    what_question: str
    what_answer: str
    text: str
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["speaker_pair"] = list(self.speaker_pair) if self.speaker_pair else None
        return row


@dataclass(frozen=True)
class QaWhereWhatHardConfoundFrameResult:
    name: str
    output_dir: Path
    documents: tuple[dict[str, Any], ...]
    audit_report: dict[str, Any]
    outputs: dict[str, Path]
    manifest: dict[str, Any]


AGENTS = (
    "Ana",
    "Ben",
    "Maya",
    "Omar",
    "Lia",
    "Nora",
    "Theo",
    "Iris",
    "Jonah",
    "Rina",
    "Eli",
    "Sofia",
    "Mateo",
    "Clara",
    "Noah",
    "Ava",
    "Milo",
    "Zara",
    "Tariq",
    "June",
    "Leo",
    "Mina",
    "Sam",
    "Tessa",
    "Ravi",
    "Grace",
    "Hugo",
    "Nina",
    "Caleb",
    "Ivy",
)

SPEAKER_PAIRS = (
    ("A", "B"),
    ("David", "Sarah"),
    ("Maya", "Omar"),
    ("Lia", "Ben"),
    ("Nora", "Theo"),
    ("Iris", "Leo"),
)

PORTABLE_THEMES = (
    "the letter",
    "the parcel",
    "the small box",
    "the blue bag",
    "the red cup",
    "the toy car",
    "the notebook",
    "the lunch bag",
    "the silver key",
    "the old map",
    "the green ball",
    "the glass bowl",
    "the photo album",
    "the folded blanket",
    "the basket",
    "the yellow scarf",
    "the phone charger",
    "the grocery bag",
    "the library book",
    "the empty jar",
    "the paper lantern",
    "the cardboard sign",
    "the camera case",
    "the water bottle",
    "the puzzle box",
    "the raincoat",
    "the cookie tin",
    "the tool kit",
    "the clay pot",
    "the gift bag",
    "the chess board",
    "the cereal box",
    "the drawing pad",
    "the winter coat",
    "the lunch tray",
    "the flower pot",
    "the folded sign",
    "the soup bowl",
    "the art folder",
    "the toolbox",
    "the beach bag",
    "the lantern",
    "the package",
)

MAILABLE_THEMES = (
    "the letter",
    "the parcel",
    "the small box",
    "the notebook",
    "the silver key",
    "the old map",
    "the photo album",
    "the library book",
    "the cardboard sign",
    "the camera case",
    "the puzzle box",
    "the cookie tin",
    "the gift bag",
    "the chess board",
    "the drawing pad",
    "the folded sign",
    "the art folder",
    "the phone charger",
    "the folded blanket",
    "the yellow scarf",
    "the package",
)

MOVABLE_THEMES = (
    "the small box",
    "the blue bag",
    "the red cup",
    "the toy car",
    "the notebook",
    "the lunch bag",
    "the silver key",
    "the old map",
    "the green ball",
    "the wooden chair",
    "the glass bowl",
    "the folded blanket",
    "the basket",
    "the grocery bag",
    "the library book",
    "the music stand",
    "the picnic basket",
    "the empty jar",
    "the cardboard sign",
    "the camera case",
    "the water bottle",
    "the puzzle box",
    "the raincoat",
    "the cookie tin",
    "the tool kit",
    "the clay pot",
    "the gift bag",
    "the chess board",
    "the cereal box",
    "the drawing pad",
    "the winter coat",
    "the lunch tray",
    "the flower pot",
    "the violin case",
    "the soup bowl",
    "the toolbox",
    "the beach bag",
    "the package",
)

VEHICLE_THEMES = (
    "the van",
    "the truck",
    "the bus",
    "the scooter",
    "the bicycle",
    "the small boat",
)

PUSHABLE_VEHICLES = ("the cart", "the wagon", "the stroller", "the sled")

COMMON_DESTINATIONS = (
    "to the library",
    "to school",
    "to the office",
    "to the kitchen",
    "to the garden",
    "to the clinic",
    "to the bakery",
    "to the museum",
    "to the station",
    "to the harbor",
    "to the playground",
    "to the studio",
    "to the workshop",
    "to the theater",
    "to the market",
    "to the post office",
    "to the front porch",
    "to the back room",
    "to the music room",
    "to the art room",
    "to the gym",
    "to the nurse's office",
    "to the craft room",
    "to the science room",
    "to the train platform",
    "to the bus stop",
    "to the town square",
    "to the dentist's office",
    "to the grocery store",
    "to the community center",
    "to the storage closet",
    "to the side entrance",
    "to the reading room",
    "to the family room",
    "to the laundry room",
    "to the neighbor's house",
    "to the corner shop",
    "to the city park",
    "to the river dock",
    "to the school office",
)

PATH_LOCATIONS = (
    "into the garage",
    "into the hallway",
    "into the closet",
    "into the pantry",
    "into the courtyard",
    "into the elevator",
    "into the storage room",
    "into the shed",
    "into the classroom",
    "into the lobby",
    "onto the porch",
    "onto the balcony",
    "onto the shelf",
    "onto the bench",
    "onto the counter",
    "toward the station",
    "toward the river",
    "toward the entrance",
    "toward the garden gate",
    "toward the front desk",
    "toward the side door",
    "through the hallway",
    "through the side door",
    "through the lobby",
    "through the doorway",
    "through the gate",
    "through the kitchen",
    "across the room",
    "across the yard",
    "across the hallway",
    "across the porch",
    "up the stairs",
    "down the hallway",
    "around the corner",
    "past the fountain",
    "over the bridge",
    "along the sidewalk",
    "along the fence",
    "beside the doorway",
    "near the checkout counter",
    "by the front desk",
)

NAMED_DESTINATIONS = (
    "to Denver",
    "to Boston",
    "to Seattle",
    "to Chicago",
    "to Austin",
    "to Dallas",
    "to Portland",
    "to Miami",
    "to Atlanta",
    "to Texas",
    "to Oregon",
    "to Florida",
    "to Arizona",
    "to Spain",
    "to Japan",
    "to Mexico",
    "to Brazil",
    "to India",
    "to China",
    "to France",
    "to Germany",
)

LOCAL_LOCATIONS = (
    "under the blanket",
    "under the chair",
    "under the sink",
    "under the bed",
    "under the towel",
    "under the stairs",
    "under the bench",
    "behind the cabinet",
    "behind the curtain",
    "behind the door",
    "behind the plant",
    "behind the bookcase",
    "behind the screen",
    "next to the shelf",
    "next to the lamp",
    "next to the sink",
    "next to the window",
    "next to the doorway",
    "next to the coat rack",
    "near the window",
    "near the cabinet",
    "near the front desk",
    "near the closet",
    "near the garden gate",
    "beside the couch",
    "beside the bed",
    "beside the sink",
    "beside the bookshelf",
    "on the counter",
    "on the bench",
    "on the shelf",
    "on the porch",
    "on the rug",
    "on the windowsill",
    "inside the box",
    "inside the drawer",
    "inside the basket",
    "inside the locker",
    "inside the cabinet",
    "between the chairs",
    "between the cushions",
    "against the wall",
    "against the fence",
    "above the sink",
    "below the window",
    "by the door",
    "by the lamp",
    "in the closet",
    "in the pantry",
    "in the shed",
)

VERB_FRAMES_BY_GROUP: dict[str, tuple[HardConfoundVerbFrame, ...]] = {
    DESTINATION_PATH_NAMED: (
        HardConfoundVerbFrame("send_mail_ship_deliver", "send", "sent", MAILABLE_THEMES),
        HardConfoundVerbFrame("send_mail_ship_deliver", "mail", "mailed", MAILABLE_THEMES),
        HardConfoundVerbFrame("send_mail_ship_deliver", "deliver", "delivered", PORTABLE_THEMES),
        HardConfoundVerbFrame("send_mail_ship_deliver", "ship", "shipped", MAILABLE_THEMES),
        HardConfoundVerbFrame("carry_bring_take_move", "bring", "brought", PORTABLE_THEMES),
        HardConfoundVerbFrame("carry_bring_take_move", "take", "took", PORTABLE_THEMES),
        HardConfoundVerbFrame("vehicle_motion", "drive", "drove", VEHICLE_THEMES),
    ),
    DESTINATION_PATH_COMMON: (
        HardConfoundVerbFrame("send_mail_ship_deliver", "send", "sent", MAILABLE_THEMES),
        HardConfoundVerbFrame("send_mail_ship_deliver", "mail", "mailed", MAILABLE_THEMES),
        HardConfoundVerbFrame("send_mail_ship_deliver", "deliver", "delivered", PORTABLE_THEMES),
        HardConfoundVerbFrame("send_mail_ship_deliver", "ship", "shipped", MAILABLE_THEMES),
        HardConfoundVerbFrame("carry_bring_take_move", "bring", "brought", PORTABLE_THEMES),
        HardConfoundVerbFrame("carry_bring_take_move", "take", "took", PORTABLE_THEMES),
        HardConfoundVerbFrame("carry_bring_take_move", "carry", "carried", PORTABLE_THEMES),
        HardConfoundVerbFrame("carry_bring_take_move", "move", "moved", MOVABLE_THEMES),
        HardConfoundVerbFrame("path_manipulation", "push", "pushed", MOVABLE_THEMES),
        HardConfoundVerbFrame("path_manipulation", "pull", "pulled", MOVABLE_THEMES),
        HardConfoundVerbFrame("path_manipulation", "slide", "slid", MOVABLE_THEMES),
        HardConfoundVerbFrame("vehicle_motion", "drive", "drove", VEHICLE_THEMES),
        HardConfoundVerbFrame("vehicle_motion", "push", "pushed", PUSHABLE_VEHICLES),
        HardConfoundVerbFrame("vehicle_motion", "pull", "pulled", PUSHABLE_VEHICLES),
    ),
    OBJECT_ANCHORED_LOCAL: (
        HardConfoundVerbFrame("local_placement", "put", "put", PORTABLE_THEMES),
        HardConfoundVerbFrame("local_placement", "place", "placed", PORTABLE_THEMES),
        HardConfoundVerbFrame("local_placement", "set", "set", PORTABLE_THEMES),
        HardConfoundVerbFrame("local_placement", "lay", "laid", PORTABLE_THEMES),
        HardConfoundVerbFrame("local_storage", "leave", "left", PORTABLE_THEMES),
        HardConfoundVerbFrame("local_storage", "drop", "dropped", PORTABLE_THEMES),
        HardConfoundVerbFrame("local_storage", "hide", "hid", PORTABLE_THEMES),
        HardConfoundVerbFrame("local_storage", "store", "stored", PORTABLE_THEMES),
        HardConfoundVerbFrame("local_storage", "keep", "kept", PORTABLE_THEMES),
        HardConfoundVerbFrame("local_find", "find", "found", PORTABLE_THEMES),
    ),
}

def build_qa_wherewhat_hardconfound_frames(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> QaWhereWhatHardConfoundFrameResult:
    """Build deterministic same-event where/what hard-confound documents."""

    if not isinstance(mapping, dict):
        raise TypeError("QA where/what hard-confound config must be a mapping.")

    name = str(mapping["name"])
    seed = int(mapping.get("seed", 42))
    document_count = int(mapping.get("document_count", 500))
    if document_count <= 0:
        raise ValueError("document_count must be positive.")

    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    scenario_group_weights = _scenario_group_weights(mapping)
    style_weights = _style_weights(mapping)
    second_person_ratio = _ratio(mapping, "second_person_ratio", 0.80)
    where_first_ratio = _ratio(mapping, "where_first_ratio", 0.50)
    agents = _agents(mapping)
    speaker_pairs = _speaker_pairs(mapping)
    banned_phrases = _banned_phrases(mapping)

    rng = random.Random(seed)
    plans = _build_plans(
        name=name,
        document_count=document_count,
        scenario_group_weights=scenario_group_weights,
        style_weights=style_weights,
        second_person_ratio=second_person_ratio,
        where_first_ratio=where_first_ratio,
        agents=agents,
        speaker_pairs=speaker_pairs,
        rng=rng,
    )
    documents = tuple(_document_from_plan(plan, banned_phrases=banned_phrases) for plan in plans)
    audit_report = _audit_documents(
        name,
        documents,
        scenario_group_names=tuple(
            group for group in SCENARIO_GROUP_ORDER if scenario_group_weights[group] > 0.0
        ),
    )

    paths = {
        "documents": output_dir / "documents.jsonl",
        "train": output_dir / "train.txt",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
        "config": output_dir / "qa_wherewhat_hardconfound_frames_config.yaml",
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
        "scenario_group_weights": scenario_group_weights,
        "style_weights": style_weights,
        "second_person_ratio": second_person_ratio,
        "where_first_ratio": where_first_ratio,
        "speaker_pairs": [list(pair) for pair in speaker_pairs],
        "agents": list(agents),
        "excluded_benchmark_phrases": sorted(banned_phrases),
        "qwen_rewrite_ready": True,
        "role_fields": [
            "verb",
            "verb_past",
            "theme",
            "location",
            "where_question",
            "where_answer",
            "what_question",
            "what_answer",
            "style",
            "order",
            "person",
        ],
        "counts": {
            "documents": len(documents),
            "train_documents": sum(1 for row in documents if row["audit_status"] == "pass"),
            **audit_report["counts"],
            "scenario_group_counts": audit_report["scenario_group_counts"],
            "style_counts": audit_report["style_counts"],
            "order_counts": audit_report["order_counts"],
            "person_counts": audit_report["person_counts"],
            "speaker_pair_counts": audit_report["speaker_pair_counts"],
            "verb_family_counts": audit_report["verb_family_counts"],
            "verb_counts": audit_report["verb_counts"],
            "location_type_counts": audit_report["location_type_counts"],
            "answer_preposition_counts": audit_report["answer_preposition_counts"],
            "named_location_counts": audit_report["named_location_counts"],
            "question_count_counts": audit_report["question_count_counts"],
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    return QaWhereWhatHardConfoundFrameResult(
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
    scenario_group_weights: dict[str, float],
    style_weights: dict[str, float],
    second_person_ratio: float,
    where_first_ratio: float,
    agents: tuple[str, ...],
    speaker_pairs: tuple[tuple[str, str], ...],
    rng: random.Random,
) -> list[WhereWhatHardConfoundPlan]:
    scenario_groups = _weighted_sequence(
        document_count,
        values=SCENARIO_GROUP_ORDER,
        weights=scenario_group_weights,
        rng=rng,
    )
    styles = _weighted_sequence(
        document_count,
        values=STYLE_ORDER,
        weights=style_weights,
        rng=rng,
    )
    orders = _ratio_sequence(
        document_count,
        positive=WHERE_FIRST,
        negative=WHAT_FIRST,
        positive_ratio=where_first_ratio,
        rng=rng,
    )
    person_modes = _ratio_sequence(
        document_count,
        positive=SECOND_PERSON,
        negative=THIRD_PERSON,
        positive_ratio=second_person_ratio,
        rng=rng,
    )
    group_offsets: Counter[str] = Counter()
    location_offsets: Counter[str] = Counter()
    plans: list[WhereWhatHardConfoundPlan] = []
    for index, scenario_group in enumerate(scenario_groups):
        frames = VERB_FRAMES_BY_GROUP[scenario_group]
        frame = frames[group_offsets[scenario_group] % len(frames)]
        group_offsets[scenario_group] += 1
        location = _location_for_frame(
            scenario_group,
            frame=frame,
            offset=location_offsets[scenario_group],
            rng=rng,
        )
        location_offsets[scenario_group] += 1
        person = person_modes[index]
        theme = _theme_for_frame(frame, index=index, person=person, rng=rng)
        agent = agents[(index * 7 + rng.randrange(len(agents))) % len(agents)]
        speaker_pair = (
            speaker_pairs[(index + rng.randrange(len(speaker_pairs))) % len(speaker_pairs)]
            if styles[index] in {DIALOGUE_FRAGMENT, DIALOGUE_FULL_SENTENCE}
            else None
        )
        question_agent = "you" if person == SECOND_PERSON else agent
        answer_agent = "I" if person == SECOND_PERSON else agent
        where_question = f"Where did {question_agent} {frame.verb} {theme}?"
        what_question = f"What did {question_agent} {frame.verb} {location}?"
        full_answer = styles[index] in {BARE_FULL_SENTENCE, DIALOGUE_FULL_SENTENCE}
        if full_answer:
            where_answer = _sentence(f"{answer_agent} {frame.past} {theme} {location}")
            what_answer = _sentence(f"{answer_agent} {frame.past} {theme}")
        else:
            where_answer = _sentence(location)
            what_answer = _sentence(theme)
        text = _render_text(
            style=styles[index],
            order=orders[index],
            speaker_pair=speaker_pair,
            where_question=where_question,
            where_answer=where_answer,
            what_question=what_question,
            what_answer=what_answer,
        )
        plans.append(
            WhereWhatHardConfoundPlan(
                id=f"{name}_{index:05d}",
                index=index,
                scenario_group=scenario_group,
                style=styles[index],
                order=orders[index],
                person=person,
                speaker_pair=speaker_pair,
                verb_family=frame.family,
                verb=frame.verb,
                verb_past=frame.past,
                agent=agent,
                question_agent=question_agent,
                answer_agent=answer_agent,
                theme=theme,
                location=location,
                location_type=_location_type(scenario_group),
                answer_preposition=_first_word(location),
                named_location=scenario_group == DESTINATION_PATH_NAMED,
                where_question=where_question,
                where_answer=where_answer,
                what_question=what_question,
                what_answer=what_answer,
                text=text,
                template_id=styles[index],
            )
        )
    return plans


def _location_for_frame(
    scenario_group: str,
    *,
    frame: HardConfoundVerbFrame,
    offset: int,
    rng: random.Random,
) -> str:
    pool = _location_pool_for_frame(scenario_group, frame)
    return pool[(offset + rng.randrange(len(pool))) % len(pool)]


def _location_pool_for_frame(
    scenario_group: str,
    frame: HardConfoundVerbFrame,
) -> tuple[str, ...]:
    if scenario_group == DESTINATION_PATH_NAMED:
        return NAMED_DESTINATIONS
    if scenario_group == OBJECT_ANCHORED_LOCAL:
        return LOCAL_LOCATIONS
    if frame.family == "send_mail_ship_deliver":
        return COMMON_DESTINATIONS
    if frame.family == "path_manipulation":
        return PATH_LOCATIONS
    if frame.family == "vehicle_motion" and frame.verb in {"push", "pull"}:
        return PATH_LOCATIONS
    if scenario_group == DESTINATION_PATH_COMMON:
        return (*COMMON_DESTINATIONS, *PATH_LOCATIONS)
    raise ValueError(f"Unknown scenario group: {scenario_group!r}")


def _theme_for_frame(
    frame: HardConfoundVerbFrame,
    *,
    index: int,
    person: str,
    rng: random.Random,
) -> str:
    theme = frame.theme_pool[(index * 5 + rng.randrange(len(frame.theme_pool))) % len(frame.theme_pool)]
    if frame.verb == "send" and theme == "the package":
        return "the parcel"
    if frame.verb == "send" and person == SECOND_PERSON and theme == "the parcel":
        return "the letter"
    return theme


def _render_text(
    *,
    style: str,
    order: str,
    speaker_pair: tuple[str, str] | None,
    where_question: str,
    where_answer: str,
    what_question: str,
    what_answer: str,
) -> str:
    if order == WHERE_FIRST:
        qa_parts = (
            (where_question, where_answer),
            (what_question, what_answer),
        )
    elif order == WHAT_FIRST:
        qa_parts = (
            (what_question, what_answer),
            (where_question, where_answer),
        )
    else:
        raise ValueError(f"Unsupported order: {order!r}")

    if style in {BARE_FRAGMENT, BARE_FULL_SENTENCE}:
        return " ".join(f"{question} {answer}" for question, answer in qa_parts)
    if style in {DIALOGUE_FRAGMENT, DIALOGUE_FULL_SENTENCE}:
        if speaker_pair is None:
            raise ValueError("dialogue styles require a speaker pair.")
        q_speaker, a_speaker = speaker_pair
        return " ".join(
            f"{q_speaker}: {question} {a_speaker}: {answer}"
            for question, answer in qa_parts
        )
    raise ValueError(f"Unsupported style: {style!r}")


def _document_from_plan(
    plan: WhereWhatHardConfoundPlan,
    *,
    banned_phrases: frozenset[str],
) -> dict[str, Any]:
    row = plan.to_dict()
    row["question_type"] = QUESTION_TYPE
    row["word_count"] = _word_count(plan.text)
    audit = _audit_document(row, banned_phrases=banned_phrases)
    row["audit_status"] = audit["status"]
    row["audit_flags"] = audit["flags"]
    return row


def _audit_documents(
    name: str,
    documents: tuple[dict[str, Any], ...],
    *,
    scenario_group_names: tuple[str, ...],
) -> dict[str, Any]:
    counts = Counter(row["audit_status"] for row in documents)
    scenario_group_counts = Counter(str(row["scenario_group"]) for row in documents)
    style_counts = Counter(str(row["style"]) for row in documents)
    order_counts = Counter(str(row["order"]) for row in documents)
    person_counts = Counter(str(row["person"]) for row in documents)
    speaker_pair_counts = Counter(_speaker_pair_key(row) for row in documents)
    verb_family_counts = Counter(str(row["verb_family"]) for row in documents)
    verb_counts = Counter(str(row["verb"]) for row in documents)
    location_type_counts = Counter(str(row["location_type"]) for row in documents)
    preposition_counts = Counter(str(row["answer_preposition"]) for row in documents)
    named_location_counts = Counter(
        "named" if row["named_location"] else "generic" for row in documents
    )
    question_count_counts = Counter(str(row["text"].count("?")) for row in documents)
    group_values = [scenario_group_counts[group] for group in scenario_group_names]
    group_delta = max(group_values) - min(group_values) if group_values else 0
    flags = []
    if group_delta > 1 and len(set(group_values)) > 1:
        flags.append(
            {
                "severity": "info",
                "code": "scenario_group_weighted",
                "message": "Scenario groups are intentionally weighted rather than balanced.",
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
        "scenario_group_counts": dict(sorted(scenario_group_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
        "order_counts": dict(sorted(order_counts.items())),
        "person_counts": dict(sorted(person_counts.items())),
        "speaker_pair_counts": dict(sorted(speaker_pair_counts.items())),
        "verb_family_counts": dict(sorted(verb_family_counts.items())),
        "verb_counts": dict(sorted(verb_counts.items())),
        "location_type_counts": dict(sorted(location_type_counts.items())),
        "answer_preposition_counts": dict(sorted(preposition_counts.items())),
        "named_location_counts": dict(sorted(named_location_counts.items())),
        "question_count_counts": dict(sorted(question_count_counts.items())),
        "flags": flags,
    }


def _audit_document(row: dict[str, Any], *, banned_phrases: frozenset[str]) -> dict[str, Any]:
    flags = []
    text = str(row["text"])
    lower = _normalize_text(text)
    where_question = str(row["where_question"])
    what_question = str(row["what_question"])
    where_answer = str(row["where_answer"])
    what_answer = str(row["what_answer"])
    theme = str(row["theme"])
    location = str(row["location"])
    if text.count("?") != 2:
        flags.append(_flag("error", "question_count", "Expected exactly two questions."))
    if not where_question.startswith("Where did "):
        flags.append(_flag("error", "where_shape", "Where question must start with 'Where did'."))
    if not what_question.startswith("What did "):
        flags.append(_flag("error", "what_shape", "What question must start with 'What did'."))
    for marker in BANNED_CONTRAST_MARKERS:
        if marker in lower:
            flags.append(_flag("error", "contrast_marker", f"Found contrast marker: {marker}"))
    for phrase in banned_phrases:
        if phrase in lower:
            flags.append(_flag("error", "benchmark_phrase_overlap", f"Found banned phrase: {phrase}"))
    for planned in (where_question, what_question, where_answer, what_answer):
        if planned not in text:
            flags.append(_flag("error", "missing_planned_text", f"Missing planned text: {planned}"))
    if not _contains_normalized(where_answer, location):
        flags.append(_flag("error", "where_answer_role", "Where answer must contain location."))
    if not _contains_normalized(what_answer, theme):
        flags.append(_flag("error", "what_answer_role", "What answer must contain theme."))
    if not _contains_normalized(where_question, theme):
        flags.append(_flag("error", "where_question_theme", "Where question must contain theme."))
    if not _contains_normalized(what_question, location):
        flags.append(_flag("error", "what_question_location", "What question must contain location."))
    if "\n" in text:
        flags.append(_flag("error", "multiline_text", "Train text must stay one line per example."))
    status = "fail" if any(flag["severity"] == "error" for flag in flags) else "pass"
    return {"status": status, "flags": flags}


def _scenario_group_weights(mapping: dict[str, Any]) -> dict[str, float]:
    return _weights(
        mapping.get(
            "scenario_group_weights",
            {
                DESTINATION_PATH_NAMED: 0.15,
                DESTINATION_PATH_COMMON: 0.35,
                OBJECT_ANCHORED_LOCAL: 0.50,
            },
        ),
        values=SCENARIO_GROUP_ORDER,
        label="scenario_group_weights",
    )


def _style_weights(mapping: dict[str, Any]) -> dict[str, float]:
    return _weights(
        mapping.get(
            "style_weights",
            {
                BARE_FRAGMENT: 0.50,
                DIALOGUE_FRAGMENT: 0.25,
                BARE_FULL_SENTENCE: 0.25,
                DIALOGUE_FULL_SENTENCE: 0.0,
            },
        ),
        values=STYLE_ORDER,
        label="style_weights",
    )


def _weights(raw: Any, *, values: tuple[str, ...], label: str) -> dict[str, float]:
    weights = {str(key): float(value) for key, value in dict(raw).items()}
    unknown = sorted(set(weights) - set(values))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}")
    if any(value < 0.0 for value in weights.values()):
        raise ValueError(f"{label} must not contain negative weights.")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError(f"{label} must have positive total weight.")
    return {value: weights.get(value, 0.0) / total for value in values}


def _ratio(mapping: dict[str, Any], key: str, default: float) -> float:
    value = float(mapping.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1.")
    return value


def _agents(mapping: dict[str, Any]) -> tuple[str, ...]:
    raw_agents = mapping.get("agents", AGENTS)
    agents = tuple(str(agent) for agent in raw_agents)
    if not agents:
        raise ValueError("agents must not be empty.")
    return agents


def _speaker_pairs(mapping: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    raw_pairs = mapping.get("speaker_pairs", SPEAKER_PAIRS)
    pairs = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            raise ValueError("speaker_pairs entries must be two-item lists.")
        pairs.append((str(raw_pair[0]), str(raw_pair[1])))
    if not pairs:
        raise ValueError("speaker_pairs must not be empty.")
    return tuple(pairs)


def _banned_phrases(mapping: dict[str, Any]) -> frozenset[str]:
    raw = mapping.get("banned_exact_phrases", BANNED_EXACT_PHRASES)
    return frozenset(_normalize_text(str(phrase)) for phrase in raw)


def _weighted_sequence(
    count: int,
    *,
    values: tuple[str, ...],
    weights: dict[str, float],
    rng: random.Random,
) -> list[str]:
    raw_counts = {value: count * weights[value] for value in values}
    value_counts = {value: int(raw_counts[value]) for value in values}
    assigned = sum(value_counts.values())
    remainders = sorted(
        values,
        key=lambda value: (raw_counts[value] - value_counts[value], value),
        reverse=True,
    )
    for value in remainders[: count - assigned]:
        value_counts[value] += 1
    sequence: list[str] = []
    for value in values:
        sequence.extend([value] * value_counts[value])
    rng.shuffle(sequence)
    return sequence


def _ratio_sequence(
    count: int,
    *,
    positive: str,
    negative: str,
    positive_ratio: float,
    rng: random.Random,
) -> list[str]:
    positive_count = min(max(round(count * positive_ratio), 0), count)
    sequence = [positive] * positive_count + [negative] * (count - positive_count)
    rng.shuffle(sequence)
    return sequence


def _speaker_pair_key(row: dict[str, Any]) -> str:
    pair = row.get("speaker_pair")
    if not pair:
        return "none"
    return f"{pair[0]}/{pair[1]}"


def _location_type(scenario_group: str) -> str:
    if scenario_group == DESTINATION_PATH_NAMED:
        return "named_destination"
    if scenario_group == DESTINATION_PATH_COMMON:
        return "common_destination_path"
    if scenario_group == OBJECT_ANCHORED_LOCAL:
        return "object_anchored_local"
    raise ValueError(f"Unknown scenario group: {scenario_group!r}")


def _first_word(text: str) -> str:
    return text.split(maxsplit=1)[0].lower()


def _sentence(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}."


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_normalized(text: str, phrase: str) -> bool:
    return _normalize_text(phrase) in _normalize_text(text)


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
