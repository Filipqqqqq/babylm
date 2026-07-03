"""Deterministic turn-taking pronoun-frame repair data."""

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
NAMED_REPORT = "named_report"
REPORT_ASKED_SAID = "asked_said"
REPORT_ASKED_ANSWERED = "asked_answered"
REPORT_QUOTE_FIRST_REPLIED = "quote_first_replied"
REPORT_TEMPLATES = (REPORT_ASKED_SAID, REPORT_ASKED_ANSWERED, REPORT_QUOTE_FIRST_REPLIED)
WITH_POLARITY_MARKER = "with_polarity_marker"
BARE_POLARITY_ANSWER = "bare_polarity_answer"
CONTRACTED_NEGATIVE = "contracted_negative"
FULL_NEGATIVE = "full_negative"
NO_NEGATIVE = "not_applicable"
YES_POLARITY = "yes"
NO_POLARITY = "no"
GENERATOR_NAME = "deterministic_template"
BANNED_CONTRAST_MARKERS = ("bad:", "wrong:", "incorrect:", "bad answer", "wrong answer")
BANNED_EVAL_FRAGMENTS = (
    "should you quit",
    "should she quit",
    "did you quit",
    "did she quit",
    "meet him",
    "arrive",
    "go somewhere after the movie",
    "try to finish it on time or not",
    "figure it out",
)


@dataclass(frozen=True)
class PredicateFrame:
    family: str
    frame_id: str
    aux_family: str
    predicate: str


@dataclass(frozen=True)
class MappingFrame:
    name: str
    question_subject: str
    answer_subject: str
    role: str
    weak_target: bool


@dataclass(frozen=True)
class TurnTakingPronounFramePlan:
    id: str
    index: int
    style: str
    mapping: str
    weak_target: bool
    polarity: str
    answer_surface: str
    negative_form: str
    aux_family: str
    predicate_family: str
    predicate_id: str
    asker: str
    responder: str
    referent: str | None
    question_subject: str
    answer_subject: str
    report_template: str
    question: str
    answer: str
    text: str
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QaTurnTakingPronounFrameResult:
    name: str
    output_dir: Path
    documents: tuple[dict[str, Any], ...]
    audit_report: dict[str, Any]
    outputs: dict[str, Path]
    manifest: dict[str, Any]


MAPPING_FRAMES: dict[str, MappingFrame] = {
    "i_to_you": MappingFrame("i_to_you", "I", "you", "singular_to_singular", True),
    "we_to_you": MappingFrame("we_to_you", "we", "you", "group_to_outside_responder", True),
    "you_to_we": MappingFrame("you_to_we", "you", "we", "speaker_to_group", True),
    "you_to_i": MappingFrame("you_to_i", "you", "I", "speaker_to_singular_responder", False),
    "she_to_she": MappingFrame("she_to_she", "she", "she", "third_person_singular_control", False),
    "they_to_they": MappingFrame("they_to_they", "they", "they", "third_person_plural_control", False),
}

WEAK_MAPPINGS = tuple(name for name, frame in MAPPING_FRAMES.items() if frame.weak_target)
CONTROL_MAPPINGS = tuple(name for name, frame in MAPPING_FRAMES.items() if not frame.weak_target)
AUX_FAMILIES = ("did", "have", "can", "should", "will", "be_past")

NAMES = (
    "Noor",
    "Maya",
    "Leo",
    "Sam",
    "Priya",
    "Luis",
    "Jamal",
    "Ana",
    "Iris",
    "Omar",
    "Nina",
    "Theo",
    "Rina",
    "Ben",
    "Elena",
    "Kai",
    "Sofia",
    "Mateo",
    "Ari",
    "Mina",
)

GROUPS = (
    "the helpers",
    "the students",
    "the neighbors",
    "the teammates",
    "the volunteers",
    "the campers",
    "the cousins",
    "the twins",
    "the kids",
    "the visitors",
)

FEMALE_REFERENTS = (
    "Maya",
    "Priya",
    "Ana",
    "Iris",
    "Nina",
    "Rina",
    "Elena",
    "Sofia",
    "Mina",
    "Hana",
)

DID_PREDICATES = (
    PredicateFrame("object_handling", "pack_map", "did", "pack the map"),
    PredicateFrame("object_handling", "close_gate", "did", "close the gate"),
    PredicateFrame("everyday_task", "check_list", "did", "check the list"),
    PredicateFrame("object_handling", "bring_folder", "did", "bring the folder"),
    PredicateFrame("everyday_task", "fold_cloth", "did", "fold the cloth"),
    PredicateFrame("communication", "text_coach", "did", "text the coach"),
    PredicateFrame("everyday_task", "clean_counter", "did", "clean the counter"),
    PredicateFrame("object_handling", "save_receipt", "did", "save the receipt"),
    PredicateFrame("plans_permission", "choose_folder", "did", "choose the blue folder"),
    PredicateFrame("everyday_task", "start_timer", "did", "start the timer"),
    PredicateFrame("object_handling", "pack_lunch", "did", "pack the lunch"),
    PredicateFrame("object_handling", "close_window", "did", "close the window"),
    PredicateFrame("object_handling", "return_notebook", "did", "return the notebook"),
    PredicateFrame("object_handling", "bring_tickets", "did", "bring the tickets"),
    PredicateFrame("object_handling", "move_basket", "did", "move the basket"),
    PredicateFrame("object_handling", "carry_bags", "did", "carry the bags"),
    PredicateFrame("object_handling", "open_box", "did", "open the small box"),
    PredicateFrame("object_handling", "find_badge", "did", "find the badge"),
    PredicateFrame("everyday_task", "wash_plate", "did", "wash the plate"),
    PredicateFrame("everyday_task", "wipe_board", "did", "wipe the board"),
    PredicateFrame("everyday_task", "sort_cards", "did", "sort the cards"),
    PredicateFrame("everyday_task", "label_box", "did", "label the box"),
    PredicateFrame("everyday_task", "hang_coat", "did", "hang the coat"),
    PredicateFrame("everyday_task", "water_plant", "did", "water the plant"),
    PredicateFrame("everyday_task", "count_pencils", "did", "count the pencils"),
    PredicateFrame("communication", "call_grandma", "did", "call Grandma"),
    PredicateFrame("communication", "answer_note", "did", "answer the note"),
    PredicateFrame("communication", "read_message", "did", "read the message"),
    PredicateFrame("communication", "remind_coach", "did", "remind the coach"),
    PredicateFrame("movement", "walk_to_gate", "did", "walk to the gate"),
    PredicateFrame("movement", "wait_by_door", "did", "wait by the door"),
    PredicateFrame("movement", "return_to_room", "did", "return to the room"),
    PredicateFrame("plans_permission", "choose_seat", "did", "choose a seat"),
    PredicateFrame("plans_permission", "join_line", "did", "join the line"),
    PredicateFrame("plans_permission", "finish_signs", "did", "finish the signs"),
)

HAVE_PREDICATES = (
    PredicateFrame("object_handling", "packed_map", "have", "packed the map"),
    PredicateFrame("object_handling", "locked_door", "have", "locked the door"),
    PredicateFrame("object_handling", "closed_gate", "have", "closed the gate"),
    PredicateFrame("everyday_task", "checked_list", "have", "checked the list"),
    PredicateFrame("object_handling", "brought_folder", "have", "brought the folder"),
    PredicateFrame("everyday_task", "folded_cloth", "have", "folded the cloth"),
    PredicateFrame("object_handling", "saved_receipt", "have", "saved the receipt"),
    PredicateFrame("communication", "called_before_dinner", "have", "called before dinner"),
    PredicateFrame("plans_permission", "chosen_folder", "have", "chosen the blue folder"),
    PredicateFrame("everyday_task", "washed_cups", "have", "washed the cups"),
    PredicateFrame("object_handling", "packed_lunch", "have", "packed the lunch"),
    PredicateFrame("object_handling", "closed_window", "have", "closed the window"),
    PredicateFrame("object_handling", "returned_notebook", "have", "returned the notebook"),
    PredicateFrame("object_handling", "brought_tickets", "have", "brought the tickets"),
    PredicateFrame("object_handling", "moved_basket", "have", "moved the basket"),
    PredicateFrame("object_handling", "carried_bags", "have", "carried the bags"),
    PredicateFrame("object_handling", "opened_box", "have", "opened the small box"),
    PredicateFrame("object_handling", "found_badge", "have", "found the badge"),
    PredicateFrame("everyday_task", "washed_plate", "have", "washed the plate"),
    PredicateFrame("everyday_task", "wiped_board", "have", "wiped the board"),
    PredicateFrame("everyday_task", "sorted_cards", "have", "sorted the cards"),
    PredicateFrame("everyday_task", "labeled_box", "have", "labeled the box"),
    PredicateFrame("everyday_task", "hung_coat", "have", "hung the coat"),
    PredicateFrame("everyday_task", "watered_plant", "have", "watered the plant"),
    PredicateFrame("everyday_task", "counted_pencils", "have", "counted the pencils"),
    PredicateFrame("communication", "called_grandma", "have", "called Grandma"),
    PredicateFrame("communication", "answered_note", "have", "answered the note"),
    PredicateFrame("communication", "read_message", "have", "read the message"),
    PredicateFrame("communication", "reminded_coach", "have", "reminded the coach"),
    PredicateFrame("movement", "walked_to_gate", "have", "walked to the gate"),
    PredicateFrame("movement", "waited_by_door", "have", "waited by the door"),
    PredicateFrame("movement", "returned_to_room", "have", "returned to the room"),
    PredicateFrame("plans_permission", "chosen_seat", "have", "chosen a seat"),
    PredicateFrame("plans_permission", "joined_line", "have", "joined the line"),
    PredicateFrame("plans_permission", "finished_signs", "have", "finished the signs"),
)

MODAL_PREDICATES = (
    PredicateFrame("object_handling", "carry_boxes", "can", "carry the boxes"),
    PredicateFrame("plans_permission", "borrow_charger", "can", "borrow the charger"),
    PredicateFrame("movement", "wait_by_gate", "can", "wait by the gate"),
    PredicateFrame("communication", "call_after_lunch", "can", "call after lunch"),
    PredicateFrame("everyday_task", "open_supply_box", "can", "open the supply box"),
    PredicateFrame("object_handling", "return_key_card", "can", "return the key card"),
    PredicateFrame("plans_permission", "use_side_door", "can", "use the side door"),
    PredicateFrame("everyday_task", "clean_table", "can", "clean the table"),
    PredicateFrame("object_handling", "move_chairs", "can", "move the chairs"),
    PredicateFrame("communication", "remind_teacher", "can", "remind the teacher"),
    PredicateFrame("object_handling", "carry_bags", "can", "carry the bags"),
    PredicateFrame("object_handling", "pack_lunch", "can", "pack the lunch"),
    PredicateFrame("object_handling", "bring_tickets", "can", "bring the tickets"),
    PredicateFrame("object_handling", "return_notebook", "can", "return the notebook"),
    PredicateFrame("object_handling", "hold_folder", "can", "hold the folder"),
    PredicateFrame("object_handling", "find_badge", "can", "find the badge"),
    PredicateFrame("object_handling", "close_window", "can", "close the window"),
    PredicateFrame("object_handling", "open_small_box", "can", "open the small box"),
    PredicateFrame("object_handling", "move_basket", "can", "move the basket"),
    PredicateFrame("everyday_task", "check_sign", "can", "check the sign"),
    PredicateFrame("everyday_task", "wipe_board", "can", "wipe the board"),
    PredicateFrame("everyday_task", "sort_cards", "can", "sort the cards"),
    PredicateFrame("everyday_task", "label_box", "can", "label the box"),
    PredicateFrame("everyday_task", "hang_coat", "can", "hang the coat"),
    PredicateFrame("everyday_task", "water_plant", "can", "water the plant"),
    PredicateFrame("everyday_task", "count_pencils", "can", "count the pencils"),
    PredicateFrame("everyday_task", "fold_cloth", "can", "fold the cloth"),
    PredicateFrame("communication", "answer_note", "can", "answer the note"),
    PredicateFrame("communication", "read_message", "can", "read the message"),
    PredicateFrame("communication", "text_parent", "can", "text the parent"),
    PredicateFrame("communication", "ask_librarian", "can", "ask the librarian"),
    PredicateFrame("movement", "walk_to_gate", "can", "walk to the gate"),
    PredicateFrame("movement", "wait_near_door", "can", "wait near the door"),
    PredicateFrame("movement", "return_to_room", "can", "return to the room"),
    PredicateFrame("movement", "stand_by_desk", "can", "stand by the desk"),
    PredicateFrame("plans_permission", "choose_seat", "can", "choose a seat"),
    PredicateFrame("plans_permission", "join_line", "can", "join the line"),
    PredicateFrame("plans_permission", "finish_signs", "can", "finish the signs"),
    PredicateFrame("plans_permission", "use_red_marker", "can", "use the red marker"),
    PredicateFrame("plans_permission", "share_tape", "can", "share the tape"),
)

SHOULD_PREDICATES = tuple(
    PredicateFrame(frame.family, frame.frame_id, "should", frame.predicate) for frame in MODAL_PREDICATES
)
WILL_PREDICATES = tuple(
    PredicateFrame(frame.family, frame.frame_id, "will", frame.predicate) for frame in MODAL_PREDICATES
)

BE_PAST_PREDICATES = (
    PredicateFrame("movement", "waiting_by_gate", "be_past", "waiting by the gate"),
    PredicateFrame("object_handling", "holding_folder", "be_past", "holding the folder"),
    PredicateFrame("plans_permission", "ready_for_trip", "be_past", "ready for the trip"),
    PredicateFrame("movement", "near_front_desk", "be_past", "near the front desk"),
    PredicateFrame("everyday_task", "using_charger", "be_past", "using the charger"),
    PredicateFrame("movement", "standing_by_door", "be_past", "standing by the door"),
    PredicateFrame("plans_permission", "early_for_lesson", "be_past", "early for the lesson"),
    PredicateFrame("everyday_task", "quiet_in_hall", "be_past", "quiet in the hall"),
    PredicateFrame("object_handling", "looking_for_notebook", "be_past", "looking for the notebook"),
    PredicateFrame("everyday_task", "cleaning_table", "be_past", "cleaning the table"),
    PredicateFrame("object_handling", "holding_tickets", "be_past", "holding the tickets"),
    PredicateFrame("object_handling", "carrying_bags", "be_past", "carrying the bags"),
    PredicateFrame("object_handling", "packing_lunch", "be_past", "packing the lunch"),
    PredicateFrame("object_handling", "using_red_marker", "be_past", "using the red marker"),
    PredicateFrame("object_handling", "looking_for_badge", "be_past", "looking for the badge"),
    PredicateFrame("object_handling", "opening_small_box", "be_past", "opening the small box"),
    PredicateFrame("object_handling", "closing_window", "be_past", "closing the window"),
    PredicateFrame("object_handling", "moving_basket", "be_past", "moving the basket"),
    PredicateFrame("everyday_task", "checking_sign", "be_past", "checking the sign"),
    PredicateFrame("everyday_task", "wiping_board", "be_past", "wiping the board"),
    PredicateFrame("everyday_task", "sorting_cards", "be_past", "sorting the cards"),
    PredicateFrame("everyday_task", "labeling_box", "be_past", "labeling the box"),
    PredicateFrame("everyday_task", "hanging_coat", "be_past", "hanging the coat"),
    PredicateFrame("everyday_task", "watering_plant", "be_past", "watering the plant"),
    PredicateFrame("everyday_task", "counting_pencils", "be_past", "counting the pencils"),
    PredicateFrame("everyday_task", "using_restroom", "be_past", "using the restroom"),
    PredicateFrame("communication", "calling_after_lunch", "be_past", "calling after lunch"),
    PredicateFrame("communication", "reading_message", "be_past", "reading the message"),
    PredicateFrame("communication", "answering_note", "be_past", "answering the note"),
    PredicateFrame("movement", "near_side_door", "be_past", "near the side door"),
    PredicateFrame("movement", "beside_coat_rack", "be_past", "beside the coat rack"),
    PredicateFrame("movement", "standing_by_desk", "be_past", "standing by the desk"),
    PredicateFrame("movement", "walking_to_gate", "be_past", "walking to the gate"),
    PredicateFrame("plans_permission", "ready_for_lesson", "be_past", "ready for the lesson"),
    PredicateFrame("plans_permission", "next_in_line", "be_past", "next in line"),
)

PREDICATES_BY_AUX: dict[str, tuple[PredicateFrame, ...]] = {
    "did": DID_PREDICATES,
    "have": HAVE_PREDICATES,
    "can": MODAL_PREDICATES,
    "should": SHOULD_PREDICATES,
    "will": WILL_PREDICATES,
    "be_past": BE_PAST_PREDICATES,
}


def build_qa_turn_taking_pronoun_frames(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> QaTurnTakingPronounFrameResult:
    """Build deterministic turn-taking pronoun repair documents."""

    if not isinstance(mapping, dict):
        raise TypeError("QA turn-taking pronoun frame config must be a mapping.")

    name = str(mapping["name"])
    seed = int(mapping.get("seed", 42))
    document_count = int(mapping.get("document_count", 500))
    if document_count <= 0:
        raise ValueError("document_count must be positive.")

    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    direct_ratio = float(mapping.get("direct_dialogue_ratio", 0.5))
    yes_ratio = float(mapping.get("yes_ratio", 0.5))
    answer_marker_ratio = float(mapping.get("answer_marker_ratio", 0.75))
    negative_contraction_ratio = float(mapping.get("negative_contraction_ratio", 0.5))
    weak_mapping_ratio = float(mapping.get("weak_mapping_ratio", 0.7))
    allow_duplicate_text = bool(mapping.get("allow_duplicate_text", False))
    mapping_names = tuple(str(name) for name in mapping.get("mapping_families", MAPPING_FRAMES))
    unknown = sorted(set(mapping_names) - set(MAPPING_FRAMES))
    if unknown:
        raise ValueError(f"Unknown turn-taking mapping families: {unknown}")
    if not mapping_names:
        raise ValueError("mapping_families must not be empty.")
    aux_families = tuple(str(name) for name in mapping.get("aux_families", AUX_FAMILIES))
    unknown_aux = sorted(set(aux_families) - set(AUX_FAMILIES))
    if unknown_aux:
        raise ValueError(f"Unknown auxiliary families: {unknown_aux}")
    if not aux_families:
        raise ValueError("aux_families must not be empty.")

    rng = random.Random(seed)
    plans = _build_plans(
        name=name,
        document_count=document_count,
        mapping_names=mapping_names,
        aux_families=aux_families,
        direct_ratio=direct_ratio,
        yes_ratio=yes_ratio,
        answer_marker_ratio=answer_marker_ratio,
        negative_contraction_ratio=negative_contraction_ratio,
        weak_mapping_ratio=weak_mapping_ratio,
        names=tuple(str(value) for value in mapping.get("names", NAMES)),
        groups=tuple(str(value) for value in mapping.get("groups", GROUPS)),
        female_referents=tuple(str(value) for value in mapping.get("female_referents", FEMALE_REFERENTS)),
        allow_duplicate_text=allow_duplicate_text,
        rng=rng,
    )
    documents = tuple(_document_from_plan(plan) for plan in plans)
    audit_report = _audit_documents(
        name,
        documents,
        mapping_names=mapping_names,
        aux_families=aux_families,
    )

    paths = {
        "documents": output_dir / "documents.jsonl",
        "train": output_dir / "train.txt",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
        "config": output_dir / "qa_turn_taking_pronoun_frames_config.yaml",
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
        "generator": GENERATOR_NAME,
        "direct_dialogue_ratio": direct_ratio,
        "yes_ratio": yes_ratio,
        "answer_marker_ratio": answer_marker_ratio,
        "negative_contraction_ratio": negative_contraction_ratio,
        "weak_mapping_ratio": weak_mapping_ratio,
        "allow_duplicate_text": allow_duplicate_text,
        "mapping_families": list(mapping_names),
        "aux_families": list(aux_families),
        "counts": {
            "documents": len(documents),
            "train_documents": sum(1 for row in documents if row["audit_status"] == "pass"),
            **audit_report["counts"],
            "mapping_counts": audit_report["mapping_counts"],
            "style_counts": audit_report["style_counts"],
            "polarity_counts": audit_report["polarity_counts"],
            "answer_surface_counts": audit_report["answer_surface_counts"],
            "negative_form_counts": audit_report["negative_form_counts"],
            "aux_family_counts": audit_report["aux_family_counts"],
            "predicate_family_counts": audit_report["predicate_family_counts"],
            "report_template_counts": audit_report["report_template_counts"],
            "weak_target_counts": audit_report["weak_target_counts"],
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    return QaTurnTakingPronounFrameResult(
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
    mapping_names: tuple[str, ...],
    aux_families: tuple[str, ...],
    direct_ratio: float,
    yes_ratio: float,
    answer_marker_ratio: float,
    negative_contraction_ratio: float,
    weak_mapping_ratio: float,
    names: tuple[str, ...],
    groups: tuple[str, ...],
    female_referents: tuple[str, ...],
    allow_duplicate_text: bool,
    rng: random.Random,
) -> list[TurnTakingPronounFramePlan]:
    mapping_sequence = _mapping_sequence(
        mapping_names,
        document_count,
        weak_mapping_ratio=weak_mapping_ratio,
        rng=rng,
    )
    styles = _balanced_label_sequence_by_group(
        mapping_sequence,
        positive_label=DIRECT_DIALOGUE,
        negative_label=NAMED_REPORT,
        positive_ratio=direct_ratio,
        group_names=mapping_names,
        rng=rng,
    )
    report_templates = _report_template_sequence(
        mapping_sequence,
        styles,
        group_names=mapping_names,
        rng=rng,
    )
    polarities = _balanced_label_sequence_by_group(
        mapping_sequence,
        positive_label=YES_POLARITY,
        negative_label=NO_POLARITY,
        positive_ratio=yes_ratio,
        group_names=mapping_names,
        rng=rng,
    )
    answer_surfaces = _balanced_label_sequence_by_group(
        mapping_sequence,
        positive_label=WITH_POLARITY_MARKER,
        negative_label=BARE_POLARITY_ANSWER,
        positive_ratio=answer_marker_ratio,
        group_names=mapping_names,
        rng=rng,
    )
    negative_forms = _negative_form_sequence(
        mapping_sequence,
        polarities,
        group_names=mapping_names,
        contraction_ratio=negative_contraction_ratio,
        rng=rng,
    )
    aux_sequence = _balanced_values_by_group(
        mapping_sequence,
        values=aux_families,
        group_names=mapping_names,
        rng=rng,
    )
    predicate_offsets: Counter[str] = Counter()
    plans: list[TurnTakingPronounFramePlan] = []
    seen_texts: set[str] = set()
    for index, mapping_name in enumerate(mapping_sequence):
        mapping_frame = MAPPING_FRAMES[mapping_name]
        aux_family = aux_sequence[index]
        predicates = PREDICATES_BY_AUX[aux_family]
        predicate_start = predicate_offsets[aux_family]
        style = styles[index]
        polarity = polarities[index]
        selected: tuple[PredicateFrame, str, str, str | None, str, str, str] | None = None
        max_attempts = max(200, len(predicates) * 200)
        for attempt in range(max_attempts):
            predicate = predicates[(predicate_start + attempt) % len(predicates)]
            asker, responder, referent = _choose_roles(
                mapping_frame,
                names=names,
                groups=groups,
                female_referents=female_referents,
                rng=rng,
            )
            question = _question(
                subject=mapping_frame.question_subject,
                aux_family=aux_family,
                predicate=predicate.predicate,
            )
            answer = _answer(
                subject=mapping_frame.answer_subject,
                aux_family=aux_family,
                polarity=polarity,
                answer_surface=answer_surfaces[index],
                negative_form=negative_forms[index],
            )
            text = _render_text(
                question=question,
                answer=answer,
                style=style,
                mapping=mapping_frame,
                asker=asker,
                responder=responder,
                referent=referent,
                report_template=report_templates[index],
            )
            if allow_duplicate_text or text not in seen_texts:
                selected = (predicate, asker, responder, referent, question, answer, text)
                break
        predicate_offsets[aux_family] += 1
        if selected is None:
            raise RuntimeError(
                f"Could not generate unique turn-taking text after {max_attempts} attempts "
                f"for mapping={mapping_name!r}, aux_family={aux_family!r}."
            )
        predicate, asker, responder, referent, question, answer, text = selected
        seen_texts.add(text)
        plans.append(
            TurnTakingPronounFramePlan(
                id=f"{name}_{index:05d}",
                index=index,
                style=style,
                mapping=mapping_name,
                weak_target=mapping_frame.weak_target,
                polarity=polarity,
                answer_surface=answer_surfaces[index],
                negative_form=negative_forms[index],
                aux_family=aux_family,
                predicate_family=predicate.family,
                predicate_id=predicate.frame_id,
                asker=asker,
                responder=responder,
                referent=referent,
                question_subject=mapping_frame.question_subject,
                answer_subject=mapping_frame.answer_subject,
                report_template=report_templates[index],
                question=question,
                answer=answer,
                text=text,
                template_id=report_templates[index],
            )
        )
    return plans


def _mapping_sequence(
    mapping_names: tuple[str, ...],
    count: int,
    *,
    weak_mapping_ratio: float,
    rng: random.Random,
) -> list[str]:
    weak_names = tuple(name for name in mapping_names if MAPPING_FRAMES[name].weak_target)
    control_names = tuple(name for name in mapping_names if not MAPPING_FRAMES[name].weak_target)
    if weak_names and control_names:
        weak_count = round(count * weak_mapping_ratio)
        weak_count = min(max(weak_count, 0), count)
        control_count = count - weak_count
        sequence = _balanced_sequence(weak_names, weak_count, rng=rng)
        sequence.extend(_balanced_sequence(control_names, control_count, rng=rng))
        rng.shuffle(sequence)
        return sequence
    return _balanced_sequence(mapping_names, count, rng=rng)


def _balanced_sequence(values: tuple[str, ...], count: int, *, rng: random.Random) -> list[str]:
    if not values:
        return []
    base, remainder = divmod(count, len(values))
    sequence: list[str] = []
    for index, value in enumerate(values):
        sequence.extend([value] * (base + (1 if index < remainder else 0)))
    rng.shuffle(sequence)
    return sequence


def _balanced_label_sequence_by_group(
    group_sequence: list[str],
    *,
    positive_label: str,
    negative_label: str,
    positive_ratio: float,
    group_names: tuple[str, ...],
    rng: random.Random,
) -> list[str]:
    group_counts = Counter(group_sequence)
    target_positive_total = round(len(group_sequence) * positive_ratio)
    target_positive_total = min(max(target_positive_total, 0), len(group_sequence))
    group_positive_counts: dict[str, int] = {}
    running_total = 0
    fractional: list[tuple[float, str]] = []
    for group_name in group_names:
        exact = group_counts[group_name] * positive_ratio
        positive_count = int(exact)
        group_positive_counts[group_name] = positive_count
        running_total += positive_count
        fractional.append((exact - positive_count, group_name))
    for _, group_name in sorted(fractional, reverse=True)[: target_positive_total - running_total]:
        group_positive_counts[group_name] += 1

    group_labels: dict[str, list[str]] = {}
    for group_name in group_names:
        positive_count = group_positive_counts[group_name]
        negative_count = group_counts[group_name] - positive_count
        labels = [positive_label] * positive_count + [negative_label] * negative_count
        rng.shuffle(labels)
        group_labels[group_name] = labels

    offsets: Counter[str] = Counter()
    labels_for_sequence: list[str] = []
    for group_name in group_sequence:
        offset = offsets[group_name]
        labels_for_sequence.append(group_labels[group_name][offset])
        offsets[group_name] += 1
    return labels_for_sequence


def _balanced_values_by_group(
    group_sequence: list[str],
    *,
    values: tuple[str, ...],
    group_names: tuple[str, ...],
    rng: random.Random,
) -> list[str]:
    group_counts = Counter(group_sequence)
    group_values: dict[str, list[str]] = {}
    for index, group_name in enumerate(group_names):
        offset = index % len(values)
        rotated_values = values[offset:] + values[:offset]
        group_values[group_name] = _balanced_sequence(rotated_values, group_counts[group_name], rng=rng)

    offsets: Counter[str] = Counter()
    values_for_sequence: list[str] = []
    for group_name in group_sequence:
        offset = offsets[group_name]
        values_for_sequence.append(group_values[group_name][offset])
        offsets[group_name] += 1
    return values_for_sequence


def _report_template_sequence(
    group_sequence: list[str],
    styles: list[str],
    *,
    group_names: tuple[str, ...],
    rng: random.Random,
) -> list[str]:
    named_counts = Counter(
        group_name for group_name, style in zip(group_sequence, styles) if style == NAMED_REPORT
    )
    group_templates: dict[str, list[str]] = {}
    for index, group_name in enumerate(group_names):
        offset = index % len(REPORT_TEMPLATES)
        rotated_templates = REPORT_TEMPLATES[offset:] + REPORT_TEMPLATES[:offset]
        group_templates[group_name] = _balanced_sequence(rotated_templates, named_counts[group_name], rng=rng)

    offsets: Counter[str] = Counter()
    templates: list[str] = []
    for group_name, style in zip(group_sequence, styles):
        if style == DIRECT_DIALOGUE:
            templates.append(DIRECT_DIALOGUE)
            continue
        offset = offsets[group_name]
        templates.append(group_templates[group_name][offset])
        offsets[group_name] += 1
    return templates


def _negative_form_sequence(
    group_sequence: list[str],
    polarities: list[str],
    *,
    group_names: tuple[str, ...],
    contraction_ratio: float,
    rng: random.Random,
) -> list[str]:
    no_counts = Counter(
        group_name for group_name, polarity in zip(group_sequence, polarities) if polarity == NO_POLARITY
    )
    total_no = sum(no_counts.values())
    target_contracted_total = round(total_no * contraction_ratio)
    target_contracted_total = min(max(target_contracted_total, 0), total_no)
    group_contracted_counts: dict[str, int] = {}
    running_total = 0
    fractional: list[tuple[float, str]] = []
    for group_name in group_names:
        exact = no_counts[group_name] * contraction_ratio
        contracted_count = int(exact)
        group_contracted_counts[group_name] = contracted_count
        running_total += contracted_count
        fractional.append((exact - contracted_count, group_name))
    for _, group_name in sorted(fractional, reverse=True)[: target_contracted_total - running_total]:
        group_contracted_counts[group_name] += 1

    group_forms: dict[str, list[str]] = {}
    for group_name in group_names:
        contracted_count = group_contracted_counts[group_name]
        full_count = no_counts[group_name] - contracted_count
        forms = [CONTRACTED_NEGATIVE] * contracted_count + [FULL_NEGATIVE] * full_count
        rng.shuffle(forms)
        group_forms[group_name] = forms

    offsets: Counter[str] = Counter()
    forms_for_sequence: list[str] = []
    for group_name, polarity in zip(group_sequence, polarities):
        if polarity == YES_POLARITY:
            forms_for_sequence.append(NO_NEGATIVE)
            continue
        offset = offsets[group_name]
        forms_for_sequence.append(group_forms[group_name][offset])
        offsets[group_name] += 1
    return forms_for_sequence


def _choose_roles(
    mapping: MappingFrame,
    *,
    names: tuple[str, ...],
    groups: tuple[str, ...],
    female_referents: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, str, str | None]:
    if mapping.name == "we_to_you":
        asker = rng.choice(groups)
        responder = rng.choice(names)
        return asker, responder, None
    if mapping.name == "you_to_we":
        asker = rng.choice(names)
        responder = rng.choice(groups)
        return asker, responder, None
    asker, responder = rng.sample(names, 2)
    if mapping.name == "she_to_she":
        candidates = [name for name in female_referents if name not in {asker, responder}]
        referent = rng.choice(candidates or female_referents)
        return asker, responder, referent
    if mapping.name == "they_to_they":
        referent = rng.choice(groups)
        return asker, responder, referent
    return asker, responder, None


def _question(*, subject: str, aux_family: str, predicate: str) -> str:
    aux = _question_aux(aux_family, subject)
    return _normalize_spaces(f"{aux} {subject} {predicate}?")


def _answer(
    *,
    subject: str,
    aux_family: str,
    polarity: str,
    answer_surface: str,
    negative_form: str,
) -> str:
    positive_aux = _positive_answer_aux(aux_family, subject)
    if polarity == YES_POLARITY:
        if answer_surface == WITH_POLARITY_MARKER:
            return _normalize_spaces(f"Yes, {subject} {positive_aux}.")
        if answer_surface == BARE_POLARITY_ANSWER:
            return _normalize_spaces(f"{_sentence_initial(subject)} {positive_aux}.")
    if polarity == NO_POLARITY:
        negative_aux = _negative_answer_aux(aux_family, subject, negative_form=negative_form)
        if answer_surface == WITH_POLARITY_MARKER:
            return _normalize_spaces(f"No, {subject} {negative_aux}.")
        if answer_surface == BARE_POLARITY_ANSWER:
            return _normalize_spaces(f"{_sentence_initial(subject)} {negative_aux}.")
    if answer_surface not in {WITH_POLARITY_MARKER, BARE_POLARITY_ANSWER}:
        raise ValueError(f"Unsupported answer surface: {answer_surface!r}")
    raise ValueError(f"Unsupported polarity: {polarity!r}")


def _question_aux(aux_family: str, subject: str) -> str:
    if aux_family == "did":
        return "Did"
    if aux_family == "have":
        return "Has" if subject == "she" else "Have"
    if aux_family == "can":
        return "Can"
    if aux_family == "should":
        return "Should"
    if aux_family == "will":
        return "Will"
    if aux_family == "be_past":
        return "Was" if subject in {"I", "she"} else "Were"
    raise ValueError(f"Unsupported auxiliary family: {aux_family!r}")


def _positive_answer_aux(aux_family: str, subject: str) -> str:
    if aux_family == "did":
        return "did"
    if aux_family == "have":
        return "has" if subject == "she" else "have"
    if aux_family == "can":
        return "can"
    if aux_family == "should":
        return "should"
    if aux_family == "will":
        return "will"
    if aux_family == "be_past":
        return "was" if subject in {"I", "she"} else "were"
    raise ValueError(f"Unsupported auxiliary family: {aux_family!r}")


def _negative_answer_aux(aux_family: str, subject: str, *, negative_form: str) -> str:
    if negative_form == FULL_NEGATIVE:
        if aux_family == "did":
            return "did not"
        if aux_family == "have":
            return "has not" if subject == "she" else "have not"
        if aux_family == "can":
            return "can not"
        if aux_family == "should":
            return "should not"
        if aux_family == "will":
            return "will not"
        if aux_family == "be_past":
            return "was not" if subject in {"I", "she"} else "were not"
        raise ValueError(f"Unsupported auxiliary family: {aux_family!r}")
    if negative_form != CONTRACTED_NEGATIVE:
        raise ValueError(f"Unsupported negative form: {negative_form!r}")
    if aux_family == "did":
        return "didn't"
    if aux_family == "have":
        return "hasn't" if subject == "she" else "haven't"
    if aux_family == "can":
        return "can't"
    if aux_family == "should":
        return "shouldn't"
    if aux_family == "will":
        return "won't"
    if aux_family == "be_past":
        return "wasn't" if subject in {"I", "she"} else "weren't"
    raise ValueError(f"Unsupported auxiliary family: {aux_family!r}")


def _render_text(
    *,
    question: str,
    answer: str,
    style: str,
    mapping: MappingFrame,
    asker: str,
    responder: str,
    referent: str | None,
    report_template: str,
) -> str:
    if style == DIRECT_DIALOGUE:
        return f"A: {question} B: {answer}"
    if style != NAMED_REPORT:
        raise ValueError(f"Unsupported style: {style!r}")
    asker_text = _sentence_initial(asker)
    responder_text = _sentence_initial(responder)
    about = f" about {referent}" if mapping.name in {"she_to_she", "they_to_they"} and referent else ""
    if report_template == REPORT_ASKED_SAID:
        return f'{asker_text} asked {responder}{about}, "{question}" {responder_text} said, "{answer}"'
    if report_template == REPORT_ASKED_ANSWERED:
        return f'{asker_text} asked {responder}{about}, "{question}" {responder_text} answered, "{answer}"'
    if report_template == REPORT_QUOTE_FIRST_REPLIED:
        return f'"{question}" {asker_text} asked {responder}{about}. "{answer}" {responder_text} replied.'
    raise ValueError(f"Unsupported report template: {report_template!r}")


def _sentence_initial(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}"


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([?.!,])", r"\1", text)


def _document_from_plan(plan: TurnTakingPronounFramePlan) -> dict[str, Any]:
    row = plan.to_dict()
    row["word_count"] = _word_count(plan.text)
    audit = _audit_document(row)
    row["audit_status"] = audit["status"]
    row["audit_flags"] = audit["flags"]
    return row


def _audit_documents(
    name: str,
    documents: tuple[dict[str, Any], ...],
    *,
    mapping_names: tuple[str, ...],
    aux_families: tuple[str, ...],
) -> dict[str, Any]:
    counts = Counter(row["audit_status"] for row in documents)
    mapping_counts = Counter(str(row["mapping"]) for row in documents)
    style_counts = Counter(str(row["style"]) for row in documents)
    polarity_counts = Counter(str(row["polarity"]) for row in documents)
    answer_surface_counts = Counter(str(row["answer_surface"]) for row in documents)
    negative_form_counts = Counter(str(row["negative_form"]) for row in documents)
    aux_family_counts = Counter(str(row["aux_family"]) for row in documents)
    predicate_family_counts = Counter(str(row["predicate_family"]) for row in documents)
    report_template_counts = Counter(str(row["report_template"]) for row in documents)
    weak_target_counts = Counter(str(bool(row["weak_target"])).lower() for row in documents)
    mapping_values = [mapping_counts[name] for name in mapping_names]
    aux_values = [aux_family_counts[name] for name in aux_families]
    mapping_delta = max(mapping_values) - min(mapping_values) if mapping_values else 0
    aux_delta = max(aux_values) - min(aux_values) if aux_values else 0
    flags = []
    if mapping_delta > 100:
        flags.append(
            {
                "severity": "warn",
                "code": "mapping_balance",
                "message": "Mapping family counts differ more than expected for weighted repair.",
            }
        )
    if aux_delta > 1:
        flags.append(
            {
                "severity": "warn",
                "code": "aux_family_balance",
                "message": "Auxiliary family counts differ by more than one.",
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
        "mapping_counts": dict(sorted(mapping_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
        "polarity_counts": dict(sorted(polarity_counts.items())),
        "answer_surface_counts": dict(sorted(answer_surface_counts.items())),
        "negative_form_counts": dict(sorted(negative_form_counts.items())),
        "aux_family_counts": dict(sorted(aux_family_counts.items())),
        "predicate_family_counts": dict(sorted(predicate_family_counts.items())),
        "report_template_counts": dict(sorted(report_template_counts.items())),
        "weak_target_counts": dict(sorted(weak_target_counts.items())),
        "balance": {
            "mapping_family_max_minus_min": mapping_delta,
            "aux_family_max_minus_min": aux_delta,
        },
        "flags": flags,
    }


def _audit_document(row: dict[str, Any]) -> dict[str, Any]:
    flags = []
    text = str(row["text"])
    lower = text.lower()
    question = str(row["question"])
    answer = str(row["answer"])
    answer_surface = str(row["answer_surface"])
    if text.count("?") != 1:
        flags.append(_flag("error", "question_count", "Expected exactly one question."))
    if not question.endswith("?"):
        flags.append(_flag("error", "question_shape", "Question must end with a question mark."))
    if answer_surface == WITH_POLARITY_MARKER and not answer.startswith(("Yes, ", "No, ")):
        flags.append(_flag("error", "answer_shape", "Answer must begin with yes/no polarity."))
    if answer_surface == BARE_POLARITY_ANSWER and answer.startswith(("Yes, ", "No, ")):
        flags.append(_flag("error", "answer_shape", "Bare answer should omit yes/no polarity marker."))
    for marker in BANNED_CONTRAST_MARKERS:
        if marker in lower:
            flags.append(_flag("error", "contrast_marker", f"Found contrast marker: {marker}"))
    for fragment in BANNED_EVAL_FRAGMENTS:
        if fragment in lower:
            flags.append(_flag("error", "eval_like_fragment", f"Found banned eval-like fragment: {fragment}"))
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
