"""Seeded scene and question planning for QA-congruence repair."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from babysteps.linguistic_repair.numbers import (
    quantified_object_phrase,
    quantity_answer,
    quantity_surface,
    simple_plural,
)


QUESTION_TYPES = ("who", "what", "where", "when", "why", "how_many")
SCENE_SENTENCE_TEMPLATE_IDS = ("agent_time_reason", "time_fronted", "reason_fronted")
SCENELESS_STYLES = frozenset({"minimal_qa_pair"})
DEFAULT_MINIMAL_QA_PAIR_QUESTION_COUNT_WEIGHTS = ((1, 0.6), (2, 0.3), (3, 0.1))
QUESTION_TEMPLATE_IDS_BY_TYPE = {
    "who": (
        "who_action_location",
        "who_did_that",
        "who_action_object",
        "who_was_it",
    ),
    "what": (
        "what_action_location",
        "what_action_there",
        "what_action_short",
        "what_was_it",
    ),
    "where": (
        "where_action_object",
        "where_did_that_happen",
        "where_did_agent_do_that",
        "where_was_it",
    ),
    "when": (
        "when_action_object_location",
        "when_was_that",
        "when_action_object",
        "when_did_that_happen",
    ),
    "why": (
        "why_action_object_location",
        "why_do_that",
        "why_action_object",
        "why_did_that_happen",
    ),
    "how_many": (
        "how_many_action_location",
        "how_many_were_there",
        "how_many_action_short",
        "how_many_object_short",
    ),
}
MINIMAL_QUESTION_TEMPLATE_IDS_BY_TYPE = {
    "who": ("who_action_location",),
    "what": ("what_action_location",),
    "where": ("where_action_object",),
    "when": ("when_action_object_location",),
    "why": ("why_action_object_location",),
    "how_many": ("how_many_action_location",),
}
STYLES = (
    "clean_dialogue",
    "simple_prose",
    "classroom_tutor",
    "parent_child_dialogue",
    "friend_dialogue",
    "coach_player_dialogue",
    "nurse_child_dialogue",
    "clerk_customer_dialogue",
    "subtitle_clean",
    "minimal_qa_pair",
)
DEFAULT_QUANTITIES = tuple(range(0, 21)) + (25, 30, 40, 50, 75, 100)
DEFAULT_QUANTITY_MODES = ("word", "digit", "no_for_zero")
DEFAULT_TIMES = (
    "yesterday",
    "this morning",
    "this afternoon",
    "after lunch",
    "before dinner",
    "after school",
    "before bedtime",
    "a minute ago",
    "five minutes ago",
    "an hour ago",
    "two hours ago",
    "two days ago",
    "a few days ago",
    "on Sunday",
    "on Monday",
    "on Tuesday",
    "on Wednesday",
    "on Thursday",
    "on Friday",
    "on Saturday",
)


@dataclass(frozen=True)
class TopicTemplate:
    topic: str
    agents: tuple[str, ...]
    action_base: str
    action_past: str
    objects: tuple[tuple[str, str], ...]
    locations: tuple[str, ...]
    reasons: tuple[str, ...]


TOPICS = (
    TopicTemplate(
        topic="kitchen_snacks",
        agents=("Mia", "Mateo", "Aisha", "Grandpa", "Leila"),
        action_base="put",
        action_past="put",
        objects=(("apple", "apples"), ("cracker", "crackers"), ("cup", "cups")),
        locations=("in the blue bowl", "on the kitchen table", "in the lunch bag"),
        reasons=(
            "because a snack was needed",
            "because lunch was getting ready",
            "so the table would be ready",
        ),
    ),
    TopicTemplate(
        topic="cleaning",
        agents=("Dad", "Nora", "Jamal", "Mei", "the teacher"),
        action_base="move",
        action_past="moved",
        objects=(("chair", "chairs"), ("book", "books"), ("box", "boxes")),
        locations=("into the hallway", "beside the door", "near the window"),
        reasons=(
            "because the floor needed sweeping",
            "so there was room to walk",
            "because the room was messy",
        ),
    ),
    TopicTemplate(
        topic="laundry",
        agents=("Sarah", "Mom", "Priya", "David", "Hana"),
        action_base="put",
        action_past="put",
        objects=(("shirt", "shirts"), ("sock", "socks"), ("towel", "towels")),
        locations=("in the laundry basket", "on the chair", "in the closet"),
        reasons=(
            "because laundry time had started",
            "because the basket was ready",
            "so the room would be tidy",
        ),
    ),
    TopicTemplate(
        topic="school_supplies",
        agents=("Tom", "Mia", "the teacher", "Lily", "Omar", "Sofia"),
        action_base="pack",
        action_past="packed",
        objects=(("pencil", "pencils"), ("notebook", "notebooks"), ("eraser", "erasers")),
        locations=("in the school bag", "on the desk", "inside the red box"),
        reasons=(
            "because class was starting soon",
            "because the supplies were needed for math",
            "so the lesson could begin",
        ),
    ),
    TopicTemplate(
        topic="classroom_lesson",
        agents=("the teacher", "Omar", "Lily", "Mia", "Sofia"),
        action_base="place",
        action_past="placed",
        objects=(("card", "cards"), ("notebook", "notebooks"), ("pencil", "pencils")),
        locations=("on the lesson table", "near the board", "in the supply tray"),
        reasons=(
            "because the lesson was starting",
            "so the class could practice",
            "because the materials were ready",
        ),
    ),
    TopicTemplate(
        topic="art_crafts",
        agents=("Lily", "Sam", "Emma", "the child", "Amara", "Ravi"),
        action_base="place",
        action_past="placed",
        objects=(("sticker", "stickers"), ("star", "stars"), ("bead", "beads")),
        locations=("on the card", "in the small cup", "on the craft table"),
        reasons=(
            "because the card needed decoration",
            "because the project needed color",
            "for the art project",
        ),
    ),
    TopicTemplate(
        topic="toys_tidying",
        agents=("Ben", "Mia", "Jack", "Sarah", "Yuna", "Diego"),
        action_base="put",
        action_past="put",
        objects=(("block", "blocks"), ("toy car", "toy cars"), ("shell", "shells")),
        locations=("in the blue bin", "under the shelf", "beside the toy box"),
        reasons=(
            "because the room was messy",
            "because bedtime was close",
            "so the floor would stay clear",
        ),
    ),
    TopicTemplate(
        topic="sports_practice",
        agents=("Diego", "Yuna", "Sam", "Mia", "Ben"),
        action_base="place",
        action_past="placed",
        objects=(("cone", "cones"), ("ball", "balls"), ("practice bib", "practice bibs")),
        locations=("near the goal", "beside the bench", "on the sideline"),
        reasons=(
            "because practice was starting",
            "so the drill could begin",
            "because the field needed markers",
        ),
    ),
    TopicTemplate(
        topic="gym_warmup",
        agents=("Diego", "Yuna", "Sam", "Mia", "Ben"),
        action_base="set",
        action_past="set",
        objects=(("jump rope", "jump ropes"), ("water bottle", "water bottles"), ("mat", "mats")),
        locations=("beside the gym wall", "near the bench", "on the gym floor"),
        reasons=(
            "because warmups were starting",
            "so the team could stretch",
            "because the drill needed space",
        ),
    ),
    TopicTemplate(
        topic="game_equipment",
        agents=("Kenji", "Amina", "Diego", "Yuna", "Sam"),
        action_base="put",
        action_past="put",
        objects=(("jersey", "jerseys"), ("ball", "balls"), ("score card", "score cards")),
        locations=("in the equipment bag", "near the scoreboard", "by the sideline"),
        reasons=(
            "because the game was starting",
            "so the team could play",
            "because the equipment needed sorting",
        ),
    ),
    TopicTemplate(
        topic="library",
        agents=("David", "Nora", "the librarian", "Grandpa", "Fatima", "Min"),
        action_base="place",
        action_past="placed",
        objects=(("book", "books"), ("card", "cards"), ("magazine", "magazines")),
        locations=("on the library shelf", "beside the reading chair", "on the front desk"),
        reasons=(
            "because reading time was finished",
            "so another child could use the shelf",
            "because the shelf was labeled",
        ),
    ),
    TopicTemplate(
        topic="clinic_visit",
        agents=("Nia", "Omar", "Mom", "Aisha", "Luis"),
        action_base="place",
        action_past="placed",
        objects=(("bandage", "bandages"), ("sticker", "stickers"), ("cup", "cups")),
        locations=("on the clinic tray", "beside the sink", "in the small box"),
        reasons=(
            "because the checkup was starting",
            "so the room would be ready",
            "because the supplies were needed",
        ),
    ),
    TopicTemplate(
        topic="school_nurse",
        agents=("Nia", "Omar", "Aisha", "Luis", "Priya"),
        action_base="place",
        action_past="placed",
        objects=(("ice pack", "ice packs"), ("tissue", "tissues"), ("bandage", "bandages")),
        locations=("on the nurse's desk", "beside the cot", "in the first-aid box"),
        reasons=(
            "because recess had ended",
            "so the visit could start",
            "because the supplies were needed",
        ),
    ),
    TopicTemplate(
        topic="health_check",
        agents=("Nia", "Omar", "Aisha", "Luis", "Priya"),
        action_base="put",
        action_past="put",
        objects=(("sticker", "stickers"), ("cup", "cups"), ("form", "forms")),
        locations=("on the checkup table", "beside the scale", "in the paper tray"),
        reasons=(
            "because the checkup was ready",
            "so the nurse could help",
            "because the room needed supplies",
        ),
    ),
    TopicTemplate(
        topic="garden_plants",
        agents=("Grandpa", "Mia", "the gardener", "Mom", "Carlos", "Nia"),
        action_base="pour",
        action_past="poured",
        objects=(("cup of water", "cups of water"), ("seed", "seeds"), ("flower", "flowers")),
        locations=("into the plant pot", "beside the garden path", "near the small tree"),
        reasons=(
            "because the soil was dry",
            "so the plants could grow",
            "because planting time had started",
        ),
    ),
    TopicTemplate(
        topic="store_checkout",
        agents=("Amina", "Luis", "Mom", "Dad", "Noor"),
        action_base="put",
        action_past="put",
        objects=(("orange", "oranges"), ("bag", "bags"), ("receipt", "receipts")),
        locations=("on the counter", "in the shopping bag", "beside the register"),
        reasons=(
            "because the customer was checking out",
            "so the bag would be ready",
            "because the shelf was being restocked",
        ),
    ),
    TopicTemplate(
        topic="market_stall",
        agents=("Amina", "Luis", "Noor", "Dad", "Mom"),
        action_base="set",
        action_past="set",
        objects=(("apple", "apples"), ("bag", "bags"), ("price tag", "price tags")),
        locations=("on the market table", "beside the cash box", "in the paper bag"),
        reasons=(
            "because the customer was waiting",
            "so the sale could finish",
            "because the table needed restocking",
        ),
    ),
    TopicTemplate(
        topic="bakery_counter",
        agents=("Amina", "Luis", "Noor", "Mom", "Dad"),
        action_base="put",
        action_past="put",
        objects=(("muffin", "muffins"), ("cookie", "cookies"), ("napkin", "napkins")),
        locations=("on the bakery counter", "inside the small bag", "beside the display case"),
        reasons=(
            "because the order was ready",
            "so the order would be easy to carry",
            "because the counter was being arranged",
        ),
    ),
    TopicTemplate(
        topic="mail_delivery",
        agents=("Amina", "Luis", "Noor", "Dad", "Mom"),
        action_base="send",
        action_past="sent",
        objects=(("package", "packages"), ("letter", "letters"), ("postcard", "postcards")),
        locations=(
            "to another city",
            "to another country",
            "to a nearby town",
            "to the far side of the city",
            "across the state",
        ),
        reasons=(
            "because delivery time was close",
            "so the delivery would arrive",
            "because the address was ready",
        ),
    ),
    TopicTemplate(
        topic="study_places",
        agents=("Omar", "Lily", "Mia", "Sofia", "Kenji"),
        action_base="study",
        action_past="studied",
        objects=(("card", "cards"), ("word", "words"), ("problem", "problems")),
        locations=(
            "at the learning center",
            "at the community college",
            "at the preschool",
            "in the study room",
            "in the upstairs classroom",
        ),
        reasons=(
            "because the lesson was starting",
            "so the class could practice",
            "because the homework was due",
        ),
    ),
    TopicTemplate(
        topic="reading_places",
        agents=("David", "Nora", "Fatima", "Min", "Aisha"),
        action_base="read",
        action_past="read",
        objects=(("book", "books"), ("letter", "letters"), ("story", "stories")),
        locations=(
            "in the workroom",
            "in a small office",
            "under the trees",
            "at the neighborhood park",
            "at the workplace",
            "beside the pond",
            "in the reading room",
        ),
        reasons=(
            "because reading time had started",
            "so the group could discuss the reading",
            "because the work needed checking",
        ),
    ),
    TopicTemplate(
        topic="writing_places",
        agents=("David", "Nora", "Fatima", "Min", "Ravi"),
        action_base="write",
        action_past="wrote",
        objects=(("letter", "letters"), ("essay", "essays"), ("story", "stories")),
        locations=(
            "in the workroom",
            "in a small office",
            "under the trees",
            "at the neighborhood park",
            "at the workplace",
            "beside the pond",
            "in the reading room",
        ),
        reasons=(
            "because the notes were needed",
            "so the group could read the work",
            "because the deadline was close",
        ),
    ),
    TopicTemplate(
        topic="party",
        agents=("Sarah", "Dad", "Lily", "Tom", "Amina", "Kenji"),
        action_base="set",
        action_past="set",
        objects=(("cup", "cups"), ("plate", "plates"), ("napkin", "napkins")),
        locations=("on the party table", "beside the cake", "near the door"),
        reasons=(
            "because guests were coming",
            "so everyone could eat",
            "because the party was starting soon",
        ),
    ),
    TopicTemplate(
        topic="lost_and_found",
        agents=("Jack", "Mia", "the teacher", "Anna", "Noor", "Luis"),
        action_base="put",
        action_past="put",
        objects=(("key", "keys"), ("ticket", "tickets"), ("glove", "gloves")),
        locations=("on the counter", "in the lost-and-found box", "beside the door"),
        reasons=(
            "because an item was found on the floor",
            "so the owner could check the box",
            "because the lost item needed a place",
        ),
    ),
)


STYLE_TOPIC_NAMES = {
    "classroom_tutor": (
        "classroom_lesson",
        "school_supplies",
        "study_places",
        "reading_places",
        "writing_places",
    ),
    "parent_child_dialogue": (
        "kitchen_snacks",
        "laundry",
        "toys_tidying",
        "garden_plants",
        "party",
    ),
    "friend_dialogue": (
        "art_crafts",
        "toys_tidying",
        "party",
        "lost_and_found",
        "mail_delivery",
        "reading_places",
        "writing_places",
    ),
    "coach_player_dialogue": ("sports_practice", "gym_warmup", "game_equipment"),
    "nurse_child_dialogue": ("clinic_visit", "school_nurse", "health_check"),
    "clerk_customer_dialogue": (
        "store_checkout",
        "market_stall",
        "bakery_counter",
        "mail_delivery",
    ),
}


@dataclass(frozen=True)
class QuestionPlan:
    question_type: str
    question: str
    answer: str
    expected_slot: str
    question_template_id: str = "canonical"


@dataclass(frozen=True)
class ScenePlan:
    id: str
    style: str
    topic: str
    agent: str
    action_base: str
    action_past: str
    object_singular: str
    object_plural: str
    object_reference: str
    quantity_value: int
    quantity_mode: str
    quantity_surface: str
    object_phrase: str
    location: str
    time: str
    reason: str
    scene_sentence: str
    questions: tuple[QuestionPlan, ...]
    scene_template_id: str = "agent_time_reason"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["questions"] = [asdict(question) for question in self.questions]
        return row


def _question_for_type(
    question_type: str,
    *,
    scene: dict[str, Any],
    template_id: str,
) -> QuestionPlan:
    agent = scene["agent"]
    action_base = scene["action_base"]
    action_past = scene["action_past"]
    object_plural = scene["object_plural"]
    object_reference = scene["object_reference"]
    object_phrase = scene["object_phrase"]
    location = scene["location"]
    time = scene["time"]
    reason = scene["reason"]
    quantity = quantity_answer(scene["quantity_value"], mode=scene["quantity_mode"])

    if question_type == "who":
        question = f"Who {action_past} {object_reference} {location}?"
        if template_id == "who_did_that":
            question = "Who did that?"
        elif template_id == "who_action_object":
            question = f"Who {action_past} {object_reference}?"
        elif template_id == "who_was_it":
            question = "Who was it?"
        return QuestionPlan(
            question_type=question_type,
            question=question,
            answer=_answer_sentence(f"{agent} did"),
            expected_slot=agent,
            question_template_id=template_id,
        )
    if question_type == "what":
        question = f"What did {agent} {action_base} {location}?"
        if template_id == "what_action_there":
            question = f"What did {agent} {action_base} there?"
        elif template_id == "what_action_short":
            question = f"What did {agent} {action_base}?"
        elif template_id == "what_was_it":
            question = "What was it?"
        return QuestionPlan(
            question_type=question_type,
            question=question,
            answer=_answer_sentence(object_phrase),
            expected_slot=object_phrase,
            question_template_id=template_id,
        )
    if question_type == "where":
        question = f"Where did {agent} {action_base} {object_reference}?"
        if template_id == "where_did_that_happen":
            question = (
                _destination_where_question(object_reference)
                if _is_destination(location)
                else "Where did that happen?"
            )
        elif template_id == "where_did_agent_do_that":
            question = (
                f"Where did {agent} {action_base} {object_reference}?"
                if _is_destination(location)
                else f"Where did {agent} do that?"
            )
        elif template_id == "where_was_it":
            question = (
                _destination_where_question(object_reference)
                if _is_destination(location)
                else "Where was it?"
            )
        return QuestionPlan(
            question_type=question_type,
            question=question,
            answer=_answer_sentence(location),
            expected_slot=location,
            question_template_id=template_id,
        )
    if question_type == "when":
        question = f"When did {agent} {action_base} {object_reference} {location}?"
        if template_id == "when_was_that":
            question = "When was that?"
        elif template_id == "when_action_object":
            question = f"When did {agent} {action_base} {object_reference}?"
        elif template_id == "when_did_that_happen":
            question = "When did that happen?"
        return QuestionPlan(
            question_type=question_type,
            question=question,
            answer=_answer_sentence(time),
            expected_slot=time,
            question_template_id=template_id,
        )
    if question_type == "why":
        question = f"Why did {agent} {action_base} {object_reference} {location}?"
        if template_id == "why_do_that":
            question = f"Why did {agent} do that?"
        elif template_id == "why_action_object":
            question = f"Why did {agent} {action_base} {object_reference}?"
        elif template_id == "why_did_that_happen":
            question = "Why did that happen?"
        return QuestionPlan(
            question_type=question_type,
            question=question,
            answer=_answer_sentence(reason),
            expected_slot=reason,
            question_template_id=template_id,
        )
    if question_type == "how_many":
        question = f"How many {object_plural} did {agent} {action_base} {location}?"
        if template_id == "how_many_were_there":
            question = f"How many {object_plural} were there?"
        elif template_id == "how_many_action_short":
            question = f"How many {object_plural} did {agent} {action_base}?"
        elif template_id == "how_many_object_short":
            question = f"How many {object_plural}?"
        return QuestionPlan(
            question_type=question_type,
            question=question,
            answer=quantity,
            expected_slot=quantity.rstrip("."),
            question_template_id=template_id,
        )
    raise ValueError(f"Unsupported question type: {question_type!r}")


def _question_template_ids(question_type: str, *, style: str) -> tuple[str, ...]:
    if style in SCENELESS_STYLES:
        try:
            return MINIMAL_QUESTION_TEMPLATE_IDS_BY_TYPE[question_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported question type: {question_type!r}") from exc
    try:
        return QUESTION_TEMPLATE_IDS_BY_TYPE[question_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported question type: {question_type!r}") from exc


def _question_count_weights_from_config(raw: Any) -> tuple[tuple[int, float], ...]:
    if raw is None:
        return DEFAULT_MINIMAL_QA_PAIR_QUESTION_COUNT_WEIGHTS

    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = raw

    weights: list[tuple[int, float]] = []
    for item in items:
        if isinstance(item, dict):
            count = int(item["count"])
            weight = float(item["weight"])
        else:
            count, weight = item
            count = int(count)
            weight = float(weight)
        if count <= 0:
            raise ValueError("minimal_qa_pair question counts must be positive.")
        if weight <= 0:
            raise ValueError("minimal_qa_pair question count weights must be positive.")
        weights.append((count, weight))
    if not weights:
        raise ValueError("minimal_qa_pair question count weights must not be empty.")
    return tuple(weights)


def _weighted_question_count(
    rng: random.Random,
    weights: tuple[tuple[int, float], ...],
    *,
    max_count: int,
) -> int:
    available = tuple((count, weight) for count, weight in weights if count <= max_count)
    if not available:
        raise ValueError(
            "minimal_qa_pair question count weights exceed available question_types."
        )
    counts, count_weights = zip(*available)
    return int(rng.choices(counts, weights=count_weights, k=1)[0])


def _scene_sentence(template_id: str, *, scene: dict[str, Any]) -> str:
    agent = scene["agent"]
    action_past = scene["action_past"]
    object_phrase = scene["object_phrase"]
    location = scene["location"]
    time = scene["time"]
    reason = scene["reason"]

    if template_id == "agent_time_reason":
        return (
            f"{_sentence_initial(agent)} {action_past} {object_phrase} {location} "
            f"{time} {reason}."
        )
    if template_id == "time_fronted":
        return (
            f"{_sentence_initial(time)}, {agent} {action_past} {object_phrase} "
            f"{location} {reason}."
        )
    if template_id == "reason_fronted":
        return (
            f"{_sentence_initial(reason)}, {agent} {action_past} {object_phrase} "
            f"{location} {time}."
        )
    raise ValueError(f"Unsupported scene sentence template: {template_id!r}")


def _answer_sentence(text: str) -> str:
    return f"{_sentence_initial(text)}."


def _is_destination(location: str) -> bool:
    return location.startswith(("to ", "across "))


def _destination_where_question(object_reference: str) -> str:
    return f"Where did {object_reference} go?"


def _object_reference_for_quantity(value: int, *, singular: str, plural: str) -> str:
    noun = singular if int(value) == 1 else plural
    return f"the {noun}"


def _sentence_initial(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}"


def _topics_for_style(style: str) -> tuple[TopicTemplate, ...]:
    topic_names = STYLE_TOPIC_NAMES.get(style)
    if topic_names is None:
        return TOPICS
    by_name = {topic.topic: topic for topic in TOPICS}
    try:
        return tuple(by_name[name] for name in topic_names)
    except KeyError as exc:
        raise ValueError(f"STYLE_TOPIC_NAMES references unknown topic: {exc.args[0]!r}") from exc


def build_scene_plans(config: dict[str, Any]) -> list[ScenePlan]:
    """Build deterministic scene plans from a config mapping."""

    seed = int(config.get("seed", 42))
    document_count = int(config.get("document_count", 50))
    if document_count <= 0:
        raise ValueError("document_count must be positive.")

    rng = random.Random(seed)
    styles = tuple(str(style) for style in config.get("styles", STYLES))
    question_types = tuple(str(kind) for kind in config.get("question_types", QUESTION_TYPES))
    quantities = tuple(int(value) for value in config.get("quantities", DEFAULT_QUANTITIES))
    quantity_modes = tuple(str(mode) for mode in config.get("quantity_modes", DEFAULT_QUANTITY_MODES))
    times = tuple(str(value) for value in config.get("times", DEFAULT_TIMES))
    min_questions = int(config.get("min_questions_per_document", 2))
    max_questions = int(config.get("max_questions_per_document", 5))
    minimal_question_count_weights = _question_count_weights_from_config(
        config.get("minimal_qa_pair_question_count_weights")
    )

    if not styles:
        raise ValueError("styles must not be empty.")
    if not question_types:
        raise ValueError("question_types must not be empty.")
    if min_questions <= 0 or max_questions < min_questions:
        raise ValueError("Question count bounds are invalid.")
    if max_questions > len(question_types):
        raise ValueError("max_questions_per_document cannot exceed available question_types.")
    if not quantities:
        raise ValueError("quantities must not be empty.")
    if not quantity_modes:
        raise ValueError("quantity_modes must not be empty.")
    if not times:
        raise ValueError("times must not be empty.")

    plans: list[ScenePlan] = []
    for index in range(document_count):
        style = styles[index % len(styles)]
        compatible_topics = _topics_for_style(style)
        topic = compatible_topics[(index // len(styles)) % len(compatible_topics)]
        agent = rng.choice(topic.agents)
        singular, plural = rng.choice(topic.objects)
        value = int(rng.choice(quantities))
        mode = str(rng.choice(quantity_modes))
        if value != 0 and mode == "no_for_zero":
            mode = "word"
        location = rng.choice(topic.locations)
        time = rng.choice(times)
        reason = rng.choice(topic.reasons)
        object_phrase = quantified_object_phrase(
            value,
            singular=singular,
            plural=plural or simple_plural(singular),
            mode=mode,
        )
        object_reference = _object_reference_for_quantity(
            value,
            singular=singular,
            plural=plural,
        )
        scene = {
            "agent": agent,
            "action_base": topic.action_base,
            "action_past": topic.action_past,
            "object_plural": plural,
            "object_reference": object_reference,
            "object_phrase": object_phrase,
            "quantity_value": value,
            "quantity_mode": mode,
            "location": location,
            "time": time,
            "reason": reason,
        }
        selected_questions = list(question_types)
        rng.shuffle(selected_questions)
        question_count = (
            _weighted_question_count(
                rng,
                minimal_question_count_weights,
                max_count=len(question_types),
            )
            if style in SCENELESS_STYLES
            else rng.randint(min_questions, max_questions)
        )
        selected_questions = selected_questions[:question_count]
        questions = tuple(
            _question_for_type(
                question_type,
                scene=scene,
                template_id=rng.choice(_question_template_ids(question_type, style=style)),
            )
            for question_type in selected_questions
        )
        scene_template_id = rng.choice(SCENE_SENTENCE_TEMPLATE_IDS)
        scene_sentence = _scene_sentence(scene_template_id, scene=scene)
        plans.append(
            ScenePlan(
                id=f"qa_scene_{index:05d}",
                style=style,
                topic=topic.topic,
                agent=agent,
                action_base=topic.action_base,
                action_past=topic.action_past,
                object_singular=singular,
                object_plural=plural,
                object_reference=object_reference,
                quantity_value=value,
                quantity_mode=mode,
                quantity_surface=quantity_surface(value, mode=mode),
                object_phrase=object_phrase,
                location=location,
                time=time,
                reason=reason,
                scene_sentence=scene_sentence,
                questions=questions,
                scene_template_id=scene_template_id,
            )
        )
    return plans
