"""Generate synthetic spatial-relation rows from symbolic scenes."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from babysteps.spatial_synth.world import (
    HORIZONTAL,
    OPPOSITE,
    Facing,
    State,
    Turn,
    compute_relations,
    position_for_world_relation,
    relative_to_world_relation,
    turn_facing,
)


DEFAULT_FAMILIES: tuple[str, ...] = (
    "static_egocentric",
    "absolute_role_swap",
    "cardinal_pass_by",
    "cardinal_rotation_invariance",
    "cardinal_farther_motion",
    "vertical_paraphrase_role_swap",
    "vertical_motion_update",
    "vertical_motion_distractor",
    "egocentric_same_facing_role_swap",
    "close_far_distance",
    "turn_update_basic",
)

RankObjective = Literal["completion_rank", "context_rank"]
DEFAULT_RANK_OBJECTIVE: RankObjective = "completion_rank"
CONTEXT_RANK_OBJECTIVE: RankObjective = "context_rank"

WORLD_RELATIONS: tuple[str, ...] = ("north", "east", "south", "west", "above", "below")
HORIZONTAL_WORLD_RELATIONS: tuple[str, ...] = ("north", "east", "south", "west")
RELATIVE_RELATIONS: tuple[str, ...] = ("front", "behind", "left", "right")
DISTANCE_RELATIONS: tuple[str, ...] = ("close", "far")
DISTANCE_TEMPLATE_MODES: tuple[str, ...] = (
    "metric_distance",
    "movement_toward_away",
    "symmetric_role_swap",
)
CARDINAL_PASS_OUTCOME_MODES: tuple[str, ...] = (
    "stops_before",
    "passes_beyond",
)
CARDINAL_PASS_TARGET_MODES: tuple[str, ...] = (
    "direct_relation",
    "role_swap",
)
CARDINAL_PASS_VERBS: tuple[str, ...] = (
    "walks",
    "runs",
    "drives",
    "crawls",
    "moves",
    "goes",
)
CARDINAL_INVARIANCE_TARGET_MODES: tuple[str, ...] = (
    "direct_relation",
    "role_swap",
)
CARDINAL_ROTATION_TURNS: tuple[Turn, ...] = ("left", "right", "around")
CARDINAL_CONTINUE_MOTION_VERBS: tuple[str, ...] = (
    "walking",
    "running",
    "driving",
    "crawling",
    "moving",
    "going",
)
VERTICAL_RELATIONS: tuple[str, ...] = ("above", "below")
VERTICAL_TEMPLATE_MODES: tuple[str, ...] = (
    "literal_role_swap",
    "paraphrase_to_literal",
    "paraphrased_role_swap",
)
VERTICAL_MOTION_TARGET_MODES: tuple[str, ...] = (
    "direct_relation",
    "role_swap",
)
VERTICAL_MOTION_MOVER_MODES: tuple[str, ...] = (
    "figure_moves",
    "ground_moves",
)
VERTICAL_DISTRACTOR_RELATION_MODES: tuple[str, ...] = (
    "same_distractor",
    "opposite_distractor",
)
VERTICAL_DISTRACTOR_ORDER_MODES: tuple[str, ...] = (
    "target_first",
    "distractor_first",
)
VERTICAL_PARAPHRASE_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "over_under",
        "{figure} is over {ground}.",
        "{figure} is under {ground}.",
    ),
    (
        "on_top_beneath",
        "{figure} is on top of {ground}.",
        "{figure} is beneath {ground}.",
    ),
    (
        "higher_lower",
        "{figure} is higher than {ground}.",
        "{figure} is lower than {ground}.",
    ),
)

AGENTS: tuple[str, ...] = (
    "Alex",
    "Blair",
    "Casey",
    "Dana",
    "Eli",
    "Harper",
    "Jordan",
    "Morgan",
    "Riley",
    "Taylor",
)

OBJECTS: tuple[str, ...] = (
    "bag",
    "book",
    "box",
    "chair",
    "coin",
    "cup",
    "key",
    "lamp",
    "mug",
    "notebook",
    "plate",
    "sign",
)


FIELDNAMES: tuple[str, ...] = (
    "item_id",
    "template_family",
    "relation_type",
    "rank_objective",
    "context",
    "completion",
    "hard_negative",
    "text",
    "negative_text",
    "true_relation",
    "negative_relation",
    "world_relation",
    "relative_relation",
    "figure",
    "ground",
    "observer",
    "observer_facing",
    "before_facing",
    "after_facing",
    "turn",
    "ntp_weight",
)


CONTEXT_PAIR_FIELDNAMES: tuple[str, ...] = (
    "item_id",
    "template_family",
    "relation_type",
    "rank_objective",
    "Domain",
    "ConceptA",
    "ConceptB",
    "ContextType",
    "ContextDiff",
    "TargetDiff",
    "Context1",
    "Context2",
    "Target1",
    "Target2",
    "text11",
    "text21",
    "text22",
    "text12",
    "target1_relation",
    "target2_relation",
    "context1_world_relation",
    "context2_world_relation",
    "figure",
    "ground",
    "observer",
    "observer_facing",
    "before_facing1",
    "after_facing1",
    "turn1",
    "before_facing2",
    "after_facing2",
    "turn2",
    "ntp_weight",
)


@dataclass(frozen=True)
class SpatialItem:
    """One synthetic spatial item with a correct and hard-negative completion."""

    item_id: str
    template_family: str
    relation_type: str
    context: str
    completion: str
    hard_negative: str
    true_relation: str
    negative_relation: str
    world_relation: str
    relative_relation: str
    figure: str
    ground: str
    observer: str
    observer_facing: str
    before_facing: str = ""
    after_facing: str = ""
    turn: str = ""
    ntp_weight: float = 1.0

    def to_row(self) -> dict[str, str]:
        return {
            "item_id": self.item_id,
            "template_family": self.template_family,
            "relation_type": self.relation_type,
            "rank_objective": DEFAULT_RANK_OBJECTIVE,
            "context": self.context,
            "completion": self.completion,
            "hard_negative": self.hard_negative,
            "text": join_context_completion(self.context, self.completion),
            "negative_text": join_context_completion(self.context, self.hard_negative),
            "true_relation": self.true_relation,
            "negative_relation": self.negative_relation,
            "world_relation": self.world_relation,
            "relative_relation": self.relative_relation,
            "figure": self.figure,
            "ground": self.ground,
            "observer": self.observer,
            "observer_facing": self.observer_facing,
            "before_facing": self.before_facing,
            "after_facing": self.after_facing,
            "turn": self.turn,
            "ntp_weight": f"{self.ntp_weight:g}",
        }


@dataclass(frozen=True)
class SpatialContextPair:
    """One EWoK-style 2x2 context-rank pair."""

    item_id: str
    template_family: str
    relation_type: str
    context1: str
    context2: str
    target1: str
    target2: str
    concept_a: str
    concept_b: str
    context_type: str
    context_diff: str
    target_diff: str
    target1_relation: str
    target2_relation: str
    context1_world_relation: str
    context2_world_relation: str
    figure: str
    ground: str
    observer: str = ""
    observer_facing: str = ""
    before_facing1: str = ""
    after_facing1: str = ""
    turn1: str = ""
    before_facing2: str = ""
    after_facing2: str = ""
    turn2: str = ""
    ntp_weight: float = 1.0

    def to_row(self) -> dict[str, str]:
        return {
            "item_id": self.item_id,
            "template_family": self.template_family,
            "relation_type": self.relation_type,
            "rank_objective": CONTEXT_RANK_OBJECTIVE,
            "Domain": "synthetic-spatial-relations",
            "ConceptA": self.concept_a,
            "ConceptB": self.concept_b,
            "ContextType": self.context_type,
            "ContextDiff": self.context_diff,
            "TargetDiff": self.target_diff,
            "Context1": self.context1,
            "Context2": self.context2,
            "Target1": self.target1,
            "Target2": self.target2,
            "text11": join_context_completion(self.context1, self.target1),
            "text21": join_context_completion(self.context2, self.target1),
            "text22": join_context_completion(self.context2, self.target2),
            "text12": join_context_completion(self.context1, self.target2),
            "target1_relation": self.target1_relation,
            "target2_relation": self.target2_relation,
            "context1_world_relation": self.context1_world_relation,
            "context2_world_relation": self.context2_world_relation,
            "figure": self.figure,
            "ground": self.ground,
            "observer": self.observer,
            "observer_facing": self.observer_facing,
            "before_facing1": self.before_facing1,
            "after_facing1": self.after_facing1,
            "turn1": self.turn1,
            "before_facing2": self.before_facing2,
            "after_facing2": self.after_facing2,
            "turn2": self.turn2,
            "ntp_weight": f"{self.ntp_weight:g}",
        }


SpatialExample = SpatialItem | SpatialContextPair


def join_context_completion(context: str, completion: str) -> str:
    return f"{context.rstrip()} {completion.lstrip()}"


def noun_phrase(noun: str) -> str:
    return f"the {noun}"


def sentence_case(text: str) -> str:
    if not text:
        return text
    return text[:1].upper() + text[1:]


def render_world_relation(figure: str, ground: str, relation: str) -> str:
    if relation in {"above", "below"}:
        return sentence_case(f"{figure} is {relation} {ground}.")
    return sentence_case(f"{figure} is {relation} of {ground}.")


def render_relative_relation(figure: str, observer: str, relation: str) -> str:
    if relation == "front":
        return sentence_case(f"{figure} is in front of {observer}.")
    if relation == "behind":
        return sentence_case(f"{figure} is behind {observer}.")
    if relation in {"left", "right"}:
        return sentence_case(f"{figure} is to {observer}'s {relation}.")
    if relation in {"above", "below"}:
        return sentence_case(f"{figure} is {relation} {observer}.")
    raise ValueError(f"Unsupported relative relation {relation!r}.")


def render_distance_relation(figure: str, ground: str, relation: str) -> str:
    if relation == "close":
        return sentence_case(f"{figure} is close to {ground}.")
    if relation == "far":
        return sentence_case(f"{figure} is far from {ground}.")
    raise ValueError(f"Unsupported distance relation {relation!r}.")


def _sample_object_pair(rng: random.Random) -> tuple[str, str]:
    first, second = rng.sample(OBJECTS, 2)
    return noun_phrase(first), noun_phrase(second)


def _sample_agent_pair(rng: random.Random) -> tuple[str, str]:
    return rng.sample(AGENTS, 2)


def _sample_two_object_pairs(rng: random.Random) -> tuple[str, str, str, str]:
    first, second, third, fourth = rng.sample(OBJECTS, 4)
    return (
        noun_phrase(first),
        noun_phrase(second),
        noun_phrase(third),
        noun_phrase(fourth),
    )


def _render_static_egocentric_context(
    *,
    observer: str,
    figure: str,
    facing: str,
    world_relation_name: str,
    rng: random.Random,
) -> str:
    context_templates = (
        "{observer} is facing {facing}. {world_sentence}",
        "{observer} faces {facing}. {world_sentence}",
    )
    world_sentence = render_world_relation(figure, observer, world_relation_name)
    return rng.choice(context_templates).format(
        observer=observer,
        facing=facing,
        world_sentence=world_sentence,
    )


def generate_static_egocentric_item(index: int, rng: random.Random) -> SpatialItem:
    observer = rng.choice(AGENTS)
    figure = noun_phrase(rng.choice(OBJECTS))
    facing: Facing = rng.choice(HORIZONTAL)
    world = rng.choice(HORIZONTAL_WORLD_RELATIONS)

    state = State(
        a_pos=(0, 0, 0),
        b_pos=position_for_world_relation(world),
        a_facing=facing,
        b_facing="north",
    )
    relations = compute_relations(state, "a")
    negative = OPPOSITE[relations.relative]

    context = _render_static_egocentric_context(
        observer=observer,
        figure=figure,
        facing=facing,
        world_relation_name=relations.world,
        rng=rng,
    )

    return SpatialItem(
        item_id=f"static_egocentric:{index:06d}",
        template_family="static_egocentric",
        relation_type="egocentric",
        context=context,
        completion=render_relative_relation(figure, observer, relations.relative),
        hard_negative=render_relative_relation(figure, observer, negative),
        true_relation=relations.relative,
        negative_relation=negative,
        world_relation=relations.world,
        relative_relation=relations.relative,
        figure=figure,
        ground=observer,
        observer=observer,
        observer_facing=facing,
        before_facing=facing,
        after_facing=facing,
    )


def generate_static_egocentric_pair(index: int, rng: random.Random) -> SpatialContextPair:
    observer = rng.choice(AGENTS)
    figure = noun_phrase(rng.choice(OBJECTS))
    facing: Facing = rng.choice(HORIZONTAL)
    world1 = rng.choice(HORIZONTAL_WORLD_RELATIONS)
    world2 = OPPOSITE[world1]

    state1 = State(
        a_pos=(0, 0, 0),
        b_pos=position_for_world_relation(world1),
        a_facing=facing,
        b_facing="north",
    )
    state2 = State(
        a_pos=(0, 0, 0),
        b_pos=position_for_world_relation(world2),
        a_facing=facing,
        b_facing="north",
    )
    relations1 = compute_relations(state1, "a")
    relations2 = compute_relations(state2, "a")

    return SpatialContextPair(
        item_id=f"static_egocentric:{index:06d}",
        template_family="static_egocentric",
        relation_type="egocentric",
        context1=_render_static_egocentric_context(
            observer=observer,
            figure=figure,
            facing=facing,
            world_relation_name=relations1.world,
            rng=rng,
        ),
        context2=_render_static_egocentric_context(
            observer=observer,
            figure=figure,
            facing=facing,
            world_relation_name=relations2.world,
            rng=rng,
        ),
        target1=render_relative_relation(figure, observer, relations1.relative),
        target2=render_relative_relation(figure, observer, relations2.relative),
        concept_a=relations1.relative,
        concept_b=relations2.relative,
        context_type="static_egocentric",
        context_diff="world_offset",
        target_diff="relative_relation",
        target1_relation=relations1.relative,
        target2_relation=relations2.relative,
        context1_world_relation=relations1.world,
        context2_world_relation=relations2.world,
        figure=figure,
        ground=observer,
        observer=observer,
        observer_facing=facing,
        before_facing1=facing,
        after_facing1=facing,
        before_facing2=facing,
        after_facing2=facing,
    )


def generate_absolute_role_swap_item(
    index: int,
    rng: random.Random,
    *,
    family_name: str = "absolute_role_swap",
) -> SpatialItem:
    ground, figure = _sample_object_pair(rng)
    world = WORLD_RELATIONS[index % len(WORLD_RELATIONS)]
    inverse = OPPOSITE[world]

    context = render_world_relation(figure, ground, world)
    completion = render_world_relation(ground, figure, inverse)
    hard_negative = render_world_relation(ground, figure, world)

    return SpatialItem(
        item_id=f"{family_name}:{index:06d}",
        template_family=family_name,
        relation_type="world",
        context=context,
        completion=completion,
        hard_negative=hard_negative,
        true_relation=inverse,
        negative_relation=world,
        world_relation=world,
        relative_relation="",
        figure=figure,
        ground=ground,
        observer="",
        observer_facing="",
    )


def generate_absolute_role_swap_pair(
    index: int,
    rng: random.Random,
    *,
    family_name: str = "absolute_role_swap",
) -> SpatialContextPair:
    ground, figure = _sample_object_pair(rng)
    world1 = WORLD_RELATIONS[index % len(WORLD_RELATIONS)]
    world2 = OPPOSITE[world1]
    target1_relation = OPPOSITE[world1]
    target2_relation = OPPOSITE[world2]

    return SpatialContextPair(
        item_id=f"{family_name}:{index:06d}",
        template_family=family_name,
        relation_type="world",
        context1=render_world_relation(figure, ground, world1),
        context2=render_world_relation(figure, ground, world2),
        target1=render_world_relation(ground, figure, target1_relation),
        target2=render_world_relation(ground, figure, target2_relation),
        concept_a=target1_relation,
        concept_b=target2_relation,
        context_type=family_name,
        context_diff="world_relation",
        target_diff="inverse_world_relation",
        target1_relation=target1_relation,
        target2_relation=target2_relation,
        context1_world_relation=world1,
        context2_world_relation=world2,
        figure=figure,
        ground=ground,
    )


def _cardinal_pass_target_mode(index: int) -> str:
    return CARDINAL_PASS_TARGET_MODES[index % len(CARDINAL_PASS_TARGET_MODES)]


def _cardinal_pass_outcome_mode(index: int) -> str:
    return CARDINAL_PASS_OUTCOME_MODES[
        (index // len(CARDINAL_PASS_TARGET_MODES))
        % len(CARDINAL_PASS_OUTCOME_MODES)
    ]


def _cardinal_pass_relation(index: int) -> str:
    cycle = len(CARDINAL_PASS_TARGET_MODES) * len(CARDINAL_PASS_OUTCOME_MODES)
    return HORIZONTAL_WORLD_RELATIONS[
        (index // cycle) % len(HORIZONTAL_WORLD_RELATIONS)
    ]


def _cardinal_pass_verb(index: int) -> str:
    cycle = (
        len(CARDINAL_PASS_TARGET_MODES)
        * len(CARDINAL_PASS_OUTCOME_MODES)
        * len(HORIZONTAL_WORLD_RELATIONS)
    )
    return CARDINAL_PASS_VERBS[(index // cycle) % len(CARDINAL_PASS_VERBS)]


def cardinal_pass_start_and_move(
    final_relation: str,
    outcome_mode: str,
) -> tuple[str, str]:
    """Return start relation and movement direction for a cardinal pass-by event."""

    if final_relation not in HORIZONTAL_WORLD_RELATIONS:
        raise ValueError("cardinal pass-by only supports north/east/south/west.")
    if outcome_mode == "stops_before":
        return final_relation, OPPOSITE[final_relation]
    if outcome_mode == "passes_beyond":
        return OPPOSITE[final_relation], final_relation
    raise ValueError(f"Unsupported cardinal pass-by outcome: {outcome_mode!r}.")


def render_cardinal_pass_context(
    *,
    figure: str,
    ground: str,
    final_relation: str,
    outcome_mode: str,
    verb: str,
) -> str:
    """Render a motion event where an agent does or does not pass a referent."""

    if verb not in CARDINAL_PASS_VERBS:
        raise ValueError(f"Unsupported cardinal pass-by verb: {verb!r}.")
    start_relation, move_direction = cardinal_pass_start_and_move(
        final_relation,
        outcome_mode,
    )
    start_sentence = render_world_relation(figure, ground, start_relation)
    if outcome_mode == "stops_before":
        return (
            f"{start_sentence} "
            f"{figure} {verb} {move_direction} toward {ground} but stops before reaching it."
        )
    if outcome_mode == "passes_beyond":
        return (
            f"{start_sentence} "
            f"{figure} {verb} {move_direction} past {ground} to the other side."
        )
    raise ValueError(f"Unsupported cardinal pass-by outcome: {outcome_mode!r}.")


def _cardinal_pass_target(
    *,
    figure: str,
    ground: str,
    final_relation: str,
    target_mode: str,
) -> tuple[str, str, str, str]:
    if target_mode == "direct_relation":
        return (
            render_world_relation(figure, ground, final_relation),
            final_relation,
            figure,
            ground,
        )
    if target_mode == "role_swap":
        target_relation = OPPOSITE[final_relation]
        return (
            render_world_relation(ground, figure, target_relation),
            target_relation,
            ground,
            figure,
        )
    raise ValueError(f"Unsupported cardinal pass-by target mode: {target_mode!r}.")


def _cardinal_pass_context_target(
    *,
    figure: str,
    ground: str,
    final_relation: str,
    outcome_mode: str,
    target_mode: str,
    verb: str,
) -> tuple[str, str, str, str, str, str, str]:
    start_relation, move_direction = cardinal_pass_start_and_move(
        final_relation,
        outcome_mode,
    )
    context = render_cardinal_pass_context(
        figure=figure,
        ground=ground,
        final_relation=final_relation,
        outcome_mode=outcome_mode,
        verb=verb,
    )
    target, target_relation, target_figure, target_ground = _cardinal_pass_target(
        figure=figure,
        ground=ground,
        final_relation=final_relation,
        target_mode=target_mode,
    )
    return (
        context,
        target,
        target_relation,
        target_figure,
        target_ground,
        start_relation,
        move_direction,
    )


def generate_cardinal_pass_by_item(index: int, rng: random.Random) -> SpatialItem:
    figure = rng.choice(AGENTS)
    ground = noun_phrase(rng.choice(OBJECTS))
    final_relation = _cardinal_pass_relation(index)
    outcome_mode = _cardinal_pass_outcome_mode(index)
    target_mode = _cardinal_pass_target_mode(index)
    verb = _cardinal_pass_verb(index)
    (
        context,
        completion,
        target_relation,
        target_figure,
        target_ground,
        _,
        move_direction,
    ) = _cardinal_pass_context_target(
        figure=figure,
        ground=ground,
        final_relation=final_relation,
        outcome_mode=outcome_mode,
        target_mode=target_mode,
        verb=verb,
    )
    negative_relation = OPPOSITE[target_relation]

    return SpatialItem(
        item_id=f"cardinal_pass_by:{index:06d}",
        template_family="cardinal_pass_by",
        relation_type="world",
        context=context,
        completion=completion,
        hard_negative=render_world_relation(
            target_figure,
            target_ground,
            negative_relation,
        ),
        true_relation=target_relation,
        negative_relation=negative_relation,
        world_relation=final_relation,
        relative_relation="",
        figure=figure,
        ground=ground,
        observer="",
        observer_facing="",
        turn=f"{outcome_mode}:{move_direction}",
    )


def generate_cardinal_pass_by_pair(
    index: int,
    rng: random.Random,
) -> SpatialContextPair:
    figure = rng.choice(AGENTS)
    ground = noun_phrase(rng.choice(OBJECTS))
    final_relation1 = _cardinal_pass_relation(index)
    final_relation2 = OPPOSITE[final_relation1]
    outcome_mode = _cardinal_pass_outcome_mode(index)
    target_mode = _cardinal_pass_target_mode(index)
    verb = _cardinal_pass_verb(index)
    context1, target1, target1_relation, _, _, start1, move1 = (
        _cardinal_pass_context_target(
            figure=figure,
            ground=ground,
            final_relation=final_relation1,
            outcome_mode=outcome_mode,
            target_mode=target_mode,
            verb=verb,
        )
    )
    context2, target2, target2_relation, _, _, start2, move2 = (
        _cardinal_pass_context_target(
            figure=figure,
            ground=ground,
            final_relation=final_relation2,
            outcome_mode=outcome_mode,
            target_mode=target_mode,
            verb=verb,
        )
    )

    return SpatialContextPair(
        item_id=f"cardinal_pass_by:{index:06d}",
        template_family="cardinal_pass_by",
        relation_type="world",
        context1=context1,
        context2=context2,
        target1=target1,
        target2=target2,
        concept_a=target1_relation,
        concept_b=target2_relation,
        context_type="cardinal_pass_by",
        context_diff=outcome_mode,
        target_diff=target_mode,
        target1_relation=target1_relation,
        target2_relation=target2_relation,
        context1_world_relation=final_relation1,
        context2_world_relation=final_relation2,
        figure=figure,
        ground=ground,
        turn1=f"start:{start1};move:{move1}",
        turn2=f"start:{start2};move:{move2}",
    )


def _cardinal_invariance_target_mode(index: int) -> str:
    return CARDINAL_INVARIANCE_TARGET_MODES[
        index % len(CARDINAL_INVARIANCE_TARGET_MODES)
    ]


def _cardinal_rotation_turn(index: int) -> Turn:
    return CARDINAL_ROTATION_TURNS[
        (index // len(CARDINAL_INVARIANCE_TARGET_MODES))
        % len(CARDINAL_ROTATION_TURNS)
    ]


def _cardinal_rotation_relation(index: int) -> str:
    cycle = len(CARDINAL_INVARIANCE_TARGET_MODES) * len(CARDINAL_ROTATION_TURNS)
    return HORIZONTAL_WORLD_RELATIONS[
        (index // cycle) % len(HORIZONTAL_WORLD_RELATIONS)
    ]


def _cardinal_rotation_facing(index: int) -> Facing:
    cycle = (
        len(CARDINAL_INVARIANCE_TARGET_MODES)
        * len(CARDINAL_ROTATION_TURNS)
        * len(HORIZONTAL_WORLD_RELATIONS)
    )
    return HORIZONTAL[(index // cycle) % len(HORIZONTAL)]


def _turn_phrase(turn: str) -> str:
    return "turns around" if turn == "around" else f"turns {turn}"


def render_cardinal_rotation_context(
    *,
    figure: str,
    ground: str,
    relation: str,
    before_facing: Facing,
    turn: Turn,
) -> str:
    """Render an absolute-relation scene where only body orientation changes."""

    start_sentence = render_world_relation(figure, ground, relation)
    return (
        f"{start_sentence} "
        f"{figure} faces {before_facing} and {_turn_phrase(turn)}."
    )


def _cardinal_rotation_context_target(
    *,
    figure: str,
    ground: str,
    relation: str,
    target_mode: str,
    before_facing: Facing,
    turn: Turn,
) -> tuple[str, str, str, str, Facing]:
    context = render_cardinal_rotation_context(
        figure=figure,
        ground=ground,
        relation=relation,
        before_facing=before_facing,
        turn=turn,
    )
    target, target_relation, target_figure, target_ground = _cardinal_pass_target(
        figure=figure,
        ground=ground,
        final_relation=relation,
        target_mode=target_mode,
    )
    return (
        context,
        target,
        target_relation,
        target_figure,
        target_ground,
        turn_facing(before_facing, turn),
    )


def generate_cardinal_rotation_invariance_item(
    index: int,
    rng: random.Random,
) -> SpatialItem:
    figure = rng.choice(AGENTS)
    ground = noun_phrase(rng.choice(OBJECTS))
    relation = _cardinal_rotation_relation(index)
    target_mode = _cardinal_invariance_target_mode(index)
    before_facing = _cardinal_rotation_facing(index)
    turn = _cardinal_rotation_turn(index)
    (
        context,
        completion,
        target_relation,
        target_figure,
        target_ground,
        after_facing,
    ) = _cardinal_rotation_context_target(
        figure=figure,
        ground=ground,
        relation=relation,
        target_mode=target_mode,
        before_facing=before_facing,
        turn=turn,
    )
    negative_relation = OPPOSITE[target_relation]

    return SpatialItem(
        item_id=f"cardinal_rotation_invariance:{index:06d}",
        template_family="cardinal_rotation_invariance",
        relation_type="world",
        context=context,
        completion=completion,
        hard_negative=render_world_relation(
            target_figure,
            target_ground,
            negative_relation,
        ),
        true_relation=target_relation,
        negative_relation=negative_relation,
        world_relation=relation,
        relative_relation="",
        figure=figure,
        ground=ground,
        observer=figure,
        observer_facing=after_facing,
        before_facing=before_facing,
        after_facing=after_facing,
        turn=turn,
    )


def generate_cardinal_rotation_invariance_pair(
    index: int,
    rng: random.Random,
) -> SpatialContextPair:
    figure = rng.choice(AGENTS)
    ground = noun_phrase(rng.choice(OBJECTS))
    relation1 = _cardinal_rotation_relation(index)
    relation2 = OPPOSITE[relation1]
    target_mode = _cardinal_invariance_target_mode(index)
    before_facing = _cardinal_rotation_facing(index)
    turn = _cardinal_rotation_turn(index)
    context1, target1, target1_relation, _, _, after_facing = (
        _cardinal_rotation_context_target(
            figure=figure,
            ground=ground,
            relation=relation1,
            target_mode=target_mode,
            before_facing=before_facing,
            turn=turn,
        )
    )
    context2, target2, target2_relation, _, _, _ = (
        _cardinal_rotation_context_target(
            figure=figure,
            ground=ground,
            relation=relation2,
            target_mode=target_mode,
            before_facing=before_facing,
            turn=turn,
        )
    )

    return SpatialContextPair(
        item_id=f"cardinal_rotation_invariance:{index:06d}",
        template_family="cardinal_rotation_invariance",
        relation_type="world",
        context1=context1,
        context2=context2,
        target1=target1,
        target2=target2,
        concept_a=target1_relation,
        concept_b=target2_relation,
        context_type="cardinal_rotation_invariance",
        context_diff="world_relation_after_rotation",
        target_diff=target_mode,
        target1_relation=target1_relation,
        target2_relation=target2_relation,
        context1_world_relation=relation1,
        context2_world_relation=relation2,
        figure=figure,
        ground=ground,
        observer=figure,
        observer_facing=after_facing,
        before_facing1=before_facing,
        after_facing1=after_facing,
        turn1=turn,
        before_facing2=before_facing,
        after_facing2=after_facing,
        turn2=turn,
    )


def _cardinal_farther_relation(index: int) -> str:
    return HORIZONTAL_WORLD_RELATIONS[
        (index // len(CARDINAL_INVARIANCE_TARGET_MODES))
        % len(HORIZONTAL_WORLD_RELATIONS)
    ]


def _cardinal_farther_verb(index: int) -> str:
    cycle = len(CARDINAL_INVARIANCE_TARGET_MODES) * len(HORIZONTAL_WORLD_RELATIONS)
    return CARDINAL_CONTINUE_MOTION_VERBS[
        (index // cycle) % len(CARDINAL_CONTINUE_MOTION_VERBS)
    ]


def render_cardinal_farther_motion_context(
    *,
    figure: str,
    ground: str,
    relation: str,
    verb: str,
) -> str:
    """Render motion farther along the same cardinal axis."""

    if verb not in CARDINAL_CONTINUE_MOTION_VERBS:
        raise ValueError(f"Unsupported cardinal continuation verb: {verb!r}.")
    start_sentence = render_world_relation(figure, ground, relation)
    return (
        f"{start_sentence} "
        f"{figure} continues {verb} {relation}, farther from {ground}."
    )


def _cardinal_farther_context_target(
    *,
    figure: str,
    ground: str,
    relation: str,
    target_mode: str,
    verb: str,
) -> tuple[str, str, str, str, str]:
    context = render_cardinal_farther_motion_context(
        figure=figure,
        ground=ground,
        relation=relation,
        verb=verb,
    )
    target, target_relation, target_figure, target_ground = _cardinal_pass_target(
        figure=figure,
        ground=ground,
        final_relation=relation,
        target_mode=target_mode,
    )
    return context, target, target_relation, target_figure, target_ground


def generate_cardinal_farther_motion_item(
    index: int,
    rng: random.Random,
) -> SpatialItem:
    figure = rng.choice(AGENTS)
    ground = noun_phrase(rng.choice(OBJECTS))
    relation = _cardinal_farther_relation(index)
    target_mode = _cardinal_invariance_target_mode(index)
    verb = _cardinal_farther_verb(index)
    context, completion, target_relation, target_figure, target_ground = (
        _cardinal_farther_context_target(
            figure=figure,
            ground=ground,
            relation=relation,
            target_mode=target_mode,
            verb=verb,
        )
    )
    negative_relation = OPPOSITE[target_relation]

    return SpatialItem(
        item_id=f"cardinal_farther_motion:{index:06d}",
        template_family="cardinal_farther_motion",
        relation_type="world",
        context=context,
        completion=completion,
        hard_negative=render_world_relation(
            target_figure,
            target_ground,
            negative_relation,
        ),
        true_relation=target_relation,
        negative_relation=negative_relation,
        world_relation=relation,
        relative_relation="",
        figure=figure,
        ground=ground,
        observer="",
        observer_facing="",
        turn=f"continue:{relation}",
    )


def generate_cardinal_farther_motion_pair(
    index: int,
    rng: random.Random,
) -> SpatialContextPair:
    figure = rng.choice(AGENTS)
    ground = noun_phrase(rng.choice(OBJECTS))
    relation1 = _cardinal_farther_relation(index)
    relation2 = OPPOSITE[relation1]
    target_mode = _cardinal_invariance_target_mode(index)
    verb = _cardinal_farther_verb(index)
    context1, target1, target1_relation, _, _ = _cardinal_farther_context_target(
        figure=figure,
        ground=ground,
        relation=relation1,
        target_mode=target_mode,
        verb=verb,
    )
    context2, target2, target2_relation, _, _ = _cardinal_farther_context_target(
        figure=figure,
        ground=ground,
        relation=relation2,
        target_mode=target_mode,
        verb=verb,
    )

    return SpatialContextPair(
        item_id=f"cardinal_farther_motion:{index:06d}",
        template_family="cardinal_farther_motion",
        relation_type="world",
        context1=context1,
        context2=context2,
        target1=target1,
        target2=target2,
        concept_a=target1_relation,
        concept_b=target2_relation,
        context_type="cardinal_farther_motion",
        context_diff="world_relation_after_farther_motion",
        target_diff=target_mode,
        target1_relation=target1_relation,
        target2_relation=target2_relation,
        context1_world_relation=relation1,
        context2_world_relation=relation2,
        figure=figure,
        ground=ground,
        turn1=f"continue:{relation1}",
        turn2=f"continue:{relation2}",
    )


def generate_reciprocal_inversion_item(index: int, rng: random.Random) -> SpatialItem:
    return generate_absolute_role_swap_item(
        index,
        rng,
        family_name="reciprocal_inversion",
    )


def generate_reciprocal_inversion_pair(
    index: int,
    rng: random.Random,
) -> SpatialContextPair:
    return generate_absolute_role_swap_pair(
        index,
        rng,
        family_name="reciprocal_inversion",
    )


def _vertical_mode(index: int) -> str:
    return VERTICAL_TEMPLATE_MODES[index % len(VERTICAL_TEMPLATE_MODES)]


def _vertical_relation(index: int) -> str:
    relation_index = (index // len(VERTICAL_TEMPLATE_MODES)) % len(VERTICAL_RELATIONS)
    return VERTICAL_RELATIONS[relation_index]


def _vertical_paraphrase_index(index: int) -> int:
    cycle = len(VERTICAL_TEMPLATE_MODES) * len(VERTICAL_RELATIONS)
    return (index // cycle) % len(VERTICAL_PARAPHRASE_PAIRS)


def render_vertical_paraphrase(
    figure: str,
    ground: str,
    relation: str,
    paraphrase_index: int,
) -> str:
    """Render a vertical relation with a balanced paraphrase pair."""

    if relation not in {"above", "below"}:
        raise ValueError("vertical paraphrases only support above/below.")
    _, above_template, below_template = VERTICAL_PARAPHRASE_PAIRS[
        paraphrase_index % len(VERTICAL_PARAPHRASE_PAIRS)
    ]
    template = above_template if relation == "above" else below_template
    return sentence_case(template.format(figure=figure, ground=ground))


def _vertical_context_target(
    *,
    figure: str,
    ground: str,
    relation: str,
    mode: str,
    paraphrase_index: int,
) -> tuple[str, str, str, str, str]:
    if mode == "literal_role_swap":
        target_figure = ground
        target_ground = figure
        target_relation = OPPOSITE[relation]
        return (
            render_world_relation(figure, ground, relation),
            render_world_relation(target_figure, target_ground, target_relation),
            target_relation,
            target_figure,
            target_ground,
        )
    if mode == "paraphrase_to_literal":
        return (
            render_vertical_paraphrase(figure, ground, relation, paraphrase_index),
            render_world_relation(figure, ground, relation),
            relation,
            figure,
            ground,
        )
    if mode == "paraphrased_role_swap":
        target_figure = ground
        target_ground = figure
        target_relation = OPPOSITE[relation]
        return (
            render_vertical_paraphrase(figure, ground, relation, paraphrase_index),
            render_world_relation(target_figure, target_ground, target_relation),
            target_relation,
            target_figure,
            target_ground,
        )
    raise ValueError(f"Unsupported vertical template mode: {mode!r}.")


def generate_vertical_paraphrase_role_swap_item(
    index: int,
    rng: random.Random,
) -> SpatialItem:
    ground, figure = _sample_object_pair(rng)
    mode = _vertical_mode(index)
    relation = _vertical_relation(index)
    paraphrase_index = _vertical_paraphrase_index(index)
    context, completion, target_relation, target_figure, target_ground = (
        _vertical_context_target(
            figure=figure,
            ground=ground,
            relation=relation,
            mode=mode,
            paraphrase_index=paraphrase_index,
        )
    )
    negative_relation = OPPOSITE[target_relation]

    return SpatialItem(
        item_id=f"vertical_paraphrase_role_swap:{index:06d}",
        template_family="vertical_paraphrase_role_swap",
        relation_type="vertical",
        context=context,
        completion=completion,
        hard_negative=render_world_relation(
            target_figure,
            target_ground,
            negative_relation,
        ),
        true_relation=target_relation,
        negative_relation=negative_relation,
        world_relation=relation,
        relative_relation="",
        figure=figure,
        ground=ground,
        observer="",
        observer_facing="",
    )


def generate_vertical_paraphrase_role_swap_pair(
    index: int,
    rng: random.Random,
) -> SpatialContextPair:
    ground, figure = _sample_object_pair(rng)
    mode = _vertical_mode(index)
    relation1 = _vertical_relation(index)
    relation2 = OPPOSITE[relation1]
    paraphrase_index = _vertical_paraphrase_index(index)
    context1, target1, target1_relation, _, _ = _vertical_context_target(
        figure=figure,
        ground=ground,
        relation=relation1,
        mode=mode,
        paraphrase_index=paraphrase_index,
    )
    context2, target2, target2_relation, _, _ = _vertical_context_target(
        figure=figure,
        ground=ground,
        relation=relation2,
        mode=mode,
        paraphrase_index=paraphrase_index,
    )

    return SpatialContextPair(
        item_id=f"vertical_paraphrase_role_swap:{index:06d}",
        template_family="vertical_paraphrase_role_swap",
        relation_type="vertical",
        context1=context1,
        context2=context2,
        target1=target1,
        target2=target2,
        concept_a=target1_relation,
        concept_b=target2_relation,
        context_type="vertical_paraphrase_role_swap",
        context_diff=mode,
        target_diff="vertical_relation",
        target1_relation=target1_relation,
        target2_relation=target2_relation,
        context1_world_relation=relation1,
        context2_world_relation=relation2,
        figure=figure,
        ground=ground,
    )


def _vertical_motion_target_mode(index: int) -> str:
    return VERTICAL_MOTION_TARGET_MODES[index % len(VERTICAL_MOTION_TARGET_MODES)]


def _vertical_motion_mover_mode(index: int) -> str:
    return VERTICAL_MOTION_MOVER_MODES[
        (index // len(VERTICAL_MOTION_TARGET_MODES))
        % len(VERTICAL_MOTION_MOVER_MODES)
    ]


def _vertical_motion_relation(index: int) -> str:
    cycle = len(VERTICAL_MOTION_TARGET_MODES) * len(VERTICAL_MOTION_MOVER_MODES)
    return VERTICAL_RELATIONS[(index // cycle) % len(VERTICAL_RELATIONS)]


def _vertical_distractor_relation_mode(index: int) -> str:
    cycle = (
        len(VERTICAL_MOTION_TARGET_MODES)
        * len(VERTICAL_MOTION_MOVER_MODES)
        * len(VERTICAL_RELATIONS)
    )
    return VERTICAL_DISTRACTOR_RELATION_MODES[
        (index // cycle) % len(VERTICAL_DISTRACTOR_RELATION_MODES)
    ]


def _vertical_distractor_order_mode(index: int) -> str:
    cycle = (
        len(VERTICAL_MOTION_TARGET_MODES)
        * len(VERTICAL_MOTION_MOVER_MODES)
        * len(VERTICAL_RELATIONS)
        * len(VERTICAL_DISTRACTOR_RELATION_MODES)
    )
    return VERTICAL_DISTRACTOR_ORDER_MODES[
        (index // cycle) % len(VERTICAL_DISTRACTOR_ORDER_MODES)
    ]


def _vertical_distractor_mover_mode(index: int) -> str:
    cycle = (
        len(VERTICAL_MOTION_TARGET_MODES)
        * len(VERTICAL_MOTION_MOVER_MODES)
        * len(VERTICAL_RELATIONS)
        * len(VERTICAL_DISTRACTOR_RELATION_MODES)
        * len(VERTICAL_DISTRACTOR_ORDER_MODES)
    )
    return VERTICAL_MOTION_MOVER_MODES[
        (index // cycle) % len(VERTICAL_MOTION_MOVER_MODES)
    ]


def vertical_move_direction(relation: str, mover_mode: str) -> str:
    """Return the movement that creates ``figure relation ground``."""

    if relation not in {"above", "below"}:
        raise ValueError("vertical motion only supports above/below.")
    if mover_mode == "figure_moves":
        return "up" if relation == "above" else "down"
    if mover_mode == "ground_moves":
        return "down" if relation == "above" else "up"
    raise ValueError(f"Unsupported vertical mover mode: {mover_mode!r}.")


def render_vertical_motion_event(
    *,
    figure: str,
    ground: str,
    relation: str,
    mover_mode: str,
) -> str:
    """Render a state update that creates a vertical relation."""

    direction = vertical_move_direction(relation, mover_mode)
    mover = figure if mover_mode == "figure_moves" else ground
    return (
        f"{sentence_case(figure)} and {ground} start at the same height. "
        f"{sentence_case(mover)} moves {direction}."
    )


def _vertical_motion_target(
    *,
    figure: str,
    ground: str,
    relation: str,
    target_mode: str,
) -> tuple[str, str, str, str]:
    if target_mode == "direct_relation":
        return (
            render_world_relation(figure, ground, relation),
            relation,
            figure,
            ground,
        )
    if target_mode == "role_swap":
        target_relation = OPPOSITE[relation]
        return (
            render_world_relation(ground, figure, target_relation),
            target_relation,
            ground,
            figure,
        )
    raise ValueError(f"Unsupported vertical motion target mode: {target_mode!r}.")


def _vertical_motion_context_target(
    *,
    figure: str,
    ground: str,
    relation: str,
    mover_mode: str,
    target_mode: str,
) -> tuple[str, str, str, str, str, str]:
    context = render_vertical_motion_event(
        figure=figure,
        ground=ground,
        relation=relation,
        mover_mode=mover_mode,
    )
    target, target_relation, target_figure, target_ground = _vertical_motion_target(
        figure=figure,
        ground=ground,
        relation=relation,
        target_mode=target_mode,
    )
    move_direction = vertical_move_direction(relation, mover_mode)
    return context, target, target_relation, target_figure, target_ground, move_direction


def generate_vertical_motion_update_item(
    index: int,
    rng: random.Random,
) -> SpatialItem:
    figure, ground = _sample_object_pair(rng)
    relation = _vertical_motion_relation(index)
    mover_mode = _vertical_motion_mover_mode(index)
    target_mode = _vertical_motion_target_mode(index)
    context, completion, target_relation, target_figure, target_ground, move_direction = (
        _vertical_motion_context_target(
            figure=figure,
            ground=ground,
            relation=relation,
            mover_mode=mover_mode,
            target_mode=target_mode,
        )
    )
    negative_relation = OPPOSITE[target_relation]

    return SpatialItem(
        item_id=f"vertical_motion_update:{index:06d}",
        template_family="vertical_motion_update",
        relation_type="vertical",
        context=context,
        completion=completion,
        hard_negative=render_world_relation(
            target_figure,
            target_ground,
            negative_relation,
        ),
        true_relation=target_relation,
        negative_relation=negative_relation,
        world_relation=relation,
        relative_relation="",
        figure=figure,
        ground=ground,
        observer="",
        observer_facing="",
        turn=f"{mover_mode}:{move_direction}",
    )


def generate_vertical_motion_update_pair(
    index: int,
    rng: random.Random,
) -> SpatialContextPair:
    figure, ground = _sample_object_pair(rng)
    relation1 = _vertical_motion_relation(index)
    relation2 = OPPOSITE[relation1]
    mover_mode = _vertical_motion_mover_mode(index)
    target_mode = _vertical_motion_target_mode(index)
    context1, target1, target1_relation, _, _, move1 = (
        _vertical_motion_context_target(
            figure=figure,
            ground=ground,
            relation=relation1,
            mover_mode=mover_mode,
            target_mode=target_mode,
        )
    )
    context2, target2, target2_relation, _, _, move2 = (
        _vertical_motion_context_target(
            figure=figure,
            ground=ground,
            relation=relation2,
            mover_mode=mover_mode,
            target_mode=target_mode,
        )
    )

    return SpatialContextPair(
        item_id=f"vertical_motion_update:{index:06d}",
        template_family="vertical_motion_update",
        relation_type="vertical",
        context1=context1,
        context2=context2,
        target1=target1,
        target2=target2,
        concept_a=target1_relation,
        concept_b=target2_relation,
        context_type="vertical_motion_update",
        context_diff=mover_mode,
        target_diff=target_mode,
        target1_relation=target1_relation,
        target2_relation=target2_relation,
        context1_world_relation=relation1,
        context2_world_relation=relation2,
        figure=figure,
        ground=ground,
        turn1=f"{mover_mode}:{move1}",
        turn2=f"{mover_mode}:{move2}",
    )


def _render_vertical_distractor_context(
    *,
    figure: str,
    ground: str,
    relation: str,
    mover_mode: str,
    distractor_figure: str,
    distractor_ground: str,
    distractor_relation: str,
    distractor_mover_mode: str,
    order_mode: str,
) -> str:
    target_event = render_vertical_motion_event(
        figure=figure,
        ground=ground,
        relation=relation,
        mover_mode=mover_mode,
    )
    distractor_event = render_vertical_motion_event(
        figure=distractor_figure,
        ground=distractor_ground,
        relation=distractor_relation,
        mover_mode=distractor_mover_mode,
    )
    if order_mode == "target_first":
        return f"{target_event} {distractor_event}"
    if order_mode == "distractor_first":
        return f"{distractor_event} {target_event}"
    raise ValueError(f"Unsupported vertical distractor order mode: {order_mode!r}.")


def _vertical_distractor_context_target(
    *,
    figure: str,
    ground: str,
    relation: str,
    mover_mode: str,
    target_mode: str,
    distractor_figure: str,
    distractor_ground: str,
    distractor_relation_mode: str,
    distractor_mover_mode: str,
    order_mode: str,
) -> tuple[str, str, str, str, str, str, str]:
    if distractor_relation_mode == "same_distractor":
        distractor_relation = relation
    elif distractor_relation_mode == "opposite_distractor":
        distractor_relation = OPPOSITE[relation]
    else:
        raise ValueError(
            f"Unsupported vertical distractor relation mode: {distractor_relation_mode!r}."
        )

    context = _render_vertical_distractor_context(
        figure=figure,
        ground=ground,
        relation=relation,
        mover_mode=mover_mode,
        distractor_figure=distractor_figure,
        distractor_ground=distractor_ground,
        distractor_relation=distractor_relation,
        distractor_mover_mode=distractor_mover_mode,
        order_mode=order_mode,
    )
    target, target_relation, target_figure, target_ground = _vertical_motion_target(
        figure=figure,
        ground=ground,
        relation=relation,
        target_mode=target_mode,
    )
    move_direction = vertical_move_direction(relation, mover_mode)
    distractor_direction = vertical_move_direction(
        distractor_relation,
        distractor_mover_mode,
    )
    return (
        context,
        target,
        target_relation,
        target_figure,
        target_ground,
        f"{mover_mode}:{move_direction}",
        f"{distractor_mover_mode}:{distractor_direction}",
    )


def generate_vertical_motion_distractor_item(
    index: int,
    rng: random.Random,
) -> SpatialItem:
    figure, ground, distractor_figure, distractor_ground = _sample_two_object_pairs(rng)
    relation = _vertical_motion_relation(index)
    mover_mode = _vertical_motion_mover_mode(index)
    target_mode = _vertical_motion_target_mode(index)
    distractor_relation_mode = _vertical_distractor_relation_mode(index)
    distractor_mover_mode = _vertical_distractor_mover_mode(index)
    order_mode = _vertical_distractor_order_mode(index)
    context, completion, target_relation, target_figure, target_ground, move, _ = (
        _vertical_distractor_context_target(
            figure=figure,
            ground=ground,
            relation=relation,
            mover_mode=mover_mode,
            target_mode=target_mode,
            distractor_figure=distractor_figure,
            distractor_ground=distractor_ground,
            distractor_relation_mode=distractor_relation_mode,
            distractor_mover_mode=distractor_mover_mode,
            order_mode=order_mode,
        )
    )
    negative_relation = OPPOSITE[target_relation]

    return SpatialItem(
        item_id=f"vertical_motion_distractor:{index:06d}",
        template_family="vertical_motion_distractor",
        relation_type="vertical",
        context=context,
        completion=completion,
        hard_negative=render_world_relation(
            target_figure,
            target_ground,
            negative_relation,
        ),
        true_relation=target_relation,
        negative_relation=negative_relation,
        world_relation=relation,
        relative_relation="",
        figure=figure,
        ground=ground,
        observer="",
        observer_facing="",
        turn=move,
    )


def generate_vertical_motion_distractor_pair(
    index: int,
    rng: random.Random,
) -> SpatialContextPair:
    figure, ground, distractor_figure, distractor_ground = _sample_two_object_pairs(rng)
    relation1 = _vertical_motion_relation(index)
    relation2 = OPPOSITE[relation1]
    mover_mode = _vertical_motion_mover_mode(index)
    target_mode = _vertical_motion_target_mode(index)
    distractor_relation_mode = _vertical_distractor_relation_mode(index)
    distractor_mover_mode = _vertical_distractor_mover_mode(index)
    order_mode = _vertical_distractor_order_mode(index)
    context1, target1, target1_relation, _, _, move1, distractor_move1 = (
        _vertical_distractor_context_target(
            figure=figure,
            ground=ground,
            relation=relation1,
            mover_mode=mover_mode,
            target_mode=target_mode,
            distractor_figure=distractor_figure,
            distractor_ground=distractor_ground,
            distractor_relation_mode=distractor_relation_mode,
            distractor_mover_mode=distractor_mover_mode,
            order_mode=order_mode,
        )
    )
    context2, target2, target2_relation, _, _, move2, distractor_move2 = (
        _vertical_distractor_context_target(
            figure=figure,
            ground=ground,
            relation=relation2,
            mover_mode=mover_mode,
            target_mode=target_mode,
            distractor_figure=distractor_figure,
            distractor_ground=distractor_ground,
            distractor_relation_mode=distractor_relation_mode,
            distractor_mover_mode=distractor_mover_mode,
            order_mode=order_mode,
        )
    )

    return SpatialContextPair(
        item_id=f"vertical_motion_distractor:{index:06d}",
        template_family="vertical_motion_distractor",
        relation_type="vertical",
        context1=context1,
        context2=context2,
        target1=target1,
        target2=target2,
        concept_a=target1_relation,
        concept_b=target2_relation,
        context_type="vertical_motion_distractor",
        context_diff=f"{mover_mode}:{distractor_relation_mode}:{order_mode}",
        target_diff=target_mode,
        target1_relation=target1_relation,
        target2_relation=target2_relation,
        context1_world_relation=relation1,
        context2_world_relation=relation2,
        figure=figure,
        ground=ground,
        turn1=f"{move1};{distractor_move1}",
        turn2=f"{move2};{distractor_move2}",
    )


def _distance_mode(index: int) -> str:
    return DISTANCE_TEMPLATE_MODES[index % len(DISTANCE_TEMPLATE_MODES)]


def _distance_relation(index: int) -> str:
    relation_index = (index // len(DISTANCE_TEMPLATE_MODES)) % len(DISTANCE_RELATIONS)
    return DISTANCE_RELATIONS[relation_index]


def _distance_steps(relation: str) -> int:
    if relation == "close":
        return 1
    if relation == "far":
        return 8
    raise ValueError(f"Unsupported distance relation {relation!r}.")


def _distance_context_target(
    *,
    figure: str,
    ground: str,
    relation: str,
    mode: str,
) -> tuple[str, str, str, str, str]:
    if mode == "metric_distance":
        steps = _distance_steps(relation)
        return (
            f"{ground} stands at the marker. "
            f"{sentence_case(figure)} is {steps} step"
            f"{'' if steps == 1 else 's'} from {ground}.",
            render_distance_relation(figure, ground, relation),
            relation,
            figure,
            ground,
        )
    if mode == "movement_toward_away":
        if relation == "close":
            context = (
                f"{render_distance_relation(figure, ground, 'far')} "
                f"{ground} walks toward {figure} until it is 1 step away."
            )
        elif relation == "far":
            context = (
                f"{render_distance_relation(figure, ground, 'close')} "
                f"{ground} walks away from {figure} until it is 8 steps away."
            )
        else:
            raise ValueError(f"Unsupported distance relation {relation!r}.")
        return (
            context,
            render_distance_relation(figure, ground, relation),
            relation,
            figure,
            ground,
        )
    if mode == "symmetric_role_swap":
        return (
            render_distance_relation(figure, ground, relation),
            render_distance_relation(ground, figure, relation),
            relation,
            ground,
            figure,
        )
    raise ValueError(f"Unsupported distance template mode: {mode!r}.")


def generate_close_far_distance_item(index: int, rng: random.Random) -> SpatialItem:
    ground = rng.choice(AGENTS)
    figure = noun_phrase(rng.choice(OBJECTS))
    mode = _distance_mode(index)
    relation = _distance_relation(index)
    context, completion, target_relation, target_figure, target_ground = (
        _distance_context_target(
            figure=figure,
            ground=ground,
            relation=relation,
            mode=mode,
        )
    )
    negative_relation = OPPOSITE[target_relation]

    return SpatialItem(
        item_id=f"close_far_distance:{index:06d}",
        template_family="close_far_distance",
        relation_type="distance",
        context=context,
        completion=completion,
        hard_negative=render_distance_relation(
            target_figure,
            target_ground,
            negative_relation,
        ),
        true_relation=target_relation,
        negative_relation=negative_relation,
        world_relation=relation,
        relative_relation="",
        figure=figure,
        ground=ground,
        observer=ground,
        observer_facing="",
    )


def generate_close_far_distance_pair(
    index: int,
    rng: random.Random,
) -> SpatialContextPair:
    ground = rng.choice(AGENTS)
    figure = noun_phrase(rng.choice(OBJECTS))
    mode = _distance_mode(index)
    relation1 = _distance_relation(index)
    relation2 = OPPOSITE[relation1]
    context1, target1, target1_relation, _, _ = _distance_context_target(
        figure=figure,
        ground=ground,
        relation=relation1,
        mode=mode,
    )
    context2, target2, target2_relation, _, _ = _distance_context_target(
        figure=figure,
        ground=ground,
        relation=relation2,
        mode=mode,
    )

    return SpatialContextPair(
        item_id=f"close_far_distance:{index:06d}",
        template_family="close_far_distance",
        relation_type="distance",
        context1=context1,
        context2=context2,
        target1=target1,
        target2=target2,
        concept_a=target1_relation,
        concept_b=target2_relation,
        context_type="close_far_distance",
        context_diff=mode,
        target_diff="distance_relation",
        target1_relation=target1_relation,
        target2_relation=target2_relation,
        context1_world_relation=relation1,
        context2_world_relation=relation2,
        figure=figure,
        ground=ground,
        observer=ground,
    )


def _render_same_facing_role_swap_context(
    *,
    figure: str,
    observer: str,
    facing: str,
    relation: str,
    rng: random.Random,
) -> str:
    context_templates = (
        "{observer} and {figure} both face {facing}. {relation_sentence}",
        "{observer} faces {facing}, and {figure} faces {facing}. {relation_sentence}",
    )
    return rng.choice(context_templates).format(
        observer=observer,
        figure=figure,
        facing=facing,
        relation_sentence=render_relative_relation(figure, observer, relation),
    )


def _egocentric_same_facing_state(
    relation: str,
    facing: Facing,
) -> State:
    world = relative_to_world_relation(relation, facing)
    return State(
        a_pos=(0, 0, 0),
        b_pos=position_for_world_relation(world),
        a_facing=facing,
        b_facing=facing,
    )


def generate_egocentric_same_facing_role_swap_item(
    index: int,
    rng: random.Random,
) -> SpatialItem:
    observer, figure = _sample_agent_pair(rng)
    facing: Facing = HORIZONTAL[(index // len(RELATIVE_RELATIONS)) % len(HORIZONTAL)]
    relation = RELATIVE_RELATIONS[index % len(RELATIVE_RELATIONS)]
    state = _egocentric_same_facing_state(relation, facing)
    context_relations = compute_relations(state, "a")
    target_relations = compute_relations(state, "b")
    negative = OPPOSITE[target_relations.relative]

    return SpatialItem(
        item_id=f"egocentric_same_facing_role_swap:{index:06d}",
        template_family="egocentric_same_facing_role_swap",
        relation_type="egocentric",
        context=_render_same_facing_role_swap_context(
            figure=figure,
            observer=observer,
            facing=facing,
            relation=context_relations.relative,
            rng=rng,
        ),
        completion=render_relative_relation(observer, figure, target_relations.relative),
        hard_negative=render_relative_relation(observer, figure, negative),
        true_relation=target_relations.relative,
        negative_relation=negative,
        world_relation=target_relations.world,
        relative_relation=target_relations.relative,
        figure=figure,
        ground=observer,
        observer=observer,
        observer_facing=facing,
        before_facing=facing,
        after_facing=facing,
    )


def generate_egocentric_same_facing_role_swap_pair(
    index: int,
    rng: random.Random,
) -> SpatialContextPair:
    observer, figure = _sample_agent_pair(rng)
    facing: Facing = HORIZONTAL[(index // len(RELATIVE_RELATIONS)) % len(HORIZONTAL)]
    relation1 = RELATIVE_RELATIONS[index % len(RELATIVE_RELATIONS)]
    relation2 = OPPOSITE[relation1]
    state1 = _egocentric_same_facing_state(relation1, facing)
    state2 = _egocentric_same_facing_state(relation2, facing)
    context_relations1 = compute_relations(state1, "a")
    context_relations2 = compute_relations(state2, "a")
    target_relations1 = compute_relations(state1, "b")
    target_relations2 = compute_relations(state2, "b")

    return SpatialContextPair(
        item_id=f"egocentric_same_facing_role_swap:{index:06d}",
        template_family="egocentric_same_facing_role_swap",
        relation_type="egocentric",
        context1=_render_same_facing_role_swap_context(
            figure=figure,
            observer=observer,
            facing=facing,
            relation=context_relations1.relative,
            rng=rng,
        ),
        context2=_render_same_facing_role_swap_context(
            figure=figure,
            observer=observer,
            facing=facing,
            relation=context_relations2.relative,
            rng=rng,
        ),
        target1=render_relative_relation(
            observer,
            figure,
            target_relations1.relative,
        ),
        target2=render_relative_relation(
            observer,
            figure,
            target_relations2.relative,
        ),
        concept_a=target_relations1.relative,
        concept_b=target_relations2.relative,
        context_type="egocentric_same_facing_role_swap",
        context_diff="relative_relation",
        target_diff="inverse_relative_relation",
        target1_relation=target_relations1.relative,
        target2_relation=target_relations2.relative,
        context1_world_relation=context_relations1.world,
        context2_world_relation=context_relations2.world,
        figure=figure,
        ground=observer,
        observer=observer,
        observer_facing=facing,
        before_facing1=facing,
        after_facing1=facing,
        before_facing2=facing,
        after_facing2=facing,
    )


def _render_turn_update_context(
    *,
    figure: str,
    observer: str,
    before_relative: str,
    turn: str,
) -> str:
    turn_phrase = "turns around" if turn == "around" else f"turns {turn}"
    return (
        f"{render_relative_relation(figure, observer, before_relative)} "
        f"{observer} {turn_phrase}."
    )


def generate_turn_update_basic_item(index: int, rng: random.Random) -> SpatialItem:
    observer = rng.choice(AGENTS)
    figure = noun_phrase(rng.choice(OBJECTS))
    before_facing: Facing = rng.choice(HORIZONTAL)
    before_relative = rng.choice(RELATIVE_RELATIONS)
    turn: Turn = rng.choice(("left", "right", "around"))
    after_facing = turn_facing(before_facing, turn)
    world = relative_to_world_relation(before_relative, before_facing)

    state = State(
        a_pos=(0, 0, 0),
        b_pos=position_for_world_relation(world),
        a_facing=after_facing,
        b_facing="north",
    )
    after_relations = compute_relations(state, "a")
    negative = OPPOSITE[after_relations.relative]

    context = _render_turn_update_context(
        figure=figure,
        observer=observer,
        before_relative=before_relative,
        turn=turn,
    )

    return SpatialItem(
        item_id=f"turn_update_basic:{index:06d}",
        template_family="turn_update_basic",
        relation_type="egocentric",
        context=context,
        completion=render_relative_relation(figure, observer, after_relations.relative),
        hard_negative=render_relative_relation(figure, observer, negative),
        true_relation=after_relations.relative,
        negative_relation=negative,
        world_relation=after_relations.world,
        relative_relation=after_relations.relative,
        figure=figure,
        ground=observer,
        observer=observer,
        observer_facing=after_facing,
        before_facing=before_facing,
        after_facing=after_facing,
        turn=turn,
    )


def generate_turn_update_basic_pair(index: int, rng: random.Random) -> SpatialContextPair:
    observer = rng.choice(AGENTS)
    figure = noun_phrase(rng.choice(OBJECTS))
    before_facing: Facing = rng.choice(HORIZONTAL)
    before_relative = rng.choice(RELATIVE_RELATIONS)
    turn1, turn2 = rng.sample(("left", "right", "around"), 2)
    after_facing1 = turn_facing(before_facing, turn1)
    after_facing2 = turn_facing(before_facing, turn2)
    world = relative_to_world_relation(before_relative, before_facing)

    state1 = State(
        a_pos=(0, 0, 0),
        b_pos=position_for_world_relation(world),
        a_facing=after_facing1,
        b_facing="north",
    )
    state2 = State(
        a_pos=(0, 0, 0),
        b_pos=position_for_world_relation(world),
        a_facing=after_facing2,
        b_facing="north",
    )
    after_relations1 = compute_relations(state1, "a")
    after_relations2 = compute_relations(state2, "a")

    return SpatialContextPair(
        item_id=f"turn_update_basic:{index:06d}",
        template_family="turn_update_basic",
        relation_type="egocentric",
        context1=_render_turn_update_context(
            figure=figure,
            observer=observer,
            before_relative=before_relative,
            turn=turn1,
        ),
        context2=_render_turn_update_context(
            figure=figure,
            observer=observer,
            before_relative=before_relative,
            turn=turn2,
        ),
        target1=render_relative_relation(figure, observer, after_relations1.relative),
        target2=render_relative_relation(figure, observer, after_relations2.relative),
        concept_a=after_relations1.relative,
        concept_b=after_relations2.relative,
        context_type="turn_update_basic",
        context_diff="turn",
        target_diff="updated_relative_relation",
        target1_relation=after_relations1.relative,
        target2_relation=after_relations2.relative,
        context1_world_relation=after_relations1.world,
        context2_world_relation=after_relations2.world,
        figure=figure,
        ground=observer,
        observer=observer,
        observer_facing=after_facing1,
        before_facing1=before_facing,
        after_facing1=after_facing1,
        turn1=turn1,
        before_facing2=before_facing,
        after_facing2=after_facing2,
        turn2=turn2,
    )


def generate_item(index: int, family: str, rng: random.Random) -> SpatialItem:
    if family == "static_egocentric":
        return generate_static_egocentric_item(index, rng)
    if family == "absolute_role_swap":
        return generate_absolute_role_swap_item(index, rng)
    if family == "cardinal_pass_by":
        return generate_cardinal_pass_by_item(index, rng)
    if family == "cardinal_rotation_invariance":
        return generate_cardinal_rotation_invariance_item(index, rng)
    if family == "cardinal_farther_motion":
        return generate_cardinal_farther_motion_item(index, rng)
    if family == "vertical_paraphrase_role_swap":
        return generate_vertical_paraphrase_role_swap_item(index, rng)
    if family == "vertical_motion_update":
        return generate_vertical_motion_update_item(index, rng)
    if family == "vertical_motion_distractor":
        return generate_vertical_motion_distractor_item(index, rng)
    if family == "egocentric_same_facing_role_swap":
        return generate_egocentric_same_facing_role_swap_item(index, rng)
    if family == "close_far_distance":
        return generate_close_far_distance_item(index, rng)
    if family == "reciprocal_inversion":
        return generate_reciprocal_inversion_item(index, rng)
    if family == "turn_update_basic":
        return generate_turn_update_basic_item(index, rng)
    raise ValueError(f"Unsupported spatial template family: {family!r}.")


def generate_context_pair(index: int, family: str, rng: random.Random) -> SpatialContextPair:
    if family == "static_egocentric":
        return generate_static_egocentric_pair(index, rng)
    if family == "absolute_role_swap":
        return generate_absolute_role_swap_pair(index, rng)
    if family == "cardinal_pass_by":
        return generate_cardinal_pass_by_pair(index, rng)
    if family == "cardinal_rotation_invariance":
        return generate_cardinal_rotation_invariance_pair(index, rng)
    if family == "cardinal_farther_motion":
        return generate_cardinal_farther_motion_pair(index, rng)
    if family == "vertical_paraphrase_role_swap":
        return generate_vertical_paraphrase_role_swap_pair(index, rng)
    if family == "vertical_motion_update":
        return generate_vertical_motion_update_pair(index, rng)
    if family == "vertical_motion_distractor":
        return generate_vertical_motion_distractor_pair(index, rng)
    if family == "egocentric_same_facing_role_swap":
        return generate_egocentric_same_facing_role_swap_pair(index, rng)
    if family == "close_far_distance":
        return generate_close_far_distance_pair(index, rng)
    if family == "reciprocal_inversion":
        return generate_reciprocal_inversion_pair(index, rng)
    if family == "turn_update_basic":
        return generate_turn_update_basic_pair(index, rng)
    raise ValueError(f"Unsupported spatial template family: {family!r}.")


def generate_items(
    count: int,
    *,
    seed: int = 0,
    families: Sequence[str] = DEFAULT_FAMILIES,
) -> list[SpatialItem]:
    """Generate a balanced list of synthetic spatial items."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    if not families:
        raise ValueError("families must contain at least one family.")

    rng = random.Random(seed)
    family_list = tuple(families)
    family_counts = dict.fromkeys(family_list, 0)
    items: list[SpatialItem] = []
    for index in range(count):
        family = family_list[index % len(family_list)]
        family_index = family_counts[family]
        family_counts[family] += 1
        items.append(generate_item(family_index, family, rng))
    return items


def generate_context_pairs(
    count: int,
    *,
    seed: int = 0,
    families: Sequence[str] = DEFAULT_FAMILIES,
) -> list[SpatialContextPair]:
    """Generate balanced EWoK-style context-rank pairs."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    if not families:
        raise ValueError("families must contain at least one family.")

    rng = random.Random(seed)
    family_list = tuple(families)
    family_counts = dict.fromkeys(family_list, 0)
    pairs: list[SpatialContextPair] = []
    for index in range(count):
        family = family_list[index % len(family_list)]
        family_index = family_counts[family]
        family_counts[family] += 1
        pairs.append(generate_context_pair(family_index, family, rng))
    return pairs


def _validate_family_counts(
    family_counts: Mapping[str, int],
) -> tuple[tuple[str, int], ...]:
    if not family_counts:
        raise ValueError("family_counts must contain at least one family.")

    normalized: list[tuple[str, int]] = []
    for family, count in family_counts.items():
        count = int(count)
        if count < 0:
            raise ValueError("family_counts values must be non-negative.")
        if count > 0:
            normalized.append((family, count))

    if not normalized:
        raise ValueError("family_counts must include at least one positive count.")
    return tuple(normalized)


def generate_items_from_family_counts(
    family_counts: Mapping[str, int],
    *,
    seed: int = 0,
) -> list[SpatialItem]:
    """Generate completion-rank items with explicit per-family counts."""

    remaining = dict(_validate_family_counts(family_counts))
    local_indices = dict.fromkeys(remaining, 0)
    rng = random.Random(seed)
    items: list[SpatialItem] = []

    while any(count > 0 for count in remaining.values()):
        for family in remaining:
            if remaining[family] <= 0:
                continue
            family_index = local_indices[family]
            items.append(generate_item(family_index, family, rng))
            local_indices[family] += 1
            remaining[family] -= 1
    return items


def generate_context_pairs_from_family_counts(
    family_counts: Mapping[str, int],
    *,
    seed: int = 0,
) -> list[SpatialContextPair]:
    """Generate context-rank pairs with explicit per-family counts."""

    remaining = dict(_validate_family_counts(family_counts))
    local_indices = dict.fromkeys(remaining, 0)
    rng = random.Random(seed)
    pairs: list[SpatialContextPair] = []

    while any(count > 0 for count in remaining.values()):
        for family in remaining:
            if remaining[family] <= 0:
                continue
            family_index = local_indices[family]
            pairs.append(generate_context_pair(family_index, family, rng))
            local_indices[family] += 1
            remaining[family] -= 1
    return pairs


def generate_examples(
    count: int,
    *,
    seed: int = 0,
    families: Sequence[str] = DEFAULT_FAMILIES,
    objective: RankObjective = DEFAULT_RANK_OBJECTIVE,
) -> list[SpatialExample]:
    """Generate spatial examples for the requested rank objective."""

    if objective == "completion_rank":
        return generate_items(count, seed=seed, families=families)
    if objective == "context_rank":
        return generate_context_pairs(count, seed=seed, families=families)
    raise ValueError("objective must be 'completion_rank' or 'context_rank'.")


def generate_examples_from_family_counts(
    family_counts: Mapping[str, int],
    *,
    seed: int = 0,
    objective: RankObjective = DEFAULT_RANK_OBJECTIVE,
) -> list[SpatialExample]:
    """Generate spatial examples with explicit per-family counts."""

    if objective == "completion_rank":
        return generate_items_from_family_counts(family_counts, seed=seed)
    if objective == "context_rank":
        return generate_context_pairs_from_family_counts(family_counts, seed=seed)
    raise ValueError("objective must be 'completion_rank' or 'context_rank'.")


def write_jsonl(items: Iterable[SpatialExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_row(), ensure_ascii=True) + "\n")


def fieldnames_for_items(items: Sequence[SpatialExample]) -> tuple[str, ...]:
    if items and isinstance(items[0], SpatialContextPair):
        return CONTEXT_PAIR_FIELDNAMES
    return FIELDNAMES


def write_csv(items: Iterable[SpatialExample], path: str | Path) -> None:
    materialized_items = list(items)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames_for_items(materialized_items)))
        writer.writeheader()
        for item in materialized_items:
            writer.writerow(item.to_row())


def infer_format(path: str | Path, output_format: str | None = None) -> str:
    if output_format:
        output_format = output_format.lower()
        if output_format not in {"jsonl", "csv"}:
            raise ValueError("output_format must be 'jsonl' or 'csv'.")
        return output_format
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    return "jsonl"


def write_items(
    items: Iterable[SpatialExample],
    path: str | Path,
    *,
    output_format: str | None = None,
) -> None:
    fmt = infer_format(path, output_format)
    if fmt == "csv":
        write_csv(items, path)
    else:
        write_jsonl(items, path)


__all__ = [
    "AGENTS",
    "CARDINAL_CONTINUE_MOTION_VERBS",
    "CARDINAL_INVARIANCE_TARGET_MODES",
    "CARDINAL_PASS_OUTCOME_MODES",
    "CARDINAL_PASS_TARGET_MODES",
    "CARDINAL_PASS_VERBS",
    "CARDINAL_ROTATION_TURNS",
    "CONTEXT_PAIR_FIELDNAMES",
    "CONTEXT_RANK_OBJECTIVE",
    "DEFAULT_RANK_OBJECTIVE",
    "DEFAULT_FAMILIES",
    "DISTANCE_RELATIONS",
    "DISTANCE_TEMPLATE_MODES",
    "FIELDNAMES",
    "HORIZONTAL_WORLD_RELATIONS",
    "OBJECTS",
    "RELATIVE_RELATIONS",
    "RankObjective",
    "SpatialContextPair",
    "SpatialExample",
    "VERTICAL_PARAPHRASE_PAIRS",
    "VERTICAL_DISTRACTOR_ORDER_MODES",
    "VERTICAL_DISTRACTOR_RELATION_MODES",
    "VERTICAL_MOTION_MOVER_MODES",
    "VERTICAL_MOTION_TARGET_MODES",
    "VERTICAL_RELATIONS",
    "VERTICAL_TEMPLATE_MODES",
    "WORLD_RELATIONS",
    "SpatialItem",
    "cardinal_pass_start_and_move",
    "fieldnames_for_items",
    "generate_absolute_role_swap_pair",
    "generate_absolute_role_swap_item",
    "generate_cardinal_pass_by_pair",
    "generate_cardinal_pass_by_item",
    "generate_cardinal_farther_motion_pair",
    "generate_cardinal_farther_motion_item",
    "generate_cardinal_rotation_invariance_pair",
    "generate_cardinal_rotation_invariance_item",
    "generate_close_far_distance_pair",
    "generate_close_far_distance_item",
    "generate_context_pair",
    "generate_context_pairs",
    "generate_context_pairs_from_family_counts",
    "generate_egocentric_same_facing_role_swap_pair",
    "generate_egocentric_same_facing_role_swap_item",
    "generate_examples",
    "generate_examples_from_family_counts",
    "generate_item",
    "generate_items",
    "generate_items_from_family_counts",
    "generate_reciprocal_inversion_pair",
    "generate_reciprocal_inversion_item",
    "generate_static_egocentric_pair",
    "generate_static_egocentric_item",
    "generate_turn_update_basic_pair",
    "generate_turn_update_basic_item",
    "generate_vertical_motion_distractor_pair",
    "generate_vertical_motion_distractor_item",
    "generate_vertical_motion_update_pair",
    "generate_vertical_motion_update_item",
    "generate_vertical_paraphrase_role_swap_pair",
    "generate_vertical_paraphrase_role_swap_item",
    "infer_format",
    "join_context_completion",
    "noun_phrase",
    "render_cardinal_pass_context",
    "render_cardinal_farther_motion_context",
    "render_cardinal_rotation_context",
    "render_distance_relation",
    "render_relative_relation",
    "render_vertical_paraphrase",
    "render_vertical_motion_event",
    "render_world_relation",
    "sentence_case",
    "vertical_move_direction",
    "write_csv",
    "write_items",
    "write_jsonl",
]
