from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from babysteps.spatial_synth.generate import (
    CARDINAL_CONTINUE_MOTION_VERBS,
    CARDINAL_INVARIANCE_TARGET_MODES,
    CARDINAL_PASS_OUTCOME_MODES,
    CARDINAL_PASS_TARGET_MODES,
    CARDINAL_PASS_VERBS,
    CARDINAL_ROTATION_TURNS,
    DEFAULT_FAMILIES,
    DISTANCE_TEMPLATE_MODES,
    HORIZONTAL_WORLD_RELATIONS,
    VERTICAL_DISTRACTOR_ORDER_MODES,
    VERTICAL_DISTRACTOR_RELATION_MODES,
    VERTICAL_MOTION_MOVER_MODES,
    VERTICAL_MOTION_TARGET_MODES,
    VERTICAL_PARAPHRASE_PAIRS,
    VERTICAL_TEMPLATE_MODES,
    generate_context_pairs,
    generate_examples_from_family_counts,
    generate_examples,
    generate_items,
    render_relative_relation,
    render_world_relation,
    write_items,
)
from babysteps.spatial_synth.world import OPPOSITE
from babysteps.evaluation.ewok_context_sensitivity import read_ewok_items


def test_generate_items_balances_families_and_writes_hard_negatives() -> None:
    items = generate_items(len(DEFAULT_FAMILIES) * 2, seed=7)

    assert [item.template_family for item in items] == list(DEFAULT_FAMILIES) * 2
    for item in items:
        row = item.to_row()
        assert row["rank_objective"] == "completion_rank"
        assert row["context"]
        assert row["completion"]
        assert row["hard_negative"]
        assert row["completion"] != row["hard_negative"]
        assert row["text"] == f"{row['context']} {row['completion']}"
        assert row["negative_text"] == f"{row['context']} {row['hard_negative']}"
        assert item.negative_relation == OPPOSITE[item.true_relation]


def test_generate_examples_can_emit_context_rank_pairs() -> None:
    examples = generate_examples(
        len(DEFAULT_FAMILIES),
        seed=7,
        objective="context_rank",
    )

    assert [item.template_family for item in examples] == list(DEFAULT_FAMILIES)
    for item in examples:
        row = item.to_row()
        assert row["rank_objective"] == "context_rank"
        assert row["Domain"] == "synthetic-spatial-relations"
        assert row["Context1"]
        assert row["Context2"]
        assert row["Target1"]
        assert row["Target2"]
        assert row["Target1"] != row["Target2"]
        assert row["text11"] == f"{row['Context1']} {row['Target1']}"
        assert row["text21"] == f"{row['Context2']} {row['Target1']}"
        assert row["text22"] == f"{row['Context2']} {row['Target2']}"
        assert row["text12"] == f"{row['Context1']} {row['Target2']}"


def test_generate_examples_from_family_counts_uses_explicit_counts() -> None:
    examples = generate_examples_from_family_counts(
        {
            "static_egocentric": 2,
            "close_far_distance": 4,
        },
        seed=7,
        objective="context_rank",
    )

    assert len(examples) == 6
    assert Counter(item.template_family for item in examples) == {
        "static_egocentric": 2,
        "close_far_distance": 4,
    }
    assert all(item.to_row()["rank_objective"] == "context_rank" for item in examples)


def test_static_context_rank_pair_flips_world_and_relative_relations() -> None:
    pair = generate_context_pairs(1, seed=3, families=("static_egocentric",))[0]

    assert pair.template_family == "static_egocentric"
    assert pair.context2_world_relation == OPPOSITE[pair.context1_world_relation]
    assert pair.target2_relation == OPPOSITE[pair.target1_relation]
    assert pair.observer_facing in pair.context1
    assert pair.observer_facing in pair.context2


def test_absolute_role_swap_pair_inverts_world_relation() -> None:
    pair = generate_context_pairs(1, seed=5, families=("absolute_role_swap",))[0]

    assert pair.template_family == "absolute_role_swap"
    assert pair.relation_type == "world"
    assert pair.context2_world_relation == OPPOSITE[pair.context1_world_relation]
    assert pair.target1_relation == OPPOSITE[pair.context1_world_relation]
    assert pair.target2_relation == OPPOSITE[pair.context2_world_relation]
    assert pair.target2_relation == OPPOSITE[pair.target1_relation]


def test_legacy_reciprocal_inversion_family_still_works() -> None:
    pair = generate_context_pairs(1, seed=5, families=("reciprocal_inversion",))[0]

    assert pair.template_family == "reciprocal_inversion"
    assert pair.item_id.startswith("reciprocal_inversion:")
    assert pair.target1_relation == OPPOSITE[pair.context1_world_relation]


def test_cardinal_pass_by_pair_covers_crossing_and_stop_modes() -> None:
    pairs = generate_context_pairs(
        len(CARDINAL_PASS_TARGET_MODES)
        * len(CARDINAL_PASS_OUTCOME_MODES)
        * len(HORIZONTAL_WORLD_RELATIONS)
        * len(CARDINAL_PASS_VERBS),
        seed=24,
        families=("cardinal_pass_by",),
    )

    assert {pair.template_family for pair in pairs} == {"cardinal_pass_by"}
    assert {pair.context_diff for pair in pairs} == set(CARDINAL_PASS_OUTCOME_MODES)
    assert {pair.target_diff for pair in pairs} == set(CARDINAL_PASS_TARGET_MODES)
    assert {pair.context1_world_relation for pair in pairs} == set(HORIZONTAL_WORLD_RELATIONS)
    assert all(pair.context2_world_relation == OPPOSITE[pair.context1_world_relation] for pair in pairs)
    assert all(pair.target2_relation == OPPOSITE[pair.target1_relation] for pair in pairs)

    inverse_pair_counts = Counter(
        tuple(sorted((pair.target1_relation, pair.target2_relation)))
        for pair in pairs
    )
    assert inverse_pair_counts[("north", "south")] == 8 * len(CARDINAL_PASS_VERBS)
    assert inverse_pair_counts[("east", "west")] == 8 * len(CARDINAL_PASS_VERBS)

    contexts = " ".join(f"{pair.context1} {pair.context2}" for pair in pairs)
    assert "stops before reaching it" in contexts
    assert "past" in contexts
    assert all(f"{verb} north" in contexts for verb in CARDINAL_PASS_VERBS)
    assert all(f"{verb} south" in contexts for verb in CARDINAL_PASS_VERBS)
    assert all(f"{verb} east" in contexts for verb in CARDINAL_PASS_VERBS)
    assert all(f"{verb} west" in contexts for verb in CARDINAL_PASS_VERBS)


def test_cardinal_pass_by_completion_rows_use_cardinal_targets() -> None:
    items = generate_items(
        len(CARDINAL_PASS_TARGET_MODES)
        * len(CARDINAL_PASS_OUTCOME_MODES)
        * len(HORIZONTAL_WORLD_RELATIONS)
        * len(CARDINAL_PASS_VERBS),
        seed=25,
        families=("cardinal_pass_by",),
    )

    assert {item.template_family for item in items} == {"cardinal_pass_by"}
    assert {item.world_relation for item in items} == set(HORIZONTAL_WORLD_RELATIONS)
    assert {item.true_relation for item in items} == set(HORIZONTAL_WORLD_RELATIONS)
    for item in items:
        assert item.relation_type == "world"
        assert item.negative_relation == OPPOSITE[item.true_relation]
        assert any(f" {verb} " in item.context for verb in CARDINAL_PASS_VERBS)
        assert item.true_relation in item.completion
        assert item.negative_relation in item.hard_negative


def test_cardinal_rotation_invariance_pair_keeps_world_relation_after_turns() -> None:
    pairs = generate_context_pairs(
        len(CARDINAL_INVARIANCE_TARGET_MODES)
        * len(CARDINAL_ROTATION_TURNS)
        * len(HORIZONTAL_WORLD_RELATIONS),
        seed=26,
        families=("cardinal_rotation_invariance",),
    )

    assert {pair.template_family for pair in pairs} == {"cardinal_rotation_invariance"}
    assert {pair.target_diff for pair in pairs} == set(CARDINAL_INVARIANCE_TARGET_MODES)
    assert {pair.turn1 for pair in pairs} == set(CARDINAL_ROTATION_TURNS)
    assert {pair.context1_world_relation for pair in pairs} == set(HORIZONTAL_WORLD_RELATIONS)
    assert all(pair.turn1 == pair.turn2 for pair in pairs)
    assert all(pair.before_facing1 == pair.before_facing2 for pair in pairs)
    assert all(pair.after_facing1 == pair.after_facing2 for pair in pairs)
    assert all(pair.context2_world_relation == OPPOSITE[pair.context1_world_relation] for pair in pairs)
    assert all(pair.target2_relation == OPPOSITE[pair.target1_relation] for pair in pairs)

    contexts = " ".join(f"{pair.context1} {pair.context2}" for pair in pairs)
    assert "turns left" in contexts
    assert "turns right" in contexts
    assert "turns around" in contexts


def test_cardinal_rotation_invariance_completion_rows_use_cardinal_targets() -> None:
    items = generate_items(
        len(CARDINAL_INVARIANCE_TARGET_MODES)
        * len(CARDINAL_ROTATION_TURNS)
        * len(HORIZONTAL_WORLD_RELATIONS),
        seed=27,
        families=("cardinal_rotation_invariance",),
    )

    assert {item.template_family for item in items} == {"cardinal_rotation_invariance"}
    assert {item.world_relation for item in items} == set(HORIZONTAL_WORLD_RELATIONS)
    assert {item.true_relation for item in items} == set(HORIZONTAL_WORLD_RELATIONS)
    for item in items:
        assert item.relation_type == "world"
        assert item.negative_relation == OPPOSITE[item.true_relation]
        assert "turns" in item.context
        assert item.true_relation in item.completion
        assert item.negative_relation in item.hard_negative


def test_cardinal_farther_motion_pair_keeps_relation_on_same_side() -> None:
    pairs = generate_context_pairs(
        len(CARDINAL_INVARIANCE_TARGET_MODES)
        * len(HORIZONTAL_WORLD_RELATIONS)
        * len(CARDINAL_CONTINUE_MOTION_VERBS),
        seed=28,
        families=("cardinal_farther_motion",),
    )

    assert {pair.template_family for pair in pairs} == {"cardinal_farther_motion"}
    assert {pair.target_diff for pair in pairs} == set(CARDINAL_INVARIANCE_TARGET_MODES)
    assert {pair.context1_world_relation for pair in pairs} == set(HORIZONTAL_WORLD_RELATIONS)
    assert all(pair.context2_world_relation == OPPOSITE[pair.context1_world_relation] for pair in pairs)
    assert all(pair.target2_relation == OPPOSITE[pair.target1_relation] for pair in pairs)
    assert all(pair.turn1.startswith("continue:") for pair in pairs)
    assert all(pair.turn2.startswith("continue:") for pair in pairs)

    contexts = " ".join(f"{pair.context1} {pair.context2}" for pair in pairs)
    assert "farther from" in contexts
    assert all(
        f"continues {verb}" in contexts
        for verb in CARDINAL_CONTINUE_MOTION_VERBS
    )


def test_cardinal_farther_motion_completion_rows_use_cardinal_targets() -> None:
    items = generate_items(
        len(CARDINAL_INVARIANCE_TARGET_MODES)
        * len(HORIZONTAL_WORLD_RELATIONS)
        * len(CARDINAL_CONTINUE_MOTION_VERBS),
        seed=29,
        families=("cardinal_farther_motion",),
    )

    assert {item.template_family for item in items} == {"cardinal_farther_motion"}
    assert {item.world_relation for item in items} == set(HORIZONTAL_WORLD_RELATIONS)
    assert {item.true_relation for item in items} == set(HORIZONTAL_WORLD_RELATIONS)
    for item in items:
        assert item.relation_type == "world"
        assert item.negative_relation == OPPOSITE[item.true_relation]
        assert "continues" in item.context
        assert "farther from" in item.context
        assert item.true_relation in item.completion
        assert item.negative_relation in item.hard_negative


def test_vertical_paraphrase_role_swap_pair_covers_modes_and_phrases() -> None:
    pairs = generate_context_pairs(
        len(VERTICAL_TEMPLATE_MODES) * 2 * len(VERTICAL_PARAPHRASE_PAIRS),
        seed=8,
        families=("vertical_paraphrase_role_swap",),
    )

    assert {pair.template_family for pair in pairs} == {
        "vertical_paraphrase_role_swap"
    }
    assert {pair.context_diff for pair in pairs} == set(VERTICAL_TEMPLATE_MODES)
    assert {pair.target1_relation for pair in pairs} == {"above", "below"}
    assert {pair.target2_relation for pair in pairs} == {"above", "below"}
    assert all(pair.target1_relation == OPPOSITE[pair.target2_relation] for pair in pairs)
    assert all(pair.concept_a in {"above", "below"} for pair in pairs)
    assert all(pair.concept_b in {"above", "below"} for pair in pairs)

    contexts = " ".join(
        f"{pair.context1} {pair.context2}"
        for pair in pairs
    )
    assert "over" in contexts
    assert "under" in contexts
    assert "on top of" in contexts
    assert "beneath" in contexts
    assert "higher than" in contexts
    assert "lower than" in contexts


def test_vertical_paraphrase_role_swap_completion_rows_use_literal_targets() -> None:
    items = generate_items(
        len(VERTICAL_TEMPLATE_MODES) * 2,
        seed=9,
        families=("vertical_paraphrase_role_swap",),
    )

    assert {item.template_family for item in items} == {"vertical_paraphrase_role_swap"}
    assert {item.true_relation for item in items} == {"above", "below"}
    for item in items:
        assert item.relation_type == "vertical"
        assert item.negative_relation == OPPOSITE[item.true_relation]
        assert item.true_relation in item.completion
        assert item.negative_relation in item.hard_negative


def test_vertical_motion_update_pair_covers_motion_and_role_swap_modes() -> None:
    pairs = generate_context_pairs(
        len(VERTICAL_MOTION_TARGET_MODES)
        * len(VERTICAL_MOTION_MOVER_MODES)
        * 2,
        seed=21,
        families=("vertical_motion_update",),
    )

    assert {pair.template_family for pair in pairs} == {"vertical_motion_update"}
    assert {pair.context_diff for pair in pairs} == set(VERTICAL_MOTION_MOVER_MODES)
    assert {pair.target_diff for pair in pairs} == set(VERTICAL_MOTION_TARGET_MODES)
    assert {pair.target1_relation for pair in pairs} == {"above", "below"}
    assert {pair.target2_relation for pair in pairs} == {"above", "below"}
    assert all(pair.target1_relation == OPPOSITE[pair.target2_relation] for pair in pairs)
    assert all("start at the same height" in pair.context1 for pair in pairs)
    assert all("start at the same height" in pair.context2 for pair in pairs)

    contexts = " ".join(f"{pair.context1} {pair.context2}" for pair in pairs)
    assert "moves up" in contexts
    assert "moves down" in contexts


def test_vertical_motion_distractor_pair_balances_distractors_and_order() -> None:
    pairs = generate_context_pairs(
        len(VERTICAL_MOTION_TARGET_MODES)
        * len(VERTICAL_MOTION_MOVER_MODES)
        * 2
        * len(VERTICAL_DISTRACTOR_RELATION_MODES)
        * len(VERTICAL_DISTRACTOR_ORDER_MODES),
        seed=22,
        families=("vertical_motion_distractor",),
    )

    assert {pair.template_family for pair in pairs} == {"vertical_motion_distractor"}
    assert {pair.target_diff for pair in pairs} == set(VERTICAL_MOTION_TARGET_MODES)
    assert {pair.target1_relation for pair in pairs} == {"above", "below"}
    assert {pair.target2_relation for pair in pairs} == {"above", "below"}
    assert all(pair.target1_relation == OPPOSITE[pair.target2_relation] for pair in pairs)

    context_diffs = " ".join(pair.context_diff for pair in pairs)
    assert all(mode in context_diffs for mode in VERTICAL_MOTION_MOVER_MODES)
    assert all(mode in context_diffs for mode in VERTICAL_DISTRACTOR_RELATION_MODES)
    assert all(mode in context_diffs for mode in VERTICAL_DISTRACTOR_ORDER_MODES)
    assert all(pair.context1.count("start at the same height") == 2 for pair in pairs)
    assert all(pair.context2.count("start at the same height") == 2 for pair in pairs)
    assert all(pair.figure in pair.target1 or pair.ground in pair.target1 for pair in pairs)
    assert all(pair.figure in pair.target2 or pair.ground in pair.target2 for pair in pairs)


def test_vertical_motion_completion_rows_use_vertical_targets() -> None:
    items = generate_items(
        len(VERTICAL_MOTION_TARGET_MODES)
        * len(VERTICAL_MOTION_MOVER_MODES)
        * 2,
        seed=23,
        families=("vertical_motion_update",),
    )

    assert {item.template_family for item in items} == {"vertical_motion_update"}
    assert {item.true_relation for item in items} == {"above", "below"}
    for item in items:
        assert item.relation_type == "vertical"
        assert item.negative_relation == OPPOSITE[item.true_relation]
        assert "start at the same height" in item.context
        assert "moves " in item.context
        assert item.true_relation in item.completion
        assert item.negative_relation in item.hard_negative


def test_close_far_distance_pair_covers_minimal_subtypes() -> None:
    pairs = generate_context_pairs(
        len(DISTANCE_TEMPLATE_MODES) * 2,
        seed=10,
        families=("close_far_distance",),
    )

    assert {pair.template_family for pair in pairs} == {"close_far_distance"}
    assert {pair.context_diff for pair in pairs} == set(DISTANCE_TEMPLATE_MODES)
    assert {pair.target1_relation for pair in pairs} == {"close", "far"}
    assert {pair.target2_relation for pair in pairs} == {"close", "far"}
    assert all(pair.target1_relation == OPPOSITE[pair.target2_relation] for pair in pairs)

    contexts = " ".join(f"{pair.context1} {pair.context2}" for pair in pairs)
    assert "1 step" in contexts
    assert "8 steps" in contexts
    assert "walks toward" in contexts
    assert "walks away" in contexts


def test_close_far_symmetric_role_swap_preserves_distance_relation() -> None:
    # The third local index is the first symmetric-role-swap item.
    pair = generate_context_pairs(
        3,
        seed=11,
        families=("close_far_distance",),
    )[2]

    assert pair.context_diff == "symmetric_role_swap"
    assert pair.target1_relation in pair.context1
    assert pair.target1_relation in pair.target1
    assert pair.target2_relation in pair.context2
    assert pair.target2_relation in pair.target2


def test_close_far_distance_completion_rows_use_distance_targets() -> None:
    items = generate_items(
        len(DISTANCE_TEMPLATE_MODES) * 2,
        seed=12,
        families=("close_far_distance",),
    )

    assert {item.template_family for item in items} == {"close_far_distance"}
    assert {item.true_relation for item in items} == {"close", "far"}
    for item in items:
        assert item.relation_type == "distance"
        assert item.negative_relation == OPPOSITE[item.true_relation]
        assert item.true_relation in item.completion
        assert item.negative_relation in item.hard_negative


def test_egocentric_same_facing_role_swap_pair_inverts_relative_relation() -> None:
    pair = generate_context_pairs(
        1,
        seed=6,
        families=("egocentric_same_facing_role_swap",),
    )[0]

    assert pair.template_family == "egocentric_same_facing_role_swap"
    assert pair.relation_type == "egocentric"
    assert pair.context_type == "egocentric_same_facing_role_swap"
    assert pair.before_facing1 == pair.after_facing1
    assert pair.before_facing2 == pair.after_facing2
    assert pair.before_facing1 == pair.before_facing2
    assert pair.target1_relation == OPPOSITE[pair.target2_relation]
    assert pair.target1_relation in pair.target1
    assert pair.target2_relation in pair.target2
    assert pair.before_facing1 in pair.context1
    assert pair.before_facing1 in pair.context2


def test_egocentric_same_facing_role_swap_does_not_oversample_left_right() -> None:
    pairs = generate_context_pairs(
        16,
        seed=6,
        families=("egocentric_same_facing_role_swap",),
    )

    inverse_pair_counts = Counter(
        tuple(sorted((pair.target1_relation, pair.target2_relation)))
        for pair in pairs
    )
    assert inverse_pair_counts[("behind", "front")] == 8
    assert inverse_pair_counts[("left", "right")] == 8


def test_turn_context_rank_pair_uses_context_turn_as_the_contrast() -> None:
    pair = generate_context_pairs(1, seed=4, families=("turn_update_basic",))[0]

    assert pair.template_family == "turn_update_basic"
    assert pair.turn1 != pair.turn2
    assert pair.before_facing1 == pair.before_facing2
    assert pair.after_facing1 != pair.after_facing2
    assert pair.target1_relation != pair.target2_relation


def test_static_egocentric_rows_ground_facing_to_relative_relation() -> None:
    item = generate_items(1, seed=3, families=("static_egocentric",))[0]

    assert item.template_family == "static_egocentric"
    assert item.relation_type == "egocentric"
    assert item.observer in item.context
    assert item.observer_facing in item.context
    assert item.completion == render_relative_relation(
        item.figure,
        item.observer,
        item.true_relation,
    )


def test_world_renderer_handles_cardinal_and_vertical_phrases() -> None:
    assert render_world_relation("the cup", "the box", "east") == "The cup is east of the box."
    assert render_world_relation("the cup", "the box", "above") == "The cup is above the box."


def test_turn_update_rows_change_facing_and_keep_world_offset() -> None:
    item = generate_items(1, seed=4, families=("turn_update_basic",))[0]

    assert item.template_family == "turn_update_basic"
    assert item.before_facing
    assert item.after_facing
    assert item.turn in {"left", "right", "around"}
    assert item.before_facing != item.after_facing
    assert item.world_relation in {"north", "east", "south", "west"}


def test_write_items_jsonl_and_csv(tmp_path: Path) -> None:
    items = generate_items(len(DEFAULT_FAMILIES), seed=11)
    jsonl_path = tmp_path / "spatial.jsonl"
    csv_path = tmp_path / "spatial.csv"

    write_items(items, jsonl_path)
    write_items(items, csv_path)

    json_rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(json_rows) == len(DEFAULT_FAMILIES)
    assert json_rows[0]["item_id"].startswith("static_egocentric:")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(DEFAULT_FAMILIES)
    assert csv_rows[1]["item_id"].startswith(f"{DEFAULT_FAMILIES[1]}:")


def test_context_rank_jsonl_is_ewok_readable(tmp_path: Path) -> None:
    pairs = generate_context_pairs(3, seed=13)
    jsonl_path = tmp_path / "spatial_context_rank.jsonl"

    write_items(pairs, jsonl_path)

    ewok_items = read_ewok_items(jsonl_path)
    assert len(ewok_items) == 3
    assert ewok_items[0].domain == "synthetic-spatial-relations"
    assert ewok_items[0].context1 == pairs[0].context1
    assert ewok_items[0].target2 == pairs[0].target2
