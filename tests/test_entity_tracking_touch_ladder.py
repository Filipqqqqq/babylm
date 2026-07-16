from __future__ import annotations

from collections import Counter, defaultdict

from babysteps.evaluation.entity_tracking_touch_ladder import (
    EXPECTED_COUNTS,
    EXTENDED_LADDER_CONDITIONS,
    LADDER_CONDITIONS,
    build_scored_records,
    generate_touch_ladder_items,
    read_touch_ladder_jsonl,
    summarize_scored_records,
    validate_ladders,
    write_touch_ladder_jsonl,
)


def test_touch_ladder_is_deterministic_balanced_and_valid() -> None:
    first = generate_touch_ladder_items(worlds=8, seed=42)
    second = generate_touch_ladder_items(worlds=8, seed=42)

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == 8 * len(LADDER_CONDITIONS)
    assert Counter(item.condition for item in first) == Counter(
        {condition: 8 for condition in LADDER_CONDITIONS}
    )
    validate_ladders(first, expected_worlds=8)


def test_each_world_holds_state_answer_options_and_last_touch_fixed() -> None:
    items = generate_touch_ladder_items(worlds=5, seed=7)
    worlds = defaultdict(list)
    for item in items:
        worlds[item.base_id].append(item)

    for world in worlds.values():
        reference = world[0]
        assert {item.condition for item in world} == set(LADDER_CONDITIONS)
        for item in world:
            assert item.initial_state == reference.initial_state
            assert item.target_contents == reference.target_contents
            assert item.options == reference.options
            assert item.target_index == reference.target_index
            assert item.last_touch == reference.last_touch
            assert item.last_touch_distance == 0
            assert "nothing." not in item.options
            assert len({option.count("the ") for option in item.options}) == 1
            assert (
                item.total_operation_count,
                item.query_touch_count,
            ) == EXPECTED_COUNTS[item.condition]


def test_touch_ladder_jsonl_round_trip(tmp_path) -> None:
    items = generate_touch_ladder_items(worlds=3, seed=13)
    path = write_touch_ladder_jsonl(tmp_path / "ladder.jsonl", items)

    restored = read_touch_ladder_jsonl(path)

    assert [item.to_dict() for item in restored] == [item.to_dict() for item in items]


def test_extended_ladder_adds_matched_transient_query_touches() -> None:
    items = generate_touch_ladder_items(worlds=5, seed=31, include_transient=True)
    worlds = defaultdict(dict)
    for item in items:
        worlds[item.base_id][item.condition] = item

    assert len(items) == 5 * len(EXTENDED_LADDER_CONDITIONS)
    validate_ladders(items, expected_worlds=5)
    for world in worlds.values():
        assert set(world) == set(EXTENDED_LADDER_CONDITIONS)
        for transient, control in (("t3", "c3"), ("t5", "c5")):
            transient_item = world[transient]
            control_item = world[control]
            assert transient_item.target_contents == control_item.target_contents
            assert transient_item.options == control_item.options
            assert [op.objects for op in transient_item.operations] == [
                op.objects for op in control_item.operations
            ]
            assert transient_item.operations[-1] == control_item.operations[-1]
            assert transient_item.query_touch_count == transient_item.total_operation_count
            assert control_item.query_touch_count == 1
            transient_objects = {
                obj
                for operation in transient_item.operations[:-1]
                for obj in operation.objects
            }
            assert transient_objects.isdisjoint(transient_item.target_contents)


def test_summary_computes_matched_query_touch_contrasts() -> None:
    items = generate_touch_ladder_items(worlds=2, seed=21)
    target_scores = {"q1": 1.0, "q3": 3.0, "q5": 5.0, "c3": 2.0, "c5": 4.0}
    option_scores = []
    token_counts = []
    for item in items:
        scores = [0.0] * len(item.options)
        scores[item.target_index] = target_scores[item.condition]
        option_scores.append(scores)
        token_counts.append([1] * len(item.options))
    records = build_scored_records(
        items=items,
        option_scores=option_scores,
        scored_token_counts=token_counts,
    )

    summary = summarize_scored_records(records, bootstrap_seed=5)

    assert summary["world_count"] == 2
    assert summary["contrasts"]["q3_minus_c3"]["mean_margin_difference"] == 1.0
    assert summary["contrasts"]["q5_minus_c5"]["mean_margin_difference"] == 1.0
    assert all(
        summary["conditions"][condition]["accuracy"]["estimate"] == 1.0
        for condition in LADDER_CONDITIONS
    )


def test_extended_summary_separates_box_focus_and_answer_repetition() -> None:
    items = generate_touch_ladder_items(worlds=2, seed=22, include_transient=True)
    target_scores = {
        "q1": 1.0,
        "q3": 4.0,
        "q5": 7.0,
        "t3": 3.0,
        "t5": 5.0,
        "c3": 2.0,
        "c5": 3.0,
    }
    option_scores = []
    token_counts = []
    for item in items:
        scores = [0.0] * len(item.options)
        scores[item.target_index] = target_scores[item.condition]
        option_scores.append(scores)
        token_counts.append([1] * len(item.options))
    records = build_scored_records(
        items=items,
        option_scores=option_scores,
        scored_token_counts=token_counts,
    )

    summary = summarize_scored_records(records, bootstrap_seed=5)

    assert summary["condition_order"] == list(EXTENDED_LADDER_CONDITIONS)
    assert summary["contrasts"]["t3_minus_c3"]["mean_margin_difference"] == 1.0
    assert summary["contrasts"]["t5_minus_c5"]["mean_margin_difference"] == 2.0
    assert summary["contrasts"]["q3_minus_t3"]["mean_margin_difference"] == 1.0
    assert summary["contrasts"]["q5_minus_t5"]["mean_margin_difference"] == 2.0
