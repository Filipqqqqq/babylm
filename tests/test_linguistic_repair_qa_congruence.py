from __future__ import annotations

from collections import Counter
import json
import subprocess
import sys
from pathlib import Path

from babysteps.linguistic_repair import (
    DeterministicQaMockGenerator,
    audit_document,
    build_qa_congruence_repair,
    build_scene_plans,
    number_to_words,
    quantity_answer,
)
from babysteps.linguistic_repair.planning import (
    DEFAULT_MINIMAL_QA_PAIR_QUESTION_COUNT_WEIGHTS,
    MINIMAL_QUESTION_TEMPLATE_IDS_BY_TYPE,
    QUESTION_TEMPLATE_IDS_BY_TYPE,
    SCENELESS_STYLES,
    SCENE_SENTENCE_TEMPLATE_IDS,
    STYLE_TOPIC_NAMES,
    STYLES,
    TOPICS,
)
from babysteps.linguistic_repair.rendering import SPEAKER_STYLE_PAIRS, render_template_text


ROOT = Path(__file__).resolve().parents[1]


def _base_config(tmp_path: Path, *, renderer_type: str = "template") -> dict:
    return {
        "name": "toy_qa_repair",
        "seed": 42,
        "document_count": 4,
        "min_questions_per_document": 6,
        "max_questions_per_document": 6,
        "output_dir": str(tmp_path / "qa_repair"),
        "renderer": {
            "type": renderer_type,
            "model_id": "fake-qwen",
            "documents_per_request": 1,
        },
        "styles": ["clean_dialogue", "simple_prose"],
        "question_types": ["who", "what", "where", "when", "why", "how_many"],
        "quantity_modes": ["word", "digit", "no_for_zero"],
        "quantities": [0, 1, 7, 20, 100],
        "times": ["yesterday", "after lunch", "on Monday"],
    }


def test_number_to_words_supports_repair_range() -> None:
    assert number_to_words(0) == "zero"
    assert quantity_answer(0, mode="no_for_zero") == "None."
    assert number_to_words(1) == "one"
    assert number_to_words(7) == "seven"
    assert number_to_words(20) == "twenty"
    assert number_to_words(100) == "one hundred"


def test_scene_planning_is_seeded_and_balanced_enough(tmp_path: Path) -> None:
    config = _base_config(tmp_path)

    first = [plan.to_dict() for plan in build_scene_plans(config)]
    second = [plan.to_dict() for plan in build_scene_plans(config)]
    changed = [plan.to_dict() for plan in build_scene_plans({**config, "seed": 43})]

    assert first == second
    assert first != changed
    assert len(first) == 4
    for row in first:
        assert [question["question_type"] for question in row["questions"]] != [
            "who",
            "what",
            "where",
            "when",
            "why",
            "how_many",
        ]
        assert set(question["question_type"] for question in row["questions"]) == {
            "who",
            "what",
            "where",
            "when",
            "why",
            "how_many",
        }


def test_surface_template_inventory_is_intentional() -> None:
    assert SCENE_SENTENCE_TEMPLATE_IDS == (
        "agent_time_reason",
        "time_fronted",
        "reason_fronted",
    )
    assert set(QUESTION_TEMPLATE_IDS_BY_TYPE) == {
        "who",
        "what",
        "where",
        "when",
        "why",
        "how_many",
    }
    assert all(
        len(template_ids) == 4
        for template_ids in QUESTION_TEMPLATE_IDS_BY_TYPE.values()
    )
    assert "who_was_it" in QUESTION_TEMPLATE_IDS_BY_TYPE["who"]
    assert "what_was_it" in QUESTION_TEMPLATE_IDS_BY_TYPE["what"]
    assert "where_was_it" in QUESTION_TEMPLATE_IDS_BY_TYPE["where"]
    assert "when_did_that_happen" in QUESTION_TEMPLATE_IDS_BY_TYPE["when"]
    assert "why_did_that_happen" in QUESTION_TEMPLATE_IDS_BY_TYPE["why"]
    assert "how_many_object_short" in QUESTION_TEMPLATE_IDS_BY_TYPE["how_many"]
    assert DEFAULT_MINIMAL_QA_PAIR_QUESTION_COUNT_WEIGHTS == (
        (1, 0.6),
        (2, 0.3),
        (3, 0.1),
    )
    assert MINIMAL_QUESTION_TEMPLATE_IDS_BY_TYPE == {
        "who": ("who_action_location",),
        "what": ("what_action_location",),
        "where": ("where_action_object",),
        "when": ("when_action_object_location",),
        "why": ("why_action_object_location",),
        "how_many": ("how_many_action_location",),
    }
    for question_type, template_ids in MINIMAL_QUESTION_TEMPLATE_IDS_BY_TYPE.items():
        assert set(template_ids) <= set(QUESTION_TEMPLATE_IDS_BY_TYPE[question_type])


def test_planner_uses_scene_and_question_surface_templates(tmp_path: Path) -> None:
    config = _base_config(tmp_path, renderer_type="template")
    config["document_count"] = 120
    config["styles"] = ["subtitle_clean"]
    plans = build_scene_plans(config)

    assert {plan.scene_template_id for plan in plans} == set(SCENE_SENTENCE_TEMPLATE_IDS)
    seen_question_templates = {question_type: set() for question_type in QUESTION_TEMPLATE_IDS_BY_TYPE}
    for plan in plans:
        assert plan.scene_sentence.startswith(
            (
                plan.agent[:1].upper() + plan.agent[1:],
                plan.time[:1].upper() + plan.time[1:] + ",",
                plan.reason[:1].upper() + plan.reason[1:] + ",",
            )
        )
        for question in plan.questions:
            seen_question_templates[question.question_type].add(question.question_template_id)

    assert seen_question_templates == {
        question_type: set(template_ids)
        for question_type, template_ids in QUESTION_TEMPLATE_IDS_BY_TYPE.items()
    }


def test_role_styles_use_compatible_topic_pools(tmp_path: Path) -> None:
    config = _base_config(tmp_path, renderer_type="template")
    config["document_count"] = 180
    config["styles"] = list(STYLES)
    plans = build_scene_plans(config)
    known_topics = {topic.topic for topic in TOPICS}

    assert all(set(topic_names) <= known_topics for topic_names in STYLE_TOPIC_NAMES.values())
    for plan in plans:
        if plan.style in STYLE_TOPIC_NAMES:
            assert plan.topic in STYLE_TOPIC_NAMES[plan.style]

    by_style = {style: {plan.topic for plan in plans if plan.style == style} for style in STYLES}
    assert by_style["coach_player_dialogue"] == {
        "game_equipment",
        "gym_warmup",
        "sports_practice",
    }
    assert by_style["nurse_child_dialogue"] == {
        "clinic_visit",
        "health_check",
        "school_nurse",
    }
    assert by_style["clerk_customer_dialogue"] == {
        "bakery_counter",
        "mail_delivery",
        "market_stall",
        "store_checkout",
    }
    assert "toys_tidying" not in by_style["clerk_customer_dialogue"]
    assert {
        plan.agent
        for plan in plans
        if plan.style in {"coach_player_dialogue", "nurse_child_dialogue", "clerk_customer_dialogue"}
    }.isdisjoint({"Coach Lee", "the nurse", "the clerk"})


def test_topic_locations_include_broad_where_targets() -> None:
    by_topic = {topic.topic: topic for topic in TOPICS}

    assert {
        "mail_delivery",
        "study_places",
        "reading_places",
        "writing_places",
    } <= set(by_topic)
    assert {
        "to another city",
        "to another country",
        "to a nearby town",
        "to the far side of the city",
        "across the state",
    } <= set(by_topic["mail_delivery"].locations)
    assert {
        "at the learning center",
        "at the community college",
        "at the preschool",
    } <= set(by_topic["study_places"].locations)
    assert {
        "in the workroom",
        "in a small office",
        "under the trees",
        "at the neighborhood park",
        "at the workplace",
        "beside the pond",
    } <= set(by_topic["reading_places"].locations)


def test_agent_pools_include_diverse_names_and_neutral_reasons() -> None:
    agents = {agent for topic in TOPICS for agent in topic.agents}
    reasons = " ".join(reason for topic in TOPICS for reason in topic.reasons).lower()
    reason_tokens = {token.strip(".,;:!?") for token in reasons.split()}

    assert {
        "Aisha",
        "Mateo",
        "Jamal",
        "Mei",
        "Priya",
        "Omar",
        "Amara",
        "Ravi",
        "Yuna",
        "Diego",
        "Fatima",
        "Min",
        "Carlos",
        "Amina",
        "Kenji",
        "Noor",
        "Luis",
    } <= agents
    assert not {"she", "he", "her", "his", "him", "they", "them", "their", "it"} & reason_tokens


def test_object_references_are_number_aware(tmp_path: Path) -> None:
    config = _base_config(tmp_path, renderer_type="template")
    config["document_count"] = 36
    config["min_questions_per_document"] = 6
    config["max_questions_per_document"] = 6
    config["quantities"] = [1]
    config["quantity_modes"] = ["word"]

    plans = build_scene_plans(config)

    for plan in plans:
        assert plan.object_phrase.startswith("one ")
        assert plan.object_reference == f"the {plan.object_singular}"
        for question in plan.questions:
            if question.question_type != "how_many":
                assert f"the {plan.object_plural}" not in question.question


def test_minimal_qa_pair_is_sceneless_and_explicit(tmp_path: Path) -> None:
    config = _base_config(tmp_path, renderer_type="template")
    config["document_count"] = 120
    config["styles"] = ["minimal_qa_pair"]

    plans = build_scene_plans(config)
    question_counts = Counter(len(plan.questions) for plan in plans)

    assert {plan.style for plan in plans} == {"minimal_qa_pair"}
    assert set(question_counts) <= {1, 2, 3}
    assert question_counts[1] > question_counts[2] > question_counts[3]
    assert {
        question.question_type
        for plan in plans
        for question in plan.questions
    } == set(config["question_types"])
    for plan in plans:
        rendered = render_template_text(plan)
        assert audit_document(rendered, plan).status == "pass"
        assert plan.scene_sentence not in rendered
        assert rendered.count("?") == len(plan.questions)
        assert "A:" not in rendered
        assert "B:" not in rendered
        assert "Teacher:" not in rendered
        for question in plan.questions:
            assert (
                question.question_template_id
                in MINIMAL_QUESTION_TEMPLATE_IDS_BY_TYPE[question.question_type]
            )
            assert "that happen" not in question.question
            assert "was that" not in question.question
            assert "was it" not in question.question


def test_minimal_qa_pair_question_count_weights_are_configurable(tmp_path: Path) -> None:
    config = _base_config(tmp_path, renderer_type="template")
    config["document_count"] = 12
    config["styles"] = ["minimal_qa_pair"]
    config["minimal_qa_pair_question_count_weights"] = {2: 1.0}

    plans = build_scene_plans(config)

    assert {len(plan.questions) for plan in plans} == {2}


def test_template_documents_pass_answer_type_audit(tmp_path: Path) -> None:
    plan = build_scene_plans(_base_config(tmp_path, renderer_type="template"))[0]
    text = render_template_text(plan)

    result = audit_document(text, plan)

    assert result.status == "pass"


def test_template_speaker_styles_pass_audit(tmp_path: Path) -> None:
    config = _base_config(tmp_path, renderer_type="template")
    config["document_count"] = len(STYLES)
    config["styles"] = list(STYLES)
    plans = build_scene_plans(config)

    rendered_by_style = {plan.style: render_template_text(plan) for plan in plans}
    for plan in plans:
        rendered = rendered_by_style[plan.style]
        assert audit_document(rendered, plan).status == "pass"
        if plan.style in SCENELESS_STYLES:
            assert not rendered.startswith(plan.scene_sentence)
            assert plan.scene_sentence not in rendered
        else:
            assert rendered.startswith(plan.scene_sentence)
    for style, (question_speaker, answer_speaker) in SPEAKER_STYLE_PAIRS.items():
        rendered = rendered_by_style[style]
        plan = next(plan for plan in plans if plan.style == style)
        assert not rendered.startswith(f"{question_speaker}:")
        assert f"{question_speaker}: {plan.questions[0].question}" in rendered
        assert f"{answer_speaker}: {plan.questions[0].answer}" in rendered


def test_answer_casing_preserves_weekday_names(tmp_path: Path) -> None:
    config = _base_config(tmp_path, renderer_type="template")
    config["document_count"] = 1
    config["times"] = ["on Saturday"]
    plan = build_scene_plans(config)[0]
    when_question = next(question for question in plan.questions if question.question_type == "when")

    assert when_question.answer == "On Saturday."


def test_audit_rejects_changed_answer_slots(tmp_path: Path) -> None:
    plan = build_scene_plans(_base_config(tmp_path, renderer_type="template"))[0]
    text = render_template_text(plan)
    where = next(question for question in plan.questions if question.question_type == "where")
    broken = text.replace(where.answer, "A book.", 1)

    result = audit_document(broken, plan)

    assert result.status == "fail"
    assert {flag.code for flag in result.flags} >= {"missing_answer"}


def test_audit_rejects_speaker_label_before_scene(tmp_path: Path) -> None:
    config = _base_config(tmp_path, renderer_type="template")
    config["styles"] = ["classroom_tutor"]
    plan = build_scene_plans(config)[0]
    broken = f"Teacher: {render_template_text(plan)}"

    result = audit_document(broken, plan)

    assert result.status == "fail"
    assert {flag.code for flag in result.flags} >= {"scene_prefix"}


def test_fireworks_mock_renderer_writes_checkpoints(tmp_path: Path) -> None:
    result = build_qa_congruence_repair(
        _base_config(tmp_path, renderer_type="fireworks"),
        base_dir=ROOT,
        generator=DeterministicQaMockGenerator(),
    )

    assert result.audit_report["counts"]["fail"] == 0
    assert result.audit_report["train_documents"] == 4
    assert (result.output_dir / "render_checkpoints" / "batch_000000_n001.jsonl").exists()
    assert (result.output_dir / "scenario_plans.jsonl").exists()
    assert (result.output_dir / "documents.jsonl").exists()
    assert (result.output_dir / "train.txt").exists()
    assert (result.output_dir / "audit_report.json").exists()
    assert (result.output_dir / "manifest.json").exists()


def test_fireworks_renderer_uses_template_for_local_styles(tmp_path: Path) -> None:
    class NoLocalStyleGenerator:
        def __call__(self, **kwargs: object) -> str:
            prompt = str(kwargs["prompt"])
            assert '"style": "simple_prose"' not in prompt
            assert '"style": "minimal_qa_pair"' not in prompt
            return DeterministicQaMockGenerator()(**kwargs)

    config = _base_config(tmp_path, renderer_type="fireworks")
    config["document_count"] = 3
    config["min_questions_per_document"] = 2
    config["max_questions_per_document"] = 2
    config["styles"] = ["simple_prose", "minimal_qa_pair", "clean_dialogue"]
    config["renderer"]["documents_per_request"] = 3

    result = build_qa_congruence_repair(
        config,
        base_dir=ROOT,
        generator=NoLocalStyleGenerator(),
    )

    by_style = {plan.style: document for plan, document in zip(result.plans, result.documents)}
    assert result.audit_report["counts"]["fail"] == 0
    assert by_style["simple_prose"]["renderer"] == "template"
    assert by_style["simple_prose"]["render_model_id"] is None
    assert by_style["minimal_qa_pair"]["renderer"] == "template"
    assert by_style["minimal_qa_pair"]["render_model_id"] is None
    assert by_style["clean_dialogue"]["renderer"] == "fireworks"
    assert by_style["clean_dialogue"]["render_model_id"] == "fake-qwen"


def test_stale_checkpoints_are_rerendered(tmp_path: Path) -> None:
    config = _base_config(tmp_path, renderer_type="fireworks")
    config["document_count"] = 1
    config["styles"] = ["classroom_tutor"]
    plans = build_scene_plans(config)
    output_dir = Path(config["output_dir"])
    checkpoint_path = output_dir / "render_checkpoints" / "batch_000000_n001.jsonl"
    checkpoint_path.parent.mkdir(parents=True)
    stale_row = {
        "id": plans[0].id,
        "renderer": "fireworks",
        "render_model_id": "fake-qwen",
        "text": f"Teacher: {render_template_text(plans[0])}",
    }
    checkpoint_path.write_text(json.dumps(stale_row) + "\n", encoding="utf-8")

    result = build_qa_congruence_repair(
        config,
        base_dir=ROOT,
        generator=DeterministicQaMockGenerator(),
    )

    assert result.audit_report["counts"]["fail"] == 0
    assert result.documents[0]["text"].startswith(plans[0].scene_sentence)
    assert not result.documents[0]["text"].startswith("Teacher:")


def test_cli_smoke_writes_artifacts_without_network(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_qa_repair",
                "seed: 42",
                "document_count: 2",
                "min_questions_per_document: 2",
                "max_questions_per_document: 3",
                f"output_dir: {output_dir}",
                "renderer:",
                "  type: template",
                "  documents_per_request: 1",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "synthetic_data_generation"
                / "linguistic_repair"
                / "generate_qa_congruence.py"
            ),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "QA repair: cli_qa_repair" in completed.stdout
    assert (output_dir / "scenario_plans.jsonl").exists()
    assert (output_dir / "render_checkpoints" / "batch_000000_n001.jsonl").exists()
    assert (output_dir / "train.txt").exists()
