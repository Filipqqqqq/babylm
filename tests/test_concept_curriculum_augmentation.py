from __future__ import annotations

import json
from pathlib import Path

from babysteps.concept_curriculum.augmentation import (
    DeterministicMotChiMockGenerator,
    DocumentAugmenter,
    RegexTermMatcher,
    assigned_agent_for_spec,
    assigned_deictic_agent_for_spec,
    assigned_speaker_perspective_for_spec,
    augmentation_config_from_mapping,
    build_augmentation_render_requests,
    build_document_augmentations,
    build_mot_chi_render_prompt,
    build_mot_chi_render_requests,
    build_mot_chi_repair_prompt,
    render_augmented_text,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def _counts(values) -> dict:
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


class _RepairingGenerator:
    def __call__(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        response_format: dict | None = None,
    ) -> str:
        del system_prompt, temperature, max_tokens, response_format
        if stage == "augmentation_mot_chi_repair":
            return json.dumps(
                {
                    "items": [
                        {
                            "id": "phys-pass__caregiver_qa",
                            "text": (
                                "*MOT:\twhat happened to the bead? "
                                "*CHI:\tit rolled. "
                                "*MOT:\tyes, the bead rolled; mat, rolling, turning."
                            ),
                        }
                    ]
                }
            )
        return json.dumps(
            {
                "items": [
                    {
                        "id": "phys-pass__caregiver_qa",
                        "text": (
                            "*MOT:\twhat happened to the bead? "
                            "*CHI:\tit moved. "
                            "*MOT:\tyes, the bead moved."
                        ),
                    }
                ]
            }
        )


def test_material_augmentation_preserves_required_terms() -> None:
    spec = {
        "id": "mat-1",
        "domain": "material-dynamics",
        "material": "foam",
        "required_materials": ["foam"],
        "required_affordances": ["tap", "compress", "rebound"],
        "target_affordance": "tap",
    }

    text = render_augmented_text(spec, "caregiver_qa")

    assert "*MOT:" in text
    assert "*CHI:" in text
    assert "?" in text
    assert "foam" in text
    assert "tap" in text
    assert "compress" in text
    assert "rebound" in text
    assert "not" not in text.lower()


def test_physical_augmentation_preserves_lexical_and_evidence_terms() -> None:
    spec = {
        "id": "phys-1",
        "domain": "physical-dynamics",
        "object": "foam plug",
        "lexical_target": "floats",
        "required_cause_terms": ["foam"],
        "required_process_terms": ["floating"],
        "required_outcome_terms": ["same height"],
    }

    text = render_augmented_text(spec, "state_tracking_dialogue")

    assert "foam plug" in text
    assert "floats" in text
    assert "floating" in text
    assert "same height" in text
    assert "*MOT:" in text
    assert "*CHI:" in text


def test_object_and_agent_qa_styles_are_positive_and_type_specific() -> None:
    physical = {
        "id": "phys-1",
        "domain": "physical-dynamics",
        "object": "bead",
        "lexical_target": "rolled",
        "required_cause_terms": ["mat"],
        "required_process_terms": ["rolling"],
        "required_outcome_terms": ["turning"],
    }
    material = {
        "id": "mat-1",
        "domain": "material-dynamics",
        "material": "foam",
        "required_materials": ["foam"],
        "required_affordances": ["tap", "compress"],
        "target_affordance": "tap",
    }

    object_text = render_augmented_text(physical, "what_object_qa")
    agent_text = render_augmented_text(material, "who_agent_qa", assigned_agent="the gardener")

    assert "*MOT:\twhat did we watch?" in object_text
    assert "*CHI:\tthe bead." in object_text
    assert "rolled" in object_text
    assert "rolling" in object_text
    assert "*MOT:\twho used the foam?" in agent_text
    assert "*CHI:\tthe gardener." in agent_text
    assert "the gardener used the foam" in agent_text
    assert "tap" in agent_text
    assert "compress" in agent_text
    assert "not" not in object_text.lower()
    assert "not" not in agent_text.lower()


def test_where_location_qa_style_preserves_rules() -> None:
    physical = {
        "id": "phys-1",
        "domain": "physical-dynamics",
        "object": "foam plug",
        "lexical_target": "floats",
        "bridge_clause": "on the tank water",
        "required_cause_terms": ["foam"],
        "required_process_terms": ["floating"],
        "required_outcome_terms": ["same height"],
    }
    material = {
        "id": "mat-1",
        "domain": "material-dynamics",
        "material": "paper bag",
        "required_materials": ["paper bag"],
        "required_affordances": ["rip", "tear"],
        "target_affordance": "rip",
    }

    physical_text = render_augmented_text(physical, "where_location_qa")
    material_text = render_augmented_text(material, "where_location_qa")

    assert "*MOT:\twhere is the foam plug?" in physical_text
    assert "*CHI:\ton the tank water." in physical_text
    assert "floats" in physical_text
    assert "floating" in physical_text
    assert "same height" in physical_text
    assert "*MOT:\twhere is the paper bag?" in material_text
    assert "*CHI:\ton the table." in material_text
    assert "rip" in material_text
    assert "tear" in material_text


def test_where_location_qa_uses_speech_like_physical_locations() -> None:
    after_release = {
        "id": "phys-1",
        "domain": "physical-dynamics",
        "object": "paper scrap",
        "lexical_target": "fell",
        "material_or_surface": "paper in room air",
        "bridge_clause": "after release",
        "required_cause_terms": ["gravity", "release"],
        "required_process_terms": ["falling"],
        "required_outcome_terms": ["height", "lower"],
    }
    past_edge = {
        "id": "phys-2",
        "domain": "physical-dynamics",
        "object": "button",
        "lexical_target": "fell",
        "material_or_surface": "button in air",
        "bridge_clause": "past the table edge",
        "required_cause_terms": ["edge"],
        "required_process_terms": ["falling"],
        "required_outcome_terms": ["lower", "mat"],
    }

    after_release_text = render_augmented_text(after_release, "where_location_qa")
    past_edge_text = render_augmented_text(past_edge, "where_location_qa")

    assert "*CHI:\tin the room air." in after_release_text
    assert "on the after release" not in after_release_text
    assert "*CHI:\tin the air." in past_edge_text
    assert "on the past the table edge" not in past_edge_text


def test_positive_only_audit_rejects_negative_markers() -> None:
    config = {
        "name": "toy",
        "input_dir": ".",
        "output_dir": ".",
        "styles": ["caregiver_qa"],
    }
    augmenter = DocumentAugmenter(matcher=RegexTermMatcher())
    parsed = augmentation_config_from_mapping(config, base_dir=Path("."))
    spec = {
        "id": "mat-1",
        "domain": "material-dynamics",
        "required_materials": ["foam"],
        "required_affordances": ["tap"],
    }
    document = {
        "text": "*MOT:\twhat can happen? *CHI:\tnot tap. *MOT:\tfoam tap.",
        "augmentation_style": "caregiver_qa",
    }
    avoid_document = {
        "text": (
            "*MOT:\twhat can happen? "
            "*CHI:\tavoid tap. "
            "*MOT:\tfoam can tap."
        ),
        "augmentation_style": "caregiver_qa",
    }

    audit = augmenter.audit_document(document, spec=spec, config=parsed)
    avoid_audit = augmenter.audit_document(avoid_document, spec=spec, config=parsed)

    assert audit["audit_status"] == "fail"
    assert any(flag["code"] == "positive_only_marker" for flag in audit["audit_flags"])
    assert avoid_audit["audit_status"] == "fail"
    assert any(
        "avoid" in flag["message"]
        for flag in avoid_audit["audit_flags"]
        if flag["code"] == "positive_only_marker"
    )


def test_material_audit_rejects_bad_bare_affordance_verb_frames() -> None:
    parsed = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["who_agent_qa"],
        },
        base_dir=Path("."),
    )
    augmenter = DocumentAugmenter(matcher=RegexTermMatcher())
    spec = {
        "id": "mat-1",
        "domain": "material-dynamics",
        "material": "paper strip",
        "required_materials": ["paper strip"],
        "required_affordances": ["bend", "press"],
        "target_affordance": "bend",
    }
    bad_document = {
        "text": (
            "*MOT:\tWho bend the paper strip? "
            "*CHI:\tHe press. "
            "*MOT:\tHe press and bend the paper strip."
        ),
        "augmentation_style": "who_agent_qa",
    }
    good_document = {
        "text": (
            "*MOT:\tWho can bend the paper strip? "
            "*CHI:\tYou can bend it. "
            "*MOT:\tYes, you can bend and press the paper strip."
        ),
        "augmentation_style": "who_agent_qa",
    }
    perception_document = {
        "text": (
            "*MOT:\tWhat did you see? "
            "*CHI:\tI saw it bend. "
            "*MOT:\tThe paper strip can bend and press."
        ),
        "augmentation_style": "caregiver_qa",
    }

    bad_audit = augmenter.audit_document(bad_document, spec=spec, config=parsed)
    good_audit = augmenter.audit_document(good_document, spec=spec, config=parsed)
    perception_audit = augmenter.audit_document(
        perception_document,
        spec=spec,
        config=parsed,
    )

    assert bad_audit["audit_status"] == "fail"
    assert any(
        flag["code"] == "bad_material_affordance_verb_frame"
        for flag in bad_audit["audit_flags"]
    )
    assert good_audit["audit_status"] == "pass"
    assert perception_audit["audit_status"] == "pass"


def test_who_agent_audit_requires_caregiver_binding() -> None:
    parsed = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["who_agent_qa"],
            "agent_pool": ["the gardener"],
        },
        base_dir=Path("."),
    )
    augmenter = DocumentAugmenter(matcher=RegexTermMatcher())
    spec = {
        "id": "phys-pass",
        "domain": "physical-dynamics",
        "object": "bead",
        "lexical_target": "rolled",
        "required_cause_terms": ["mat"],
        "required_process_terms": ["rolling"],
        "required_outcome_terms": ["turning"],
    }
    weak_document = {
        "text": (
            "*MOT:\twho watched the bead? "
            "*CHI:\tthe gardener. "
            "*MOT:\tyes, the bead rolled on the mat, rolling and turning."
        ),
        "augmentation_style": "who_agent_qa",
        "assigned_agent": "the gardener",
    }
    bound_document = {
        "text": render_augmented_text(
            spec,
            "who_agent_qa",
            assigned_agent="the gardener",
        ),
        "augmentation_style": "who_agent_qa",
        "assigned_agent": "the gardener",
    }

    weak_audit = augmenter.audit_document(weak_document, spec=spec, config=parsed)
    bound_audit = augmenter.audit_document(bound_document, spec=spec, config=parsed)

    assert weak_audit["audit_status"] == "fail"
    assert any(flag["code"] == "missing_who_agent_binding" for flag in weak_audit["audit_flags"])
    assert bound_audit["audit_status"] == "pass"


def test_speaker_perspective_who_qa_maps_i_and_you_across_speakers() -> None:
    parsed = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["speaker_perspective_who_qa"],
        },
        base_dir=Path("."),
    )
    augmenter = DocumentAugmenter(matcher=RegexTermMatcher())
    spec = {
        "id": "phys-pass",
        "domain": "physical-dynamics",
        "object": "bead",
        "lexical_target": "rolled",
        "required_cause_terms": ["mat"],
        "required_process_terms": ["rolling"],
        "required_outcome_terms": ["turning"],
    }
    perspective = {
        "role": "child",
        "child_who_answer": "I",
        "caregiver_followup_subject": "you",
    }
    caregiver_perspective = {
        "role": "caregiver",
        "child_who_answer": "you",
        "caregiver_followup_subject": "I",
    }
    good_document = {
        "text": render_augmented_text(
            spec,
            "speaker_perspective_who_qa",
            speaker_perspective=perspective,
        ),
        "augmentation_style": "speaker_perspective_who_qa",
        "speaker_perspective": perspective,
    }
    caregiver_document = {
        "text": render_augmented_text(
            spec,
            "speaker_perspective_who_qa",
            speaker_perspective=caregiver_perspective,
        ),
        "augmentation_style": "speaker_perspective_who_qa",
        "speaker_perspective": caregiver_perspective,
    }
    bad_document = {
        "text": (
            "*MOT:\twho watched the bead? "
            "*CHI:\tI watched it. "
            "*MOT:\tyes, I watched the bead; it rolled on the mat, rolling and turning."
        ),
        "augmentation_style": "speaker_perspective_who_qa",
        "speaker_perspective": perspective,
    }

    good_audit = augmenter.audit_document(good_document, spec=spec, config=parsed)
    caregiver_audit = augmenter.audit_document(caregiver_document, spec=spec, config=parsed)
    bad_audit = augmenter.audit_document(bad_document, spec=spec, config=parsed)

    assert "*CHI:\tI did." in good_document["text"]
    assert "*MOT:\tyes, you watched the bead" in good_document["text"]
    assert "*CHI:\tyou did." in caregiver_document["text"]
    assert "*MOT:\tyes, I watched the bead" in caregiver_document["text"]
    assert good_audit["audit_status"] == "pass"
    assert caregiver_audit["audit_status"] == "pass"
    assert bad_audit["audit_status"] == "fail"
    assert any(
        flag["code"] == "missing_caregiver_perspective_pronoun"
        for flag in bad_audit["audit_flags"]
    )


def test_possessive_deixis_who_qa_maps_my_to_your_across_speakers() -> None:
    parsed = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["possessive_deixis_who_qa"],
        },
        base_dir=Path("."),
    )
    augmenter = DocumentAugmenter(matcher=RegexTermMatcher())
    spec = {
        "id": "phys-pass",
        "domain": "physical-dynamics",
        "object": "bead",
        "lexical_target": "rolled",
        "required_cause_terms": ["mat"],
        "required_process_terms": ["rolling"],
        "required_outcome_terms": ["turning"],
    }
    deictic_agent = {
        "child_who_answer": "my brother",
        "caregiver_followup_subject": "your brother",
    }
    good_document = {
        "text": render_augmented_text(
            spec,
            "possessive_deixis_who_qa",
            deictic_agent=deictic_agent,
        ),
        "augmentation_style": "possessive_deixis_who_qa",
        "deictic_agent": deictic_agent,
    }
    bad_document = {
        "text": (
            "*MOT:\twho watched the bead? "
            "*CHI:\tmy brother. "
            "*MOT:\tyes, my brother watched the bead; it rolled on the mat, "
            "rolling and turning."
        ),
        "augmentation_style": "possessive_deixis_who_qa",
        "deictic_agent": deictic_agent,
    }

    good_audit = augmenter.audit_document(good_document, spec=spec, config=parsed)
    bad_audit = augmenter.audit_document(bad_document, spec=spec, config=parsed)

    assert "*CHI:\tmy brother." in good_document["text"]
    assert "*MOT:\tyes, your brother watched the bead" in good_document["text"]
    assert good_audit["audit_status"] == "pass"
    assert bad_audit["audit_status"] == "fail"
    assert any(
        flag["code"] == "caregiver_repeats_child_deictic_agent"
        for flag in bad_audit["audit_flags"]
    )


def test_present_process_who_qa_uses_present_observation_frame() -> None:
    parsed = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["possessive_deixis_who_qa"],
        },
        base_dir=Path("."),
    )
    augmenter = DocumentAugmenter(matcher=RegexTermMatcher())
    spec = {
        "id": "phys-pass",
        "domain": "physical-dynamics",
        "object": "toy cart",
        "lexical_target": "rolls",
        "required_cause_terms": ["wheels"],
        "required_process_terms": ["rolling"],
        "required_outcome_terms": ["turning"],
    }
    deictic_agent = {
        "child_who_answer": "my brother",
        "caregiver_followup_subject": "your brother",
    }
    good_text = render_augmented_text(
        spec,
        "possessive_deixis_who_qa",
        child_answer_mode="short_clause",
        deictic_agent=deictic_agent,
    )
    bad_document = {
        "text": (
            "*MOT:\twho watched the toy cart? "
            "*CHI:\tmy brother. "
            "*MOT:\tyes, your brother watched the toy cart rolls with wheels, "
            "rolling and turning."
        ),
        "augmentation_style": "possessive_deixis_who_qa",
        "deictic_agent": deictic_agent,
    }
    good_document = {
        "text": good_text,
        "augmentation_style": "possessive_deixis_who_qa",
        "deictic_agent": deictic_agent,
    }

    good_audit = augmenter.audit_document(good_document, spec=spec, config=parsed)
    bad_audit = augmenter.audit_document(bad_document, spec=spec, config=parsed)

    assert "*MOT:\twho is watching the toy cart now?" in good_text
    assert "*CHI:\tmy brother is watching." in good_text
    assert "*MOT:\tyes, your brother is watching. the toy cart rolls" in good_text
    assert "watched the toy cart rolls" not in good_text
    assert good_audit["audit_status"] == "pass"
    assert bad_audit["audit_status"] == "fail"
    assert any(
        flag["code"] == "present_target_embedded_in_agent_clause"
        for flag in bad_audit["audit_flags"]
    )


def test_build_document_augmentations_writes_outputs_and_filters_failed_sources(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "source"
    output_dir = tmp_path / "augmented"
    specs = [
        {
            "id": "mat-pass",
            "domain": "material-dynamics",
            "material": "foam",
            "required_materials": ["foam"],
            "required_affordances": ["tap", "compress"],
            "target_affordance": "tap",
        },
        {
            "id": "mat-fail",
            "domain": "material-dynamics",
            "material": "paper",
            "required_materials": ["paper"],
            "required_affordances": ["fold"],
            "target_affordance": "fold",
        },
    ]
    source_docs = [
        {"id": "mat-pass", "text": "foam text", "audit_status": "pass"},
        {"id": "mat-fail", "text": "paper text", "audit_status": "fail"},
    ]
    _write_jsonl(input_dir / "scenario_plans.jsonl", specs)
    _write_jsonl(input_dir / "documents_audited.jsonl", source_docs)

    result = build_document_augmentations(
        {
            "name": "toy_aug",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "styles": ["caregiver_qa", "fragment_answer"],
            "selection": {"include_audit_status": ["pass", "warn"]},
            "audit": {"max_words": 80},
        },
        base_dir=tmp_path,
    )

    assert len(result.documents) == 2
    assert {row["source_id"] for row in result.documents} == {"mat-pass"}
    assert result.audit_report["counts"]["fail"] == 0
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "augmented_documents.jsonl").exists()
    assert (output_dir / "augmentation_manifest.json").exists()
    train_lines = [
        line
        for line in (output_dir / "train.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(train_lines) == 2


def test_mot_chi_prompt_includes_style_and_required_terms() -> None:
    parsed = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "renderer": {"type": "fireworks", "model_id": "fake-qwen"},
            "styles": ["who_agent_qa"],
            "child_answer_modes": {
                "strategy": "balanced_quota",
                "modes": ["explanatory_clause"],
            },
            "agent_pool": ["the gardener"],
        },
        base_dir=Path("."),
    )
    spec = {
        "id": "phys-pass",
        "domain": "physical-dynamics",
        "object": "foam plug",
        "lexical_target": "floats",
        "required_cause_terms": ["foam"],
        "required_process_terms": ["floating"],
        "required_outcome_terms": ["same height"],
        "preferred_process_terms": ["float", "floated", "floating"],
    }

    requests = build_mot_chi_render_requests([spec], parsed)
    prompt = build_mot_chi_render_prompt(requests, config=parsed)

    assert requests[0]["must_include_exact_terms"] == [
        "foam plug",
        "floats",
        "foam",
        "floating",
        "same height",
    ]
    assert requests[0]["source_spec"]["lexical_target_frame"] == "present_process_observation"
    assert requests[0]["child_answer_mode"] == "explanatory_clause"
    assert "*MOT:" in prompt
    assert "*CHI:" in prompt
    assert "who_agent_qa" in prompt
    assert '"child_answer_mode": "explanatory_clause"' in prompt
    assert '"child_answer_mode_guidance"' in prompt
    assert '"assigned_agent": "the gardener"' in prompt
    assert '"must_include_exact_terms"' in prompt
    assert "foam plug" in prompt
    assert "floats" in prompt
    assert "floating" in prompt
    assert "same height" in prompt
    assert "preferred_process_terms" not in prompt
    assert "Return only valid JSON" in prompt
    assert "use that exact third-person string" in prompt
    assert "must repeat the exact assigned_agent" in prompt
    assert "lexical_target_frame='present_process_observation'" in prompt
    assert "watched the X rolls" in prompt
    assert "including avoid, prevent, or keep from" in prompt
    assert "Who bend" in prompt
    assert "He press" in prompt
    assert "It spill" in prompt
    assert "final *MOT: turn must include every string" in prompt
    assert "Every text must include at least one question mark" in prompt
    assert "Do not substitute synonyms" in prompt


def test_mot_chi_prompt_includes_speaker_perspective_mapping() -> None:
    parsed = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "renderer": {"type": "fireworks", "model_id": "fake-qwen"},
            "styles": ["speaker_perspective_who_qa"],
        },
        base_dir=Path("."),
    )
    spec = {
        "id": "phys-pass",
        "domain": "physical-dynamics",
        "object": "foam plug",
        "lexical_target": "floats",
        "required_cause_terms": ["foam"],
        "required_process_terms": ["floating"],
        "required_outcome_terms": ["same height"],
    }

    requests = build_mot_chi_render_requests([spec], parsed)
    prompt = build_mot_chi_render_prompt(requests, config=parsed)

    assert requests[0]["augmentation_style"] == "speaker_perspective_who_qa"
    assert requests[0]["speaker_perspective"] in [
        {
            "role": "child",
            "child_who_answer": "I",
            "caregiver_followup_subject": "you",
        },
        {
            "role": "caregiver",
            "child_who_answer": "you",
            "caregiver_followup_subject": "I",
        },
    ]
    assert '"speaker_perspective"' in prompt
    assert "child 'I' maps to caregiver 'you'" in prompt
    assert "child 'you' maps to caregiver 'I'" in prompt


def test_mot_chi_prompt_includes_possessive_deixis_mapping() -> None:
    parsed = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "renderer": {"type": "fireworks", "model_id": "fake-qwen"},
            "styles": ["possessive_deixis_who_qa"],
            "deictic_agent_pool": [
                {
                    "child_who_answer": "my brother",
                    "caregiver_followup_subject": "your brother",
                }
            ],
        },
        base_dir=Path("."),
    )
    spec = {
        "id": "phys-pass",
        "domain": "physical-dynamics",
        "object": "foam plug",
        "lexical_target": "floats",
        "required_cause_terms": ["foam"],
        "required_process_terms": ["floating"],
        "required_outcome_terms": ["same height"],
    }

    requests = build_mot_chi_render_requests([spec], parsed)
    prompt = build_mot_chi_render_prompt(requests, config=parsed)

    assert requests[0]["augmentation_style"] == "possessive_deixis_who_qa"
    assert requests[0]["deictic_agent"] == {
        "child_who_answer": "my brother",
        "caregiver_followup_subject": "your brother",
    }
    assert '"deictic_agent"' in prompt
    assert "child 'my brother' maps to caregiver 'your brother'" in prompt
    assert "do not make the caregiver repeat child-relative 'my'" in prompt


def test_mot_chi_repair_prompt_requires_exact_final_caregiver_terms() -> None:
    parsed = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "renderer": {"type": "fireworks", "model_id": "fake-qwen"},
            "styles": ["caregiver_qa"],
        },
        base_dir=Path("."),
    )
    prompt = build_mot_chi_repair_prompt(
        [
            {
                "id": "phys-pass__caregiver_qa",
                "source_id": "phys-pass",
                "augmentation_style": "caregiver_qa",
                "child_answer_mode": "fragment",
                "assigned_agent": None,
                "speaker_perspective": None,
                "existing_text": (
                    "*MOT:\twhat happened to the bead? "
                    "*CHI:\tit moved. "
                    "*MOT:\tyes, the bead moved."
                ),
                "audit_flags": [
                    {
                        "severity": "error",
                        "code": "missing_required_process",
                        "message": "Missing required_process_terms: rolling",
                    }
                ],
                "missing_terms": ["rolling"],
                "must_include_exact_terms": ["bead", "rolled", "mat", "rolling", "turning"],
                "source_spec": {
                    "id": "phys-pass",
                    "object": "bead",
                    "lexical_target": "rolled",
                    "required_cause_terms": ["mat"],
                    "required_process_terms": ["rolling"],
                    "required_outcome_terms": ["turning"],
                },
            }
        ],
        config=parsed,
    )

    assert "Repair each failed child-directed dialogue" in prompt
    assert "Fix only the listed audit_flags" in prompt
    assert "final *MOT: turn must include every string" in prompt
    assert "including avoid, prevent, or keep from" in prompt
    assert "Who bend" in prompt
    assert "He press" in prompt
    assert "It spill" in prompt
    assert '"missing_terms": [' in prompt
    assert "rolling" in prompt


def test_balanced_quota_assigns_styles_and_child_answer_modes() -> None:
    config = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["caregiver_qa", "who_agent_qa", "where_location_qa"],
            "style_assignment": {"strategy": "balanced_quota"},
            "child_answer_modes": {
                "strategy": "balanced_quota",
                "modes": ["fragment", "short_clause", "explanatory_clause"],
            },
            "selection": {"max_augmented_documents": 9},
            "agent_pool": ["the gardener"],
        },
        base_dir=Path("."),
    )
    specs = [
        {"id": f"phys-{index}", "domain": "physical-dynamics", "object": "bead"}
        for index in range(20)
    ]

    first = build_augmentation_render_requests(specs, config)
    second = build_augmentation_render_requests(specs, config)

    assert first == second
    assert len(first) == 9
    assert _counts(row["augmentation_style"] for row in first) == {
        "caregiver_qa": 3,
        "where_location_qa": 3,
        "who_agent_qa": 3,
    }
    assert _counts(row["child_answer_mode"] for row in first) == {
        "explanatory_clause": 3,
        "fragment": 3,
        "short_clause": 3,
    }
    for style in {"caregiver_qa", "where_location_qa", "who_agent_qa"}:
        style_rows = [row for row in first if row["augmentation_style"] == style]
        assert _counts(row["child_answer_mode"] for row in style_rows) == {
            "explanatory_clause": 1,
            "fragment": 1,
            "short_clause": 1,
        }


def test_balanced_quota_build_writes_answer_mode_counts(tmp_path: Path) -> None:
    input_dir = tmp_path / "source"
    output_dir = tmp_path / "augmented"
    specs = [
        {
            "id": f"mat-{index}",
            "domain": "material-dynamics",
            "material": "foam",
            "required_materials": ["foam"],
            "required_affordances": ["tap", "compress"],
            "target_affordance": "tap",
        }
        for index in range(6)
    ]
    source_docs = [
        {"id": spec["id"], "text": "foam text", "audit_status": "pass"}
        for spec in specs
    ]
    _write_jsonl(input_dir / "scenario_plans.jsonl", specs)
    _write_jsonl(input_dir / "documents_audited.jsonl", source_docs)

    result = build_document_augmentations(
        {
            "name": "toy_balanced",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "styles": ["caregiver_qa", "prediction_qa"],
            "style_assignment": {"strategy": "balanced_quota"},
            "child_answer_modes": {
                "strategy": "balanced_quota",
                "modes": ["fragment", "short_clause", "explanatory_clause"],
            },
            "selection": {"include_audit_status": ["pass"]},
            "audit": {"max_words": 90},
        },
        base_dir=tmp_path,
    )

    assert len(result.documents) == 6
    assert _counts(row["augmentation_style"] for row in result.documents) == {
        "caregiver_qa": 3,
        "prediction_qa": 3,
    }
    assert _counts(row["child_answer_mode"] for row in result.documents) == {
        "explanatory_clause": 2,
        "fragment": 2,
        "short_clause": 2,
    }
    assert result.audit_report["counts_by_child_answer_mode"]["fragment"]["pass"] == 2
    assert result.manifest["style_assignment"]["strategy"] == "balanced_quota"
    assert result.manifest["child_answer_modes"]["modes"] == [
        "fragment",
        "short_clause",
        "explanatory_clause",
    ]


def test_who_agent_assignment_is_deterministic_and_configured() -> None:
    config = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["who_agent_qa"],
            "agent_pool": ["the gardener", "the mailman", "the teacher", "the helper"],
        },
        base_dir=Path("."),
    )
    spec = {"id": "phys-pass", "domain": "physical-dynamics", "object": "bead"}

    first = assigned_agent_for_spec(spec, "who_agent_qa", config)
    second = assigned_agent_for_spec(spec, "who_agent_qa", config)

    assert first == second
    assert first in config.agent_pool
    assert assigned_agent_for_spec(spec, "caregiver_qa", config) is None


def test_default_who_agent_pool_avoids_speaker_relative_pronouns() -> None:
    config = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["who_agent_qa"],
        },
        base_dir=Path("."),
    )
    agents = {
        assigned_agent_for_spec(
            {"id": f"phys-{index}", "domain": "physical-dynamics", "object": "bead"},
            "who_agent_qa",
            config,
        )
        for index in range(100)
    }

    assert "I" not in agents
    assert "you" not in agents
    assert "my friend" not in config.agent_pool
    assert "my brother" not in config.agent_pool
    assert "my sister" not in config.agent_pool


def test_deictic_agent_assignment_is_deterministic_and_configured() -> None:
    config = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["possessive_deixis_who_qa"],
            "deictic_agent_pool": [
                {
                    "child_who_answer": "my friend",
                    "caregiver_followup_subject": "your friend",
                },
                {
                    "child_who_answer": "my sister",
                    "caregiver_followup_subject": "your sister",
                },
            ],
        },
        base_dir=Path("."),
    )
    spec = {"id": "phys-pass", "domain": "physical-dynamics", "object": "bead"}

    first = assigned_deictic_agent_for_spec(
        spec,
        "possessive_deixis_who_qa",
        config,
    )
    second = assigned_deictic_agent_for_spec(
        spec,
        "possessive_deixis_who_qa",
        config,
    )

    assert first == second
    assert first in config.deictic_agent_pool
    assert assigned_deictic_agent_for_spec(spec, "who_agent_qa", config) is None


def test_speaker_perspective_assignment_is_deterministic() -> None:
    config = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["speaker_perspective_who_qa"],
        },
        base_dir=Path("."),
    )
    spec = {"id": "phys-pass", "domain": "physical-dynamics", "object": "bead"}

    first = assigned_speaker_perspective_for_spec(
        spec,
        "speaker_perspective_who_qa",
        config,
    )
    second = assigned_speaker_perspective_for_spec(
        spec,
        "speaker_perspective_who_qa",
        config,
    )

    assert first == second
    assert first in [
        {
            "role": "child",
            "child_who_answer": "I",
            "caregiver_followup_subject": "you",
        },
        {
            "role": "caregiver",
            "child_who_answer": "you",
            "caregiver_followup_subject": "I",
        },
    ]
    assert assigned_speaker_perspective_for_spec(spec, "who_agent_qa", config) is None


def test_speaker_perspective_roles_are_quota_balanced_in_requests() -> None:
    config = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["speaker_perspective_who_qa"],
            "style_assignment": {"strategy": "balanced_quota"},
            "selection": {"max_augmented_documents": 6},
        },
        base_dir=Path("."),
    )
    specs = [
        {"id": f"phys-{index}", "domain": "physical-dynamics", "object": "bead"}
        for index in range(10)
    ]

    requests = build_augmentation_render_requests(specs, config)

    assert _counts(row["speaker_perspective"]["role"] for row in requests) == {
        "caregiver": 3,
        "child": 3,
    }


def test_deictic_agents_are_quota_balanced_in_requests() -> None:
    config = augmentation_config_from_mapping(
        {
            "name": "toy",
            "input_dir": ".",
            "output_dir": ".",
            "styles": ["possessive_deixis_who_qa"],
            "style_assignment": {"strategy": "balanced_quota"},
            "selection": {"max_augmented_documents": 6},
            "deictic_agent_pool": [
                {
                    "child_who_answer": "my friend",
                    "caregiver_followup_subject": "your friend",
                },
                {
                    "child_who_answer": "my brother",
                    "caregiver_followup_subject": "your brother",
                },
                {
                    "child_who_answer": "my sister",
                    "caregiver_followup_subject": "your sister",
                },
            ],
        },
        base_dir=Path("."),
    )
    specs = [
        {"id": f"phys-{index}", "domain": "physical-dynamics", "object": "bead"}
        for index in range(10)
    ]

    requests = build_augmentation_render_requests(specs, config)

    assert _counts(row["deictic_agent"]["child_who_answer"] for row in requests) == {
        "my brother": 2,
        "my friend": 2,
        "my sister": 2,
    }


def test_fireworks_mot_chi_renderer_uses_generator_and_audits_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "source"
    output_dir = tmp_path / "augmented"
    specs = [
        {
            "id": "phys-pass",
            "domain": "physical-dynamics",
            "object": "foam plug",
            "lexical_target": "floats",
            "required_cause_terms": ["foam"],
            "required_process_terms": ["floating"],
            "required_outcome_terms": ["same height"],
        }
    ]
    source_docs = [{"id": "phys-pass", "text": "foam text", "audit_status": "pass"}]
    _write_jsonl(input_dir / "scenario_plans.jsonl", specs)
    _write_jsonl(input_dir / "documents_audited.jsonl", source_docs)

    result = build_document_augmentations(
        {
            "name": "toy_qwen",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "renderer": {
                "type": "fireworks",
                "model_id": "fake-qwen",
                "documents_per_request": 1,
            },
            "styles": ["caregiver_qa", "where_location_qa"],
            "selection": {"include_audit_status": ["pass"]},
            "agent_pool": ["the mailman"],
            "audit": {"max_words": 90},
        },
        base_dir=tmp_path,
        generator=DeterministicMotChiMockGenerator(),
    )

    assert len(result.documents) == 2
    assert result.audit_report["counts"]["fail"] == 0
    assert {row["renderer"] for row in result.documents} == {"fireworks"}
    assert {row["render_model_id"] for row in result.documents} == {"fake-qwen"}
    assert (output_dir / "render_checkpoints" / "batch_000000_n001.jsonl").exists()
    train_text = (output_dir / "train.txt").read_text(encoding="utf-8")
    assert "*MOT:" in train_text
    assert "*CHI:" in train_text
    assert "foam plug" in train_text


def test_fireworks_repair_recovers_failed_documents(tmp_path: Path) -> None:
    input_dir = tmp_path / "source"
    output_dir = tmp_path / "augmented"
    specs = [
        {
            "id": "phys-pass",
            "domain": "physical-dynamics",
            "object": "bead",
            "lexical_target": "rolled",
            "required_cause_terms": ["mat"],
            "required_process_terms": ["rolling"],
            "required_outcome_terms": ["turning"],
        }
    ]
    source_docs = [{"id": "phys-pass", "text": "bead text", "audit_status": "pass"}]
    _write_jsonl(input_dir / "scenario_plans.jsonl", specs)
    _write_jsonl(input_dir / "documents_audited.jsonl", source_docs)

    result = build_document_augmentations(
        {
            "name": "toy_qwen_repair",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "renderer": {
                "type": "fireworks",
                "model_id": "fake-qwen",
                "documents_per_request": 1,
            },
            "styles": ["caregiver_qa"],
            "selection": {"include_audit_status": ["pass"]},
            "repair": {"enabled": True, "max_rounds": 1},
            "audit": {"max_words": 90},
        },
        base_dir=tmp_path,
        generator=_RepairingGenerator(),
    )

    assert result.audit_report["counts"] == {"fail": 0, "pass": 1, "warn": 0}
    assert result.audit_report["repair_counts"] == {
        "attempted": 1,
        "fail": 0,
        "pass": 1,
        "warn": 0,
    }
    assert result.documents[0]["repair_attempted"] is True
    assert result.documents[0]["repair_original_audit_status"] == "fail"
    assert "rolling" in result.documents[0]["text"]
    assert (output_dir / "repair_checkpoints" / "round_01_batch_000000_n001.jsonl").exists()
    train_text = (output_dir / "train.txt").read_text(encoding="utf-8")
    assert "mat, rolling, turning" in train_text


def test_fireworks_mot_chi_renderer_carries_assigned_agent(tmp_path: Path) -> None:
    input_dir = tmp_path / "source"
    output_dir = tmp_path / "augmented"
    specs = [
        {
            "id": "mat-pass",
            "domain": "material-dynamics",
            "material": "foam",
            "required_materials": ["foam"],
            "required_affordances": ["tap", "compress"],
            "target_affordance": "tap",
        }
    ]
    source_docs = [{"id": "mat-pass", "text": "foam text", "audit_status": "pass"}]
    _write_jsonl(input_dir / "scenario_plans.jsonl", specs)
    _write_jsonl(input_dir / "documents_audited.jsonl", source_docs)

    result = build_document_augmentations(
        {
            "name": "toy_qwen_agent",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "renderer": {
                "type": "fireworks",
                "model_id": "fake-qwen",
                "documents_per_request": 1,
            },
            "styles": ["who_agent_qa"],
            "selection": {"include_audit_status": ["pass"]},
            "agent_pool": ["the gardener"],
            "audit": {"max_words": 90},
        },
        base_dir=tmp_path,
        generator=DeterministicMotChiMockGenerator(),
    )

    assert result.documents[0]["assigned_agent"] == "the gardener"
    assert "*CHI:\tthe gardener." in result.documents[0]["text"]
    assert "the gardener used the foam" in result.documents[0]["text"]
