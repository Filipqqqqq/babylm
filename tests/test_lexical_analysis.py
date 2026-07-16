from __future__ import annotations

import csv
import json

import pytest

from babysteps.analysis.lexical.comps_targets import (
    build_concept_targets_payload,
    build_property_targets_payload,
    export_comps_targets,
    read_comps_targets,
    unsupported_rows as comps_unsupported_rows,
)
from babysteps.analysis.lexical.domain_frequency import (
    DomainConcept,
    build_domain_frequency_rows,
    write_domain_frequency_csv,
)
from babysteps.analysis.lexical.ewok_targets import (
    build_domain_concepts_payload,
    build_lexical_targets_payload,
    read_ewok_concepts,
    unsupported_rows,
)
from babysteps.analysis.lexical.pipeline import run_lexical_analysis
from babysteps.analysis.lexical.targets import TargetMatcher, parse_targets
from babysteps.analysis.lexical.text import RegexTextProcessor
from babysteps.analysis.lexical.tokenization import TokenizerAnalyzer


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[str]:
        self.calls += 1
        word = text.strip().lower()
        if word == "running":
            return ["run", "ning"]
        if word == "puppies":
            return ["pup", "pies"]
        if len(word) > 6:
            return [word[:3], word[3:]]
        return [word]


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_csv(path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_parse_targets_matches_surface_and_lemma() -> None:
    targets = parse_targets(
        {
            "version": 1,
            "targets": [
                {
                    "id": "dog",
                    "surfaces": ["dogs"],
                    "lemmas": ["dog"],
                }
            ],
        }
    )

    matcher = TargetMatcher(targets)

    assert matcher.match(surface="dogs", lemma="dog") == {"dog": {"surface", "lemma"}}
    assert matcher.match(surface="dog", lemma="dog") == {"dog": {"lemma"}}


def test_parse_targets_rejects_multiword_entries() -> None:
    with pytest.raises(ValueError, match="multiword"):
        parse_targets(
            {
                "version": 1,
                "targets": [
                    {
                        "id": "ice_cream",
                        "surfaces": ["ice cream"],
                    }
                ],
            }
        )


def test_ewok_target_export_payloads_keep_unsupported_concepts_visible(tmp_path) -> None:
    ewok_dir = tmp_path / "ewok_filtered"
    ewok_dir.mkdir()
    _write_jsonl(
        ewok_dir / "toy.jsonl",
        [
            {
                "Domain": "material-dynamics",
                "ConceptA": "tap",
                "ConceptB": "stir",
            },
            {
                "Domain": "physical-dynamics",
                "ConceptA": "slow down",
                "ConceptB": "rise",
            },
            {
                "Domain": "agent-properties",
                "ConceptA": "-",
                "ConceptB": "believe",
            },
        ],
    )

    summaries = read_ewok_concepts(ewok_dir)
    lexical_payload = build_lexical_targets_payload(summaries)
    domain_payload = build_domain_concepts_payload(summaries)
    unsupported = unsupported_rows(summaries)

    target_labels = {target["label"] for target in lexical_payload["targets"]}
    assert target_labels == {"believe", "rise", "stir", "tap"}

    physical_concepts = domain_payload["domains"]["physical-dynamics"]["concepts"]
    assert physical_concepts == ["rise", "slow down"]

    unsupported_by_concept = {row["concept"]: row for row in unsupported}
    assert unsupported_by_concept["slow down"]["reason"] == "multiword"
    assert unsupported_by_concept["-"]["reason"] == "placeholder"


def test_comps_target_export_payloads_aggregate_roles_and_properties(tmp_path) -> None:
    comps_dir = tmp_path / "comps"
    comps_dir.mkdir()
    _write_jsonl(
        comps_dir / "comps_base.jsonl",
        [
            {
                "id": 1,
                "property": "absorbs sweat",
                "acceptable_concept": "sock",
                "unacceptable_concept": "stocking",
                "property_phrase": "absorbs sweat.",
                "negative_sample_type": "taxonomic",
            },
            {
                "id": 2,
                "property": "absorbs sweat",
                "acceptable_concept": "sock",
                "unacceptable_concept": "ice cream",
                "property_phrase": "absorbs sweat.",
                "negative_sample_type": "overlap",
            },
        ],
    )
    _write_jsonl(
        comps_dir / "comps_wugs_dist-before.jsonl",
        [
            {
                "id": 3,
                "property": "attaches to rocks",
                "acceptable_concept": "mussel",
                "unacceptable_concept": "clam",
                "property_phrase": "attaches to rocks.",
                "negative_sample_type": "taxonomic",
                "distraction_type": "before",
            },
        ],
    )

    concept_summaries, property_summaries = read_comps_targets(comps_dir)
    concept_payload = build_concept_targets_payload(concept_summaries)
    property_payload = build_property_targets_payload(property_summaries)
    unsupported = comps_unsupported_rows(
        concept_summaries=concept_summaries,
        property_summaries=property_summaries,
    )

    targets_by_label = {
        target["label"]: target for target in concept_payload["targets"]
    }
    assert targets_by_label["sock"]["metadata"]["eval_count"] == 2
    assert targets_by_label["sock"]["metadata"]["concept_roles"] == {
        "acceptable_concept": 2
    }
    assert targets_by_label["sock"]["metadata"]["negative_sample_types"] == {
        "overlap": 1,
        "taxonomic": 1,
    }
    assert targets_by_label["ice cream"]["target_type"] == "phrase"

    properties_by_label = {
        target["label"]: target for target in property_payload["targets"]
    }
    assert properties_by_label["absorbs sweat"]["target_type"] == "phrase"
    assert properties_by_label["absorbs sweat"]["metadata"]["eval_count"] == 2
    assert properties_by_label["attaches to rocks"]["metadata"]["subtasks"] == {
        "wugs_dist_before": 1
    }
    assert unsupported == []

    concept_yaml = tmp_path / "concepts.yaml"
    property_yaml = tmp_path / "properties.yaml"
    unsupported_csv = tmp_path / "unsupported.csv"
    result = export_comps_targets(
        comps_dir=comps_dir,
        concept_targets_yaml=concept_yaml,
        property_targets_yaml=property_yaml,
        unsupported_csv=unsupported_csv,
    )

    assert result["concept_targets"] == 5
    assert result["property_targets"] == 2
    assert "comps_concept_sock" in concept_yaml.read_text(encoding="utf-8")
    assert "comps_property_absorbs_sweat" in property_yaml.read_text(encoding="utf-8")
    assert _read_csv(unsupported_csv) == []


def test_v2_targets_match_words_and_phrases_without_breaking_v1() -> None:
    targets = parse_targets(
        {
            "version": 2,
            "targets": [
                {
                    "id": "dog",
                    "target_type": "word",
                    "surfaces": ["dog"],
                    "lemmas": ["dog"],
                },
                {
                    "id": "dog_runs",
                    "target_type": "phrase",
                    "surfaces": ["dog runs"],
                    "lemmas": ["dog runs"],
                },
            ],
        }
    )
    sentence = RegexTextProcessor().process("The dog runs fast.")[0]

    matcher = TargetMatcher(targets)
    matches = matcher.match_sentence(sentence.tokens)

    assert matcher.match(surface="dog", lemma="dog") == {"dog": {"surface", "lemma"}}
    assert [
        (match.target_id, match.start_index, match.end_index, set(match.match_kinds))
        for match in matches
    ] == [
        ("dog", 1, 2, {"surface", "lemma"}),
        ("dog_runs", 1, 3, {"surface", "lemma"}),
    ]


def test_tokenizer_analyzer_caches_word_types() -> None:
    tokenizer = FakeTokenizer()
    analyzer = TokenizerAnalyzer(tokenizer)

    assert analyzer.analyze("running").pieces == ("run", "ning")
    assert analyzer.analyze("running").fertility == 2
    assert tokenizer.calls == 1


def test_domain_frequency_rows_sort_by_frequency_and_mark_nan(tmp_path) -> None:
    rows = build_domain_frequency_rows(
        [
            DomainConcept("animals", "dog", "dog", False),
            DomainConcept("animals", "ice cream", "ice cream", True),
            DomainConcept("animals", "cat", "cat", False),
            DomainConcept("actions", "run", "run", False),
            DomainConcept("actions", "jump", "jump", False),
        ],
        {"dog": 12.0, "cat": 3.0, "run": 7.0},
    )

    assert [(row["domain"], row["concept"], row["status"]) for row in rows] == [
        ("actions", "run", "counted"),
        ("actions", "jump", "missing_from_target_coverage"),
        ("animals", "dog", "counted"),
        ("animals", "cat", "counted"),
        ("animals", "ice cream", "multiword_unsupported"),
    ]

    output_csv = tmp_path / "domain_frequencies.csv"
    write_domain_frequency_csv(rows, output_csv)

    csv_rows = _read_csv(output_csv)
    assert csv_rows[1]["frequency"] == "NaN"
    assert csv_rows[-1]["frequency"] == "NaN"


def test_run_lexical_analysis_writes_target_and_context_reports(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "source": "story",
                "text": "The dogs are running. A puppy ran fast.",
            },
            {
                "source": "dialogue",
                "text": "The car and cars stopped by the car. The dogs run.",
            },
        ],
    )
    targets_path = tmp_path / "targets.yaml"
    targets_path.write_text(
        """
version: 1
targets:
  - id: dog
    surfaces: [dog, dogs, puppy, puppies]
    lemmas: [dog, puppy]
  - id: run
    surfaces: [run, running, ran]
    lemmas: [run]
  - id: car
    surfaces: [car, cars]
    lemmas: [car]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "reports"
    config = {
        "name": "toy_lexical",
        "dataset": {
            "kind": "local_jsonl",
            "files": str(corpus_path),
            "text_field": "text",
        },
        "tokenizer": {"path": "fake-tokenizer"},
        "targets": {"path": str(targets_path)},
        "lemmatizer": {"kind": "regex"},
        "analysis": {
            "source_field": "source",
            "examples_per_target": 2,
            "neighbor_window": 2,
            "context_word_radius": 3,
            "seed": 7,
            "tokenizer_mode": "isolated",
        },
        "output": {"timestamp": False},
    }

    result = run_lexical_analysis(
        config,
        root_dir=tmp_path,
        output_dir=output_dir,
        tokenizer=FakeTokenizer(),
        text_processor=RegexTextProcessor(),
    )

    assert result.output_dir == output_dir
    assert result.manifest["docs_processed"] == 2
    assert result.manifest["word_tokens"] == 19

    target_rows = {row["target_id"]: row for row in _read_csv(output_dir / "target_coverage.csv")}
    assert target_rows["dog"]["total_mentions"] == "3"
    assert target_rows["run"]["total_mentions"] == "3"
    assert target_rows["car"]["total_mentions"] == "3"
    assert target_rows["dog"]["exposure_bucket"] == "1-5"
    assert "dogs:2" in target_rows["dog"]["top_surfaces"]

    threshold_rows = {row["bucket"]: row for row in _read_csv(output_dir / "exposure_thresholds.csv")}
    assert threshold_rows["1-5"]["target_count"] == "3"

    context_rows = {row["target_id"]: row for row in _read_csv(output_dir / "context_summary.csv")}
    assert "the:2" in context_rows["dog"]["top_left_neighbors"]
    assert "the:" not in context_rows["dog"]["top_left_content_neighbors"]
    assert "are:" not in context_rows["dog"]["top_right_content_neighbors"]
    assert "running:1" in context_rows["dog"]["top_right_content_neighbors"]
    assert "story:2" in context_rows["dog"]["top_sources"]

    examples = _read_jsonl(output_dir / "context_examples.jsonl")
    example_counts = {}
    for row in examples:
        example_counts[row["target_id"]] = example_counts.get(row["target_id"], 0) + 1
        assert "sentence" in row
        assert "token_pieces" in row
        assert "left_window_words" in row
        assert "matched_word" in row
        assert "right_window_words" in row
        assert row["context_word_radius"] == 3
        assert row["example_sampling_unit"] == "sentence"
        assert "sentence_target_occurrences" in row
        assert "sentence_target_surface_counts" in row
    assert example_counts == {"dog": 2, "run": 2, "car": 1}
    car_examples = [row for row in examples if row["target_id"] == "car"]
    assert car_examples[0]["sentence_target_occurrences"] == 3
    assert car_examples[0]["sentence_target_surface_counts"] == {"car": 2, "cars": 1}
    dog_examples = [row for row in examples if row["target_id"] == "dog"]
    assert any(row["matched_word"] == "dogs" for row in dog_examples)
    assert any(row["left_window_words"] == "The" for row in dog_examples)

    summary = json.loads((output_dir / "global_tokenizer_summary.json").read_text(encoding="utf-8"))
    assert summary["total_word_tokens"] == 19
    assert summary["unique_word_types"] == 14
    assert summary["mean_subwords_per_occurrence"] > 1.0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["target_coverage"].endswith("target_coverage.csv")
    assert manifest["analysis"]["filter_context_neighbor_stopwords"] is True
    assert manifest["analysis"]["context_neighbor_stopword_count"] > 0
    assert manifest["analysis"]["progress_logging"] is True
    assert manifest["analysis"]["progress_interval_docs"] == 10000


def test_run_lexical_analysis_writes_phrase_target_reports(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "source": "comps-smoke",
                "text": "A sock absorbs sweat. The sock absorbs sweat quickly.",
            },
            {
                "source": "comps-smoke",
                "text": "Sweat absorbs water.",
            },
        ],
    )
    targets_path = tmp_path / "targets.yaml"
    targets_path.write_text(
        """
version: 2
targets:
  - id: absorbs_sweat
    label: absorbs sweat
    target_type: phrase
    surfaces: [absorbs sweat]
    lemmas: [absorbs sweat]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "reports"
    config = {
        "name": "toy_phrase_lexical",
        "dataset": {
            "kind": "local_jsonl",
            "files": str(corpus_path),
            "text_field": "text",
        },
        "tokenizer": {"path": "fake-tokenizer"},
        "targets": {"path": str(targets_path)},
        "lemmatizer": {"kind": "regex"},
        "analysis": {
            "source_field": "source",
            "examples_per_target": 5,
            "neighbor_window": 2,
            "context_word_radius": 3,
            "seed": 7,
            "tokenizer_mode": "isolated",
        },
        "output": {"timestamp": False},
    }

    run_lexical_analysis(
        config,
        root_dir=tmp_path,
        output_dir=output_dir,
        tokenizer=FakeTokenizer(),
        text_processor=RegexTextProcessor(),
    )

    target_rows = {row["target_id"]: row for row in _read_csv(output_dir / "target_coverage.csv")}
    assert target_rows["absorbs_sweat"]["total_mentions"] == "2"
    assert target_rows["absorbs_sweat"]["top_surfaces"] == "absorbs sweat:2"
    assert target_rows["absorbs_sweat"]["mean_subwords_per_occurrence"] == "3.0000"

    context_rows = {row["target_id"]: row for row in _read_csv(output_dir / "context_summary.csv")}
    assert context_rows["absorbs_sweat"]["top_left_neighbors"] == "sock:2; a:1; the:1"
    assert "quickly:1" in context_rows["absorbs_sweat"]["top_right_neighbors"]

    examples = _read_jsonl(output_dir / "context_examples.jsonl")
    assert len(examples) == 2
    assert {row["matched_phrase"] for row in examples} == {"absorbs sweat"}
    assert {row["matched_word"] for row in examples} == {"absorbs sweat"}
    assert all(row["token_pieces"] == ["abs", "orbs", "sweat"] for row in examples)
    assert all(row["sentence_target_surface_counts"] == {"absorbs sweat": 1} for row in examples)
