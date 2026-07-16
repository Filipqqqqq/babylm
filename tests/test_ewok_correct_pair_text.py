from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "build_ewok_correct_pair_text.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_ewok_correct_pair_text", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_ewok_correct_pair_text_repeats_only_correct_pairs(tmp_path: Path) -> None:
    module = load_script_module()
    source = tmp_path / "physical-dynamics.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "Domain": "physical-dynamics",
                        "ConceptA": "fall",
                        "ConceptB": "rise",
                        "Context1": "The ball moves down.",
                        "Target1": "The ball is falling.",
                        "Context2": "The ball moves up.",
                        "Target2": "The ball is rising.",
                    }
                ),
                json.dumps(
                    {
                        "Domain": "physical-dynamics",
                        "ConceptA": "roll",
                        "ConceptB": "slide",
                        "Context1": "The speck is rotating.",
                        "Target1": "The wheel is rolling.",
                        "Context2": "The speck is not rotating.",
                        "Target2": "The wheel is sliding.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = module._read_jsonl(source)
    records = module.build_correct_pair_records(rows, repeat=4)
    output_dir = tmp_path / "out"
    manifest = module.write_outputs(
        input_path=source,
        output_dir=output_dir,
        repeat=4,
        records=records,
    )

    assert manifest["items_total"] == 2
    assert manifest["unique_correct_pairs"] == 4
    assert manifest["documents_total"] == 16
    assert manifest["pair_counts"] == {"fall/rise": 8, "roll/slide": 8}
    lines = (output_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    assert lines.count("The ball moves down. The ball is falling.") == 4
    assert lines.count("The ball moves up. The ball is rising.") == 4
    assert lines.count("The speck is rotating. The wheel is rolling.") == 4
    assert lines.count("The speck is not rotating. The wheel is sliding.") == 4
    assert "The ball moves down. The ball is rising." not in lines


def test_build_ewok_correct_pair_text_reads_directory_sources(tmp_path: Path) -> None:
    module = load_script_module()
    source_dir = tmp_path / "full_eval" / "ewok_filtered"
    source_dir.mkdir(parents=True)
    (source_dir / "agent-properties.jsonl").write_text(
        json.dumps(
            {
                "Domain": "agent-properties",
                "ConceptA": "know",
                "ConceptB": "forget",
                "Context1": "Mia remembered the code.",
                "Target1": "Mia knew the code.",
                "Context2": "Mia lost the code.",
                "Target2": "Mia forgot the code.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "physical-dynamics.jsonl").write_text(
        json.dumps(
            {
                "Domain": "physical-dynamics",
                "ConceptA": "fall",
                "ConceptB": "rise",
                "Context1": "The ball moves down.",
                "Target1": "The ball is falling.",
                "Context2": "The ball moves up.",
                "Target2": "The ball is rising.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = module._read_input_rows(source_dir)
    records = module.build_correct_pair_records(rows, repeat=2)
    output_dir = tmp_path / "out"
    manifest = module.write_outputs(
        input_path=source_dir,
        output_dir=output_dir,
        repeat=2,
        records=records,
    )

    assert manifest["items_total"] == 2
    assert manifest["unique_correct_pairs"] == 4
    assert manifest["documents_total"] == 8
    assert manifest["domain_counts"] == {
        "agent-properties": 4,
        "physical-dynamics": 4,
    }
    assert manifest["source_file_counts"] == {
        "agent-properties.jsonl": 4,
        "physical-dynamics.jsonl": 4,
    }
    documents = [
        json.loads(line)
        for line in (output_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {document["metadata"]["source"] for document in documents} == {"ewok_full"}
    assert {document["metadata"]["source_file"] for document in documents} == {
        "agent-properties.jsonl",
        "physical-dynamics.jsonl",
    }
