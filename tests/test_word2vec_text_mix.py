from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "analysis" / "word2vec" / "build_text_mix.py"


def load_text_mix_module():
    spec = importlib.util.spec_from_file_location("build_text_mix", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_args(**overrides):
    defaults = {
        "base_dir": None,
        "base_text": None,
        "base_glob": "*",
        "synthetic_text": None,
        "synthetic_word_ratio": None,
        "synthetic_target_words": None,
        "synthetic_repeat": None,
        "output_dir": None,
        "name": None,
        "eval": "physical_dynamics=toy.jsonl",
        "seed": 13,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_text_mix_repeats_synthetic_to_ratio(tmp_path) -> None:
    module = load_text_mix_module()
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "b.train").write_text("six seven eight nine ten\n", encoding="utf-8")
    (base_dir / "a.train").write_text("one two three four five\n", encoding="utf-8")
    synthetic = tmp_path / "synthetic.txt"
    synthetic.write_text("speed up\nslow down\nfall rise\n", encoding="utf-8")
    output_dir = tmp_path / "mix"

    manifest = module.build_text_mix(
        make_args(
            base_dir=str(base_dir),
            base_glob="*.train",
            synthetic_text=str(synthetic),
            synthetic_word_ratio=0.5,
            output_dir=str(output_dir),
            name="toy_mix",
        )
    )

    repeated = output_dir / "synthetic_repeated.txt"
    assert manifest["base_word_count"] == 10
    assert manifest["synthetic_word_count"] == 6
    assert manifest["repeat_count"] == 1
    assert manifest["effective_synthetic_words"] == 6
    assert manifest["effective_synthetic_to_base_ratio"] == 0.6
    assert repeated.read_text(encoding="utf-8") == "speed up\nslow down\nfall rise\n"
    assert (output_dir / "base_files.txt").read_text(encoding="utf-8").splitlines() == [
        str(base_dir / "a.train"),
        str(base_dir / "b.train"),
    ]
    saved = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved["name"] == "toy_mix"
    assert saved["outputs"]["synthetic_repeated"] == str(repeated)
    command = (output_dir / "w2v_command.sh").read_text(encoding="utf-8")
    assert "--train-text" in command
    assert "--epochs 5" in command
    assert "--workers 4" in command
    assert "toy_mix_seed13" in command


def test_build_text_mix_supports_fixed_repeat_and_base_files(tmp_path) -> None:
    module = load_text_mix_module()
    base_a = tmp_path / "a.txt"
    base_b = tmp_path / "b.txt"
    base_a.write_text("alpha beta\n", encoding="utf-8")
    base_b.write_text("gamma delta\n", encoding="utf-8")
    synthetic = tmp_path / "synthetic.txt"
    synthetic.write_text("fall rise", encoding="utf-8")
    output_dir = tmp_path / "mix_repeat"

    manifest = module.build_text_mix(
        make_args(
            base_text=[str(base_b), str(base_a)],
            synthetic_text=str(synthetic),
            synthetic_repeat=3,
            output_dir=str(output_dir),
            seed=23,
        )
    )

    assert manifest["base_files"] == [str(base_b), str(base_a)]
    assert manifest["base_word_count"] == 4
    assert manifest["synthetic_word_count"] == 2
    assert manifest["repeat_count"] == 3
    assert manifest["effective_synthetic_words"] == 6
    assert (output_dir / "synthetic_repeated.txt").read_text(encoding="utf-8") == (
        "fall rise\nfall rise\nfall rise\n"
    )
    assert "--seed 23" in (output_dir / "w2v_command.sh").read_text(encoding="utf-8")
