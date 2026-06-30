from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from babysteps.data.text_mix import (
    FinalCorpusWriter,
    LineDocumentReader,
    RandomWithoutReplacementSampler,
    RegexWordCounter,
    RepeatThenPartialSampler,
    SeededDocumentShuffle,
    SelectedDocument,
    SourceDocument,
    TextMixBuilder,
    TextMixConfig,
    TextMixSourceConfig,
    build_text_mix,
)
from babysteps.paths import expand_env_vars


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def source_config(path: Path, *, target_words: int = 1, sampling: str = "random_without_replacement") -> TextMixSourceConfig:
    return TextMixSourceConfig(
        name="toy",
        role="base",
        path=path,
        target_words=target_words,
        sampling=sampling,
        path_text=str(path),
    )


def test_line_document_reader_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("first document\n\n  \nsecond document\n", encoding="utf-8")
    reader = LineDocumentReader(RegexWordCounter())

    docs = reader.read_documents(source_config(path))

    assert [doc.text for doc in docs] == ["first document", "second document"]
    assert [doc.line_number for doc in docs] == [1, 4]
    assert [doc.word_count for doc in docs] == [2, 2]


def test_repeat_then_partial_sampler_uses_full_repeats_before_partial(tmp_path) -> None:
    path = tmp_path / "synthetic.txt"
    source = source_config(
        path,
        target_words=23,
        sampling="repeat_then_partial",
    )
    docs = [
        SourceDocument("toy", "synthetic", str(path), 0, 1, "one two three four five", 5),
        SourceDocument("toy", "synthetic", str(path), 1, 2, "six seven eight nine ten", 5),
    ]

    sampled = RepeatThenPartialSampler().sample(docs, source=source, seed=42)

    assert sampled.full_repeats == 2
    assert sampled.partial_repeat_documents == 1
    assert sampled.selected_words == 25
    assert [doc.repeat_index for doc in sampled.documents] == [0, 0, 1, 1, 2]


def test_text_mix_is_deterministic_for_same_seed(tmp_path) -> None:
    base = tmp_path / "base.txt"
    synthetic = tmp_path / "synthetic.txt"
    base.write_text(
        "\n".join(
            [
                "alpha beta gamma",
                "delta epsilon zeta",
                "eta theta iota",
                "kappa lambda mu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    synthetic.write_text("red blue\n\ngreen yellow\n", encoding="utf-8")

    def make_config(output_dir: Path) -> dict[str, Any]:
        return {
            "name": "toy_mix",
            "seed": 42,
            "max_words": 15,
            "strict_max_words": True,
            "document_mode": "line",
            "final_order": "deterministic_document_shuffle",
            "trim_policy": "trim_final_document_only_if_needed",
            "output_dir": str(output_dir),
            "sources": [
                {
                    "name": "base",
                    "role": "base",
                    "path": str(base),
                    "target_words": 9,
                    "sampling": "random_without_replacement",
                },
                {
                    "name": "synthetic",
                    "role": "synthetic",
                    "path": str(synthetic),
                    "target_words": 4,
                    "sampling": "repeat_then_partial",
                },
            ],
        }

    first = build_text_mix(make_config(tmp_path / "out_a"), base_dir=ROOT)
    second = build_text_mix(make_config(tmp_path / "out_b"), base_dir=ROOT)

    assert first.counts["output_words_total"] == second.counts["output_words_total"]
    assert first.outputs["train"].read_text(encoding="utf-8") == second.outputs[
        "train"
    ].read_text(encoding="utf-8")


def test_final_writer_trims_only_final_combined_document(tmp_path) -> None:
    counter = RegexWordCounter()
    writer = FinalCorpusWriter(counter)
    docs = [
        SelectedDocument(
            SourceDocument("base", "base", "base.txt", 0, 1, "one two three four five six", 6),
            repeat_index=0,
            selection_index=0,
        ),
        SelectedDocument(
            SourceDocument(
                "synthetic",
                "synthetic",
                "synthetic.txt",
                0,
                1,
                "seven eight nine ten eleven twelve",
                6,
            ),
            repeat_index=0,
            selection_index=1,
        ),
    ]

    written = writer.write_train_text(
        docs,
        output_path=tmp_path / "train.txt",
        max_words=10,
        strict_max_words=True,
    )

    assert written.total_words == 10
    assert len(written.documents) == 2
    assert written.trimmed_document is not None
    assert written.trimmed_document.selected_document.document.source_name == "synthetic"
    assert (tmp_path / "train.txt").read_text(encoding="utf-8").splitlines() == [
        "one two three four five six",
        "seven eight nine ten",
    ]


def test_builder_components_are_swappable(tmp_path) -> None:
    path = tmp_path / "base.txt"
    path.write_text("one two three\nfour five six\n", encoding="utf-8")
    counter = RegexWordCounter()
    builder = TextMixBuilder(
        reader=LineDocumentReader(counter),
        samplers={"random_without_replacement": RandomWithoutReplacementSampler()},
        orderer=SeededDocumentShuffle(),
        writer=FinalCorpusWriter(counter),
    )
    config = TextMixConfig(
        name="toy",
        output_dir=tmp_path / "out",
        max_words=6,
        seed=42,
        strict_max_words=True,
        document_mode="line",
        final_order="deterministic_document_shuffle",
        trim_policy="trim_final_document_only_if_needed",
        word_pattern=counter.pattern,
        sources=(source_config(path, target_words=6),),
    )

    result = builder.build(config)

    assert result.counts["output_words_total"] == 6
    assert result.outputs["mix_manifest"].exists()
    assert result.outputs["counts"].exists()
    assert result.outputs["mix_documents"].exists()


def test_replace4pct_gpt_bert_tokenization_config_points_at_mixed_train(monkeypatch) -> None:
    monkeypatch.delenv("BABYSTEPS_GPT_BERT_BASELINE_MIXED_TOKENIZER_DIR", raising=False)
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")

    config = load_yaml(
        ROOT
        / "configs"
        / "tokenization"
        / "babylm2026_strict_small_replace4pct_phys2_mat2_gpt_bert_mixed_shards.yaml"
    )

    assert config["dataset"]["kind"] == "local_text"
    assert config["dataset"]["document_mode"] == "line"
    assert config["dataset"]["files"] == [
        "data/text_mixes/babylm2026_strict_small_replace4pct_phys2_mat2_seed42/train.txt"
    ]
    assert config["tokenizer"]["source"] == "/cache/tokenizers/gpt_bert_baseline_mixed"
    assert config["tokenizer"]["output_dir"] == "/cache/tokenizers/gpt_bert_baseline_mixed"
    assert (
        config["shards"]["output_dir"]
        == "/cache/token_shards/babylm2026_strict_small_replace4pct_phys2_mat2_gpt_bert_mixed"
    )


def test_replace4pct_train_config_matches_mixed_muon_baseline_except_identity(monkeypatch) -> None:
    monkeypatch.delenv("BABYSTEPS_GPT_BERT_BASELINE_MIXED_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_GPT_BERT_BASELINE_MIXED_SHARDS_DIR", raising=False)
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    baseline = load_yaml(
        ROOT / "configs" / "train" / "gpt_bert_baseline_mixed_babylm2026_strict_small_muon.yaml"
    )
    mixed = load_yaml(
        ROOT
        / "configs"
        / "train"
        / "gpt_bert_mixed_babylm2026_strict_small_replace4pct_phys2_mat2_muon.yaml"
    )

    assert mixed["run"]["name"] == "gpt_bert_mixed_babylm2026_strict_small_replace4pct_phys2_mat2_muon"
    assert (
        mixed["data"]["data_dir"]
        == "/cache/token_shards/babylm2026_strict_small_replace4pct_phys2_mat2_gpt_bert_mixed"
    )

    baseline["run"]["name"] = mixed["run"]["name"]
    baseline["data"]["data_dir"] = mixed["data"]["data_dir"]
    assert mixed == baseline
