"""Lightweight audits for material-world synthetic prose JSONL files."""

from __future__ import annotations

import json
import math
import re
import statistics
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "material_world" / "audit_config.json"
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


@dataclass(frozen=True)
class AuditResult:
    audited_rows: list[dict[str, Any]]
    report: dict[str, Any]
    drop_ids: list[str]


def load_audit_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_text(text: str) -> str:
    text = text.lower().translate(PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def nested_get(row: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def metadata_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    return nested_get(row, f"metadata.{key}", row.get(key, default))


def row_id(row: dict[str, Any], index: int) -> str:
    value = row.get("id", row.get("example_id", index))
    return str(value)


def phrase_count(text_lower: str, phrase: str) -> int:
    escaped = re.escape(phrase.lower())
    if re.fullmatch(r"[a-z0-9']+", phrase.lower()):
        pattern = rf"\b{escaped}\b"
    else:
        pattern = rf"(?<!\w){escaped}(?!\w)"
    return len(re.findall(pattern, text_lower))


def first_n_words(norm_text: str, n: int) -> str:
    return " ".join(norm_text.split()[:n])


def length_bucket(word_count: int, buckets: dict[str, list[int]]) -> str:
    for name, (low, high) in buckets.items():
        if low <= word_count <= high:
            return name
    if word_count < min(low for low, _high in buckets.values()):
        return "below_short"
    if word_count > max(high for _low, high in buckets.values()):
        return "above_xlong"
    return "between_targets"


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[,;/|]", value) if part.strip()]
        return parts or [value]
    return [str(value)]


def add_flag(
    flags: dict[str, list[dict[str, str]]],
    example_id: str,
    severity: str,
    code: str,
    message: str,
) -> None:
    flags[example_id].append({"severity": severity, "code": code, "message": message})


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def top_counter(counter: Counter[str], limit: int = 25, total: int | None = None) -> list[dict[str, Any]]:
    denominator = total if total is not None else sum(counter.values())
    return [
        {"value": value, "count": count, "rate": rate(count, denominator)}
        for value, count in counter.most_common(limit)
    ]


def ngrams(items: list[str], n: int) -> list[str]:
    if len(items) < n:
        return []
    return [" ".join(items[i : i + n]) for i in range(len(items) - n + 1)]


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def row_contains_negative(row: dict[str, Any], text_lower: str, phrases: list[str]) -> bool:
    metadata_value_raw = metadata_value(row, "contains_negative_reporting")
    if isinstance(metadata_value_raw, bool):
        return metadata_value_raw
    return any(phrase_count(text_lower, phrase) > 0 for phrase in phrases)


def audit_rows(rows: list[dict[str, Any]], config: dict[str, Any] | None = None) -> AuditResult:
    config = config or load_audit_config()
    total = len(rows)
    ids = [row_id(row, i) for i, row in enumerate(rows)]
    flags: dict[str, list[dict[str, str]]] = defaultdict(list)

    texts = [str(row.get("text", "")) for row in rows]
    lower_texts = [text.lower() for text in texts]
    norm_texts = [normalize_text(text) for text in texts]
    tokenized = [norm.split() for norm in norm_texts]
    word_counts = [len(items) for items in tokenized]
    word_sets = [set(items) for items in tokenized]

    total_words = sum(word_counts)
    opening5 = Counter(first_n_words(norm, 5) for norm in norm_texts if norm)
    opening8 = Counter(first_n_words(norm, 8) for norm in norm_texts if norm)
    opening12 = Counter(first_n_words(norm, 12) for norm in norm_texts if norm)

    opening_config = config["openings"]
    opening_summary: dict[str, Any] = {
        "top_first_5": top_counter(opening5, total=total),
        "top_first_8": top_counter(opening8, total=total),
        "watched_prefix_hits": {},
    }
    for prefix in opening_config.get("watched_prefixes", []):
        prefix_norm = normalize_text(prefix)
        hit_ids = [ids[i] for i, norm in enumerate(norm_texts) if norm.startswith(prefix_norm)]
        opening_summary["watched_prefix_hits"][prefix] = {
            "count": len(hit_ids),
            "rate": rate(len(hit_ids), total),
            "sample_ids": hit_ids[:10],
        }

    min_opening_total = int(opening_config.get("min_corpus_examples_for_rate_flags", 1))
    if total >= min_opening_total:
        for n, counter, warning_key in [
            (5, opening5, "first_5_warning_rate"),
            (8, opening8, "first_8_warning_rate"),
        ]:
            warning_rate = float(opening_config[warning_key])
            hard_rate = float(opening_config["hard_rate"])
            min_group_count = int(opening_config.get("min_group_count", 2))
            for opening, count in counter.items():
                if count < min_group_count:
                    continue
                opening_rate = rate(count, total)
                if opening_rate <= warning_rate:
                    continue
                severity = "severe_warning" if opening_rate > hard_rate else "warning"
                code = f"repetitive_opening_{n}"
                for i, norm in enumerate(norm_texts):
                    if first_n_words(norm, n) == opening:
                        add_flag(
                            flags,
                            ids[i],
                            severity,
                            code,
                            f"Opening '{opening}' appears in {opening_rate:.1%} of examples.",
                        )

    connective_config = config["connectives"]
    connective_phrases = connective_config["phrases"]
    connective_counts: dict[str, int] = {}
    connective_example_counts: dict[str, int] = {}
    marker_counts_per_example = []
    for i, text_lower in enumerate(lower_texts):
        marker_count = 0
        for phrase in connective_phrases:
            count = phrase_count(text_lower, phrase)
            connective_counts[phrase] = connective_counts.get(phrase, 0) + count
            if count:
                connective_example_counts[phrase] = connective_example_counts.get(phrase, 0) + 1
                marker_count += count
        marker_counts_per_example.append(marker_count)
        if marker_count >= int(connective_config["per_example_marker_warning_count"]):
            add_flag(
                flags,
                ids[i],
                "warning",
                "many_connectives",
                f"Contains {marker_count} causal/contrast markers.",
            )

    because_rate = rate(connective_example_counts.get("because", 0), total)
    if because_rate > float(connective_config["because_example_rate_warning"]):
        for i, text_lower in enumerate(lower_texts):
            if phrase_count(text_lower, "because"):
                add_flag(
                    flags,
                    ids[i],
                    "warning",
                    "because_overuse",
                    f"'because' appears in {because_rate:.1%} of examples.",
                )

    contrast_phrases = connective_config["contrast_phrases"]
    contrast_example_ids = {
        ids[i]
        for i, text_lower in enumerate(lower_texts)
        if any(phrase_count(text_lower, phrase) > 0 for phrase in contrast_phrases)
    }
    contrast_rate = rate(len(contrast_example_ids), total)
    if contrast_rate > float(connective_config["contrast_combined_example_rate_warning"]):
        for example_id in contrast_example_ids:
            add_flag(
                flags,
                example_id,
                "warning",
                "contrast_phrase_overuse",
                f"Explicit contrast phrases appear in {contrast_rate:.1%} of examples.",
            )

    style_config = config["styles"]
    style_aliases = style_config.get("aliases", {})
    raw_style_counts = Counter(str(metadata_value(row, "style_id", "missing")) for row in rows)
    style_counts = Counter(style_aliases.get(style, style) for style in raw_style_counts.elements())
    style_distribution = {
        style: {"count": count, "rate": rate(count, total)}
        for style, count in sorted(style_counts.items())
    }
    raw_style_distribution = {
        style: {"count": count, "rate": rate(count, total)}
        for style, count in sorted(raw_style_counts.items())
    }
    style_warnings = []
    if total >= int(style_config.get("min_corpus_examples_for_balance_warnings", 1)):
        styles_to_check = set(style_counts) | set(style_config["targets"])
        for style in sorted(styles_to_check):
            count = style_counts.get(style, 0)
            style_rate = rate(count, total)
            target = style_config["targets"].get(style)
            if (
                target is not None
                and abs(style_rate - float(target)) > float(style_config["target_tolerance_points"])
            ):
                style_warnings.append(
                    {
                        "style_id": style,
                        "rate": style_rate,
                        "target": target,
                        "message": f"{style} is {style_rate:.1%}, target is {target:.1%}.",
                    }
                )
            if style_rate < float(style_config["min_rate_warning"]) or style_rate > float(style_config["max_rate_warning"]):
                style_warnings.append(
                    {
                        "style_id": style,
                        "rate": style_rate,
                        "target": target,
                        "message": f"{style} is outside broad style bounds.",
                    }
                )

    negative_config = config["negative_reporting"]
    contains_negative = [
        row_contains_negative(row, lower_texts[i], negative_config["phrases"])
        for i, row in enumerate(rows)
    ]
    negative_count = sum(contains_negative)
    negative_rate = rate(negative_count, total)
    negative_warning = None
    if negative_rate < float(negative_config["warning_min_rate"]):
        negative_warning = f"Negative/contrastive reporting is low at {negative_rate:.1%}."
    elif negative_rate > float(negative_config["warning_max_rate"]):
        negative_warning = f"Negative/contrastive reporting is high at {negative_rate:.1%}."

    length_config = config["lengths"]
    length_buckets = [length_bucket(count, length_config["buckets"]) for count in word_counts]
    length_bucket_counts = Counter(length_buckets)
    length_distribution = {
        bucket: {"count": count, "rate": rate(count, total)}
        for bucket, count in sorted(length_bucket_counts.items())
    }
    length_warnings = []
    if length_bucket_counts:
        bucket, count = length_bucket_counts.most_common(1)[0]
        if rate(count, total) > float(length_config["single_bucket_warning_rate"]):
            length_warnings.append(f"{bucket} contains {rate(count, total):.1%} of examples.")
    too_short_ids = [ids[i] for i, count in enumerate(word_counts) if count < int(length_config["too_short_words"])]
    too_long_ids = [ids[i] for i, count in enumerate(word_counts) if count > int(length_config["too_long_words"])]
    if too_short_ids:
        length_warnings.append(f"{len(too_short_ids)} examples are below {length_config['too_short_words']} words.")
        for example_id in too_short_ids:
            add_flag(flags, example_id, "warning", "too_short", "Example is below target minimum length.")
    if too_long_ids:
        length_warnings.append(f"{len(too_long_ids)} examples are above {length_config['too_long_words']} words.")
        for example_id in too_long_ids:
            add_flag(flags, example_id, "warning", "too_long", "Example is above target maximum length.")

    lengths_by_style: dict[str, list[int]] = defaultdict(list)
    buckets_by_style: dict[str, Counter[str]] = defaultdict(Counter)
    for i, row in enumerate(rows):
        style = str(metadata_value(row, "style_id", "missing"))
        lengths_by_style[style].append(word_counts[i])
        buckets_by_style[style][length_buckets[i]] += 1
    style_length_stats = {}
    for style, counts in lengths_by_style.items():
        std = statistics.pstdev(counts) if len(counts) > 1 else 0.0
        style_length_stats[style] = {
            "count": len(counts),
            "mean_words": statistics.mean(counts),
            "std_words": std,
            "buckets": {
                bucket: {"count": count, "rate": rate(count, len(counts))}
                for bucket, count in sorted(buckets_by_style[style].items())
            },
        }
        if (
            len(counts) >= int(length_config["style_length_variance_min_examples"])
            and std < float(length_config["style_length_std_warning"])
        ):
            length_warnings.append(f"{style} has narrow length variance: std={std:.1f} words.")

    dedupe_config = config["dedupe"]
    exact_seen: dict[str, str] = {}
    exact_duplicate_pairs = []
    for i, norm in enumerate(norm_texts):
        if not norm:
            continue
        if norm in exact_seen:
            exact_duplicate_pairs.append({"kept_id": exact_seen[norm], "drop_id": ids[i]})
            add_flag(flags, ids[i], "hard_drop", "exact_duplicate", f"Exact duplicate of {exact_seen[norm]}.")
        else:
            exact_seen[norm] = ids[i]

    repeated_first12_groups = [
        {"opening": opening, "count": count, "rate": rate(count, total)}
        for opening, count in opening12.most_common()
        if count >= int(dedupe_config["repeated_first_12_warning_count"])
    ]
    for group in repeated_first12_groups:
        severity = (
            "severe_warning"
            if group["count"] >= int(dedupe_config["repeated_first_12_hard_count"])
            else "warning"
        )
        for i, norm in enumerate(norm_texts):
            if first_n_words(norm, 12) == group["opening"]:
                add_flag(
                    flags,
                    ids[i],
                    severity,
                    "repeated_first_12_words",
                    f"First 12 words repeat across {group['count']} examples.",
                )

    eight_gram_counts = Counter()
    eight_gram_to_ids: dict[str, set[str]] = defaultdict(set)
    for i, items in enumerate(tokenized):
        for gram in set(ngrams(items, 8)):
            eight_gram_counts[gram] += 1
            eight_gram_to_ids[gram].add(ids[i])
    repeated_eight_grams = [
        {"ngram": gram, "count": count, "sample_ids": sorted(eight_gram_to_ids[gram])[:10]}
        for gram, count in eight_gram_counts.most_common(25)
        if count >= int(dedupe_config["eight_gram_warning_count"])
    ]
    for item in repeated_eight_grams:
        for example_id in item["sample_ids"]:
            add_flag(
                flags,
                example_id,
                "warning",
                "repeated_8gram",
                f"Shares repeated 8-gram '{item['ngram']}'.",
            )

    near_duplicate_pairs = []
    max_all_pairs = int(dedupe_config["max_examples_for_all_pairs"])
    if total <= max_all_pairs:
        warning_threshold = float(dedupe_config["jaccard_warning_threshold"])
        hard_threshold = float(dedupe_config["jaccard_hard_drop_threshold"])
        min_words = int(dedupe_config["min_words_for_jaccard"])
        length_ratio_min = float(dedupe_config["length_ratio_min"])
        hard_dropped_for_similarity: set[str] = set()
        for i in range(total):
            if len(word_sets[i]) < min_words:
                continue
            for j in range(i + 1, total):
                if len(word_sets[j]) < min_words:
                    continue
                small = min(len(word_sets[i]), len(word_sets[j]))
                large = max(len(word_sets[i]), len(word_sets[j]))
                if large and small / large < length_ratio_min:
                    continue
                score = jaccard(word_sets[i], word_sets[j])
                if score < warning_threshold:
                    continue
                severity = "hard_drop" if score >= hard_threshold else "warning"
                pair = {
                    "id_a": ids[i],
                    "id_b": ids[j],
                    "jaccard": score,
                    "severity": severity,
                }
                near_duplicate_pairs.append(pair)
                if severity == "hard_drop" and ids[j] not in hard_dropped_for_similarity:
                    add_flag(
                        flags,
                        ids[j],
                        "hard_drop",
                        "near_duplicate",
                        f"High word-set Jaccard similarity to {ids[i]}: {score:.2f}.",
                    )
                    hard_dropped_for_similarity.add(ids[j])
                elif severity == "warning":
                    add_flag(
                        flags,
                        ids[i],
                        "warning",
                        "near_duplicate_candidate",
                        f"High word-set Jaccard similarity to {ids[j]}: {score:.2f}.",
                    )
                    add_flag(
                        flags,
                        ids[j],
                        "warning",
                        "near_duplicate_candidate",
                        f"High word-set Jaccard similarity to {ids[i]}: {score:.2f}.",
                    )

    collapse_config = config["collapse"]
    material_counts = Counter(str(metadata_value(row, "material", "missing")) for row in rows)
    material_class_counts = Counter(str(metadata_value(row, "material_class", "missing")) for row in rows)
    object_counts = Counter(str(metadata_value(row, "object", "missing")) for row in rows)
    material_dynamic_counts: dict[str, Counter[str]] = defaultdict(Counter)
    material_class_dynamic_counts: dict[str, Counter[str]] = defaultdict(Counter)
    object_dynamic_counts: dict[str, Counter[str]] = defaultdict(Counter)
    material_object_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        material = str(metadata_value(row, "material", "missing"))
        material_class = str(metadata_value(row, "material_class", "missing"))
        obj = str(metadata_value(row, "object", "missing"))
        dynamics = set(as_list(metadata_value(row, "dynamics")))
        material_object_counts[(material, obj)] += 1
        for dynamic in dynamics:
            material_dynamic_counts[material][dynamic] += 1
            material_class_dynamic_counts[material_class][dynamic] += 1
            object_dynamic_counts[obj][dynamic] += 1

    material_dynamic_warnings = []
    min_material_examples = int(collapse_config["min_examples_per_material"])
    dominant_dynamic_rate = float(collapse_config["dominant_dynamic_rate_warning"])
    for material, counter in material_dynamic_counts.items():
        material_total = material_counts[material]
        if material_total < min_material_examples or not counter:
            continue
        dynamic, count = counter.most_common(1)[0]
        dynamic_rate = count / material_total
        if dynamic_rate > dominant_dynamic_rate:
            material_dynamic_warnings.append(
                {
                    "material": material,
                    "dynamic": dynamic,
                    "count": count,
                    "material_examples": material_total,
                    "rate": dynamic_rate,
                }
            )
            for i, row in enumerate(rows):
                if (
                    str(metadata_value(row, "material", "missing")) == material
                    and dynamic in set(as_list(metadata_value(row, "dynamics")))
                ):
                    add_flag(
                        flags,
                        ids[i],
                        "warning",
                        "material_dynamic_collapse",
                        f"{material} examples are dominated by dynamic '{dynamic}' ({dynamic_rate:.1%}).",
                    )

    template_config = config["template_phrases"]
    template_phrase_counts: dict[str, int] = {}
    template_phrase_example_counts: dict[str, int] = {}
    for phrase in template_config["phrases"]:
        for text_lower in lower_texts:
            count = phrase_count(text_lower, phrase)
            template_phrase_counts[phrase] = template_phrase_counts.get(phrase, 0) + count
            if count:
                template_phrase_example_counts[phrase] = template_phrase_example_counts.get(phrase, 0) + 1
    for phrase, count in template_phrase_example_counts.items():
        phrase_rate = rate(count, total)
        if phrase_rate > float(template_config["phrase_rate_warning"]):
            for i, text_lower in enumerate(lower_texts):
                if phrase_count(text_lower, phrase):
                    add_flag(
                        flags,
                        ids[i],
                        "warning",
                        "template_phrase_overuse",
                        f"Phrase '{phrase}' appears in {phrase_rate:.1%} of examples.",
                    )

    ngram_n = int(template_config["top_ngram_n"])
    ngram_counter = Counter()
    for items in tokenized:
        ngram_counter.update(ngrams(items, ngram_n))

    audited_rows = []
    drop_ids = []
    warned = 0
    dropped = 0
    for i, row in enumerate(rows):
        example_id = ids[i]
        row_flags = flags.get(example_id, [])
        status = "drop" if any(flag["severity"] == "hard_drop" for flag in row_flags) else "warn" if row_flags else "keep"
        if status == "warn":
            warned += 1
        elif status == "drop":
            dropped += 1
            drop_ids.append(example_id)
        audited = dict(row)
        audited["audit_flags"] = row_flags
        audited["audit_status"] = status
        audited_rows.append(audited)

    recommendations = build_recommendations(
        style_warnings=style_warnings,
        because_rate=because_rate,
        contrast_rate=contrast_rate,
        negative_rate=negative_rate,
        negative_config=negative_config,
        opening_summary=opening_summary,
        material_dynamic_warnings=material_dynamic_warnings,
        length_warnings=length_warnings,
    )

    report = {
        "version": config.get("version", "material_world_audit"),
        "total_examples": total,
        "total_words": total_words,
        "examples_kept": total - warned - dropped,
        "examples_warned": warned,
        "examples_dropped": dropped,
        "style_distribution": style_distribution,
        "raw_style_distribution": raw_style_distribution,
        "style_warnings": style_warnings,
        "length_distribution": length_distribution,
        "length_by_style": style_length_stats,
        "length_warnings": length_warnings,
        "top_openings": opening_summary,
        "connective_rates": {
            phrase: {
                "total_count": connective_counts.get(phrase, 0),
                "example_count": connective_example_counts.get(phrase, 0),
                "example_rate": rate(connective_example_counts.get(phrase, 0), total),
            }
            for phrase in connective_phrases
        },
        "connective_summary": {
            "because_example_rate": because_rate,
            "contrast_combined_example_rate": contrast_rate,
            "examples_with_3_plus_markers": sum(
                count >= int(connective_config["per_example_marker_warning_count"])
                for count in marker_counts_per_example
            ),
        },
        "negative_reporting": {
            "count": negative_count,
            "rate": negative_rate,
            "warning": negative_warning,
            "target_rate": [
                negative_config["target_min_rate"],
                negative_config["target_max_rate"],
            ],
        },
        "dedupe": {
            "exact_duplicate_count": len(exact_duplicate_pairs),
            "exact_duplicate_pairs": exact_duplicate_pairs[:100],
            "near_duplicate_count": len(near_duplicate_pairs),
            "near_duplicate_pairs": near_duplicate_pairs[:100],
            "repeated_first_12_groups": repeated_first12_groups[:50],
            "repeated_8grams": repeated_eight_grams[:50],
            "jaccard_all_pairs_skipped": total > int(dedupe_config["max_examples_for_all_pairs"]),
        },
        "material_dynamic_summary": {
            "top_material_dynamic_pairs": top_pair_counters(material_dynamic_counts, material_counts),
            "top_material_class_dynamic_pairs": top_pair_counters(
                material_class_dynamic_counts, material_class_counts
            ),
            "top_object_dynamic_pairs": top_pair_counters(object_dynamic_counts, object_counts),
            "top_material_object_pairs": [
                {
                    "material": material,
                    "object": obj,
                    "count": count,
                    "rate_of_material": count / material_counts[material] if material_counts[material] else 0.0,
                }
                for (material, obj), count in material_object_counts.most_common(25)
            ],
            "warnings": material_dynamic_warnings,
        },
        "template_phrase_rates": {
            phrase: {
                "total_count": template_phrase_counts.get(phrase, 0),
                "example_count": template_phrase_example_counts.get(phrase, 0),
                "example_rate": rate(template_phrase_example_counts.get(phrase, 0), total),
            }
            for phrase in template_config["phrases"]
        },
        "top_repeated_ngrams": top_counter(
            ngram_counter,
            limit=int(template_config["top_ngram_limit"]),
            total=max(1, sum(ngram_counter.values())),
        ),
        "drop_ids": drop_ids,
        "recommendations": recommendations,
    }
    return AuditResult(audited_rows=audited_rows, report=report, drop_ids=drop_ids)


def top_pair_counters(
    grouped: dict[str, Counter[str]],
    group_totals: Counter[str],
    limit: int = 25,
) -> list[dict[str, Any]]:
    rows = []
    for group, counter in grouped.items():
        denominator = group_totals[group]
        for value, count in counter.items():
            rows.append(
                {
                    "group": group,
                    "dynamic": value,
                    "count": count,
                    "rate_of_group_examples": count / denominator if denominator else 0.0,
                }
            )
    rows.sort(key=lambda item: item["count"], reverse=True)
    return rows[:limit]


def build_recommendations(
    *,
    style_warnings: list[dict[str, Any]],
    because_rate: float,
    contrast_rate: float,
    negative_rate: float,
    negative_config: dict[str, Any],
    opening_summary: dict[str, Any],
    material_dynamic_warnings: list[dict[str, Any]],
    length_warnings: list[str],
) -> list[str]:
    recommendations = []
    for warning in style_warnings:
        style = warning["style_id"]
        target = warning.get("target")
        if target is not None:
            delta = warning["rate"] - target
            direction = "Reduce" if delta > 0 else "Increase"
            recommendations.append(f"{direction} {style} by about {abs(delta):.0%}.")
        else:
            recommendations.append(warning["message"])
    if because_rate > 0.25:
        recommendations.append(f"Lower use of 'because'; appears in {because_rate:.0%} of examples.")
    if contrast_rate > 0.4:
        recommendations.append(
            f"Reduce explicit contrast phrasing; contrast markers appear in {contrast_rate:.0%} of examples."
        )
    if negative_rate < float(negative_config["warning_min_rate"]):
        recommendations.append("Add more negative or contrastive reporting; current rate is below target.")
    elif negative_rate > float(negative_config["warning_max_rate"]):
        recommendations.append("Reduce negative or contrastive reporting; current rate is above target.")
    top_first5 = opening_summary.get("top_first_5", [])
    if top_first5 and top_first5[0]["rate"] > 0.03:
        recommendations.append(
            f"Too many examples start with '{top_first5[0]['value']}'; vary openings and discourse styles."
        )
    for warning in material_dynamic_warnings:
        material = warning["material"]
        dynamic = warning["dynamic"]
        recommendations.append(
            f"{material} examples are dominated by {dynamic}; add other relevant properties and dynamics."
        )
    recommendations.extend(length_warnings[:3])
    if not recommendations:
        recommendations.append("No major audit imbalance found; inspect samples before scaling.")
    deduped = []
    seen = set()
    for recommendation in recommendations:
        if recommendation in seen:
            continue
        seen.add(recommendation)
        deduped.append(recommendation)
    return deduped


def audit_jsonl_file(
    input_path: str | Path,
    *,
    output_jsonl: str | Path,
    report_json: str | Path,
    drop_ids_output: str | Path | None = None,
    config_path: str | Path | None = None,
) -> AuditResult:
    config = load_audit_config(config_path)
    rows = read_jsonl(input_path)
    result = audit_rows(rows, config=config)
    write_jsonl(output_jsonl, result.audited_rows)
    write_json(report_json, result.report)
    if drop_ids_output is not None:
        path = Path(drop_ids_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(result.drop_ids) + ("\n" if result.drop_ids else ""), encoding="utf-8")
    return result
