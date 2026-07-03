"""Deterministic audit checks for concept-curriculum documents."""

from __future__ import annotations

import re
from typing import Any


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _contains_any(text: str, values: list[str]) -> bool:
    text_lower = text.lower()
    return any(str(value).lower() in text_lower for value in values if str(value).strip())


def _contains_word_or_phrase(text: str, value: str) -> bool:
    value = str(value).strip()
    if not value:
        return False
    pattern = r"(?<!\w)" + re.escape(value) + r"(?!\w)"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


_IRREGULAR_INFLECTIONS = {
    "fall": {"fall", "falls", "fell", "fallen", "falling"},
    "fell": {"fall", "falls", "fell", "fallen", "falling"},
    "grow": {"grow", "grows", "grew", "grown", "growing"},
    "grew": {"grow", "grows", "grew", "grown", "growing"},
    "rise": {"rise", "rises", "rose", "risen", "rising"},
    "rose": {"rise", "rises", "rose", "risen", "rising"},
    "shrink": {"shrink", "shrinks", "shrank", "shrunk", "shrinking"},
    "shrank": {"shrink", "shrinks", "shrank", "shrunk", "shrinking"},
    "shrunk": {"shrink", "shrinks", "shrank", "shrunk", "shrinking"},
    "slide": {"slide", "slides", "slid", "sliding"},
    "slid": {"slide", "slides", "slid", "sliding"},
    "spin": {"spin", "spins", "spun", "spinning"},
    "spun": {"spin", "spins", "spun", "spinning"},
    "speed": {"speed", "speeds", "sped", "speeding"},
    "sped": {"speed", "speeds", "sped", "speeding"},
    "stick": {"stick", "sticks", "stuck", "sticking"},
    "stuck": {"stick", "sticks", "stuck", "sticking"},
}


def _base_candidates(value: str) -> set[str]:
    candidates = {value}
    if value.endswith("s") and len(value) > 1:
        candidates.add(value[:-1])
    if value.endswith("es") and len(value) > 2:
        candidates.add(value[:-2])
    if value.endswith("ied") and len(value) > 3:
        candidates.add(value[:-3] + "y")
    if value.endswith("ed") and len(value) > 2:
        stem = value[:-2]
        candidates.add(stem)
        if len(stem) > 1 and stem[-1] == stem[-2]:
            candidates.add(stem[:-1])
        if value[:-1].endswith("e"):
            candidates.add(value[:-1])
    if value.endswith("ing") and len(value) > 3:
        stem = value[:-3]
        candidates.add(stem)
        candidates.add(stem + "e")
        if len(stem) > 1 and stem[-1] == stem[-2]:
            candidates.add(stem[:-1])
    return {candidate for candidate in candidates if candidate}


def _inflected_forms(value: str) -> list[str]:
    value = str(value).strip().lower()
    if not value or re.search(r"\s", value):
        return [value] if value else []

    forms = set()
    bases = _base_candidates(value)
    for base in list(bases):
        forms.update(_IRREGULAR_INFLECTIONS.get(base, set()))

    for base in bases:
        forms.add(base)
        if base.endswith("e") and len(base) > 1:
            forms.add(base[:-1] + "ing")
            forms.add(base + "d")
            forms.add(base + "s")
        elif base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
            forms.add(base[:-1] + "ies")
            forms.add(base[:-1] + "ied")
            forms.add(base + "ing")
        else:
            forms.add(base + "s")
            forms.add(base + "ed")
            forms.add(base + "ing")

        if len(base) >= 3 and re.search(r"[^aeiou][aeiou][^aeiouwxy]$", base):
            forms.add(base + base[-1] + "ed")
            forms.add(base + base[-1] + "ing")

    return sorted(forms, key=lambda item: (len(item), item), reverse=True)


def _contains_word_or_inflection(text: str, value: str) -> bool:
    value = str(value).strip()
    if not value:
        return False
    if re.search(r"\s", value):
        return _contains_word_or_phrase(text, value)
    return any(_contains_word_or_phrase(text, form) for form in _inflected_forms(value))


def _source_frame_terms(concept_profile: dict[str, Any]) -> list[str]:
    terms = concept_profile.get("source_frame_terms")
    if terms:
        return [str(term) for term in terms]
    return [
        "cup",
        "container",
        "jar",
        "bag",
        "bottle",
        "pitcher",
        "bowl",
        "bucket",
        "box",
        "carton",
        "packet",
        "shaker",
        "spout",
        "opening",
        "hole",
        "mouth",
        "ladle",
        "pan",
        "mold",
        "nozzle",
        "tap",
        "tube",
        "pipe",
        "out of",
    ]


def _has_source_frame(text: str, concept_profile: dict[str, Any]) -> bool:
    if any(_contains_word_or_inflection(text, term) for term in _source_frame_terms(concept_profile)):
        return True

    can_source = r"\bfrom\s+(?:a|an|the|its|this|that)\s+can\b"
    edge_source = (
        r"\b(?:pour|pours|poured|pouring|flow|flows|flowed|flowing|stream|streams|streamed|streaming)"
        r"\s+from\s+(?:a|an|the|its|this|that)\s+(?:edge|rim|lip|side|mouth|top|bottom)\b"
    )
    return bool(re.search(can_source, text, flags=re.IGNORECASE)) or bool(
        re.search(edge_source, text, flags=re.IGNORECASE)
    )


def _needs_pour_source_frame(spec: dict[str, Any], concept_profile: dict[str, Any]) -> bool:
    concept_id = str(concept_profile.get("concept_id") or spec.get("concept_id") or "")
    if concept_id != "pour":
        return False
    document_type = str(spec.get("document_type", ""))
    return document_type in {
        "direct_positive",
        "subtype_explanation",
        "state_contrast",
        "neighbor_contrast",
    }


def _lexical_form_pattern(forms: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(form) for form in forms if str(form).strip()]
    if not escaped:
        escaped = ["stir", "stirs", "stirred", "stirring"]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def audit_banned_terms(constraints: dict[str, Any]) -> list[str]:
    """Return terms checked by audit, with legacy banned_terms support."""

    terms = constraints.get("audit_banned_terms", constraints.get("banned_terms", []))
    return [str(item) for item in terms]


def expected_materials_for_spec(spec: dict[str, Any]) -> list[str]:
    """Return material strings expected to appear in a generated document."""

    required_materials = [str(value) for value in spec.get("required_materials", []) if str(value).strip()]
    if required_materials:
        return required_materials

    materials = []
    for key in (
        "positive_material",
        "negative_material",
        "before_material",
        "after_material",
        "material",
        "companion_material",
    ):
        value = spec.get(key)
        if value:
            materials.append(str(value))
    return materials


def _is_physical_dynamics_spec(spec: dict[str, Any], concept_profile: dict[str, Any]) -> bool:
    profile_type = str(concept_profile.get("profile_type") or spec.get("profile_type") or "")
    return profile_type == "physical_dynamics" or bool(spec.get("physical_family"))


def _terms_for_spec(spec: dict[str, Any], key: str, fallback_key: str | None = None) -> list[str]:
    values = [str(value) for value in spec.get(key, []) if str(value).strip()]
    if values:
        return values
    if fallback_key and spec.get(fallback_key):
        return [str(spec[fallback_key])]
    return []


def _dedupe_terms(terms: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for term in terms:
        normalized = str(term).strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            deduped.append(normalized)
    return deduped


def _normalized_term(term: str) -> str:
    return " ".join(re.findall(r"\b[\w'-]+\b", str(term).lower()))


def _physical_phrase_variants(term: str) -> list[str]:
    normalized = _normalized_term(term)
    variants = {
        "without turning": [
            "without turning",
            "without rotating",
            "did not turn",
            "does not turn",
            "could not turn",
            "cannot turn",
            "can't turn",
            "didn't turn",
            "not turning",
            "not rotating",
            "no rotation",
            "lack of rotation",
            "kept the same side down",
            "kept the same face down",
        ],
        "moved faster": [
            "moved faster",
            "move faster",
            "moving faster",
            "went faster",
            "got faster",
            "sped up",
            "speed up",
            "speeds up",
            "accelerated",
            "accelerating",
        ],
        "accelerating": [
            "accelerating",
            "accelerated",
            "speeding up",
            "speed up",
            "speeds up",
            "sped up",
            "increasing speed",
            "gaining speed",
            "moved faster",
            "moving faster",
            "went faster",
            "got faster",
        ],
        "moved slower": [
            "moved slower",
            "move slower",
            "moving slower",
            "went slower",
            "got slower",
            "slowed down",
            "slow down",
            "slows down",
            "decelerated",
            "decelerating",
        ],
        "slowing down": [
            "slowing down",
            "slowed down",
            "slow down",
            "slows down",
            "losing speed",
            "decreasing speed",
            "moved slower",
            "moving slower",
            "went slower",
            "got slower",
            "decelerated",
            "decelerating",
        ],
        "moved back": ["moved back", "bounced back", "rebounded", "sprang back"],
        "down": ["down", "downward", "downwards", "lower"],
        "removed": [
            "removed",
            "pulled out",
            "pulled the plug out",
            "pulled the brace out",
            "taken out",
            "took out",
            "lifted out",
        ],
        "release": [
            "release",
            "released",
            "let go",
            "let go of",
            "letting go",
            "let it go",
            "let loose",
            "set free",
        ],
        "released": [
            "release",
            "released",
            "let go",
            "let go of",
            "letting go",
            "let it go",
            "let loose",
            "set free",
        ],
        "spread apart": ["spread apart", "spread out", "scattered", "moved apart"],
        "ice": ["ice", "icy"],
        "slid": ["slid", "slide", "slides", "sliding", "slip", "slips", "slipped", "slipping"],
        "warm": ["warm", "warmed", "warming", "warmth"],
        "wet": ["wet", "water"],
    }
    return variants.get(normalized, [])


def _contains_physical_term(text: str, term: str) -> bool:
    if _contains_word_or_inflection(text, term):
        return True
    return any(_contains_word_or_inflection(text, variant) for variant in _physical_phrase_variants(term))


def _missing_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if not _contains_word_or_inflection(text, term)]


def _missing_physical_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if not _contains_physical_term(text, term)]


def _any_term_present(text: str, terms: list[str]) -> bool:
    return any(_contains_word_or_inflection(text, term) for term in terms)


def _any_physical_term_present(text: str, terms: list[str]) -> bool:
    return any(_contains_physical_term(text, term) for term in terms)


def _physical_leakage_terms(constraints: dict[str, Any]) -> list[str]:
    configured = constraints.get("physical_leakage_terms")
    if configured is not None:
        return [str(term) for term in configured if str(term).strip()]
    return [
        "benchmark",
        "scoring",
        "hidden metadata",
        "scenario spec",
        "answer key",
        "target concept",
        "expected process",
        "contrast process",
        "Context1",
        "Context2",
        "Target1",
        "Target2",
    ]


def _audit_speed_geometry(
    *,
    text: str,
    spec: dict[str, Any],
) -> list[dict[str, str]]:
    """Warn when speed-change prose omits the distance-over-time bridge."""

    speed_geometry = spec.get("speed_geometry")
    if not isinstance(speed_geometry, dict) or not speed_geometry:
        return []

    required_fields = {
        "initial_rate": "initial speed/rate evidence",
        "later_rate": "later speed/rate evidence",
        "same_time_comparison": "same-time distance comparison",
        "definition_bridge": "speed definition bridge",
        "process_bridge": "speed process bridge",
    }
    missing = []
    for key, label in required_fields.items():
        value = str(speed_geometry.get(key) or "").strip()
        if value and not _contains_physical_term(text, value):
            missing.append(label)

    if not missing:
        return []

    severity = (
        "error"
        if str(spec.get("render_template") or "") == "speed_rate_bridge_5_sentence"
        or str(spec.get("family") or spec.get("physical_family") or "") == "speed_rate_definition_bridge"
        else "warning"
    )
    return [
        {
            "severity": severity,
            "code": "missing_speed_geometry",
            "message": (
                "Speed-change document may not richly define speed change; missing "
                + ", ".join(missing)
                + "."
            ),
        }
    ]


def _sentence_has_any_physical_term(sentence: str, terms: list[str]) -> bool:
    return any(_contains_physical_term(sentence, term) for term in terms if str(term).strip())


def _sentence_has_any_target_term(sentence: str, terms: list[str]) -> bool:
    return any(
        _contains_word_or_inflection(sentence, term)
        for term in terms
        if str(term).strip()
    )


_NEGATED_PHYSICAL_TERM_PREFIX = re.compile(
    r"(?:without|not|no|never|did not|does not|do not|cannot|can't|could not|couldn't|didn't|doesn't)\s+"
    r"(?:\w+\s+){0,2}$",
    flags=re.IGNORECASE,
)


def _is_negated_physical_term_match(text: str, start: int) -> bool:
    prefix = text[max(0, start - 80) : start]
    return bool(_NEGATED_PHYSICAL_TERM_PREFIX.search(prefix))


def _contains_unnegated_physical_term(text: str, term: str) -> bool:
    """Return whether a term appears as positive evidence, not under local negation."""

    term = str(term).strip()
    if not term:
        return False

    normalized = _normalized_term(term)
    if normalized.startswith(
        (
            "without ",
            "not ",
            "no ",
            "never ",
            "did not ",
            "does not ",
            "do not ",
            "cannot ",
            "can't ",
            "could not ",
            "couldn't ",
            "didn't ",
            "doesn't ",
        )
    ):
        return _contains_physical_term(text, term)

    candidates = _physical_phrase_variants(term)
    if re.search(r"\s", term):
        candidates.append(term)
    else:
        candidates.extend(_inflected_forms(term))

    for candidate in _dedupe_terms(candidates):
        pattern = r"(?<!\w)" + re.escape(candidate) + r"(?!\w)"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if not _is_negated_physical_term_match(text, match.start()):
                return True
    return False


def _audit_lexical_shortcut(
    *,
    text: str,
    spec: dict[str, Any],
) -> list[dict[str, str]]:
    """Enforce tight local evidence for W2V-oriented shortcut and bridge documents."""

    render_template = str(spec.get("render_template") or "")
    is_shortcut = render_template.startswith("lexical_shortcut_")
    is_bridge = render_template.startswith("lexical_bridge_")
    if not (is_shortcut or is_bridge):
        return []

    target_terms = _terms_for_spec(spec, "same_sentence_target_terms")
    lexical_target = str(spec.get("lexical_target") or "").strip()
    if lexical_target:
        target_terms.append(lexical_target)
    if not target_terms:
        target_terms = _terms_for_spec(spec, "required_process_terms", fallback_key="expected_process")
    target_terms = _dedupe_terms(target_terms)

    evidence_terms = _terms_for_spec(spec, "same_sentence_evidence_terms")
    contrast_bans = _terms_for_spec(spec, "contrast_evidence_bans")
    flags: list[dict[str, str]] = []

    if target_terms and evidence_terms:
        matched = any(
            _sentence_has_any_target_term(sentence, target_terms)
            and _sentence_has_any_physical_term(sentence, evidence_terms)
            for sentence in _sentence_texts(text)
        )
        if not matched:
            flags.append(
                {
                    "severity": "error",
                    "code": (
                        "missing_same_sentence_shortcut_evidence"
                        if is_shortcut
                        else "missing_same_sentence_bridge_evidence"
                    ),
                    "message": (
                        "Lexical shortcut/bridge document must place a target "
                        "term and local evidence in the same sentence."
                    ),
                }
            )

    contrast_hits = [
        term for term in contrast_bans if _contains_unnegated_physical_term(text, term)
    ]
    if contrast_hits:
        flags.append(
            {
                "severity": "error",
                "code": (
                    "shortcut_contrast_evidence_violation"
                    if is_shortcut
                    else "bridge_contrast_evidence_violation"
                ),
                "message": (
                    "Lexical shortcut/bridge document includes opposite-polarity evidence: "
                    + ", ".join(sorted(contrast_hits))
                ),
            }
        )

    return flags


def _audit_physical_dynamics(
    *,
    text: str,
    spec: dict[str, Any],
    constraints: dict[str, Any],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    lexical_shortcut = str(spec.get("render_template") or "").startswith("lexical_shortcut_")
    missing_cause_or_outcome_severity = "warning" if lexical_shortcut else "error"
    cause_terms = _terms_for_spec(spec, "required_cause_terms", fallback_key="cause_evidence")
    required_process_terms = _terms_for_spec(
        spec,
        "required_process_terms",
        fallback_key="expected_process",
    )
    process_terms = _dedupe_terms(
        _terms_for_spec(spec, "preferred_process_terms")
        + required_process_terms
        + ([str(spec["expected_process"])] if spec.get("expected_process") else [])
    )
    outcome_terms = _terms_for_spec(spec, "required_outcome_terms", fallback_key="outcome_evidence")
    contrast_terms = _dedupe_terms(
        _terms_for_spec(spec, "preferred_contrast_terms")
        + _terms_for_spec(spec, "contrast_process_terms", fallback_key="contrast_process")
    )
    shortcut_terms = _terms_for_spec(spec, "shortcut_terms")
    lexical_target = str(spec.get("lexical_target") or "").strip()
    lexical_target_missing_severity = str(
        constraints.get("missing_lexical_target_severity", "error")
    ).strip().lower()
    if lexical_target_missing_severity not in {"error", "warning"}:
        lexical_target_missing_severity = "error"
    contrast_verbalization = str(
        constraints.get("contrast_verbalization", spec.get("contrast_verbalization", "explicit"))
    ).strip().lower()

    missing_cause = _missing_physical_terms(text, cause_terms)
    missing_process = [] if _any_physical_term_present(text, process_terms) else required_process_terms
    missing_outcome = _missing_physical_terms(text, outcome_terms)
    missing_lexical_target = lexical_target and not _contains_word_or_phrase(text, lexical_target)

    if missing_cause:
        flags.append(
            {
                "severity": missing_cause_or_outcome_severity,
                "code": "missing_physical_cause",
                "message": "Document is missing required cause evidence: "
                + ", ".join(sorted(missing_cause)),
            }
        )
    if missing_process:
        flags.append(
            {
                "severity": "error",
                "code": "missing_expected_process",
                "message": "Document is missing expected process evidence: "
                + ", ".join(sorted(missing_process)),
            }
        )
    if missing_outcome:
        flags.append(
            {
                "severity": missing_cause_or_outcome_severity,
                "code": "missing_expected_outcome",
                "message": "Document is missing expected outcome evidence: "
                + ", ".join(sorted(missing_outcome)),
            }
        )
    if missing_lexical_target:
        flags.append(
            {
                "severity": lexical_target_missing_severity,
                "code": "missing_lexical_target",
                "message": f"Document is missing assigned lexical target: {lexical_target}",
            }
        )

    if contrast_terms and missing_process and _any_physical_term_present(text, contrast_terms):
        flags.append(
            {
                "severity": "error",
                "code": "physical_process_mismatch",
                "message": "Document mentions the contrast process but not the expected process.",
            }
        )

    if contrast_verbalization in {"metadata_only", "implicit"} and contrast_terms:
        protected_terms = _dedupe_terms(
            cause_terms
            + process_terms
            + outcome_terms
            + [
                str(spec[key])
                for key in (
                    "cause_evidence",
                    "outcome_evidence",
                    "direction_evidence",
                    "initial_relation_to_landmark",
                    "later_relation_to_landmark",
                    "start_position",
                    "end_position",
                )
                if spec.get(key)
            ]
        )
        protected_normalized = [
            _normalized_term(term) for term in protected_terms if _normalized_term(term)
        ]
        contrast_terms_for_policy = []
        for term in contrast_terms:
            normalized = _normalized_term(term)
            if not normalized:
                continue
            if any(
                normalized == protected
                or normalized in protected.split()
                or protected in normalized.split()
                for protected in protected_normalized
            ):
                continue
            contrast_terms_for_policy.append(term)
        contrast_hits = [
            term for term in contrast_terms_for_policy if _contains_physical_term(text, term)
        ]
        if contrast_hits:
            flags.append(
                {
                    "severity": "error" if contrast_verbalization == "metadata_only" else "warning",
                    "code": "contrast_verbalization_violation",
                    "message": (
                        f"Document uses contrast term(s) despite {contrast_verbalization} policy: "
                        + ", ".join(sorted(contrast_hits))
                    ),
                }
            )

    leakage_hits = [
        term for term in _physical_leakage_terms(constraints) if _contains_word_or_phrase(text, term)
    ]
    if leakage_hits:
        flags.append(
            {
                "severity": "error",
                "code": "target_leakage",
                "message": "Document includes benchmark-style or construction-language term(s): "
                + ", ".join(sorted(leakage_hits)),
            }
        )

    shortcut_hits = [term for term in shortcut_terms if _contains_word_or_inflection(text, term)]
    if shortcut_hits and (missing_cause or missing_process or missing_outcome):
        flags.append(
            {
                "severity": "warning",
                "code": "physical_shortcut_only",
                "message": "Document names shortcut-prone object/outcome cue(s) without full physical evidence: "
                + ", ".join(sorted(shortcut_hits)),
            }
        )

    flags.extend(_audit_speed_geometry(text=text, spec=spec))
    flags.extend(_audit_lexical_shortcut(text=text, spec=spec))

    return flags


def _opening_key(text: str, *, word_count: int = 6) -> str:
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    return " ".join(words[:word_count])


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"\b[\w'-]+\b", text.lower()))


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\b[\w'-]+\b", text.lower()))


def _sentence_texts(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _normalized_sentence(text: str) -> str:
    return " ".join(re.findall(r"\b[\w'-]+\b", text.lower()))


def _default_sentence_frame_patterns() -> list[str]:
    return [
        r"^in a simple physical scene\b",
        r"^the visible outcome was that\b",
        r"^the useful evidence links\b",
        r"^over\b.+\brather than\b",
        r"\binstead of relying only on the object name\b",
    ]


def _append_warning(row: dict[str, Any], *, code: str, message: str) -> None:
    flags = list(row.get("audit_flags", []))
    flags.append({"severity": "warning", "code": code, "message": message})
    row["audit_flags"] = flags
    if row.get("audit_status") == "pass":
        row["audit_status"] = "warn"


def _attach_repeated_opening_warnings(
    rows: list[dict[str, Any]],
    *,
    max_reuse: int,
) -> None:
    openings: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _opening_key(str(row.get("text", "")))
        if key:
            openings.setdefault(key, []).append(row)

    for key, grouped_rows in openings.items():
        if len(grouped_rows) > max_reuse:
            for row in grouped_rows[max_reuse:]:
                _append_warning(
                    row,
                    code="overused_opening",
                    message=f"Opening is reused more than {max_reuse} times: {key!r}.",
                )


def _attach_near_duplicate_warnings(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> None:
    seen_exact: dict[str, dict[str, Any]] = {}
    seen_sets: list[tuple[set[str], dict[str, Any]]] = []
    for row in rows:
        text = str(row.get("text", ""))
        normalized = _normalized_text(text)
        if normalized in seen_exact:
            _append_warning(
                row,
                code="near_duplicate",
                message=f"Text is an exact duplicate of {seen_exact[normalized].get('id')}.",
            )
            continue

        current_set = _word_set(text)
        for previous_set, previous_row in seen_sets:
            union = current_set | previous_set
            if not union:
                continue
            similarity = len(current_set & previous_set) / len(union)
            if similarity >= threshold:
                _append_warning(
                    row,
                    code="near_duplicate",
                    message=(
                        "Text is very similar to "
                        f"{previous_row.get('id')} (Jaccard {similarity:.2f})."
                    ),
                )
                break

        seen_exact[normalized] = row
        seen_sets.append((current_set, row))


def _attach_repeated_sentence_warnings(
    rows: list[dict[str, Any]],
    *,
    max_reuse: int,
    min_words: int,
) -> None:
    occurrences: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for row in rows:
        for sentence in _sentence_texts(str(row.get("text", ""))):
            key = _normalized_sentence(sentence)
            if not key or _word_count(key) < min_words:
                continue
            occurrences.setdefault(key, []).append((row, sentence))

    for key, grouped in occurrences.items():
        if len(grouped) > max_reuse:
            first_row = grouped[0][0]
            for row, sentence in grouped[max_reuse:]:
                _append_warning(
                    row,
                    code="repeated_sentence",
                    message=(
                        f"Sentence repeats more than {max_reuse} time(s); "
                        f"first seen in {first_row.get('id')}: {sentence!r}."
                    ),
                )


def _attach_overused_sentence_frame_warnings(
    rows: list[dict[str, Any]],
    *,
    patterns: list[str],
    max_reuse: int,
) -> None:
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns if pattern.strip()]
    occurrences: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for row in rows:
        for sentence in _sentence_texts(str(row.get("text", ""))):
            normalized = sentence.strip()
            for pattern in compiled:
                if pattern.search(normalized):
                    occurrences.setdefault(pattern.pattern, []).append((row, sentence))
                    break

    for pattern, grouped in occurrences.items():
        if len(grouped) > max_reuse:
            for row, sentence in grouped[max_reuse:]:
                _append_warning(
                    row,
                    code="overused_sentence_frame",
                    message=(
                        f"Sentence frame {pattern!r} appears more than {max_reuse} time(s): "
                        f"{sentence!r}."
                    ),
                )


def audit_document(
    document: dict[str, Any],
    *,
    spec: dict[str, Any],
    concept_profile: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit one generated document and return flags plus status."""

    constraints = constraints or {}
    text = str(document.get("text", "")).strip()
    lexical_forms = [str(item) for item in concept_profile.get("lexical_forms", [])]
    target_pattern = _lexical_form_pattern(lexical_forms) if lexical_forms else None
    min_words = int(constraints.get("min_words", 0))
    max_words = int(constraints.get("max_words", 10_000))
    banned_terms = audit_banned_terms(constraints)

    flags: list[dict[str, str]] = []
    if not text:
        flags.append({"severity": "error", "code": "empty_text", "message": "Document text is empty."})
    if text and target_pattern is not None and not target_pattern.search(text):
        flags.append(
            {
                "severity": "error",
                "code": "missing_target_lexeme",
                "message": "Document does not include a target lexical form.",
            }
        )

    words = _word_count(text)
    if words < min_words:
        flags.append(
            {
                "severity": "warning",
                "code": "too_short",
                "message": f"Document has {words} words; minimum is {min_words}.",
            }
        )
    if words > max_words:
        flags.append(
            {
                "severity": "warning",
                "code": "too_long",
                "message": f"Document has {words} words; maximum is {max_words}.",
            }
        )

    banned_hits = [term for term in banned_terms if term.lower() in text.lower()]
    if banned_hits:
        flags.append(
            {
                "severity": "error",
                "code": "banned_term",
                "message": "Document includes banned term(s): " + ", ".join(sorted(banned_hits)),
            }
        )

    expected_materials = expected_materials_for_spec(spec)
    required_materials = [str(value) for value in spec.get("required_materials", []) if str(value).strip()]
    if required_materials:
        missing_materials = [
            material for material in required_materials if not _contains_word_or_phrase(text, material)
        ]
        if missing_materials:
            flags.append(
                {
                    "severity": "error",
                    "code": "missing_expected_material",
                    "message": "Document is missing required material(s): "
                    + ", ".join(sorted(missing_materials)),
                }
            )
    elif expected_materials and not _contains_any(text, expected_materials):
        flags.append(
            {
                "severity": "error",
                "code": "missing_expected_material",
                "message": "Document does not name an expected material/substance.",
            }
        )

    required_affordances = [
        str(value) for value in spec.get("required_affordances", []) if str(value).strip()
    ]
    companion_affordances = [
        str(value) for value in spec.get("companion_required_affordances", []) if str(value).strip()
    ]
    if required_affordances:
        min_hits = min(
            len(required_affordances),
            int(spec.get("min_required_affordances") or constraints.get("min_required_affordances", 2)),
        )
        affordance_hits = [
            value for value in required_affordances if _contains_word_or_inflection(text, value)
        ]
        if len(affordance_hits) < min_hits:
            flags.append(
                {
                    "severity": "error",
                    "code": "missing_required_affordance",
                    "message": (
                        f"Document mentions {len(affordance_hits)} required affordance(s); "
                        f"expected at least {min_hits}."
                    ),
                }
            )
    if companion_affordances:
        min_hits = min(
            len(companion_affordances),
            int(spec.get("min_companion_affordances") or constraints.get("min_required_affordances", 2)),
        )
        companion_hits = [
            value for value in companion_affordances if _contains_word_or_inflection(text, value)
        ]
        if len(companion_hits) < min_hits:
            flags.append(
                {
                    "severity": "error",
                    "code": "missing_companion_affordance",
                    "message": (
                        f"Document mentions {len(companion_hits)} companion affordance(s); "
                        f"expected at least {min_hits}."
                    ),
                }
            )

    negative_markers = [
        str(value) for value in constraints.get("negative_markers", []) if str(value).strip()
    ]
    if spec.get("material_class_id") and negative_markers:
        marker_hits = [
            marker for marker in negative_markers if _contains_word_or_phrase(text, marker)
        ]
        if marker_hits:
            flags.append(
                {
                    "severity": "warning",
                    "code": "negative_marker",
                    "message": "Correlation-first document includes negative marker(s): "
                    + ", ".join(sorted(marker_hits)),
                }
            )
    if text and _needs_pour_source_frame(spec, concept_profile) and not _has_source_frame(text, concept_profile):
        flags.append(
            {
                "severity": "warning",
                "code": "missing_source_frame",
                "message": "Pour document does not clearly name a source container, opening, spout, or outlet.",
            }
        )

    document_type = str(spec.get("document_type", ""))
    neighbor = str(spec.get("neighbor", "")).strip()
    if document_type == "neighbor_contrast" and neighbor and not _contains_word_or_inflection(text, neighbor):
        flags.append(
            {
                "severity": "error",
                "code": "missing_neighbor",
                "message": f"Neighbor contrast does not mention {neighbor!r}.",
            }
        )
    if document_type == "state_contrast":
        if not re.search(r"\b(can(?:not|'t)?|cannot|not|before|after|once|when)\b", text, flags=re.IGNORECASE):
            flags.append(
                {
                    "severity": "warning",
                    "code": "weak_state_contrast",
                    "message": "State contrast may not clearly mark the before/after condition.",
                }
            )

    style_id = str(spec.get("style_id", ""))
    style_required_terms = [str(term) for term in spec.get("style_required_terms", []) if str(term).strip()]
    style_min_terms = int(spec.get("style_min_required_terms") or 0)
    if style_id == "concept_map_paragraph" and style_required_terms and style_min_terms:
        style_hits = [term for term in style_required_terms if _contains_word_or_inflection(text, term)]
        if len(style_hits) < style_min_terms:
            flags.append(
                {
                    "severity": "warning",
                    "code": "weak_concept_map",
                    "message": (
                        f"Concept map mentions {len(style_hits)} nearby action(s); "
                        f"expected at least {style_min_terms}."
                    ),
                }
            )

    if _is_physical_dynamics_spec(spec, concept_profile):
        flags.extend(_audit_physical_dynamics(text=text, spec=spec, constraints=constraints))

    if any(flag["severity"] == "error" for flag in flags):
        status = "fail"
    elif flags:
        status = "warn"
    else:
        status = "pass"

    return {
        "audit_status": status,
        "audit_flags": flags,
        "word_count": words,
    }


def audit_documents(
    documents: list[dict[str, Any]],
    *,
    specs_by_id: dict[str, dict[str, Any]],
    concept_profile: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Audit generated documents and attach audit fields to each row."""

    audited = []
    for document in documents:
        row = dict(document)
        spec_id = str(row.get("id") or row.get("scenario_id"))
        spec = specs_by_id.get(spec_id, {})
        audit = audit_document(
            row,
            spec=spec,
            concept_profile=concept_profile,
            constraints=constraints,
        )
        row.update(audit)
        audited.append(row)

    if constraints and constraints.get("audit_repeated_openings"):
        _attach_repeated_opening_warnings(
            audited,
            max_reuse=int(constraints.get("max_opening_reuse", 4)),
        )
    if constraints and constraints.get("audit_near_duplicates"):
        _attach_near_duplicate_warnings(
            audited,
            threshold=float(constraints.get("near_duplicate_jaccard", 0.92)),
        )
    if constraints and constraints.get("audit_repeated_sentences"):
        _attach_repeated_sentence_warnings(
            audited,
            max_reuse=int(constraints.get("max_repeated_sentence_reuse", 1)),
            min_words=int(constraints.get("min_repeated_sentence_words", 6)),
        )
    if constraints and constraints.get("audit_overused_sentence_frames"):
        _attach_overused_sentence_frame_warnings(
            audited,
            patterns=[
                str(pattern)
                for pattern in constraints.get(
                    "overused_sentence_frame_patterns",
                    _default_sentence_frame_patterns(),
                )
            ],
            max_reuse=int(constraints.get("max_sentence_frame_reuse", 2)),
        )
    return audited
