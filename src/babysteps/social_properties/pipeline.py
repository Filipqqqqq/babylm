"""Toy social-property synthetic data pipeline.

This ports the scratch social-properties notebook into a small scriptable
Fireworks/Qwen pipeline. It is intentionally an MVP, not a production system.
"""

from __future__ import annotations

import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "social_properties" / "qwen3p6_toy.json"
Generator = Callable[..., str]


SOCIAL_ONTOLOGY: list[dict[str, Any]] = [
    {"axis": "warmth_affiliation", "anchor_traits": ["friendly", "hostile", "warm", "cold"]},
    {"axis": "reliability_trust", "anchor_traits": ["trustworthy", "untrustworthy"]},
    {
        "axis": "agency_status",
        "anchor_traits": ["dominant", "submissive", "confident"],
        "subfacets": {
            "prosocial_leadership": [
                "organizes the group when everyone is confused",
                "suggests a plan and checks whether others agree",
                "helps assign roles fairly",
                "speaks up so the group can finish the task",
                "takes responsibility when a decision is needed",
            ],
            "controlling_dominance": [
                "orders others around without listening",
                "interrupts to force their own plan",
                "decides roles before anyone else can speak",
                "pushes their choice even after others object",
                "takes control of shared materials without asking",
            ],
            "confident_behavior": [
                "explains their idea clearly",
                "asks a question in front of the group",
                "tries a difficult task without freezing",
                "keeps speaking calmly after a small mistake",
                "volunteers to present the group's work",
            ],
            "yielding_or_deferential_behavior": [
                "lets others choose first",
                "backs down after mild disagreement",
                "waits for someone else to decide",
                "goes along with the group's choice",
                "avoids stating a preference even when asked",
            ],
        },
        "common_confounds": [
            "dominant is not always hostile",
            "confident is not always dominant",
            "submissive or yielding is not always bad",
            "leadership can be warm and cooperative",
            "control can be cold or unfair when it ignores others",
        ],
    },
    {"axis": "sociability", "anchor_traits": ["shy", "introverted", "extroverted"]},
    {"axis": "humility_self_presentation", "anchor_traits": ["boastful", "humble"]},
    {
        "axis": "tolerance_social_attitude",
        "anchor_traits": ["tolerant", "bigoted"],
        "low_stakes_differences": ["clubs", "teams", "hobbies", "schools", "game groups", "favorite snacks"],
    },
]

DISCOURSE_STYLES: list[dict[str, str]] = [
    {"style_id": "everyday_narrative", "prompt_hint": "Write a compact everyday scene with concrete actions."},
    {"style_id": "classroom_observation", "prompt_hint": "Write like a quiet classroom observation."},
    {"style_id": "parent_child_conversation", "prompt_hint": "Write a natural parent-child conversation."},
    {"style_id": "short_dialogue", "prompt_hint": "Use brief dialogue embedded in prose."},
    {"style_id": "group_project_note", "prompt_hint": "Write like a concise group project note."},
    {"style_id": "neighbor_household_anecdote", "prompt_hint": "Write a small neighbor or household anecdote."},
    {"style_id": "before_and_after_description", "prompt_hint": "Describe before and after a social moment."},
    {"style_id": "lost_and_found_description", "prompt_hint": "Write about a lost-and-found or borrowed-item situation."},
    {"style_id": "simple_explanation", "prompt_hint": "Give a simple prose explanation, not a worksheet."},
    {"style_id": "school_club_scene", "prompt_hint": "Write a school club, game group, or shared activity scene."},
]

LENGTH_BUCKETS: dict[str, dict[str, Any]] = {
    "short_64": {"word_range": (35, 60), "description": "about 35-60 words", "max_tokens": 220},
    "medium_128": {"word_range": (70, 120), "description": "about 70-120 words", "max_tokens": 320},
    "long_256": {"word_range": (140, 220), "description": "about 140-220 words", "max_tokens": 520},
}

TRAIT_CONTRASTS = {
    "friendly": "hostile",
    "hostile": "friendly",
    "warm": "cold",
    "cold": "warm",
    "trustworthy": "untrustworthy",
    "untrustworthy": "trustworthy",
    "dominant": "submissive",
    "submissive": "dominant",
    "introverted": "extroverted",
    "extroverted": "introverted",
    "boastful": "humble",
    "humble": "boastful",
    "tolerant": "bigoted",
    "bigoted": "tolerant",
}

TRAIT_NEAR_CONTRASTS = {
    "confident": ["shy", "hesitant", "uncertain"],
    "shy": ["confident", "outgoing"],
    "dominant": ["confident", "assertive"],
    "boastful": ["confident"],
    "introverted": ["shy"],
    "cold": ["introverted"],
}

EVIDENCE_STRENGTHS = ["single_moment", "repeated_pattern"]
INFERENCE_WORDING = {
    "single_moment": ["acted in a {trait} way", "seemed {trait} in that moment", "came across as {trait}"],
    "repeated_pattern": ["was often seen as {trait}", "people came to see them as {trait}", "usually acted {trait}"],
}
LABEL_MODES = ["implicit", "evidence_then_label", "label_then_evidence"]
LABEL_MODE_SEQUENCE = ["implicit", "evidence_then_label", "implicit", "label_then_evidence"]

AVOID_SOCIAL_CUES = [
    "race",
    "ethnicity",
    "religion",
    "gender",
    "nationality",
    "accent",
    "disability",
    "appearance",
    "body size",
    "clothing price",
    "family background",
    "neighborhood",
    "name-based stereotypes",
]

TRAIT_VALENCE = {
    "friendly": "positive",
    "warm": "positive",
    "trustworthy": "positive",
    "confident": "positive",
    "extroverted": "neutral_or_positive",
    "humble": "positive",
    "tolerant": "positive",
    "hostile": "negative",
    "cold": "negative",
    "untrustworthy": "negative",
    "boastful": "negative",
    "bigoted": "negative",
    "dominant": "context_dependent",
    "submissive": "context_dependent",
    "shy": "neutral",
    "introverted": "neutral",
}

STYLE_BY_ID = {style["style_id"]: style for style in DISCOURSE_STYLES}
ALL_TRAITS = sorted({trait for card in SOCIAL_ONTOLOGY for trait in card["anchor_traits"]})

JSON_SYSTEM_PROMPT = "Return only valid JSON. Do not include markdown, comments, or extra text."
PROSE_SYSTEM_PROMPT = "Write natural prose for a behavior-grounded social-property corpus. Return valid JSON when asked."
EDITOR_SYSTEM_PROMPT = "Edit or check social-property examples for safety and behavior-grounding. Return valid JSON when asked."

CONNECTIVE_PATTERNS = ["because", "so", "therefore", "unlike", "instead of", "rather than", "this shows", "this proves", "that means"]
OVERCONFIDENT_PATTERNS = ["always", "never", "bad person", "good person", "proves", "everyone knew"]
SAFETY_CUE_RE = re.compile(
    r"\b("
    r"race|racial|ethnic|ethnicity|religion|religious|muslim|christian|jewish|hindu|buddhist|"
    r"gender|male|female|boy|girl|man|woman|trans|nonbinary|gay|lesbian|nationality|immigrant|"
    r"accent|disabled|disability|wheelchair|blind|deaf|autistic|appearance|body size|clothing price|"
    r"poor|rich|social class|working class|middle class|upper class|lower class|class marker|"
    r"skin|family background|neighborhood|name-based stereotype|name based stereotype"
    r")\b",
    flags=re.I,
)
APPEARANCE_RE = re.compile(r"\b(appearance|face|clothes|clothing|hair|skin|tall|short|pretty|handsome|ugly|body size|accent)\b", flags=re.I)
IDENTITY_INFERENCE_RE = re.compile(
    r"\b(because|since|as|due to)\b.{0,100}\b("
    r"race|ethnicity|religion|gender|nationality|accent|appearance|looked|looks|disabled|disability|"
    r"family|background|neighborhood|poor|rich|skin|name"
    r")\b",
    flags=re.I | re.S,
)
PRONOUN_RE = re.compile(r"\b(he|him|his|she|her|hers|they|them|their|theirs)\b", flags=re.I)
NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
STOP_CAPITALIZED = {"At", "After", "Before", "During", "In", "On", "When", "While", "The", "A", "An", "This", "That", "Later"}


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML configs. Use JSON or install PyYAML.") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def first_n_words(text: str, n: int) -> str:
    return " ".join(words(text)[:n])


def contains_trait_word(text: str, trait: str) -> bool:
    return bool(trait and re.search(rf"\b{re.escape(trait)}\b", text or "", flags=re.I))


def trait_word_index(text: str, trait: str) -> int | None:
    text_words = words(text)
    trait_words = words(trait)
    if not trait_words:
        return None
    width = len(trait_words)
    for idx in range(0, len(text_words) - width + 1):
        if text_words[idx : idx + width] == trait_words:
            return idx
    return None


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def safe_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        extracted = extract_json_object(text)
        if extracted is None:
            return None
        try:
            return json.loads(extracted)
        except Exception:
            return None


def _prompt_value(prompt: str, key: str, default: str = "") -> str:
    match = re.search(rf"^{re.escape(key)}:[ \t]*(.*)$", prompt, flags=re.M)
    return match.group(1).strip() if match else default


class DeterministicSocialPropertiesMockGenerator:
    """Deterministic local generator for CLI smoke tests."""

    def __call__(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        del system_prompt, temperature, max_tokens, stage, response_format
        return mock_generate_text(prompt)


def _mock_behavior_bundle(target: str, evidence_strength: str, agency_subfacet: str = "") -> tuple[str, str, list[str], str]:
    del evidence_strength
    if target == "friendly":
        return "Maya", "after-school club", ["made room at the table", "invited another student into the card game"], "Maya made room at the table and invited another student into the card game."
    if target == "dominant":
        if agency_subfacet == "controlling_dominance":
            return "Maya", "group project", ["took the poster without asking", "assigned roles before others spoke"], "Maya took the shared poster without asking and assigned roles before others spoke."
        return "Maya", "group project", ["suggested a plan", "checked whether others agreed", "assigned tasks fairly"], "Maya suggested a plan, checked for agreement, and helped assign roles fairly."
    if target == "trustworthy":
        return "Maya", "classroom supply shelf", ["admitted they forgot the calculator", "returned the missing pen"], "Maya admitted the oversight and returned the missing pen immediately."
    if target == "shy":
        return "Maya", "club meeting", ["spoke only after being invited", "stepped away from the spotlight"], "Maya spoke only after being invited and stepped away from the spotlight."
    if target == "boastful":
        return "Maya", "science fair table", ["talked mostly about their own score", "took credit for the whole project"], "Maya kept talking about their own score and took credit for the whole project."
    if target == "tolerant":
        return "Maya", "game club", ["welcomed someone from a rival game group", "judged their idea fairly"], "Maya welcomed the rival game-group member and judged the idea fairly."
    if target == "hostile":
        return "Maya", "group work", ["mocked a teammate's mistake", "refused to share the supplies"], "Maya mocked a teammate's mistake and refused to share the supplies."
    if target == "untrustworthy":
        return "Maya", "class project", ["hid a mistake", "blamed someone else for missing materials"], "Maya hid a mistake and blamed someone else for the missing materials."
    return "Maya", "club room", ["acted in a clear observable way"], "Maya acted in a clear observable way."


def mock_generate_text(prompt: str) -> str:
    if "SCENARIO_PLAN_REQUEST" in prompt:
        axis = _prompt_value(prompt, "AXIS", "warmth_affiliation")
        target = _prompt_value(prompt, "TARGET_TRAIT", "friendly")
        contrast = _prompt_value(prompt, "CONTRAST_TRAIT", "")
        style_id = _prompt_value(prompt, "STYLE_ID", "everyday_narrative")
        length_bucket = _prompt_value(prompt, "LENGTH_BUCKET", "short_64")
        scenario_id = _prompt_value(prompt, "SCENARIO_ID", f"dry_{random.randint(0, 999999):06d}")
        evidence_strength = _prompt_value(prompt, "EVIDENCE_STRENGTH", "single_moment")
        label_mode = _prompt_value(prompt, "LABEL_MODE", "implicit")
        agency_subfacet = _prompt_value(prompt, "AGENCY_SUBFACET", "")
        agent, setting, evidence, _ = _mock_behavior_bundle(target, evidence_strength, agency_subfacet)
        return json.dumps(
            {
                "scenario_id": scenario_id,
                "axis": axis,
                "target_property": target,
                "contrast_property": contrast,
                "agent": agent,
                "setting": setting,
                "behavior_evidence": evidence,
                "social_inference": INFERENCE_WORDING[evidence_strength][0].format(trait=target),
                "evidence_strength": evidence_strength,
                "label_mode": label_mode,
                "trait_word_should_appear": label_mode != "implicit",
                "agency_subfacet": agency_subfacet,
                "caveat": "One scene is limited evidence." if evidence_strength == "single_moment" else "",
                "style_id": style_id,
                "length_bucket": length_bucket,
                "avoid_cues": list(AVOID_SOCIAL_CUES),
                "safety_notes": ["Inference is based on observed behavior only."],
            }
        )
    if "RENDER_SOCIAL_PROSE_REQUEST" in prompt:
        scenario_id = _prompt_value(prompt, "SCENARIO_ID", "dry_render")
        axis = _prompt_value(prompt, "AXIS", "warmth_affiliation")
        target = _prompt_value(prompt, "TARGET_TRAIT", "friendly")
        contrast = _prompt_value(prompt, "CONTRAST_TRAIT", "")
        style_id = _prompt_value(prompt, "STYLE_ID", "everyday_narrative")
        length_bucket = _prompt_value(prompt, "LENGTH_BUCKET", "short_64")
        evidence_strength = _prompt_value(prompt, "EVIDENCE_STRENGTH", "single_moment")
        label_mode = _prompt_value(prompt, "LABEL_MODE", "implicit")
        agency_subfacet = _prompt_value(prompt, "AGENCY_SUBFACET", "")
        agent, setting, _, prose_seed = _mock_behavior_bundle(target, evidence_strength, agency_subfacet)
        if label_mode == "evidence_then_label":
            text = f"During {setting}, {prose_seed} In that moment, {agent} acted in a {target} way."
        elif label_mode == "label_then_evidence":
            text = f"{agent} seemed {target} during {setting}: {prose_seed} The label came from the actions in the scene."
        else:
            text = f"During {setting}, {prose_seed} The moment gave classmates enough behavior to read the social cue without a label."
        return json.dumps(
            {
                "scenario_id": scenario_id,
                "status": "draft",
                "text": text,
                "metadata": {
                    "axis": axis,
                    "target_property": target,
                    "contrast_property": contrast,
                    "style_id": style_id,
                    "length_bucket": length_bucket,
                    "evidence_strength": evidence_strength,
                    "label_mode": label_mode,
                    "agency_subfacet": agency_subfacet,
                    "trait_word_should_appear": label_mode != "implicit",
                    "contains_exact_trait_word": contains_trait_word(text, target),
                    "contains_negative_trait": False,
                },
            }
        )
    if "EDIT_OR_CHECK_SOCIAL_PROSE_REQUEST" in prompt:
        scenario_id = _prompt_value(prompt, "SCENARIO_ID", "dry_edit")
        text = _prompt_value(prompt, "TEXT", "") or "Maya made room at the table and invited another student into the card game."
        return json.dumps({"scenario_id": scenario_id, "verdict": "keep", "text": text, "reason": "Behavior-grounded and safe.", "metadata": {}})
    return "{}"


def choose_agency_subfacet(target_trait: str, index_hint: int = 0) -> str:
    if target_trait == "dominant":
        return ["prosocial_leadership", "controlling_dominance"][index_hint % 2]
    if target_trait == "confident":
        return "confident_behavior"
    if target_trait == "submissive":
        return "yielding_or_deferential_behavior"
    return ""


def build_generation_schedule(n: int, *, seed: int) -> list[dict[str, Any]]:
    schedule = []
    rng = random.Random(seed)
    agency_dominant_count = 0
    for i in range(int(n)):
        axis_card = SOCIAL_ONTOLOGY[i % len(SOCIAL_ONTOLOGY)]
        traits = axis_card["anchor_traits"]
        target_trait = traits[(i // len(SOCIAL_ONTOLOGY)) % len(traits)]
        style = DISCOURSE_STYLES[(i * 3) % len(DISCOURSE_STYLES)]
        length_bucket = list(LENGTH_BUCKETS)[i % len(LENGTH_BUCKETS)]
        evidence_strength = EVIDENCE_STRENGTHS[(i // 2) % len(EVIDENCE_STRENGTHS)]
        label_mode = LABEL_MODE_SEQUENCE[i % len(LABEL_MODE_SEQUENCE)]
        agency_subfacet = ""
        if axis_card["axis"] == "agency_status":
            agency_subfacet = choose_agency_subfacet(target_trait, agency_dominant_count)
            if target_trait == "dominant":
                agency_dominant_count += 1
        schedule.append(
            {
                "axis_card": axis_card,
                "target_trait": target_trait,
                "contrast_trait": TRAIT_CONTRASTS.get(target_trait, ""),
                "style_id": style["style_id"],
                "length_bucket": length_bucket,
                "evidence_strength": evidence_strength,
                "label_mode": label_mode,
                "trait_word_should_appear": label_mode != "implicit",
                "agency_subfacet": agency_subfacet,
            }
        )
    rng.shuffle(schedule)
    return schedule


def _call_json(
    generator: Generator,
    *,
    prompt: str,
    label: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    raw_log: list[dict[str, Any]],
    raw_failures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    raw = generator(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stage=label,
        response_format={"type": "json_object"},
    )
    raw_log.append({"label": label, "raw": raw})
    obj = safe_json_loads(raw)
    if isinstance(obj, dict):
        return obj
    raw_failures.append({"label": label, "raw": raw})
    return None


def make_social_property_plan(
    item: dict[str, Any],
    *,
    rng: random.Random,
    generator: Generator,
    raw_log: list[dict[str, Any]],
    raw_failures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    axis_card = item["axis_card"]
    target_trait = item["target_trait"]
    contrast_trait = item["contrast_trait"]
    style_id = item["style_id"]
    length_bucket = item["length_bucket"]
    evidence_strength = item["evidence_strength"]
    label_mode = item["label_mode"]
    agency_subfacet = item.get("agency_subfacet", "")
    scenario_id = (
        f"social_{axis_card['axis']}_{target_trait}_{style_id}_{length_bucket}_"
        f"{evidence_strength}_{rng.randint(0, 999999):06d}"
    )
    prompt = f"""
SCENARIO_PLAN_REQUEST
SCENARIO_ID: {scenario_id}
AXIS: {axis_card['axis']}
TARGET_TRAIT: {target_trait}
CONTRAST_TRAIT: {contrast_trait}
STYLE_ID: {style_id}
LENGTH_BUCKET: {length_bucket}
EVIDENCE_STRENGTH: {evidence_strength}
LABEL_MODE: {label_mode}
TRAIT_WORD_SHOULD_APPEAR: {str(label_mode != 'implicit').lower()}
AGENCY_SUBFACET: {agency_subfacet}

Create one EWoK-style social-property scenario plan.
The trait must be inferred from behavior, not identity.
Use ordinary everyday situations: school, family, neighbors, clubs, group projects, sports, chores, lost-and-found, games, shared supplies.
Reject the example if the trait is inferred from identity, appearance, accent, name, nationality, religion, gender, disability, class markers, family background, or neighborhood.
Avoid these cues entirely as trait evidence: {json.dumps(AVOID_SOCIAL_CUES)}
Avoid appearance-based trait inference. Avoid saying one weak action proves a permanent trait.

Evidence-strength rule:
- single_moment: avoid permanent labels like "is trustworthy" or "is hostile".
- repeated_pattern: stronger wording is allowed, but the behavior must be repeated.

Trait-label rule:
- implicit: imply the trait through behavior without using the exact target trait label.
- evidence_then_label: concrete behavior comes first, followed by a cautious exact trait label.
- label_then_evidence: the exact trait label appears early, then concrete behavior immediately grounds it.
- If the exact trait word appears, it must be earned by concrete behavior, not used as a naked label.

Contrast rule:
- Use only contrast_property as a strict contrast.
- TRAIT_NEAR_CONTRASTS are nearby concepts for soft comparisons or audits, not strict opposites.
- Do not treat confident as the strict opposite of shy.
- Do not treat submissive as the strict opposite of confident.

Agency/status rules:
- For dominant, include both prosocial leadership and controlling dominance across examples.
- Do not collapse dominant into bad or bossy.
- Do not collapse confident into dominant.
- Do not collapse submissive into shy.

Axis card:
{json.dumps(axis_card, indent=2)}

Return exactly this JSON schema:
{{
  "scenario_id": "{scenario_id}",
  "axis": "{axis_card['axis']}",
  "target_property": "{target_trait}",
  "contrast_property": "{contrast_trait}",
  "agent": "...",
  "setting": "...",
  "behavior_evidence": [],
  "social_inference": "...",
  "evidence_strength": "{evidence_strength}",
  "label_mode": "{label_mode}",
  "trait_word_should_appear": {str(label_mode != 'implicit').lower()},
  "agency_subfacet": "{agency_subfacet}",
  "caveat": "",
  "style_id": "{style_id}",
  "length_bucket": "{length_bucket}",
  "avoid_cues": [],
  "safety_notes": []
}}
""".strip()
    obj = _call_json(
        generator,
        prompt=prompt,
        label=f"plan:{scenario_id}",
        system_prompt=JSON_SYSTEM_PROMPT,
        max_tokens=1200,
        temperature=0.35,
        raw_log=raw_log,
        raw_failures=raw_failures,
    )
    if obj is None:
        return None
    obj.update(
        {
            "scenario_id": scenario_id,
            "axis": axis_card["axis"],
            "target_property": target_trait,
            "contrast_property": contrast_trait,
            "style_id": style_id,
            "length_bucket": length_bucket,
            "evidence_strength": evidence_strength,
            "label_mode": label_mode,
            "trait_word_should_appear": label_mode != "implicit",
            "agency_subfacet": agency_subfacet,
        }
    )
    obj.setdefault("behavior_evidence", [])
    obj.setdefault("avoid_cues", list(AVOID_SOCIAL_CUES))
    obj.setdefault("safety_notes", [])
    return obj


def render_social_prose(
    scenario_plan: dict[str, Any],
    *,
    generator: Generator,
    raw_log: list[dict[str, Any]],
    raw_failures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    scenario_id = scenario_plan["scenario_id"]
    target_trait = scenario_plan["target_property"]
    length_bucket = scenario_plan.get("length_bucket", "short_64")
    length_info = LENGTH_BUCKETS.get(length_bucket, LENGTH_BUCKETS["short_64"])
    label_mode = scenario_plan.get("label_mode", "implicit")
    style_card = STYLE_BY_ID.get(scenario_plan.get("style_id", ""), DISCOURSE_STYLES[0])
    prompt = f"""
RENDER_SOCIAL_PROSE_REQUEST
SCENARIO_ID: {scenario_id}
AXIS: {scenario_plan.get('axis', '')}
TARGET_TRAIT: {target_trait}
CONTRAST_TRAIT: {scenario_plan.get('contrast_property', '')}
STYLE_ID: {style_card.get('style_id', '')}
LENGTH_BUCKET: {length_bucket}
EVIDENCE_STRENGTH: {scenario_plan.get('evidence_strength', 'single_moment')}
LABEL_MODE: {label_mode}
TRAIT_WORD_SHOULD_APPEAR: {str(label_mode != 'implicit').lower()}
AGENCY_SUBFACET: {scenario_plan.get('agency_subfacet', '')}

Render the scenario plan as natural prose.
Requirements:
- Return only valid JSON.
- Write natural prose only inside the text field.
- No bullets, labels, tables, benchmark format, or C1/T1 formatting.
- Do not sound like a moral quiz or worksheet.
- Do not over-explain every inference.
- Do not add identity-based explanations.
- The trait must follow from behavior in the scene, not identity, appearance, accent, name, nationality, religion, gender, disability, class markers, family background, or neighborhood.
- Follow label_mode exactly.
- implicit: imply the target property without using the exact target trait word.
- evidence_then_label: describe concrete behavior first, then use the exact target trait word cautiously.
- label_then_evidence: use the exact target trait word early, then immediately explain it with concrete behavior.
- If evidence_strength is single_moment, avoid permanent labels like "is {target_trait}".
- Do not collapse quiet or introverted behavior into coldness.
- Do not turn confidence into boastfulness unless the person is actually bragging or taking undue credit.
- Do not turn dominance into hostility unless the behavior is controlling or cold.
- Avoid overusing "because", "therefore", "this shows", "this proves", and "that means".
- Target length: {length_info['description']}.

Style card:
{json.dumps(style_card, indent=2)}

Scenario plan:
{json.dumps(scenario_plan, indent=2)}

Return exactly this JSON schema:
{{
  "scenario_id": "{scenario_id}",
  "status": "draft",
  "text": "...",
  "metadata": {{
    "axis": "{scenario_plan.get('axis', '')}",
    "target_property": "{target_trait}",
    "contrast_property": "{scenario_plan.get('contrast_property', '')}",
    "style_id": "{style_card.get('style_id', '')}",
    "length_bucket": "{length_bucket}",
    "evidence_strength": "{scenario_plan.get('evidence_strength', 'single_moment')}",
    "label_mode": "{label_mode}",
    "agency_subfacet": "{scenario_plan.get('agency_subfacet', '')}",
    "trait_word_should_appear": {str(label_mode != 'implicit').lower()},
    "contains_exact_trait_word": false,
    "contains_negative_trait": false
  }}
}}
""".strip()
    obj = _call_json(
        generator,
        prompt=prompt,
        label=f"render:{scenario_id}",
        system_prompt=PROSE_SYSTEM_PROMPT,
        max_tokens=int(length_info["max_tokens"]) + 260,
        temperature=0.65,
        raw_log=raw_log,
        raw_failures=raw_failures,
    )
    if obj is None:
        return None
    text = str(obj.get("text", "")).strip()
    if not text:
        raw_failures.append({"label": f"render_empty:{scenario_id}", "raw": obj})
        return None
    metadata = dict(obj.get("metadata", {}))
    metadata.update(
        {
            "axis": scenario_plan.get("axis"),
            "target_property": target_trait,
            "contrast_property": scenario_plan.get("contrast_property", ""),
            "style_id": style_card.get("style_id"),
            "length_bucket": length_bucket,
            "evidence_strength": scenario_plan.get("evidence_strength", "single_moment"),
            "label_mode": label_mode,
            "agency_subfacet": scenario_plan.get("agency_subfacet", ""),
            "trait_word_should_appear": label_mode != "implicit",
            "contains_exact_trait_word": contains_trait_word(text, target_trait),
            "contains_negative_trait": any(contains_trait_word(text, trait) for trait, valence in TRAIT_VALENCE.items() if valence == "negative"),
        }
    )
    return {"scenario_id": scenario_id, "status": "draft", "text": text, "metadata": metadata, "scenario_plan": scenario_plan}


def edit_or_check_social_prose(
    example: dict[str, Any],
    *,
    generator: Generator,
    raw_log: list[dict[str, Any]],
    raw_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    scenario_id = example["scenario_id"]
    prompt = f"""
EDIT_OR_CHECK_SOCIAL_PROSE_REQUEST
SCENARIO_ID: {scenario_id}
TEXT: {example.get('text', '')}

Review this EWoK-style social-property example.
Editor requirements:
- Preserve the intended social-property inference.
- Reject protected-attribute or identity-based trait inference.
- The trait must be inferred from behavior in the scene.
- Respect metadata.label_mode.
- If label_mode is implicit, remove the exact target trait word and let behavior carry the inference.
- If label_mode is evidence_then_label, exact trait wording is allowed after concrete behavioral evidence.
- If label_mode is label_then_evidence, exact trait wording is allowed early, but concrete behavior must immediately ground it.
- Do not treat exact trait labels as errors when label_mode requests them; only revise naked or overconfident labels.
- Soften overconfident permanent trait labels, especially when evidence_strength is single_moment.
- Avoid moralizing phrasing like "good person" or "bad person".
- Avoid turning quiet or introverted behavior into coldness.
- Avoid turning confidence into boastfulness unless the person is actually bragging or taking undue credit.
- Avoid turning dominance into hostility unless the behavior is controlling or cold.
- Avoid adding new facts.

Return only valid JSON with this schema:
{{
  "scenario_id": "{scenario_id}",
  "verdict": "keep",
  "text": "...",
  "reason": "...",
  "metadata": {{}}
}}
Use verdict "keep", "revise", or "reject".

Example:
{json.dumps(example, indent=2)}
""".strip()
    obj = _call_json(
        generator,
        prompt=prompt,
        label=f"edit:{scenario_id}",
        system_prompt=EDITOR_SYSTEM_PROMPT,
        max_tokens=1100,
        temperature=0.2,
        raw_log=raw_log,
        raw_failures=raw_failures,
    )
    if obj is None:
        return {
            "scenario_id": scenario_id,
            "verdict": "reject",
            "text": example.get("text", ""),
            "reason": "Unparseable editor output.",
            "metadata": dict(example.get("metadata", {})),
            "scenario_plan": example.get("scenario_plan", {}),
        }
    verdict = str(obj.get("verdict", "keep")).strip().lower()
    if verdict not in {"keep", "revise", "reject"}:
        verdict = "revise"
    text = str(obj.get("text") or example.get("text", "")).strip()
    metadata = dict(example.get("metadata", {}))
    metadata.update(obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {})
    plan = example.get("scenario_plan", {}) if isinstance(example.get("scenario_plan"), dict) else {}
    metadata.setdefault("label_mode", plan.get("label_mode", "implicit"))
    metadata.setdefault("trait_word_should_appear", metadata.get("label_mode") != "implicit")
    metadata["editor_verdict"] = verdict
    metadata["editor_reason"] = str(obj.get("reason", "")).strip()
    metadata["contains_exact_trait_word"] = contains_trait_word(text, metadata.get("target_property", ""))
    metadata["contains_negative_trait"] = any(contains_trait_word(text, trait) for trait, valence in TRAIT_VALENCE.items() if valence == "negative")
    return {"scenario_id": scenario_id, "verdict": verdict, "text": text, "reason": str(obj.get("reason", "")).strip(), "metadata": metadata, "scenario_plan": plan}


def add_flag(row: dict[str, Any], code: str, severity: str, message: str) -> None:
    row.setdefault("audit_flags", []).append({"code": code, "severity": severity, "message": message})


def connective_counts(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {pattern: len(re.findall(rf"\b{re.escape(pattern)}\b", lowered)) for pattern in CONNECTIVE_PATTERNS}


def possible_names(text: str) -> list[str]:
    return [name for name in NAME_RE.findall(text) if name not in STOP_CAPITALIZED]


def length_bucket_for_word_count(count: int) -> str:
    if count <= 65:
        return "short_64"
    if count <= 130:
        return "medium_128"
    return "long_256"


def scenario_safety_text(row: dict[str, Any]) -> str:
    plan = row.get("scenario_plan", {}) if isinstance(row.get("scenario_plan"), dict) else {}
    return json.dumps(
        {
            "text": row.get("text", ""),
            "agent": plan.get("agent", ""),
            "setting": plan.get("setting", ""),
            "behavior_evidence": plan.get("behavior_evidence", []),
            "social_inference": plan.get("social_inference", ""),
        },
        ensure_ascii=False,
    )


def template_match_code(text: str, target_trait: str) -> str | None:
    lowered = text.lower()
    if target_trait == "trustworthy" and ("returned the lost wallet" in lowered or "lost wallet" in lowered):
        return "trustworthy_returned_wallet"
    if target_trait == "boastful" and ("bragged about" in lowered and "trophy" in lowered):
        return "boastful_trophy_brag"
    if target_trait == "shy" and ("hid in the corner" in lowered or "hiding in the corner" in lowered):
        return "shy_hid_in_corner"
    if target_trait == "dominant" and ("gave orders" in lowered or "ordered everyone" in lowered):
        return "dominant_gave_orders"
    return None


def uses_is_trait(text: str, trait: str) -> bool:
    return bool(trait and re.search(rf"\b(is|was|were)\s+(?:very\s+|really\s+|quite\s+)?{re.escape(trait)}\b", text, flags=re.I))


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def audit_social_examples(
    examples: list[dict[str, Any]],
    *,
    raw_failures: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_failures = raw_failures or []
    audited: list[dict[str, Any]] = []
    normalized_seen: dict[str, str] = {}
    opening5: Counter[str] = Counter()
    opening8: Counter[str] = Counter()
    opening12: Counter[str] = Counter()
    text_sets: list[set[str]] = []
    template_matches: dict[str, list[str]] = defaultdict(list)

    for index, original in enumerate(examples):
        row = copy.deepcopy(original)
        row.setdefault("scenario_id", row.get("id", f"row_{index:04d}"))
        row["audit_flags"] = []
        row["audit_status"] = "keep"
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            add_flag(row, "unparseable_or_empty_text", "hard_drop", "Missing or non-string text.")
            row["audit_status"] = "drop"
            audited.append(row)
            text_sets.append(set())
            continue

        metadata = row.setdefault("metadata", {})
        plan = row.get("scenario_plan", {}) if isinstance(row.get("scenario_plan"), dict) else {}
        target_trait = str(metadata.get("target_property") or plan.get("target_property") or "")
        contrast_trait = str(metadata.get("contrast_property") or plan.get("contrast_property") or "")
        evidence_strength = str(metadata.get("evidence_strength") or plan.get("evidence_strength") or "single_moment")
        label_mode = str(metadata.get("label_mode") or plan.get("label_mode") or "")
        if label_mode not in LABEL_MODES:
            label_mode = "evidence_then_label" if metadata.get("trait_word_should_appear") else "implicit"
        agency_subfacet = str(metadata.get("agency_subfacet") or plan.get("agency_subfacet") or "")
        metadata.update(
            {
                "target_property": target_trait,
                "contrast_property": contrast_trait,
                "evidence_strength": evidence_strength,
                "label_mode": label_mode,
                "trait_word_should_appear": label_mode != "implicit",
                "agency_subfacet": agency_subfacet,
            }
        )

        clean = normalize_text(text)
        if clean in normalized_seen:
            add_flag(row, "exact_normalized_duplicate", "hard_drop", f"Exact normalized duplicate of {normalized_seen[clean]}.")
        else:
            normalized_seen[clean] = row["scenario_id"]

        safety_text = scenario_safety_text(row)
        if IDENTITY_INFERENCE_RE.search(safety_text):
            add_flag(row, "protected_attribute_or_identity_evidence", "hard_drop", "Trait appears tied to identity or protected cues.")
        elif SAFETY_CUE_RE.search(safety_text):
            add_flag(row, "safety_cue_present", "warn", "Safety cue appears; inspect whether it is evidence or only setting text.")
        if APPEARANCE_RE.search(safety_text) and any(contains_trait_word(text, t) for t in ALL_TRAITS):
            add_flag(row, "appearance_based_trait_inference", "hard_drop", "Trait may be tied to appearance rather than behavior.")

        if target_trait == "shy" and contrast_trait == "confident":
            add_flag(row, "trait_contrast_sanity", "warn", "confident is a near contrast for shy, not a strict contrast.")
        if target_trait == "confident" and contrast_trait in {"shy", "submissive"}:
            add_flag(row, "trait_contrast_sanity", "warn", f"{contrast_trait} should not be a strict contrast for confident.")
        if target_trait == "submissive" and contrast_trait == "confident":
            add_flag(row, "trait_contrast_sanity", "warn", "confident should not be the strict contrast for submissive.")

        lowered = text.lower()
        for pattern in OVERCONFIDENT_PATTERNS:
            if re.search(rf"\b{re.escape(pattern)}\b", lowered):
                add_flag(row, "overconfident_language", "warn", f"Contains overconfident phrase: {pattern}.")
        if evidence_strength == "single_moment" and uses_is_trait(text, target_trait):
            add_flag(row, "single_moment_is_trait", "warn", "Single-moment evidence uses 'is/was [trait]' wording.")

        conn = connective_counts(text)
        if conn.get("therefore", 0) or conn.get("this proves", 0) or conn.get("that means", 0) or conn.get("this shows", 0):
            add_flag(row, "worksheet_connective", "warn", "Contains therefore/this shows/this proves/that means phrasing.")
        if sum(conn.values()) >= 4:
            add_flag(row, "connective_heavy", "warn", "Uses many explicit connective phrases.")

        exact_trait = contains_trait_word(text, target_trait) if target_trait else False
        metadata["contains_exact_trait_word"] = exact_trait
        label_idx = trait_word_index(text, target_trait) if exact_trait else None
        if label_mode == "implicit" and exact_trait:
            add_flag(row, "unexpected_exact_trait_label", "warn", "Implicit label mode contains the exact target trait.")
        if label_mode != "implicit" and not exact_trait:
            add_flag(row, "missing_requested_trait_label", "warn", "Explicit label mode omitted the exact target trait.")
        if label_mode == "evidence_then_label" and label_idx is not None and label_idx < 12:
            add_flag(row, "label_before_evidence", "warn", "Evidence-then-label placed the trait word too early.")
        if label_mode == "label_then_evidence" and label_idx is not None:
            text_word_count = len(words(text))
            if label_idx > max(12, text_word_count // 4):
                add_flag(row, "label_too_late", "warn", "Label-then-evidence placed the trait word too late.")

        word_count = len(words(text))
        metadata["word_count"] = word_count
        expected_bucket = str(metadata.get("length_bucket", ""))
        if expected_bucket in LENGTH_BUCKETS:
            low, high = LENGTH_BUCKETS[expected_bucket]["word_range"]
            if word_count < low or word_count > high:
                add_flag(row, "length_bucket_mismatch", "warn", f"{word_count} words outside {expected_bucket} range {low}-{high}.")
        metadata["observed_length_bucket"] = length_bucket_for_word_count(word_count)

        tcode = template_match_code(text, target_trait)
        if tcode is not None:
            template_matches[tcode].append(row["scenario_id"])

        if any(flag["severity"] == "hard_drop" for flag in row["audit_flags"]):
            row["audit_status"] = "drop"
        audited.append(row)
        opening5[first_n_words(text, 5)] += 1
        opening8[first_n_words(text, 8)] += 1
        opening12[first_n_words(text, 12)] += 1
        text_sets.append(set(words(text)))

    for row in audited:
        if row.get("audit_status") == "drop" or not row.get("text"):
            continue
        for n, openings in [(5, opening5), (8, opening8), (12, opening12)]:
            op = first_n_words(row["text"], n)
            if op and openings[op] > 1:
                add_flag(row, f"repeated_first_{n}_words", "warn", f"Opening {n} words repeated {openings[op]} times.")

    for i in range(len(audited)):
        for j in range(i + 1, len(audited)):
            if audited[i].get("audit_status") == "drop" or audited[j].get("audit_status") == "drop":
                continue
            sim = jaccard_similarity(text_sets[i], text_sets[j])
            if sim > 0.80:
                add_flag(audited[i], "near_duplicate_jaccard", "warn", f"Near duplicate of {audited[j]['scenario_id']} (Jaccard {sim:.2f}).")
                add_flag(audited[j], "near_duplicate_jaccard", "warn", f"Near duplicate of {audited[i]['scenario_id']} (Jaccard {sim:.2f}).")

    trait_counts = Counter(row.get("metadata", {}).get("target_property") for row in audited if row.get("metadata"))
    for code, ids in template_matches.items():
        trait = code.split("_")[0]
        if len(ids) >= 2 and len(ids) / max(1, trait_counts.get(trait, 0)) > 0.50:
            for row in audited:
                if row["scenario_id"] in ids:
                    add_flag(row, "behavior_template_collapse", "warn", f"Template pattern overused: {code}.")

    for row in audited:
        if row.get("audit_status") != "drop" and row.get("audit_flags"):
            row["audit_status"] = "warn"

    kept_rows = [row for row in audited if row.get("audit_status") != "drop"]
    kept_metadata = [row.get("metadata", {}) for row in kept_rows]
    all_metadata = [row.get("metadata", {}) for row in audited]
    exact_label_rows = [m for m in kept_metadata if m.get("contains_exact_trait_word")]
    connective_total: Counter[str] = Counter()
    pronouns: Counter[str] = Counter()
    names: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    hard_drop_counts: Counter[str] = Counter()
    for row in audited:
        for flag in row.get("audit_flags", []):
            flag_counts[flag["code"]] += 1
            if flag["severity"] == "hard_drop":
                hard_drop_counts[flag["code"]] += 1
    for row in kept_rows:
        text = row.get("text", "")
        connective_total.update(connective_counts(text))
        pronouns.update(match.lower() for match in PRONOUN_RE.findall(text))
        names.update(possible_names(text))

    def density_by(key: str) -> dict[str, float]:
        groups: dict[str, list[bool]] = defaultdict(list)
        for meta in kept_metadata:
            groups[str(meta.get(key, "<missing>"))].append(bool(meta.get("contains_exact_trait_word")))
        return {name: sum(vals) / max(1, len(vals)) for name, vals in sorted(groups.items())}

    single_moment_rows = [row for row in kept_rows if row.get("metadata", {}).get("evidence_strength") == "single_moment"]
    single_moment_is_trait_rate = sum(
        uses_is_trait(row.get("text", ""), row.get("metadata", {}).get("target_property", "")) for row in single_moment_rows
    ) / max(1, len(single_moment_rows))
    dominant_rows = [row for row in kept_rows if row.get("metadata", {}).get("target_property") == "dominant"]
    dominant_subfacets = Counter(row.get("metadata", {}).get("agency_subfacet", "<missing>") for row in dominant_rows)
    controlling_rate = dominant_subfacets.get("controlling_dominance", 0) / max(1, len(dominant_rows))

    report = {
        "total_examples": len(audited),
        "kept_examples": sum(row.get("audit_status") == "keep" for row in audited),
        "warned_examples": sum(row.get("audit_status") == "warn" for row in audited),
        "dropped_examples": sum(row.get("audit_status") == "drop" for row in audited),
        "style_distribution": dict(Counter(m.get("style_id", "<missing>") for m in all_metadata)),
        "length_distribution": dict(Counter(m.get("observed_length_bucket", "<missing>") for m in all_metadata)),
        "planned_length_distribution": dict(Counter(m.get("length_bucket", "<missing>") for m in all_metadata)),
        "axis_distribution": dict(Counter(m.get("axis", "<missing>") for m in all_metadata)),
        "target_trait_distribution": dict(Counter(m.get("target_property", "<missing>") for m in all_metadata)),
        "trait_valence_distribution": dict(Counter(TRAIT_VALENCE.get(m.get("target_property"), "<missing>") for m in all_metadata)),
        "evidence_strength_distribution": dict(Counter(m.get("evidence_strength", "<missing>") for m in all_metadata)),
        "label_mode_distribution": dict(Counter(m.get("label_mode", "<missing>") for m in all_metadata)),
        "agency_subfacet_balance": dict(Counter(m.get("agency_subfacet", "<none>") for m in all_metadata if m.get("axis") == "agency_status")),
        "dominant_subfacet_distribution": dict(dominant_subfacets),
        "controlling_dominant_rate": controlling_rate,
        "exact_trait_label_density": len(exact_label_rows) / max(1, len(kept_metadata)),
        "exact_trait_label_target": {"target": 0.50, "warn_below": 0.35, "warn_above": 0.65},
        "exact_trait_label_density_by_axis": density_by("axis"),
        "exact_trait_label_density_by_target_trait": density_by("target_property"),
        "single_moment_is_trait_rate": single_moment_is_trait_rate,
        "connective_counts": dict(connective_total),
        "connective_rate_per_example": {key: value / max(1, len(kept_rows)) for key, value in connective_total.items()},
        "pronoun_counts": dict(pronouns),
        "possible_name_counts": dict(names.most_common(30)),
        "top_repeated_opening_5": opening5.most_common(15),
        "top_repeated_opening_8": opening8.most_common(15),
        "audit_flag_counts": dict(flag_counts),
        "hard_drop_counts": dict(hard_drop_counts),
        "safety_flag_counts": {key: value for key, value in flag_counts.items() if "identity" in key or "protected" in key or "appearance" in key or "safety" in key},
        "raw_output_failures": len(raw_failures),
    }
    recommendations: list[str] = []
    if report["dropped_examples"]:
        recommendations.append("Inspect hard drops before scaling.")
    if report["exact_trait_label_density"] < 0.35:
        recommendations.append("Exact trait labels are sparse; target about 50% explicit labels with 50% implicit examples.")
    if report["exact_trait_label_density"] > 0.65:
        recommendations.append("Exact trait labels are dense; target about 50% explicit labels and increase implicit examples.")
    if report["single_moment_is_trait_rate"] > 0.25:
        recommendations.append("Single-moment examples often use 'is [trait]' wording; soften to acted/seemed/came across.")
    if controlling_rate > 0.75 and len(dominant_rows) >= 3:
        recommendations.append("Dominant examples are overwhelmingly controlling/negative; add prosocial leadership examples.")
    if any(value > 0.40 for value in report["connective_rate_per_example"].values()):
        recommendations.append("Some connective phrases are frequent; ask for more varied causal wording.")
    if flag_counts.get("behavior_template_collapse"):
        recommendations.append("Behavior-template collapse detected; diversify evidence patterns for repeated traits.")
    report["recommendations"] = recommendations
    return audited, report


def run_social_properties_pipeline(
    *,
    config: dict[str, Any],
    generator: Generator,
    model_id: str,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    progress = progress or (lambda message: None)
    n_examples = int(config.get("n_examples", 8))
    seed = int(config.get("seed", 17))
    rng = random.Random(seed)
    out_dir = resolve_path(output_dir or config.get("output_dir", "data/social_properties/qwen3p6_toy"))
    raw_log: list[dict[str, Any]] = []
    raw_failures: list[dict[str, Any]] = []

    progress(f"Scheduling {n_examples} examples")
    schedule = build_generation_schedule(n_examples, seed=seed)

    progress("Generating scenario plans")
    scenario_plans = [
        plan
        for item in schedule
        if (plan := make_social_property_plan(item, rng=rng, generator=generator, raw_log=raw_log, raw_failures=raw_failures)) is not None
    ]

    progress("Rendering prose drafts")
    prose_draft = [
        draft
        for plan in scenario_plans
        if (draft := render_social_prose(plan, generator=generator, raw_log=raw_log, raw_failures=raw_failures)) is not None
    ]

    progress("Editing/checking prose")
    prose_checked = [edit_or_check_social_prose(draft, generator=generator, raw_log=raw_log, raw_failures=raw_failures) for draft in prose_draft]

    progress("Auditing examples")
    audit_input = [row for row in prose_checked if row.get("verdict") != "reject" and row.get("text")]
    examples_audited, audit_report = audit_social_examples(audit_input, raw_failures=raw_failures)
    final_rows = [row for row in examples_audited if row.get("audit_status") != "drop" and row.get("text")]
    train_text = "\n\n".join(row["text"].strip() for row in final_rows)

    paths = {
        "raw_model_outputs": out_dir / "raw_model_outputs.jsonl",
        "scenario_plans": out_dir / "scenario_plans.jsonl",
        "prose_draft": out_dir / "prose_draft.jsonl",
        "prose_checked": out_dir / "prose_checked.jsonl",
        "examples_audited": out_dir / "examples_audited.jsonl",
        "audit_report": out_dir / "audit_report.json",
        "train_pilot": out_dir / "train_pilot.txt",
        "manifest": out_dir / "manifest.json",
    }

    progress(f"Writing outputs to {out_dir}")
    write_jsonl(paths["raw_model_outputs"], raw_log)
    write_jsonl(paths["scenario_plans"], scenario_plans)
    write_jsonl(paths["prose_draft"], prose_draft)
    write_jsonl(paths["prose_checked"], prose_checked)
    write_jsonl(paths["examples_audited"], examples_audited)
    write_json(paths["audit_report"], audit_report)
    paths["train_pilot"].write_text(train_text.strip() + "\n", encoding="utf-8")

    manifest = {
        "version": config.get("version", "social_properties_toy_v0"),
        "description": config.get("description", "Toy EWoK-style social-properties pilot."),
        "model_id": model_id,
        "config_path": str(config_path) if config_path else None,
        "output_dir": str(out_dir),
        "counts": {
            "scheduled": len(schedule),
            "scenario_plans": len(scenario_plans),
            "prose_draft": len(prose_draft),
            "prose_checked": len(prose_checked),
            "examples_audited": len(examples_audited),
            "train_documents": len(final_rows),
            "raw_output_failures": len(raw_failures),
        },
        "audit_summary": {
            "dropped_examples": audit_report.get("dropped_examples"),
            "warned_examples": audit_report.get("warned_examples"),
            "exact_trait_label_density": audit_report.get("exact_trait_label_density"),
            "label_mode_distribution": audit_report.get("label_mode_distribution"),
            "recommendations": audit_report.get("recommendations", []),
        },
        "paths": {key: str(value) for key, value in paths.items()},
    }
    write_json(paths["manifest"], manifest)
    return manifest
