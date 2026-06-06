"""BabyLM text cleaning utilities.

These rules are adapted from the earlier CustomLlama ``mrclean.py`` prototype.
They intentionally stay small and deterministic so cleaned text artifacts can
be regenerated and compared across tokenizer experiments.
"""

from __future__ import annotations

import re
from collections.abc import Callable


START_TOKEN = ""
END_TOKEN = ""
PADDING_TOKEN = ""


def _make_padding_sequence(seq_length: int) -> str:
    return "".join([END_TOKEN] + seq_length * [PADDING_TOKEN])


def cleanup_simple_wikipedia(text: str, seq_length: int) -> str:
    pad_seq = _make_padding_sequence(seq_length)
    return START_TOKEN + re.sub(r"\n\n", pad_seq + START_TOKEN, text) + pad_seq


def cleanup_wikipedia(text: str, seq_length: int) -> str:
    pad_seq = _make_padding_sequence(seq_length)
    text = re.sub(r"= = = (.+?) = = =\n", r"\1", text)
    lines = [line.strip() for line in text.splitlines()]
    return START_TOKEN + re.sub(r"\n\n", pad_seq + START_TOKEN, "\n".join(lines)[1:]) + pad_seq


def cleanup_qed(text: str, seq_length: int) -> str:
    punctuation_ex = re.compile(r"([.!?]\s*)")
    unimportant_chars_ex = re.compile(r"\(.*?\)|[.!?]")
    lines = []
    for line in text.splitlines():
        if line:
            line_body = unimportant_chars_ex.sub("", line)
            if line_body:
                f_upper = sum(c.isupper() for c in line_body) / len(line_body)
                if f_upper >= 0.5:
                    split_on_punctuation = punctuation_ex.split(line.replace("l", "I"))
                    line = "".join(sentence.capitalize() for sentence in split_on_punctuation)
        lines.append(line.strip())
    return START_TOKEN + "\n".join(lines) + END_TOKEN + "".join(seq_length * [PADDING_TOKEN])


def cleanup_extra_spaces(text: str) -> str:
    multiple_spaces_ex = re.compile(r"[ \t\u00A0]+")
    space_before_punctuation_ex = re.compile(r"[ \t\u00A0]([.,;!?])")
    text = multiple_spaces_ex.sub(" ", text)
    return space_before_punctuation_ex.sub(r"\1", text)


def cleanup_bnc_spoken(text: str, seq_length: int) -> str:
    pad_seq = _make_padding_sequence(seq_length)
    text = cleanup_extra_spaces(text)
    return START_TOKEN + re.sub(r"\n\n", pad_seq + START_TOKEN, text) + pad_seq


def cleanup_aochildes(text: str, seq_length: int) -> str:
    text = cleanup_extra_spaces(text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)


def cleanup_cbt(text: str, seq_length: int) -> str:
    text = cleanup_extra_spaces(text)
    space_before_apostrophe = re.compile(r"([\w\d])[ \t\u00A0](['’]\w)")
    text = space_before_apostrophe.sub(r"\1\2", text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)


def cleanup_children_stories(text: str, seq_length: int) -> str:
    return START_TOKEN + text + _make_padding_sequence(seq_length)


def cleanup_gutenberg(text: str, seq_length: int) -> str:
    return text + "".join(seq_length * [PADDING_TOKEN])


def cleanup_open_subtitles(text: str, seq_length: int) -> str:
    subtitle_credit_ex = re.compile(r"^.*subtitle.*$\n", re.MULTILINE | re.IGNORECASE)
    text = subtitle_credit_ex.sub("", text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)


def cleanup_switchboard(text: str, seq_length: int) -> str:
    return text + "".join(seq_length * [PADDING_TOKEN])


CLEANUP_FUNCTIONS: dict[str, Callable[[str, int], str]] = {
    "bnc_spoken": cleanup_bnc_spoken,
    "cbt": cleanup_cbt,
    "childes": cleanup_aochildes,
    "children_stories": cleanup_children_stories,
    "gutenberg": cleanup_gutenberg,
    "open_subtitles": cleanup_open_subtitles,
    "qed": cleanup_qed,
    "simple_wiki": cleanup_simple_wikipedia,
    "switchboard": cleanup_switchboard,
    "wikipedia": cleanup_wikipedia,
}


def clean_text(source: str, text: str, *, seq_length: int = 128) -> str:
    """Clean one BabyLM source by source stem."""

    try:
        cleanup_fn = CLEANUP_FUNCTIONS[source]
    except KeyError as exc:
        known = ", ".join(sorted(CLEANUP_FUNCTIONS))
        raise ValueError(f"No cleanup function registered for {source!r}. Known: {known}") from exc
    return cleanup_fn(text, seq_length)


def known_sources() -> list[str]:
    """Return source names with registered cleaning rules."""

    return sorted(CLEANUP_FUNCTIONS)
