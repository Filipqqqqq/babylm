"""Small deterministic number rendering helpers for repair data."""

from __future__ import annotations


_ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def number_to_words(value: int) -> str:
    """Return an English cardinal for integers from 0 through 100."""

    value = int(value)
    if value < 0 or value > 100:
        raise ValueError("number_to_words supports only integers from 0 through 100.")
    if value in _ONES:
        return _ONES[value]
    if value in _TENS:
        return _TENS[value]
    if value == 100:
        return "one hundred"
    tens = (value // 10) * 10
    ones = value % 10
    return f"{_TENS[tens]}-{_ONES[ones]}"


def simple_plural(singular: str) -> str:
    """Return a conservative plural for common concrete nouns."""

    if singular.endswith(("s", "x", "ch", "sh")):
        return f"{singular}es"
    if singular.endswith("y") and singular[-2:-1].lower() not in "aeiou":
        return f"{singular[:-1]}ies"
    return f"{singular}s"


def quantity_surface(value: int, *, mode: str) -> str:
    """Render a numeric quantity as a word, digit, or zero-only no-form."""

    value = int(value)
    if mode == "word":
        return number_to_words(value)
    if mode == "digit":
        return str(value)
    if mode == "no_for_zero":
        if value != 0:
            return number_to_words(value)
        return "no"
    raise ValueError(f"Unsupported quantity render mode: {mode!r}")


def quantity_answer(value: int, *, mode: str) -> str:
    """Return a short answer for a how-many question."""

    value = int(value)
    if value == 0 and mode == "no_for_zero":
        return "None."
    surface = quantity_surface(value, mode=mode)
    return f"{surface.capitalize()}."


def quantified_object_phrase(
    value: int,
    *,
    singular: str,
    plural: str | None = None,
    mode: str,
) -> str:
    """Return phrases like ``one apple``, ``seven apples``, or ``no apples``."""

    value = int(value)
    plural = simple_plural(singular) if plural is None else plural
    noun = singular if value == 1 else plural
    return f"{quantity_surface(value, mode=mode)} {noun}"
