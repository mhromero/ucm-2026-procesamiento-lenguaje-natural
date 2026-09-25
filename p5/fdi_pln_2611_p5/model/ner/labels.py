"""NER label schema used in annotation JSON (internal token format)."""

from __future__ import annotations

# o = outside; pi/pc = person (begin/continuation); li/lc = location (begin/continuation)
LABEL2ID: dict[str, int] = {
    "o": 0,
    "pi": 1,
    "pc": 2,
    "li": 3,
    "lc": 4,
}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}
IGNORE_LABEL_ID = -1

PREFIX_TO_ENTITY_TYPE: dict[str, str] = {
    "p": "PER",
    "l": "LOC",
}

CONTINUATION_LABEL: dict[str, str] = {
    "pi": "pc",
    "li": "lc",
}


def label_to_id(label: str) -> int:
    """Map a label string to its numeric id.

    Args:
        label: Label string; empty or unknown values map to ``"o"``.

    Returns:
        Numeric label id.
    """
    return LABEL2ID.get(label or "o", 0)


def entity_type_from_label(label: str) -> str | None:
    """Map a BIO label to its coarse entity type.

    Args:
        label: Internal label string (e.g. ``"pi"``, ``"lc"``).

    Returns:
        Entity type string such as ``"PER"`` or ``"LOC"``, or ``None`` for ``"o"``.
    """
    if not label or label == "o" or len(label) < 2:
        return None
    return PREFIX_TO_ENTITY_TYPE.get(label[0])


def expand_word_label_to_char_labels(token: str, label: str) -> list[str]:
    """Expand a word-level label to per-character BIO labels within the word.

    The first character receives the begin label (``pi``/``li``); remaining
    characters receive the continuation label (``pc``/``lc``).

    Args:
        token: Word text.
        label: Word-level label.

    Returns:
        Character-level label list aligned with ``token``.
    """
    if not token:
        return []
    if label == "o":
        return ["o"] * len(token)
    if label in ("pc", "lc"):
        return [label] * len(token)
    if label in CONTINUATION_LABEL:
        cont = CONTINUATION_LABEL[label]
        return [label] + [cont] * (len(token) - 1)
    return ["o"] * len(token)


def word_labels_to_char_labels(
    tokens: list[str], labels: list[str]
) -> tuple[str, list[str]]:
    """Expand word-level labels to character-level BIO labels.

    Args:
        tokens: Token strings.
        labels: Word-level labels aligned with ``tokens``.

    Returns:
        Tuple of concatenated text and character-level label list.

    Raises:
        ValueError: If ``tokens`` and ``labels`` differ in length or the expanded
            labels do not match the concatenated text length.
    """
    if len(tokens) != len(labels):
        raise ValueError("tokens y labels deben tener la misma longitud.")
    text = "".join(tokens)
    char_labels: list[str] = []
    for token, label in zip(tokens, labels):
        char_labels.extend(expand_word_label_to_char_labels(token, label))
    if len(char_labels) != len(text):
        raise ValueError("Longitud de etiquetas por carácter inconsistente.")
    return text, char_labels


def merge_subword_labels(left: int, right: int) -> int:
    """Combine label ids when merging two BPE subword tokens.

    Args:
        left: Label id of the left subword.
        right: Label id of the right subword.

    Returns:
        Merged label id for the combined token.
    """
    if left == right:
        return left
    if left == 0:
        return right
    if right == 0:
        return left

    left_label = ID2LABEL[left]
    right_label = ID2LABEL[right]

    if left_label[0] == right_label[0]:
        if left_label.endswith("c") or right_label.endswith("c"):
            return right if right_label.endswith("c") else left
        return left

    if right_label.endswith("i"):
        return right
    if left_label.endswith("i"):
        return left
    if right_label.endswith("c"):
        return right
    if left_label.endswith("c"):
        return left
    return left
