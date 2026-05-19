"""Esquema de etiquetas NER usado en los JSON de anotación (token interno)."""

from __future__ import annotations

# o = fuera; pi/pc = persona (inicio/continuación); li/lc = lugar (inicio/continuación)
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


def label_to_id(label: str) -> int:
    return LABEL2ID.get(label or "o", 0)


def entity_type_from_label(label: str) -> str | None:
    if not label or label == "o" or len(label) < 2:
        return None
    return PREFIX_TO_ENTITY_TYPE.get(label[0])


def word_labels_to_char_labels(
    tokens: list[str], labels: list[str]
) -> tuple[str, list[str]]:
    """Expande etiquetas por palabra a etiquetas por carácter."""
    if len(tokens) != len(labels):
        raise ValueError("tokens y labels deben tener la misma longitud.")
    text = "".join(tokens)
    char_labels: list[str] = []
    for token, label in zip(tokens, labels):
        char_labels.extend([label] * len(token))
    if len(char_labels) != len(text):
        raise ValueError("Longitud de etiquetas por carácter inconsistente.")
    return text, char_labels


def merge_subword_labels(left: int, right: int) -> int:
    """Combina etiquetas al fusionar dos subpalabras BPE."""
    if left == right:
        return left
    if left == 0:
        return right
    if right == 0:
        return left

    left_label = ID2LABEL[left]
    right_label = ID2LABEL[right]
    if left_label[0] == right_label[0]:
        if left_label.endswith("i"):
            return left
        if right_label.endswith("i"):
            return right
        return left
    if left_label.endswith("i"):
        return left
    if right_label.endswith("i"):
        return right
    return left
