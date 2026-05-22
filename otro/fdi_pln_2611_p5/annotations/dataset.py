from __future__ import annotations

import json
from pathlib import Path

import torch

from fdi_pln_2611_p5.BPETokenizer import BPETokenizer
from fdi_pln_2611_p5.labels import (
    IGNORE_LABEL_ID,
    label_to_id,
    word_labels_to_char_labels,
)


def char_labels_from_merged(sentence: dict) -> tuple[str, list[int]]:
    labels = sentence["labels"]
    if "tokens" in sentence:
        text, char_labels = word_labels_to_char_labels(sentence["tokens"], labels)
    else:
        text = sentence["text"]
        if len(text) != len(labels):
            raise ValueError("Longitud de texto y etiquetas inconsistente.")
        char_labels = labels
    return text, [label_to_id(label) for label in char_labels]


def build_ner_windows(
    sentences: list[dict],
    tokenizer: BPETokenizer,
    window_size: int,
    *,
    stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construye ventanas de tokens y etiquetas para entrenamiento NER.

    Frases más cortas que window_size se incluyen como ventana única con padding
    (token de espacio, etiqueta IGNORE_LABEL_ID).
    Frases más largas generan ventanas deslizantes; el último token queda cubierto.
    """
    step = stride if stride is not None else 1
    if step < 1:
        raise ValueError("stride debe ser >= 1")
    pad_id = tokenizer.padding_token_id()
    x_windows: list[list[int]] = []
    y_windows: list[list[int]] = []

    for sentence in sentences:
        text, char_label_ids = char_labels_from_merged(sentence)
        token_ids, label_ids = tokenizer.encode_with_labels(text, char_label_ids)
        if not token_ids:
            continue
        if len(token_ids) <= window_size:
            pad_len = window_size - len(token_ids)
            x_windows.append(token_ids + [pad_id] * pad_len)
            y_windows.append(label_ids + [IGNORE_LABEL_ID] * pad_len)
        else:
            last_start = len(token_ids) - window_size
            for start in range(0, last_start + 1, step):
                end = start + window_size
                x_windows.append(token_ids[start:end])
                y_windows.append(label_ids[start:end])

    if not x_windows:
        raise ValueError("No hay ventanas NER; revisa las anotaciones fusionadas.")

    return torch.tensor(x_windows, dtype=torch.long), torch.tensor(
        y_windows, dtype=torch.long
    )


def slim_sentence(sentence: dict) -> dict:
    """Solo campos necesarios para entrenamiento / dataset fusionado (por palabra)."""
    out: dict = {
        "frase_id": sentence["frase_id"],
        "text": sentence["text"],
    }
    if "tokens" in sentence:
        out["tokens"] = sentence["tokens"]
    out["labels"] = sentence["labels"]
    return out


def save_merged_dataset(path: Path, sentences: list[dict]) -> None:
    """Guarda únicamente la lista de frases anotadas (sin bloque report)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = [slim_sentence(s) for s in sentences]
    path.write_text(
        json.dumps(slim, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_merged_dataset(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload["sentences"]
