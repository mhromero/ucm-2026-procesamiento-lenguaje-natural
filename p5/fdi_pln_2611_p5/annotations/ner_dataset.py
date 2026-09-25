"""Load merged annotations and build NER training windows.

Bridges the merged annotation JSON format to BPE-tokenized tensors used by
the NER training loop, handling character-to-token label alignment and
sliding-window chunking for long sentences.
"""

from __future__ import annotations

import json
from pathlib import Path

from fdi_pln_2611_p5.paths import require_file

import torch

from fdi_pln_2611_p5.model.lm_causal.bpe_tokenizer import BPETokenizer
from fdi_pln_2611_p5.model.ner.labels import (
    IGNORE_LABEL_ID,
    label_to_id,
    word_labels_to_char_labels,
)


def char_labels_from_merged(sentence: dict) -> tuple[str, list[int]]:
    """Convert a merged sentence dict to text and integer character labels.

    Args:
        sentence: Merged sentence with ``text``, ``labels``, and optional
            ``tokens``.

    Returns:
        Tuple of text and label IDs aligned at character level.

    Raises:
        ValueError: If character-level text and labels differ in length.
    """
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
    """Build token windows and label tensors for NER training.

    Shorter sentences are padded to ``window_size`` with the space token and
    ``IGNORE_LABEL_ID``. Longer sentences produce sliding windows so the final
    token remains covered.

    Args:
        sentences: Merged sentence dicts from the annotation pipeline.
        tokenizer: BPE tokenizer shared with the NER model.
        window_size: Fixed sequence length for each training window.
        stride: Step between sliding windows (defaults to 1).

    Returns:
        Tuple of ``(input_ids, label_ids)`` tensors shaped
        ``(n_windows, window_size)``.

    Raises:
        ValueError: If ``stride`` is invalid or no windows can be built.
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
    """Keep only fields required for training and merged dataset storage.

    Args:
        sentence: Full merged sentence record.

    Returns:
        Slim dict with ``frase_id``, ``text``, optional ``tokens``, and ``labels``.
    """
    out: dict = {
        "frase_id": sentence["frase_id"],
        "text": sentence["text"],
    }
    if "tokens" in sentence:
        out["tokens"] = sentence["tokens"]
    out["labels"] = sentence["labels"]
    return out


def save_merged_dataset(path: Path, sentences: list[dict]) -> None:
    """Persist the merged sentence list without report metadata.

    Args:
        path: Output JSON file path.
        sentences: Merged sentence records to save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = [slim_sentence(s) for s in sentences]
    path.write_text(
        json.dumps(slim, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_merged_dataset(path: Path) -> list[dict]:
    """Load merged sentences from a dataset JSON file.

    Supports both a bare sentence list and a legacy wrapper with a
    ``sentences`` key.

    Args:
        path: Path to the merged dataset JSON file.

    Returns:
        List of merged sentence dicts.
    """
    path = require_file(path, label="annotation dataset")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload["sentences"]
