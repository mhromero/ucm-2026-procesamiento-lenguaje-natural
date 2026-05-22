"""Decodificación NER: argmax con P(o) + umbral opcional + restricciones BIO."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fdi_pln_2611_p5.labels import ID2LABEL


def _entity_type_char(label_id: int) -> str | None:
    label = ID2LABEL.get(label_id, "o")
    if label == "o" or len(label) < 2:
        return None
    return label[0]


def repair_bio_label_ids(label_ids: list[int]) -> list[int]:
    """Convierte secuencias BIO inválidas (p. ej. pc sin pi) a 'o'."""
    fixed: list[int] = []
    open_type: str | None = None

    for label_id in label_ids:
        label = ID2LABEL.get(label_id, "o")
        if label == "o":
            fixed.append(0)
            open_type = None
            continue

        etype = _entity_type_char(label_id)
        if etype is None:
            fixed.append(0)
            open_type = None
            continue

        if label.endswith("i"):
            fixed.append(label_id)
            open_type = etype
        elif label.endswith("c"):
            if open_type == etype:
                fixed.append(label_id)
            else:
                fixed.append(0)
                open_type = None
        else:
            fixed.append(0)
            open_type = None

    return fixed


def logits_to_label_ids(
    logits: torch.Tensor,
    entity_threshold: float = 0.5,
    apply_bio_repair: bool = True,
) -> list[int]:
    """Argmax sobre las 5 clases; entidad solo si P(etiqueta) >= umbral y P(etiqueta) > P(o)."""
    probs = F.softmax(logits, dim=-1)
    if probs.dim() == 3:
        probs = probs.squeeze(0)

    pred_ids = probs.argmax(dim=-1).tolist()

    if entity_threshold > 0:
        for i, pid in enumerate(pred_ids):
            if pid == 0:
                continue
            p_ent = probs[i, pid].item()
            p_o = probs[i, 0].item()
            if p_ent < entity_threshold or p_o >= p_ent:
                pred_ids[i] = 0

    if apply_bio_repair:
        pred_ids = repair_bio_label_ids(pred_ids)
    return pred_ids


def batch_logits_to_label_ids(
    logits: torch.Tensor,
    entity_threshold: float = 0.5,
    apply_bio_repair: bool = True,
) -> list[list[int]]:
    return [
        logits_to_label_ids(logits[i], entity_threshold, apply_bio_repair)
        for i in range(logits.size(0))
    ]
