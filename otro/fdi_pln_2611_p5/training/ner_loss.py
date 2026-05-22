"""Pérdidas y pesos de clase para NER con desbalance fuerte hacia 'o'."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fdi_pln_2611_p5.labels import IGNORE_LABEL_ID


def compute_class_weights(
    y_train: torch.Tensor,
    num_labels: int,
    *,
    mode: str = "effective_num",
    beta: float = 0.9999,
    entity_boost: float = 3.0,
) -> torch.Tensor:
    """Pesos por clase; por defecto favorece fuertemente etiquetas distintas de 'o'."""
    flat = y_train.view(-1)
    flat = flat[flat != IGNORE_LABEL_ID]
    counts = torch.bincount(flat, minlength=num_labels).float().clamp(min=1.0)

    if mode == "inverse":
        weights = flat.numel() / (num_labels * counts)
    elif mode == "sqrt_inverse":
        weights = torch.sqrt(flat.numel() / (num_labels * counts))
    elif mode == "effective_num":
        effective = (1.0 - beta**counts) / (1.0 - beta)
        weights = (1.0 - beta) / effective
    else:
        raise ValueError(f"Modo de pesos desconocido: {mode}")

    if entity_boost != 1.0 and num_labels > 1:
        # Refuerzo en inicio/continuación; lc (muy raro) sin multiplicar tanto
        for idx in (1, 2, 3):
            weights[idx] *= entity_boost
        weights[4] *= max(1.0, entity_boost**0.5)

    weights = weights / weights.mean()
    return weights


class FocalLoss(nn.Module):
    """Focal loss multiclase; reduce el peso de ejemplos fáciles (típicamente 'o')."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = IGNORE_LABEL_ID,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        mask = targets != self.ignore_index
        if not mask.any():
            return ce.sum() * 0.0
        ce = ce[mask]
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def build_ner_loss(
    loss_name: str,
    class_weights: torch.Tensor | None,
    *,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    if loss_name == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=IGNORE_LABEL_ID,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Pérdida NER desconocida: {loss_name}")
