"""Loss functions and class weights for heavily imbalanced NER training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fdi_pln_2611_p5.model.ner.labels import IGNORE_LABEL_ID


def compute_class_weights(
    y_train: torch.Tensor,
    num_labels: int,
    *,
    mode: str = "effective_num",
    beta: float = 0.9999,
    entity_boost: float = 3.0,
) -> torch.Tensor:
    """Compute per-class weights that up-weight non-``o`` entity labels.

    Args:
        y_train: Training label tensor (ignored positions excluded internally).
        num_labels: Number of NER label ids.
        mode: Weighting scheme (``inverse``, ``sqrt_inverse``, or ``effective_num``).
        beta: Effective-number smoothing parameter when ``mode`` is ``effective_num``.
        entity_boost: Multiplier applied to entity start/continuation classes.

    Returns:
        Normalized class weight tensor of shape ``(num_labels,)``.
    """
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
        raise ValueError(f"Unknown class-weight mode: {mode}")

    if entity_boost != 1.0 and num_labels > 1:
        # Boost start/continuation labels; apply a smaller boost to rare ``lc``.
        for idx in (1, 2, 3):
            weights[idx] *= entity_boost
        weights[4] *= max(1.0, entity_boost**0.5)

    weights = weights / weights.mean()
    return weights


class FocalLoss(nn.Module):
    """Multi-class focal loss that down-weights easy examples (typically ``o``)."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = IGNORE_LABEL_ID,
    ):
        """Initialize focal loss.

        Args:
            gamma: Focusing parameter; higher values emphasize hard examples.
            weight: Optional per-class weights passed to cross-entropy.
            ignore_index: Label id to exclude from the loss.
        """
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean focal loss over non-ignored targets."""
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
    """Build the NER loss module from configuration.

    Args:
        loss_name: Either ``focal`` or ``ce``.
        class_weights: Optional per-class weights.
        focal_gamma: Gamma for focal loss when ``loss_name`` is ``focal``.
        label_smoothing: Label smoothing for cross-entropy when ``loss_name`` is ``ce``.

    Returns:
        A ``nn.Module`` loss callable.

    Raises:
        ValueError: If ``loss_name`` is not recognized.
    """
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    if loss_name == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=IGNORE_LABEL_ID,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unknown NER loss: {loss_name}")
