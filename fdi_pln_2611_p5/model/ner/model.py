"""NER classification head on top of a pretrained causal language model."""

from __future__ import annotations

import torch
import torch.nn as nn

from fdi_pln_2611_p5.model.lm_causal.llm import LLM
from fdi_pln_2611_p5.model.ner.labels import ID2LABEL, LABEL2ID, entity_type_from_label
from fdi_pln_2611_p5.training.utils import predict_label_ids_for_tokens


class NERModel(nn.Module):
    """NER classification head on a pretrained causal LM backbone.

    Args:
        backbone: Pretrained ``LLM`` used as the feature extractor.
        num_labels: Number of label classes. Defaults to ``len(LABEL2ID)``.
        loss_fn: Optional loss function; defaults to cross-entropy with
            ``ignore_index=-1``.
    """

    def __init__(
        self,
        backbone: LLM,
        num_labels: int | None = None,
        loss_fn: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels or len(LABEL2ID)
        self.label_projection = nn.Linear(backbone.d_model, self.num_labels)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=-1)
        self.entity_threshold = 0.5

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-token label logits.

        Args:
            token_ids: Tensor of shape ``(batch, seq_len)``.

        Returns:
            Logits of shape ``(batch, seq_len, num_labels)``.
        """
        hidden = self.backbone.encode_tokens(token_ids, causal=False)
        return self.label_projection(hidden)

    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one NER training step and return the loss value.

        Args:
            x_batch: Input token windows.
            y_batch: Gold label ids aligned with ``x_batch``.
            optimizer: Optimizer used for the parameter update.

        Returns:
            Scalar loss value for the batch.
        """
        self.train()
        optimizer.zero_grad()
        logits = self.forward(x_batch)
        loss = self.loss_fn(logits.reshape(-1, self.num_labels), y_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict_label_ids(self, text: str) -> list[int]:
        """Predict NER label ids for a raw text string.

        Args:
            text: Input text; converted to lowercase before tokenization.

        Returns:
            Predicted label id sequence aligned with BPE tokens.
        """
        self.eval()
        device = next(self.parameters()).device
        text = text.lower()
        dummy_labels = [0] * len(text)
        token_ids, _ = self.backbone.tokenizer.encode_with_labels(text, dummy_labels)
        if not token_ids:
            return []

        return predict_label_ids_for_tokens(
            self,
            token_ids,
            self.backbone.window_size,
            device,
            self.entity_threshold,
        )


def labels_to_entities(text: str, label_ids: list[int], tokenizer) -> list[dict]:
    """Group BPE-level predictions into human-readable entity spans.

    Args:
        text: Original input text.
        label_ids: Predicted label ids aligned with BPE tokens.
        tokenizer: BPE tokenizer used to decode token strings.

    Returns:
        List of entity dictionaries with ``text`` and ``type`` keys.
    """
    token_ids, _ = tokenizer.encode_with_labels(text.lower(), [0] * len(text.lower()))
    tokens = tokenizer.decode_tokens(token_ids)
    entities: list[dict] = []
    current = ""
    current_type: str | None = None

    def flush():
        nonlocal current, current_type
        span = current.strip()
        if span and current_type:
            entities.append({"text": span, "type": current_type})
        current = ""
        current_type = None

    for token_text, label_id in zip(tokens, label_ids, strict=False):
        label = ID2LABEL.get(label_id, "o")
        if label == "o":
            flush()
            continue

        entity_type = entity_type_from_label(label)
        if entity_type is None:
            flush()
            continue

        if label.endswith("i"):
            if current_type and current_type != entity_type:
                flush()
            current_type = entity_type
            current += token_text
        elif label.endswith("c"):
            if current_type == entity_type:
                current += token_text
            else:
                flush()
                current_type = entity_type
                current = token_text
        else:
            flush()
            current_type = entity_type
            current += token_text
    flush()
    return entities
