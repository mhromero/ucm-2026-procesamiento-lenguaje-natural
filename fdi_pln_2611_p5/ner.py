from __future__ import annotations

import torch
import torch.nn as nn

from fdi_pln_2611_p5.labels import ID2LABEL, LABEL2ID, entity_type_from_label
from fdi_pln_2611_p5.LLM import LLM


class NERModel(nn.Module):
    """Cabezal de NER sobre el backbone del LLM causal preentrenado."""

    def __init__(
        self,
        backbone: LLM,
        num_labels: int | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels or len(LABEL2ID)
        self.label_projection = nn.Linear(backbone.d_model, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone.encode_tokens(token_ids, causal=False)
        return self.label_projection(hidden)

    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        optimizer.zero_grad()
        logits = self.forward(x_batch)
        loss = self.loss_fn(logits.reshape(-1, self.num_labels), y_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict_label_ids(self, text: str) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        text = text.lower()
        dummy_labels = [0] * len(text)
        token_ids, _ = self.backbone.tokenizer.encode_with_labels(text, dummy_labels)
        if not token_ids:
            return []

        preds: list[int] = []
        for start in range(0, len(token_ids), self.backbone.window_size):
            chunk_ids = token_ids[start : start + self.backbone.window_size]
            x = torch.tensor([chunk_ids], dtype=torch.long, device=device)
            logits = self.forward(x)
            chunk_preds = logits.argmax(dim=-1).squeeze(0).tolist()
            preds.extend(chunk_preds[: len(chunk_ids)])
        return preds


def labels_to_entities(text: str, label_ids: list[int], tokenizer) -> list[dict]:
    """Agrupa predicciones BPE en entidades legibles."""
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
