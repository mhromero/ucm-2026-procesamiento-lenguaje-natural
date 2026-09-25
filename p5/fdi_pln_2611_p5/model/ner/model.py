"""NER classification head on top of a pretrained causal language model."""

from __future__ import annotations

import string

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


_PUNCTUATION = set(string.punctuation)


def _word_bounds(text: str, index: int) -> tuple[int, int]:
    """Return slice indices for the word at ``index``, without edge punctuation."""
    if not text:
        return 0, 0
    index = min(max(index, 0), len(text) - 1)
    start = index
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    end = index + 1
    while end < len(text) and not text[end].isspace():
        end += 1
    while start < end and text[start] in _PUNCTUATION:
        start += 1
    while end > start and text[end - 1] in _PUNCTUATION:
        end -= 1
    return start, end


def _strip_outer_punctuation(span: str) -> str:
    """Remove leading and trailing punctuation from an entity surface form."""
    span = span.strip()
    while span and span[0] in _PUNCTUATION:
        span = span[1:]
    while span and span[-1] in _PUNCTUATION:
        span = span[:-1]
    return span


def labels_to_entities(text: str, label_ids: list[int], tokenizer) -> list[dict]:
    """Group BPE-level predictions into human-readable entity spans.

    When a subtoken is tagged with ``pi`` or ``li``, the span is expanded to the
    full whitespace-delimited word in the source text (not only the BPE piece).

    Args:
        text: Original input text.
        label_ids: Predicted label ids aligned with BPE tokens.
        tokenizer: BPE tokenizer used to decode token strings.

    Returns:
        List of entity dictionaries with ``text`` and ``type`` keys.
    """
    text = text.lower()
    token_ids, _ = tokenizer.encode_with_labels(text, [0] * len(text))
    tokens = tokenizer.decode_tokens(token_ids)
    if len(label_ids) < len(tokens):
        label_ids = label_ids + [0] * (len(tokens) - len(label_ids))
    elif len(label_ids) > len(tokens):
        label_ids = label_ids[: len(tokens)]

    entities: list[dict] = []
    open_start: int | None = None
    open_end: int | None = None
    open_type: str | None = None
    char_pos = 0

    def flush() -> None:
        nonlocal open_start, open_end, open_type
        if open_start is None or open_end is None or not open_type:
            open_start = open_end = open_type = None
            return
        span_text = _strip_outer_punctuation(text[open_start:open_end])
        if span_text:
            entities.append({"text": span_text, "type": open_type})
        open_start = open_end = open_type = None

    for token_text, label_id in zip(tokens, label_ids, strict=True):
        label = ID2LABEL.get(label_id, "o")
        token_start = char_pos
        char_pos += len(token_text)

        if label == "o":
            flush()
            continue

        entity_type = entity_type_from_label(label)
        if entity_type is None:
            flush()
            continue

        if label.endswith("i"):
            flush()
            word_start, word_end = _word_bounds(text, token_start)
            open_start, open_end, open_type = word_start, word_end, entity_type
        elif label.endswith("c"):
            if open_type == entity_type and open_start is not None:
                _, word_end = _word_bounds(text, token_start)
                open_end = max(open_end, word_end)
            else:
                flush()
                word_start, word_end = _word_bounds(text, token_start)
                open_start, open_end, open_type = word_start, word_end, entity_type
        else:
            flush()

    flush()
    return entities
