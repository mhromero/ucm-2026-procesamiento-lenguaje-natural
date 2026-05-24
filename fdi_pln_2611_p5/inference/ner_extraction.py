"""Load an NER checkpoint and extract entities from text or files."""

from __future__ import annotations

from pathlib import Path

import torch

from fdi_pln_2611_p5.config import load_config, resolve_tokenizer_path
from fdi_pln_2611_p5.paths import require_file, require_optional_file
from fdi_pln_2611_p5.model.lm_causal.bpe_tokenizer import BPETokenizer
from fdi_pln_2611_p5.model.ner.model import NERModel, labels_to_entities
from fdi_pln_2611_p5.training.causal import build_model


def _load_ner_model(
    weights_path: Path,
    tokenizer_path: Path | None = None,
) -> tuple[NERModel, BPETokenizer]:
    """Load an NER model and its tokenizer from a checkpoint file.

    Args:
        weights_path: Path to the NER checkpoint ``.pth`` file.
        tokenizer_path: Optional path to ``bpe_tokenizer.json``; when omitted, resolved
            from the checkpoint metadata and project config.

    Returns:
        Tuple of ``(ner_model, tokenizer)`` moved to the active compute device.
    """
    weights_path = require_file(weights_path, label="NER model weights")
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = load_config()
    resolved_tokenizer = require_optional_file(
        tokenizer_path, label="BPE tokenizer"
    )
    if resolved_tokenizer is None:
        resolved_tokenizer = resolve_tokenizer_path(
            payload, config, weights_path=weights_path
        )
    tokenizer = BPETokenizer.load(str(resolved_tokenizer))
    llm = build_model(config, tokenizer)
    ner_model = NERModel(llm)
    state = {
        k: v
        for k, v in payload["model_state_dict"].items()
        if not k.startswith("loss_fn.")
    }
    ner_model.load_state_dict(state)
    ner_model.entity_threshold = float(payload.get("entity_threshold", 0.5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model.to(device)
    return ner_model, tokenizer


def extract_entities_from_text(
    weights_path: Path,
    text: str,
    tokenizer_path: Path | None = None,
) -> list[dict]:
    """Predict entities in a raw text string.

    Args:
        weights_path: Path to the NER checkpoint ``.pth`` file.
        text: Input text to annotate.
        tokenizer_path: Optional path to the BPE tokenizer JSON file.

    Returns:
        List of entity dictionaries with ``text`` and ``type`` keys.
    """
    ner_model, tokenizer = _load_ner_model(weights_path, tokenizer_path=tokenizer_path)
    label_ids = ner_model.predict_label_ids(text)
    return labels_to_entities(text, label_ids, tokenizer)


def extract_entities_from_file(
    weights_path: Path,
    text_path: Path,
    tokenizer_path: Path | None = None,
) -> list[dict]:
    """Predict entities in the contents of a UTF-8 text file.

    Args:
        weights_path: Path to the NER checkpoint ``.pth`` file.
        text_path: Path to the input text file.
        tokenizer_path: Optional path to the BPE tokenizer JSON file.

    Returns:
        List of entity dictionaries with ``text`` and ``type`` keys.
    """
    text_path = require_file(text_path, label="text file")
    return extract_entities_from_text(
        weights_path,
        text_path.read_text(encoding="utf-8"),
        tokenizer_path=tokenizer_path,
    )
