"""Save and load NER model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from fdi_pln_2611_p5.model.ner.model import NERModel


def save_ner_checkpoint(
    path: Path,
    model: NERModel,
    model_config: dict,
    tokenizer_path: Path,
    extra: dict | None = None,
):
    """Save an NER model checkpoint to disk.

    Args:
        path: Destination ``.pth`` file path.
        model: Trained ``NERModel`` instance.
        model_config: Model hyperparameters used during training.
        tokenizer_path: Path to the associated BPE tokenizer JSON file.
        extra: Optional additional fields to merge into the checkpoint payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "tokenizer_path": str(tokenizer_path),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_ner_checkpoint(path: Path, model: NERModel) -> dict:
    """Load weights from an NER checkpoint into an existing model.

    Args:
        path: Checkpoint ``.pth`` file path.
        model: ``NERModel`` instance whose weights will be updated in place.

    Returns:
        Full checkpoint payload dictionary.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return payload
