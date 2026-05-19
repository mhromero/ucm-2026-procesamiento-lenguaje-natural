from __future__ import annotations

from pathlib import Path

import torch

from fdi_pln_2611_p5.LLM import LLM
from fdi_pln_2611_p5.ner import NERModel


def save_causal_checkpoint(
    path: Path,
    model: LLM,
    model_config: dict,
    tokenizer_path: Path,
    extra: dict | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "tokenizer_path": str(tokenizer_path),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_causal_checkpoint(path: Path, model: LLM) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def save_ner_checkpoint(
    path: Path,
    model: NERModel,
    model_config: dict,
    tokenizer_path: Path,
    extra: dict | None = None,
):
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
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return payload
