from __future__ import annotations

from pathlib import Path

import torch

from fdi_pln_2611_p5.BPETokenizer import BPETokenizer
from fdi_pln_2611_p5.config import load_config, resolve_tokenizer_path
from fdi_pln_2611_p5.ner import NERModel, labels_to_entities
from fdi_pln_2611_p5.training.causal import build_model


def generate_text(
    weights_path: Path, prompt: str, max_new_tokens: int, temperature: float
) -> str:
    weights_path = Path(weights_path).resolve()
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = load_config()
    tokenizer = BPETokenizer.load(
        str(resolve_tokenizer_path(payload, config, weights_path=weights_path))
    )
    model = build_model(config, tokenizer)
    model.load_state_dict(payload["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model.generate(
        prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )


def _load_ner_model(weights_path: Path) -> tuple[NERModel, BPETokenizer]:
    weights_path = Path(weights_path).resolve()
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = load_config()
    tokenizer = BPETokenizer.load(
        str(resolve_tokenizer_path(payload, config, weights_path=weights_path))
    )
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


def extract_entities_from_text(weights_path: Path, text: str) -> list[dict]:
    ner_model, tokenizer = _load_ner_model(weights_path)
    label_ids = ner_model.predict_label_ids(text)
    return labels_to_entities(text, label_ids, tokenizer)


def extract_entities_from_file(weights_path: Path, text_path: Path) -> list[dict]:
    return extract_entities_from_text(
        weights_path, text_path.read_text(encoding="utf-8")
    )
