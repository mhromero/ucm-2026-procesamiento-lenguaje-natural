from __future__ import annotations

from pathlib import Path

import torch

from fdi_pln_2611_p5.BPETokenizer import BPETokenizer
from fdi_pln_2611_p5.config import load_config, package_path
from fdi_pln_2611_p5.ner import NERModel, labels_to_entities
from fdi_pln_2611_p5.training.causal import build_model


def generate_text(
    weights_path: Path, prompt: str, max_new_tokens: int, temperature: float
) -> str:
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = load_config()
    tokenizer = BPETokenizer.load(
        payload.get(
            "tokenizer_path", str(package_path(config["tokenizer"]["cache_path"]))
        )
    )
    model = build_model(config, tokenizer)
    model.load_state_dict(payload["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model.generate(
        prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )


def _load_ner_model(weights_path: Path) -> tuple[NERModel, BPETokenizer]:
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = load_config()
    tokenizer = BPETokenizer.load(
        payload.get(
            "tokenizer_path", str(package_path(config["tokenizer"]["cache_path"]))
        )
    )
    llm = build_model(config, tokenizer)
    ner_model = NERModel(llm)
    ner_model.load_state_dict(payload["model_state_dict"])
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
