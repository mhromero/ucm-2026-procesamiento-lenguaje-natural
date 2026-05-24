from __future__ import annotations

from pathlib import Path

import torch

from fdi_pln_2611_p5.config import load_config, resolve_tokenizer_path
from fdi_pln_2611_p5.model.lm_causal.bpe_tokenizer import BPETokenizer
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
