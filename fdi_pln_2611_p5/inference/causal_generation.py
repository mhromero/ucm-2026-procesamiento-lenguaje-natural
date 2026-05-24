"""Load a causal LM checkpoint and generate text from a prompt."""

from __future__ import annotations

from pathlib import Path

import torch

from fdi_pln_2611_p5.config import load_config, resolve_tokenizer_path
from fdi_pln_2611_p5.paths import require_file, require_optional_file
from fdi_pln_2611_p5.model.lm_causal.bpe_tokenizer import BPETokenizer
from fdi_pln_2611_p5.training.causal import build_model


def generate_text(
    weights_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    tokenizer_path: Path | None = None,
) -> str:
    """Generate text with a saved causal language model.

    Args:
        weights_path: Path to the model checkpoint ``.pth`` file.
        prompt: Seed text for autoregressive generation.
        max_new_tokens: Maximum number of tokens to generate after the prompt.
        temperature: Sampling temperature passed to the model.
        tokenizer_path: Optional path to ``bpe_tokenizer.json``; when omitted, resolved
            from the checkpoint metadata and project config.

    Returns:
        Prompt followed by generated continuation text.
    """
    weights_path = require_file(weights_path, label="causal model weights")
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = load_config()
    resolved_tokenizer = require_optional_file(tokenizer_path, label="BPE tokenizer")
    if resolved_tokenizer is None:
        resolved_tokenizer = resolve_tokenizer_path(
            payload, config, weights_path=weights_path
        )
    tokenizer = BPETokenizer.load(str(resolved_tokenizer))
    model = build_model(config, tokenizer)
    model.load_state_dict(payload["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model.generate(
        prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )
