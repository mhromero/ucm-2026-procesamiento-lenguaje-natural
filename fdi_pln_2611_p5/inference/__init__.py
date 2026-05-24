"""Inferencia: generación causal y extracción de entidades NER."""

from fdi_pln_2611_p5.inference.causal_generation import generate_text
from fdi_pln_2611_p5.inference.ner_extraction import (
    extract_entities_from_file,
    extract_entities_from_text,
)

__all__ = [
    "extract_entities_from_file",
    "extract_entities_from_text",
    "generate_text",
]
