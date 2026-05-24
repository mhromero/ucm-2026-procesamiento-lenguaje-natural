"""Modelos de la práctica: LM causal y NER."""

from fdi_pln_2611_p5.model.lm_causal import (
    Attention,
    BPETokenizer,
    LLM,
    load_causal_checkpoint,
    save_causal_checkpoint,
)

__all__ = [
    "Attention",
    "BPETokenizer",
    "ID2LABEL",
    "IGNORE_LABEL_ID",
    "LABEL2ID",
    "LLM",
    "NERModel",
    "labels_to_entities",
    "load_causal_checkpoint",
    "load_ner_checkpoint",
    "logits_to_label_ids",
    "repair_bio_label_ids",
    "save_causal_checkpoint",
    "save_ner_checkpoint",
]


def __getattr__(name: str):
    from fdi_pln_2611_p5.model import ner

    return getattr(ner, name)
