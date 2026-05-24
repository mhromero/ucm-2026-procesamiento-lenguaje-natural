"""NER: etiquetas BIO, modelo y decodificación."""

from fdi_pln_2611_p5.model.ner.decode import logits_to_label_ids, repair_bio_label_ids
from fdi_pln_2611_p5.model.ner.labels import (
    ID2LABEL,
    IGNORE_LABEL_ID,
    LABEL2ID,
    PREFIX_TO_ENTITY_TYPE,
    entity_type_from_label,
    label_to_id,
    merge_subword_labels,
)

__all__ = [
    "ID2LABEL",
    "IGNORE_LABEL_ID",
    "LABEL2ID",
    "NERModel",
    "PREFIX_TO_ENTITY_TYPE",
    "entity_type_from_label",
    "label_to_id",
    "labels_to_entities",
    "logits_to_label_ids",
    "merge_subword_labels",
    "repair_bio_label_ids",
    "save_ner_checkpoint",
    "load_ner_checkpoint",
]


def __getattr__(name: str):
    if name == "NERModel":
        from fdi_pln_2611_p5.model.ner.model import NERModel

        return NERModel
    if name == "labels_to_entities":
        from fdi_pln_2611_p5.model.ner.model import labels_to_entities

        return labels_to_entities
    if name == "save_ner_checkpoint":
        from fdi_pln_2611_p5.model.ner.checkpoints import save_ner_checkpoint

        return save_ner_checkpoint
    if name == "load_ner_checkpoint":
        from fdi_pln_2611_p5.model.ner.checkpoints import load_ner_checkpoint

        return load_ner_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
