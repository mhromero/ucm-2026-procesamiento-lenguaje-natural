from __future__ import annotations

import random
from pathlib import Path

import torch
from loguru import logger

from fdi_pln_2611_p5.annotations.dataset import build_ner_windows, load_merged_dataset
from fdi_pln_2611_p5.BPETokenizer import BPETokenizer
from fdi_pln_2611_p5.checkpoints import save_ner_checkpoint
from fdi_pln_2611_p5.config import load_config, package_path
from fdi_pln_2611_p5.labels import IGNORE_LABEL_ID, LABEL2ID
from fdi_pln_2611_p5.ner import NERModel
from fdi_pln_2611_p5.training.causal import build_model
from fdi_pln_2611_p5.training.utils import evaluar_loss_ner, evaluar_metricas_ner, iter_batches


def _stratified_sentence_split(
    sentences: list[dict], val_ratio: float
) -> tuple[list[dict], list[dict]]:
    """Divide frases en train/val manteniendo la proporción de frases con entidades."""
    with_entities = [s for s in sentences if any(l != "o" for l in s.get("labels", []))]
    without_entities = [s for s in sentences if all(l == "o" for l in s.get("labels", []))]

    def take_val(group: list[dict]) -> tuple[list[dict], list[dict]]:
        n_val = max(1, round(len(group) * val_ratio)) if len(group) > 1 else 0
        # tomar cada Nth para que val esté distribuido por todo el dataset
        step = max(1, len(group) // max(n_val, 1))
        val_idx = set(range(0, len(group), step)[:n_val])
        val = [s for i, s in enumerate(group) if i in val_idx]
        train = [s for i, s in enumerate(group) if i not in val_idx]
        return train, val

    train_w, val_w = take_val(with_entities)
    train_wo, val_wo = take_val(without_entities)
    return train_w + train_wo, val_w + val_wo


def _compute_class_weights(y_train: torch.Tensor, num_labels: int) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia de cada clase, excluyendo padding."""
    flat = y_train.view(-1)
    flat = flat[flat != IGNORE_LABEL_ID]
    counts = torch.bincount(flat, minlength=num_labels).float().clamp(min=1)
    return flat.numel() / (num_labels * counts)


def _init_ner_from_causal(ner_model: NERModel, causal_state: dict):
    backbone_state = {
        key: value
        for key, value in causal_state.items()
        if not key.startswith("vocab_projection")
    }
    ner_model.backbone.load_state_dict(backbone_state, strict=False)


def train_ner(
    weights_path: Path,
    causal_weights_path: Path,
    merged_annotations_path: Path,
    config_path: Path | None = None,
) -> dict:
    """Ajusta el cabezal NER sobre el backbone causal preentrenado.

    Carga los pesos causales, sustituye la cabeza de vocabulario por una proyección
    lineal a las etiquetas NER, y entrena con CrossEntropyLoss ponderado para
    compensar el desbalance entre tokens 'o' y tokens de entidad.
    """
    config = load_config(config_path)
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Semilla: {}", seed)

    ner_cfg = config["ner_training"]
    model_cfg = config["model"]

    causal_payload = torch.load(
        causal_weights_path, map_location="cpu", weights_only=False
    )
    tokenizer = BPETokenizer.load(
        causal_payload.get(
            "tokenizer_path", str(package_path(config["tokenizer"]["cache_path"]))
        )
    )

    llm = build_model(config, tokenizer)
    llm.load_state_dict(causal_payload["model_state_dict"], strict=True)

    sentences = load_merged_dataset(merged_annotations_path)
    if not sentences:
        raise RuntimeError(
            "No hay frases anotadas fusionadas. Ejecuta merge-annotations y completa más JSON."
        )

    train_sentences, val_sentences = _stratified_sentence_split(
        sentences, val_ratio=ner_cfg["val_ratio"]
    )
    logger.info(
        "Split NER: {} frases train, {} frases val", len(train_sentences), len(val_sentences)
    )
    x_train, y_train = build_ner_windows(train_sentences, tokenizer, model_cfg["window_size"])
    x_val, y_val = build_ner_windows(val_sentences, tokenizer, model_cfg["window_size"])
    logger.info("Ventanas NER: {} train, {} val", x_train.size(0), x_val.size(0))

    class_weights = _compute_class_weights(y_train, len(LABEL2ID))
    logger.info("Pesos de clase: {}", {k: f"{v:.2f}" for k, v in zip(LABEL2ID, class_weights.tolist())})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model = NERModel(llm, class_weights=class_weights.to(device))
    _init_ner_from_causal(ner_model, causal_payload["model_state_dict"])
    ner_model.to(device)
    optimizer = torch.optim.Adam(ner_model.parameters(), lr=ner_cfg["learning_rate"])

    for epoch in range(ner_cfg["epochs"]):
        ner_model.train()
        epoch_loss = 0.0
        steps = 0
        for x_batch, y_batch in iter_batches(x_train, y_train, ner_cfg["batch_size"]):
            loss = ner_model.train_step(
                x_batch.to(device), y_batch.to(device), optimizer
            )
            epoch_loss += loss
            steps += 1
        val_loss = evaluar_loss_ner(ner_model, x_val, y_val, ner_cfg["batch_size"], device)
        metricas = evaluar_metricas_ner(ner_model, x_val, y_val, ner_cfg["batch_size"], device)
        logger.info(
            "NER epoch {}/{} train_loss={:.4f} val_loss={:.4f} acc={:.1%} entity_recall={:.1%} pred_ent={} gold_ent={}",
            epoch + 1,
            ner_cfg["epochs"],
            epoch_loss / max(steps, 1),
            val_loss,
            metricas["overall_acc"],
            metricas["entity_recall"],
            metricas["n_pred_entities"],
            metricas["n_gold_entities"],
        )

    save_ner_checkpoint(
        weights_path,
        ner_model,
        model_cfg,
        package_path(config["tokenizer"]["cache_path"]),
        extra={"val_loss": val_loss},
    )
    return {"val_loss": val_loss, "n_train_windows": x_train.size(0), "n_val_windows": x_val.size(0)}
