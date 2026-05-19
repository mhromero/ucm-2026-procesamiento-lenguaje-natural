from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from fdi_pln_2611_p5.annotations.dataset import build_ner_windows, load_merged_dataset
from fdi_pln_2611_p5.BPETokenizer import BPETokenizer
from fdi_pln_2611_p5.checkpoints import save_ner_checkpoint
from fdi_pln_2611_p5.config import load_config, package_path
from fdi_pln_2611_p5.ner import NERModel
from fdi_pln_2611_p5.training.causal import build_model
from fdi_pln_2611_p5.training.utils import evaluar_loss_ner, iter_batches


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
    config = load_config(config_path)
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
    ner_model = NERModel(llm)
    _init_ner_from_causal(ner_model, causal_payload["model_state_dict"])

    sentences = load_merged_dataset(merged_annotations_path)
    if not sentences:
        raise RuntimeError(
            "No hay frases anotadas fusionadas. Ejecuta merge-annotations y completa más JSON."
        )

    x_data, y_data = build_ner_windows(sentences, tokenizer, model_cfg["window_size"])
    split = max(1, int(x_data.size(0) * ner_cfg["val_ratio"]))
    x_train, y_train = x_data[:-split], y_data[:-split]
    x_val, y_val = x_data[-split:], y_data[-split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        val_loss = evaluar_loss_ner(
            ner_model, x_val, y_val, ner_cfg["batch_size"], device
        )
        logger.info(
            "NER epoch {}/{} train_loss={:.4f} val_loss={:.4f}",
            epoch + 1,
            ner_cfg["epochs"],
            epoch_loss / max(steps, 1),
            val_loss,
        )

    save_ner_checkpoint(
        weights_path,
        ner_model,
        model_cfg,
        package_path(config["tokenizer"]["cache_path"]),
        extra={"val_loss": val_loss},
    )
    return {"val_loss": val_loss, "n_windows": x_data.size(0)}
