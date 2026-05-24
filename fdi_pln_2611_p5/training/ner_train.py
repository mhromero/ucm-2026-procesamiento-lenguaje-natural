"""NER fine-tuning on top of a pretrained causal language-model backbone."""

from __future__ import annotations

import copy
import random
from csv import DictWriter
from pathlib import Path

import torch
from loguru import logger

from fdi_pln_2611_p5.annotations.ner_dataset import (
    build_ner_windows,
    load_merged_dataset,
)
from fdi_pln_2611_p5.model.lm_causal.bpe_tokenizer import BPETokenizer
from fdi_pln_2611_p5.model.ner.checkpoints import save_ner_checkpoint
from fdi_pln_2611_p5.config import PACKAGE_DIR, load_config, package_path
from fdi_pln_2611_p5.paths import require_file, require_optional_file
from fdi_pln_2611_p5.training.run_config import (
    resolve_config_path,
    save_reproducibility_artifacts,
)
from fdi_pln_2611_p5.model.ner.labels import IGNORE_LABEL_ID, LABEL2ID
from fdi_pln_2611_p5.model.ner.model import NERModel
from fdi_pln_2611_p5.training.causal import build_model
from fdi_pln_2611_p5.training.ner_loss import build_ner_loss, compute_class_weights
from fdi_pln_2611_p5.training.ner_report import generate_ner_report
from fdi_pln_2611_p5.training.utils import (
    build_ner_optimizer,
    evaluar_confusion_ner_sentence_level,
    evaluar_loss_ner,
    evaluar_metricas_ner,
    evaluar_metricas_ner_sentence_level,
    find_best_entity_threshold,
    iter_batches,
    ner_checkpoint_score,
    passes_overall_acc_constraint,
)


def _stratified_sentence_split(
    sentences: list[dict], val_ratio: float
) -> tuple[list[dict], list[dict]]:
    """Split sentences into train/val while preserving entity-sentence proportions.

    Args:
        sentences: Annotated sentence dicts with ``labels``.
        val_ratio: Fraction of each stratum reserved for validation.

    Returns:
        Tuple of (train sentences, validation sentences).
    """
    with_entities = [s for s in sentences if any(l != "o" for l in s.get("labels", []))]
    without_entities = [
        s for s in sentences if all(l == "o" for l in s.get("labels", []))
    ]

    def take_val(group: list[dict]) -> tuple[list[dict], list[dict]]:
        n_val = max(1, round(len(group) * val_ratio)) if len(group) > 1 else 0
        # Sample every Nth item so validation spans the full dataset.
        step = max(1, len(group) // max(n_val, 1))
        val_idx = set(range(0, len(group), step)[:n_val])
        val = [s for i, s in enumerate(group) if i in val_idx]
        train = [s for i, s in enumerate(group) if i not in val_idx]
        return train, val

    train_w, val_w = take_val(with_entities)
    train_wo, val_wo = take_val(without_entities)
    return train_w + train_wo, val_w + val_wo


def _oversample_entity_windows(
    x: torch.Tensor, y: torch.Tensor, factor: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Duplicate windows that contain at least one entity token.

    Args:
        x: Window input tensor.
        y: Window label tensor.
        factor: Total copies per entity window (1 leaves data unchanged).

    Returns:
        Possibly expanded ``(x, y)`` tensors.
    """
    if factor <= 1:
        return x, y
    extra_x: list[torch.Tensor] = []
    extra_y: list[torch.Tensor] = []
    for i in range(x.size(0)):
        labels = y[i]
        valid = labels != IGNORE_LABEL_ID
        if valid.any() and (labels[valid] != 0).any():
            for _ in range(factor - 1):
                extra_x.append(x[i])
                extra_y.append(y[i])
    if not extra_x:
        return x, y
    return torch.cat([x, torch.stack(extra_x)]), torch.cat([y, torch.stack(extra_y)])


def _init_ner_from_causal(ner_model: NERModel, causal_state: dict):
    """Load causal backbone weights into the NER model, excluding the vocab head."""
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
    tokenizer_path: Path | None = None,
) -> dict:
    """Fine-tune the NER head on a pretrained causal backbone.

    Loads causal weights, replaces the vocabulary projection with a linear NER
    head, and trains with class-weighted loss to handle ``o`` vs. entity imbalance.

    Args:
        weights_path: Destination path for NER checkpoint weights.
        causal_weights_path: Path to the pretrained causal model checkpoint.
        merged_annotations_path: Path to merged annotation JSON.
        config_path: Optional path to the configuration file.
        tokenizer_path: Optional BPE tokenizer JSON; overrides the path stored in the
            causal checkpoint when provided.

    Returns:
        Dict with best metric name, score, epoch, and window counts.
    """
    resolved_config_path = resolve_config_path(config_path)
    config = load_config(config_path)
    causal_weights_path = require_file(
        causal_weights_path, label="causal model weights"
    )
    merged_annotations_path = require_file(
        merged_annotations_path, label="merged annotations"
    )
    resolved_tokenizer_path = require_optional_file(
        tokenizer_path, label="BPE tokenizer"
    )
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Semilla: {}", seed)

    ner_cfg = config["ner_training"]
    model_cfg = config["model"]

    causal_payload = torch.load(
        causal_weights_path, map_location="cpu", weights_only=False
    )
    if resolved_tokenizer_path is None:
        resolved_tokenizer_path = require_file(
            causal_payload.get(
                "tokenizer_path", str(package_path(config["tokenizer"]["cache_path"]))
            ),
            label="BPE tokenizer",
        )
    tokenizer = BPETokenizer.load(str(resolved_tokenizer_path))
    logger.info("Tokenizador BPE: {}", resolved_tokenizer_path)

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
        "Split NER: {} frases train, {} frases val",
        len(train_sentences),
        len(val_sentences),
    )
    train_stride = ner_cfg.get("train_window_stride", 1)
    val_stride = ner_cfg.get("val_window_stride")
    if val_stride is None and ner_cfg.get("val_non_overlapping_windows", False):
        val_stride = model_cfg["window_size"]
    x_train, y_train = build_ner_windows(
        train_sentences,
        tokenizer,
        model_cfg["window_size"],
        stride=train_stride,
    )
    x_val, y_val = build_ner_windows(
        val_sentences,
        tokenizer,
        model_cfg["window_size"],
        stride=val_stride,
    )
    val_per_sentence = bool(ner_cfg.get("val_eval_per_sentence", True))
    oversample = int(ner_cfg.get("oversample_entity_windows", 1))
    n_train_before = x_train.size(0)
    x_train, y_train = _oversample_entity_windows(x_train, y_train, oversample)
    logger.info(
        "Ventanas NER: {} train ({} base × oversample {}), {} val",
        x_train.size(0),
        n_train_before,
        oversample,
        x_val.size(0),
    )

    class_weights = compute_class_weights(
        y_train,
        len(LABEL2ID),
        mode=ner_cfg.get("class_weight_mode", "effective_num"),
        entity_boost=float(ner_cfg.get("entity_class_weight_boost", 3.0)),
    )
    logger.info(
        "Pesos de clase ({}): {}",
        ner_cfg.get("class_weight_mode", "effective_num"),
        {k: f"{v:.2f}" for k, v in zip(LABEL2ID, class_weights.tolist())},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = build_ner_loss(
        ner_cfg.get("loss", "focal"),
        class_weights.to(device),
        focal_gamma=float(ner_cfg.get("focal_gamma", 2.0)),
        label_smoothing=float(ner_cfg.get("label_smoothing", 0.0)),
    )
    ner_model = NERModel(llm, loss_fn=loss_fn)
    ner_model.entity_threshold = float(ner_cfg.get("entity_prob_threshold", 0.5))
    _init_ner_from_causal(ner_model, causal_payload["model_state_dict"])
    freeze_mode = ner_cfg.get("freeze_backbone", False)
    if freeze_mode:
        for param in ner_model.backbone.parameters():
            param.requires_grad = False
        if freeze_mode == "last_block":
            for param in ner_model.backbone.blocks[-1].parameters():
                param.requires_grad = True
            logger.info("Backbone: solo último bloque Transformer entrenable")
        else:
            logger.info("Backbone congelado")
        trainable = sum(p.numel() for p in ner_model.parameters() if p.requires_grad)
        logger.info("Parámetros entrenables NER: {}", trainable)
    ner_model.to(device)
    optimizer = build_ner_optimizer(
        ner_model,
        ner_cfg["learning_rate"],
        float(ner_cfg.get("backbone_lr_factor", 1.0)),
    )

    best_metric_name = ner_cfg.get("best_metric", "entity_acc_constrained")
    min_overall_acc = float(ner_cfg.get("min_overall_acc", 0.85))
    _, higher_is_better = ner_checkpoint_score({}, best_metric_name, 0.0)
    best_score = float("-inf") if higher_is_better else float("inf")
    best_state: dict | None = None
    best_epoch = 0
    fallback_state: dict | None = None
    fallback_epoch = 0
    fallback_entity_acc = -1.0
    ner_history: list[dict] = []
    if best_metric_name == "entity_acc_constrained":
        logger.info(
            "Selección de checkpoint: max accuracy en tokens de entidad, "
            "con accuracy global ≥ {:.0%}",
            min_overall_acc,
        )

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
        train_loss_epoch = epoch_loss / max(steps, 1)
        val_loss = evaluar_loss_ner(
            ner_model, x_val, y_val, ner_cfg["batch_size"], device
        )
        if val_per_sentence:
            metricas = evaluar_metricas_ner_sentence_level(
                ner_model,
                val_sentences,
                tokenizer,
                model_cfg["window_size"],
                device,
            )
        else:
            metricas = evaluar_metricas_ner(
                ner_model, x_val, y_val, ner_cfg["batch_size"], device
            )
        score, _ = ner_checkpoint_score(metricas, best_metric_name, val_loss)
        eligible = passes_overall_acc_constraint(metricas, min_overall_acc)
        if eligible:
            is_best = (score > best_score) if higher_is_better else (score < best_score)
        else:
            is_best = False
        improved = ""
        if is_best:
            best_score = score
            best_state = copy.deepcopy(ner_model.state_dict())
            best_epoch = epoch + 1
            improved = " ★"
        ent_acc = metricas["entity_token_acc"]
        if ent_acc > fallback_entity_acc:
            fallback_entity_acc = ent_acc
            fallback_state = copy.deepcopy(ner_model.state_dict())
            fallback_epoch = epoch + 1
        constraint_mark = (
            "" if eligible else " (acc global < {:.0%})".format(min_overall_acc)
        )
        logger.info(
            "NER epoch {}/{} train_loss={:.4f} val_loss={:.4f} acc={:.1%} "
            "entity_acc={:.1%} macro_f1={:.1%} span_f1={:.1%} pred_ent={} gold_ent={}{}{}",
            epoch + 1,
            ner_cfg["epochs"],
            train_loss_epoch,
            val_loss,
            metricas["overall_acc"],
            metricas["entity_token_acc"],
            metricas["macro_f1_non_o"],
            metricas.get("span_f1", 0.0),
            metricas["n_pred_entities"],
            metricas["n_gold_entities"],
            improved,
            constraint_mark,
        )
        ner_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss_epoch,
                "val_loss": val_loss,
                "overall_acc": metricas["overall_acc"],
                "entity_recall": metricas["entity_recall"],
                "entity_token_acc": metricas["entity_token_acc"],
                "macro_f1_non_o": metricas["macro_f1_non_o"],
                "span_f1": metricas.get("span_f1", 0.0),
                "n_pred_entities": metricas["n_pred_entities"],
                "n_gold_entities": metricas["n_gold_entities"],
                "meets_overall_acc": eligible,
            }
        )

    if best_state is None and fallback_state is not None:
        logger.warning(
            "Ninguna época cumple acc global ≥ {:.0%}; checkpoint época {} "
            "(entity_acc={:.1%}).",
            min_overall_acc,
            fallback_epoch,
            fallback_entity_acc,
        )
        best_state = fallback_state
        best_epoch = fallback_epoch
        best_score = fallback_entity_acc

    if best_state is not None:
        ner_model.load_state_dict(best_state)

    tune_threshold = ner_cfg.get("tune_entity_threshold", True)
    if tune_threshold:
        best_t, tuned = find_best_entity_threshold(
            ner_model,
            x_val,
            y_val,
            ner_cfg["batch_size"],
            device,
            val_sentences=val_sentences if val_per_sentence else None,
            tokenizer=tokenizer if val_per_sentence else None,
            window_size=model_cfg["window_size"],
        )
        logger.info(
            "Umbral entidad en val: {:.2f} → macro_f1={:.1%} recall_ent={:.1%} "
            "pred_ent={} gold_ent={}",
            best_t,
            tuned["macro_f1_non_o"],
            tuned["entity_recall"],
            tuned["n_pred_entities"],
            tuned["n_gold_entities"],
        )
        ner_model.entity_threshold = best_t

    logger.info(
        "Mejor epoch NER: {} ({}={:.4f}, entity_threshold={:.2f})",
        best_epoch,
        best_metric_name,
        best_score,
        ner_model.entity_threshold,
    )

    history_path = package_path(config["metrics"]["ner_history_csv_path"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "overall_acc",
                "entity_recall",
                "entity_token_acc",
                "macro_f1_non_o",
                "span_f1",
                "n_pred_entities",
                "n_gold_entities",
                "meets_overall_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(ner_history)

    confusion = evaluar_confusion_ner_sentence_level(
        ner_model, val_sentences, tokenizer, model_cfg["window_size"], device
    )
    logger.info(
        "Val por frase (informe): span_f1={:.1%} macro_f1={:.1%}",
        confusion.get("span_f1", 0.0),
        sum(
            confusion["per_class"][c]["f1"]
            for c in ("pi", "pc", "li", "lc")
            if c in confusion.get("per_class", {})
        )
        / 4,
    )

    save_ner_checkpoint(
        weights_path,
        ner_model,
        model_cfg,
        resolved_tokenizer_path,
        extra={
            "best_metric": best_metric_name,
            "best_score": best_score,
            "entity_threshold": ner_model.entity_threshold,
            "ner_training": ner_cfg,
        },
    )
    weights_path = Path(weights_path)
    training_config_filename = None
    if weights_path.parent.resolve() == PACKAGE_DIR.resolve():
        training_config_filename = f"{weights_path.stem}_training_config.json"
    save_reproducibility_artifacts(
        weights_path.parent,
        config,
        run_type="ner",
        config_path=resolved_config_path,
        training_config_filename=training_config_filename,
        extra={
            "weights_path": str(weights_path),
            "causal_weights_path": str(causal_weights_path),
            "merged_annotations_path": str(merged_annotations_path),
            "best_metric": best_metric_name,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "entity_threshold": ner_model.entity_threshold,
            "tokenizer_path": str(resolved_tokenizer_path),
        },
    )

    report_path = package_path(config["metrics"]["ner_report_path"])
    generate_ner_report(
        history=ner_history,
        confusion=confusion,
        model_cfg=model_cfg,
        ner_cfg=ner_cfg,
        best_epoch=best_epoch,
        output_path=report_path,
    )
    logger.info("Informe NER guardado en {}", report_path)

    return {
        "best_metric": best_metric_name,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "n_train_windows": x_train.size(0),
        "n_val_windows": x_val.size(0),
    }
