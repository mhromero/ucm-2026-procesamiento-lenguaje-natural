from __future__ import annotations

import copy
import math
import sys

import torch
from loguru import logger
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from fdi_pln_2611_p5.annotations.dataset import char_labels_from_merged
from fdi_pln_2611_p5.labels import (
    ID2LABEL,
    IGNORE_LABEL_ID,
    LABEL2ID,
    entity_type_from_label,
)
from fdi_pln_2611_p5.LLM import LLM
from fdi_pln_2611_p5.ner_decode import logits_to_label_ids


def iter_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    for start in range(0, x.size(0), batch_size):
        end = start + batch_size
        yield x[start:end], y[start:end]


def mover_optimizador_a_dispositivo(
    optimizer: torch.optim.Optimizer, device: torch.device
):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


@torch.no_grad()
def evaluar_loss_causal(
    model: LLM,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    if x_data.size(0) == 0:
        return 0.0
    model.eval()
    total_loss = 0.0
    steps = 0
    for x_batch, y_batch in iter_batches(x_data, y_data, batch_size):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch, causal=True)
        loss = model.loss_fn(logits.reshape(-1, model.vocab_size), y_batch.reshape(-1))
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluar_loss_ner(
    model,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    if x_data.size(0) == 0:
        return 0.0
    model.eval()
    total_loss = 0.0
    steps = 0
    for x_batch, y_batch in iter_batches(x_data, y_data, batch_size):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = model.loss_fn(logits.reshape(-1, model.num_labels), y_batch.reshape(-1))
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def _ner_entity_threshold(model) -> float:
    return float(getattr(model, "entity_threshold", 0.5))


@torch.no_grad()
def _collect_ner_preds_labels(
    model,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    threshold = _ner_entity_threshold(model)
    pred_parts: list[int] = []
    label_parts: list[int] = []
    for x_batch, y_batch in iter_batches(x_data, y_data, batch_size):
        logits = model(x_batch.to(device))
        y_cpu = y_batch.cpu()
        for i in range(logits.size(0)):
            pred_ids = logits_to_label_ids(logits[i], entity_threshold=threshold)
            for pred_id, gold_id in zip(pred_ids, y_cpu[i].tolist()):
                if gold_id == IGNORE_LABEL_ID:
                    continue
                pred_parts.append(pred_id)
                label_parts.append(gold_id)
    return torch.tensor(pred_parts, dtype=torch.long), torch.tensor(
        label_parts, dtype=torch.long
    )


def build_ner_optimizer(
    model,
    learning_rate: float,
    backbone_lr_factor: float = 1.0,
) -> torch.optim.Optimizer:
    """Adam con LR reducido en el backbone si ``backbone_lr_factor`` < 1."""
    if backbone_lr_factor >= 1.0:
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
        )
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("backbone.")
    ]
    param_groups = []
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": learning_rate * backbone_lr_factor}
        )
    if head_params:
        param_groups.append({"params": head_params, "lr": learning_rate})
    return torch.optim.Adam(param_groups)


def _spans_from_label_ids(label_ids: list[int]) -> list[tuple[int, int, str]]:
    """Spans (inicio, fin, PER|LOC) a nivel token BPE."""
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(label_ids):
        label = ID2LABEL.get(label_ids[i], "o")
        etype = entity_type_from_label(label)
        if label == "o" or etype is None:
            i += 1
            continue
        j = i + 1
        while j < len(label_ids):
            lj = ID2LABEL.get(label_ids[j], "o")
            if lj == "o" or entity_type_from_label(lj) != etype:
                break
            j += 1
        spans.append((i, j, etype))
        i = j
    return spans


def span_f1_from_label_lists(pred_ids: list[int], gold_ids: list[int]) -> float:
    pred_spans = set(_spans_from_label_ids(pred_ids))
    gold_spans = set(_spans_from_label_ids(gold_ids))
    if not pred_spans and not gold_spans:
        return 1.0
    if not pred_spans or not gold_spans:
        return 0.0
    tp = len(pred_spans & gold_spans)
    precision = tp / len(pred_spans)
    recall = tp / len(gold_spans)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _metrics_from_pred_label_lists(
    pred_parts: list[int], label_parts: list[int]
) -> dict:
    preds = torch.tensor(pred_parts, dtype=torch.long)
    labels = torch.tensor(label_parts, dtype=torch.long)
    if labels.numel() == 0:
        return {
            "overall_acc": 0.0,
            "entity_recall": 0.0,
            "entity_token_acc": 0.0,
            "macro_f1_non_o": 0.0,
            "span_f1": 0.0,
            "n_pred_entities": 0,
            "n_gold_entities": 0,
        }
    overall_acc = (preds == labels).float().mean().item()
    entity_mask = labels != 0
    n_gold = int(entity_mask.sum().item())
    entity_recall = (
        (preds[entity_mask] == labels[entity_mask]).float().mean().item()
        if n_gold > 0
        else 0.0
    )
    return {
        "overall_acc": overall_acc,
        "entity_recall": entity_recall,
        "entity_token_acc": entity_recall,
        "macro_f1_non_o": _macro_f1_non_o(preds, labels),
        "span_f1": span_f1_from_label_lists(pred_parts, label_parts),
        "n_pred_entities": int((preds != 0).sum().item()),
        "n_gold_entities": n_gold,
    }


@torch.no_grad()
def predict_label_ids_for_tokens(
    model,
    token_ids: list[int],
    window_size: int,
    device: torch.device,
    entity_threshold: float,
    *,
    stride: int | None = None,
) -> list[int]:
    """Inferencia por token: ventanas con solapamiento y voto mayoritario."""
    n = len(token_ids)
    if n == 0:
        return []
    step = stride if stride is not None else max(1, window_size // 2)
    votes: list[list[int]] = [[] for _ in range(n)]
    model.eval()

    if n <= window_size:
        starts = [0]
    else:
        last_start = n - window_size
        starts = list(range(0, last_start + 1, step))
        if starts[-1] != last_start:
            starts.append(last_start)

    for start in starts:
        chunk = token_ids[start : start + window_size]
        x = torch.tensor([chunk], dtype=torch.long, device=device)
        logits = model(x)
        chunk_preds = logits_to_label_ids(logits[0], entity_threshold=entity_threshold)
        for offset, pid in enumerate(chunk_preds[: len(chunk)]):
            votes[start + offset].append(pid)

    result: list[int] = []
    for options in votes:
        if not options:
            result.append(0)
            continue
        counts: dict[int, int] = {}
        for pid in options:
            counts[pid] = counts.get(pid, 0) + 1
        result.append(max(counts, key=counts.get))
    from fdi_pln_2611_p5.ner_decode import repair_bio_label_ids

    return repair_bio_label_ids(result)


@torch.no_grad()
def evaluar_metricas_ner_sentence_level(
    model,
    val_sentences: list[dict],
    tokenizer,
    window_size: int,
    device: torch.device,
    *,
    infer_stride: int | None = None,
) -> dict:
    """Métricas en val: cada token BPE cuenta una vez (inferencia alineada con entrenamiento)."""
    threshold = _ner_entity_threshold(model)
    pred_parts: list[int] = []
    label_parts: list[int] = []
    for sentence in val_sentences:
        text, char_label_ids = char_labels_from_merged(sentence)
        token_ids, gold_ids = tokenizer.encode_with_labels(text, char_label_ids)
        if not token_ids:
            continue
        pred_ids = predict_label_ids_for_tokens(
            model,
            token_ids,
            window_size,
            device,
            threshold,
            stride=infer_stride,
        )
        for pred_id, gold_id in zip(pred_ids, gold_ids):
            pred_parts.append(pred_id)
            label_parts.append(gold_id)
    return _metrics_from_pred_label_lists(pred_parts, label_parts)


@torch.no_grad()
def evaluar_confusion_ner_sentence_level(
    model,
    val_sentences: list[dict],
    tokenizer,
    window_size: int,
    device: torch.device,
) -> dict:
    """Matriz de confusión sin duplicar tokens por ventanas solapadas."""
    num_labels = len(LABEL2ID)
    metrics = evaluar_metricas_ner_sentence_level(
        model, val_sentences, tokenizer, window_size, device
    )
    pred_parts: list[int] = []
    label_parts: list[int] = []
    threshold = _ner_entity_threshold(model)
    for sentence in val_sentences:
        text, char_label_ids = char_labels_from_merged(sentence)
        token_ids, gold_ids = tokenizer.encode_with_labels(text, char_label_ids)
        if not token_ids:
            continue
        pred_ids = predict_label_ids_for_tokens(
            model, token_ids, window_size, device, threshold
        )
        pred_parts.extend(pred_ids)
        label_parts.extend(gold_ids)

    matrix = [[0] * num_labels for _ in range(num_labels)]
    for true, pred in zip(label_parts, pred_parts):
        matrix[true][pred] += 1

    per_class: dict[str, dict] = {}
    for i in range(num_labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(num_labels)) - tp
        fn = sum(matrix[i][j] for j in range(num_labels)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[ID2LABEL[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    return {
        "matrix": matrix,
        "per_class": per_class,
        "labels": [ID2LABEL[i] for i in range(num_labels)],
        "span_f1": metrics["span_f1"],
    }


@torch.no_grad()
def evaluar_metricas_ner(
    model,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Métricas en ventanas de validación (puede inflar conteos si hay solapamiento)."""
    if x_val.size(0) == 0:
        return {
            "overall_acc": 0.0,
            "entity_recall": 0.0,
            "entity_token_acc": 0.0,
            "macro_f1_non_o": 0.0,
            "span_f1": 0.0,
            "n_pred_entities": 0,
            "n_gold_entities": 0,
        }
    preds, labels = _collect_ner_preds_labels(model, x_val, y_val, batch_size, device)
    return _metrics_from_pred_label_lists(preds.tolist(), labels.tolist())


def ner_checkpoint_score(
    metricas: dict,
    metric_name: str,
    val_loss: float,
) -> tuple[float, bool]:
    """Devuelve (score, higher_is_better) para elegir el mejor checkpoint."""
    if metric_name == "val_loss":
        return val_loss, False
    if metric_name == "macro_f1_non_o":
        return metricas.get("macro_f1_non_o", 0.0), True
    if metric_name == "entity_recall":
        return metricas.get("entity_recall", 0.0), True
    if metric_name == "entity_acc_constrained":
        return metricas.get("entity_token_acc", 0.0), True
    raise ValueError(f"Métrica de selección desconocida: {metric_name}")


def passes_overall_acc_constraint(metricas: dict, min_overall_acc: float) -> bool:
    if min_overall_acc <= 0:
        return True
    return metricas.get("overall_acc", 0.0) >= min_overall_acc


def _macro_f1_non_o(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """F1 macro promediando solo clases de entidad (1..n-1)."""
    f1_scores: list[float] = []
    for class_id in range(1, len(LABEL2ID)):
        tp = ((preds == class_id) & (labels == class_id)).sum().item()
        fp = ((preds == class_id) & (labels != class_id)).sum().item()
        fn = ((preds != class_id) & (labels == class_id)).sum().item()
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def entrenar_epochs_causal(
    model: LLM,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    batch_size: int,
    description: str = "Entrenando LLM",
) -> tuple[float, float, list[dict]]:
    steps_per_epoch = math.ceil(x_train.size(0) / batch_size)
    total_steps = steps_per_epoch * epochs
    train_loss = 0.0
    best_val_loss = float("inf")
    best_state: dict | None = None
    history: list[dict] = []

    show_bar = sys.stdout.isatty()
    with Progress(
        TextColumn(f"[bold green]{description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
        disable=not show_bar,
    ) as progress:
        task = progress.add_task("train", total=total_steps)
        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0
            for x_batch, y_batch in iter_batches(x_train, y_train, batch_size):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = model.train_step(x_batch, y_batch, optimizer)
                epoch_loss += loss
                steps += 1
                progress.advance(task)
            train_loss = epoch_loss / max(steps, 1)
            val_loss = evaluar_loss_causal(model, x_val, y_val, batch_size, device)
            history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            )
            logger.info(
                "{} época {}/{} train_loss={:.4f} val_loss={:.4f} ({} batches)",
                description,
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                steps,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return train_loss, best_val_loss, history


@torch.no_grad()
def find_best_entity_threshold(
    model,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    device: torch.device,
    candidates: list[float] | None = None,
    *,
    val_sentences: list[dict] | None = None,
    tokenizer=None,
    window_size: int = 128,
) -> tuple[float, dict]:
    """Busca umbral en val que maximiza macro F1 (por frase si se pasan sentences)."""
    candidates = candidates or [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
    ]
    best_t = candidates[0]
    best_metrics: dict = {}
    best_f1 = -1.0
    use_sentence = val_sentences is not None and tokenizer is not None
    for threshold in candidates:
        model.entity_threshold = threshold
        if use_sentence:
            metrics = evaluar_metricas_ner_sentence_level(
                model, val_sentences, tokenizer, window_size, device
            )
        else:
            metrics = evaluar_metricas_ner(model, x_val, y_val, batch_size, device)
        f1 = metrics["macro_f1_non_o"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = threshold
            best_metrics = metrics
    model.entity_threshold = best_t
    return best_t, best_metrics


@torch.no_grad()
def evaluar_confusion_ner(
    model,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Calcula la matriz de confusión y métricas por clase (precision/recall/F1)."""
    num_labels = len(LABEL2ID)
    if x_val.size(0) == 0:
        return {"matrix": [[0] * num_labels] * num_labels, "per_class": {}}
    preds, labels = _collect_ner_preds_labels(model, x_val, y_val, batch_size, device)
    preds_list = preds.tolist()
    labels_list = labels.tolist()

    matrix = [[0] * num_labels for _ in range(num_labels)]
    for true, pred in zip(labels_list, preds_list):
        matrix[true][pred] += 1

    per_class: dict[str, dict] = {}
    for i in range(num_labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(num_labels)) - tp
        fn = sum(matrix[i][j] for j in range(num_labels)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[ID2LABEL[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    return {
        "matrix": matrix,
        "per_class": per_class,
        "labels": [ID2LABEL[i] for i in range(num_labels)],
    }
