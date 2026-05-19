from __future__ import annotations

import math

import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from fdi_pln_2611_p5.LLM import LLM


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
) -> tuple[float, float]:
    steps_per_epoch = math.ceil(x_train.size(0) / batch_size)
    total_steps = steps_per_epoch * epochs
    train_loss = 0.0

    with Progress(
        TextColumn(f"[bold green]{description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("train", total=total_steps)
        for _epoch in range(epochs):
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

    test_loss = evaluar_loss_causal(model, x_val, y_val, batch_size, device)
    return train_loss, test_loss
