from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch
from loguru import logger
from rich.console import Console
from rich.table import Table

from fdi_pln_2611_p5.config import package_path
from fdi_pln_2611_p5.model.lm_causal.llm import LLM
from fdi_pln_2611_p5.training.grid_search_report import generate_grid_search_html
from fdi_pln_2611_p5.training.utils import entrenar_epochs_causal


def run_grid_search(
    config: dict,
    build_model: Callable[[], LLM],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
) -> dict:
    grid_cfg = config["grid_search"]
    learning_rates = grid_cfg["learning_rates"]
    batch_sizes = grid_cfg["batch_sizes"]
    epochs = grid_cfg["epochs_per_run"]

    results: list[dict] = []
    for lr in learning_rates:
        for batch_size in batch_sizes:
            model = build_model()
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            train_loss, test_loss, _history = entrenar_epochs_causal(
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_val=x_test,
                y_val=y_test,
                optimizer=optimizer,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                description=f"grid lr={lr} bs={batch_size}",
            )
            row = {
                "learning_rate": lr,
                "batch_size": batch_size,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
            results.append(row)
            logger.info("Grid {} -> test_loss={:.4f}", row, test_loss)

    best = min(results, key=lambda item: item["test_loss"])
    output_path = package_path(grid_cfg["results_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"results": results, "best": best}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _print_grid_table(results, best)
    logger.info("Resultados guardados en {}", output_path)

    report_path = output_path.parent / "informe_grid_search.html"
    generate_grid_search_html(output_path, report_path)
    logger.info("Informe HTML guardado en {}", report_path)
    return {
        "params": {
            "learning_rate": best["learning_rate"],
            "batch_size": best["batch_size"],
        },
        "best": best,
    }


def _print_grid_table(results: list[dict], best: dict) -> None:
    sorted_results = sorted(results, key=lambda r: r["test_loss"])
    table = Table(title="Resultados Grid Search", show_lines=True)
    table.add_column("lr", style="cyan", justify="right")
    table.add_column("batch", style="cyan", justify="right")
    table.add_column("train_loss", justify="right")
    table.add_column("test_loss", justify="right")
    table.add_column("", justify="center")

    for row in sorted_results:
        is_best = (
            row["learning_rate"] == best["learning_rate"]
            and row["batch_size"] == best["batch_size"]
        )
        style = "bold green" if is_best else ""
        marker = "★ mejor" if is_best else ""
        table.add_row(
            str(row["learning_rate"]),
            str(row["batch_size"]),
            f"{row['train_loss']:.4f}",
            f"{row['test_loss']:.4f}",
            marker,
            style=style,
        )

    Console().print(table)
