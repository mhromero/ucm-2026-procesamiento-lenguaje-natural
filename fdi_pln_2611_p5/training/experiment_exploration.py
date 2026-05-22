from __future__ import annotations

import copy
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from loguru import logger
from rich.console import Console
from rich.table import Table

from fdi_pln_2611_p5.checkpoints import save_causal_checkpoint
from fdi_pln_2611_p5.config import load_config, package_path
from fdi_pln_2611_p5.training.causal import build_model, prepare_tokenizer_and_tokens
from fdi_pln_2611_p5.training.experiment_report import generate_experiment_html
from fdi_pln_2611_p5.training.utils import entrenar_epochs_causal

# Ocho experimentos: corpus, ventana, vocabulario BPE y profundidad del Transformer.
# Validación siempre en Alice. Salvo corpus_alice, train = Alice + 4 primeros libros HP.
# Referencia común: window=128, vocab=300, n_blocks=4 salvo que el eje indique otra cosa.
_REF = {
    "corpus": {"extra_max_books": 4},
    "tokenizer": {"vocab_size": 300, "show_progress": False},
    "model": {"window_size": 128, "n_blocks": 4},
}

EXPERIMENT_SPECS: list[dict] = [
    {
        "id": "corpus_alice",
        "name": "Solo Alice (sin Harry Potter)",
        "variable": "corpus.extra_max_books",
        "overrides": {
            "corpus": {"extra_max_books": 0},
            "model": {"window_size": 128, "n_blocks": 4},
            "tokenizer": {"vocab_size": 300, "show_progress": False},
        },
        "research_question": (
            "¿Entrenar únicamente con Alice mejora la loss en validación (también Alice) "
            "frente a mezclar dominios con Harry Potter?"
        ),
        "expected_insight": (
            "Menos ventanas de train pero mayor homogeneidad de estilo/vocabulario con el test."
        ),
    },
    {
        "id": "corpus_alice_hp4",
        "name": "Alice + 4 libros Harry Potter",
        "variable": "corpus.extra_max_books",
        "overrides": _REF,
        "research_question": (
            "¿Añadir los 4 primeros libros HP al train reduce val_loss en Alice "
            "por más datos o lo empeora por desalineación de dominio?"
        ),
        "expected_insight": (
            "Más tokens y ventanas; posible trade-off dominio literario vs. cantidad de datos."
        ),
    },
    {
        "id": "window_64",
        "name": "Ventana 64 (Alice + 4 HP)",
        "variable": "model.window_size",
        "overrides": {**_REF, "model": {"window_size": 64, "n_blocks": 4}},
        "research_question": (
            "Con el mismo corpus (Alice+4HP), ¿una ventana más corta (64) "
            "cambia el número de batches y la generalización en Alice?"
        ),
        "expected_insight": (
            "Más ventanas deslizantes y menos contexto por paso; comparar con corpus_alice_hp4."
        ),
    },
    {
        "id": "window_128",
        "name": "Ventana 128 (Alice + 4 HP, referencia)",
        "variable": "model.window_size",
        "overrides": _REF,
        "research_question": (
            "Línea base de ventana larga con Alice+4HP; comparación directa con window_64."
        ),
        "expected_insight": (
            "Misma config que corpus_alice_hp4; aísla el efecto de window_size frente a window_64."
        ),
    },
    {
        "id": "vocab_200",
        "name": "Vocabulario BPE = 200",
        "variable": "tokenizer.vocab_size",
        "overrides": {**_REF, "tokenizer": {"vocab_size": 200, "show_progress": False}},
        "research_question": (
            "¿Un vocabulario más pequeño (más merges, secuencias más largas) "
            "empeora o mejora la loss en validación (Alice)?"
        ),
        "expected_insight": (
            "Relación entre granularidad BPE, número de ventanas y capacidad del modelo."
        ),
    },
    {
        "id": "vocab_300",
        "name": "Vocabulario BPE = 300 (referencia)",
        "variable": "tokenizer.vocab_size",
        "overrides": _REF,
        "research_question": (
            "Vocabulario intermedio del enunciado con Alice+4HP y ventana 128."
        ),
        "expected_insight": "Punto de comparación para vocab_200 y para depth_2/depth_4.",
    },
    {
        "id": "depth_2",
        "name": "Profundidad = 2 bloques Transformer",
        "variable": "model.n_blocks",
        "overrides": {**_REF, "model": {"window_size": 128, "n_blocks": 2}},
        "research_question": (
            "¿Un modelo más superficial alcanza loss similar con menos capacidad "
            "o queda corto frente al corpus combinado?"
        ),
        "expected_insight": "Trade-off capacidad vs. sobreajuste con pocos bloques.",
    },
    {
        "id": "depth_4",
        "name": "Profundidad = 4 bloques (referencia)",
        "variable": "model.n_blocks",
        "overrides": _REF,
        "research_question": (
            "Configuración por defecto del enunciado (4 bloques, 4 cabezas)."
        ),
        "expected_insight": "Punto de comparación para depth_2; misma config que vocab_300.",
    },
]


def _merge_config(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _corpus_train_label(config: dict) -> str:
    max_books = config["corpus"].get("extra_max_books")
    if max_books == 0:
        return "solo Alice"
    if max_books == 4:
        return "Alice + 4 libros HP"
    if max_books is None:
        return "Alice + HP (completo)"
    return f"Alice + HP (max_books={max_books})"


def _config_snapshot(config: dict) -> dict:
    return {
        "corpus_train": _corpus_train_label(config),
        "vocab_size": config["tokenizer"]["vocab_size"],
        "n_blocks": config["model"]["n_blocks"],
        "n_heads": config["model"]["n_heads"],
        "d_model": config["model"]["d_model"],
        "window_size": config["model"]["window_size"],
        "dropout": config["model"]["dropout"],
        "extra_max_books": config["corpus"].get("extra_max_books"),
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["batch_size"],
    }


def _tokenizer_corpus_key(config: dict) -> str:
    """Clave del corpus sobre el que se entrena el BPE (Alice ≠ Alice+HP)."""
    if config["corpus"].get("extra_max_books") == 0:
        return "alice_only"
    return f"alice_hp{config['corpus'].get('extra_max_books', 'all')}"


def _experiment_config_key(config: dict) -> tuple:
    """Dos runs solo comparten entrenamiento si BPE y modelo son equivalentes."""
    return (
        _tokenizer_corpus_key(config),
        config["tokenizer"]["vocab_size"],
        config["model"]["window_size"],
        config["model"]["n_blocks"],
    )


_TOKENIZER_CACHE_FILES = (
    "bpe_tokenizer.json",
    "train_tokens.json",
    "test_tokens.json",
)
EXPERIMENT_WEIGHTS_NAME = "model.pth"
EXPERIMENT_CONFIG_NAME = "experiment_config.json"


def _copy_experiment_artifacts(source_dir: Path, dest_dir: Path) -> None:
    """Copia BPE, tokens, pesos y config JSON al directorio del experimento."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in _TOKENIZER_CACHE_FILES + (EXPERIMENT_WEIGHTS_NAME, EXPERIMENT_CONFIG_NAME):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, dest_dir / name)


def _save_experiment_artifacts(
    cache_dir: Path,
    spec: dict,
    config: dict,
    model,
    tokenizer,
    train_loss: float,
    val_loss: float,
    best_epoch: dict,
    history: list[dict],
) -> tuple[Path, Path]:
    """Guarda pesos (.pth) y configuración (JSON) en data/experiments/<id>/."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / EXPERIMENT_WEIGHTS_NAME
    config_path = cache_dir / EXPERIMENT_CONFIG_NAME
    model_cfg = config["model"]
    model_config = {
        "d_model": model_cfg["d_model"],
        "n_blocks": model_cfg["n_blocks"],
        "n_heads": model_cfg["n_heads"],
        "window_size": model_cfg["window_size"],
        "dropout": model_cfg["dropout"],
    }
    tokenizer_path = cache_dir / "bpe_tokenizer.json"
    save_causal_checkpoint(
        weights_path,
        model,
        model_config,
        tokenizer_path,
        extra={
            "experiment_id": spec["id"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_epoch": best_epoch,
            "history": history,
            "config_snapshot": _config_snapshot(config),
        },
    )
    _write_experiment_config(config_path, spec, config)
    logger.info("Pesos guardados en {}", weights_path)
    logger.info("Config guardada en {}", config_path)
    return weights_path, config_path


def _write_experiment_config(
    config_path: Path,
    spec: dict,
    config: dict,
    *,
    reused_from: str | None = None,
) -> None:
    payload: dict = {
        "experiment_id": spec["id"],
        "experiment_name": spec["name"],
        "variable": spec["variable"],
        "research_question": spec["research_question"],
        "config": config,
        "config_snapshot": _config_snapshot(config),
    }
    if reused_from:
        payload["reused_training_from"] = reused_from
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_single_experiment(
    spec: dict,
    base_config: dict,
    epochs: int,
    device: torch.device,
    cache_root: Path,
) -> dict:
    config = _merge_config(base_config, spec["overrides"])
    cache_dir = cache_root / spec["id"]

    logger.info("══ Experimento {} — {} ══", spec["id"], spec["name"])
    random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))

    logger.info(
        "BPE: corpus={} · vocab_size={} · caché={}",
        _tokenizer_corpus_key(config),
        config["tokenizer"]["vocab_size"],
        cache_dir,
    )
    tokenizer, train_tokens, test_tokens = prepare_tokenizer_and_tokens(
        config, cache_dir=cache_dir
    )
    logger.info(
        "BPE listo: {} tokens en vocab · {} tokens train · caché {}",
        len(tokenizer.tok2id),
        len(train_tokens),
        cache_dir,
    )
    model = build_model(config, tokenizer)
    model.to(device)
    window_size = config["model"]["window_size"]
    logger.info(
        "Construyendo ventanas (window={}) sobre {} tokens train…",
        window_size,
        len(train_tokens),
    )
    x_train, y_train = model.build_windows(train_tokens)
    logger.info("Ventanas train: {}", x_train.size(0))
    logger.info(
        "Construyendo ventanas test sobre {} tokens…",
        len(test_tokens),
    )
    x_test, y_test = model.build_windows(test_tokens)
    logger.info("Ventanas test: {}", x_test.size(0))

    batch_size = config["training"]["batch_size"]
    steps_per_epoch = math.ceil(x_train.size(0) / batch_size) if x_train.size(0) else 0
    logger.info(
        "Entrenamiento: {} batches/época × {} épocas (batch_size={})",
        steps_per_epoch,
        epochs,
        batch_size,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    train_loss, val_loss, history = entrenar_epochs_causal(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_test,
        y_val=y_test,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        description=f"exp {spec['id']}",
    )

    best_epoch = min(history, key=lambda row: row["val_loss"])
    weights_path, config_path = _save_experiment_artifacts(
        cache_dir,
        spec,
        config,
        model,
        tokenizer,
        train_loss,
        val_loss,
        best_epoch,
        history,
    )
    return {
        "id": spec["id"],
        "name": spec["name"],
        "variable": spec["variable"],
        "research_question": spec["research_question"],
        "expected_insight": spec["expected_insight"],
        "config": _config_snapshot(config),
        "tokenizer_corpus_key": _tokenizer_corpus_key(config),
        "tokenizer_cache_dir": str(cache_dir),
        "weights_path": str(weights_path),
        "config_path": str(config_path),
        "bpe_vocab_actual": len(tokenizer.tok2id),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "dataset_stats": {
            "train_tokens": len(train_tokens),
            "test_tokens": len(test_tokens),
            "train_windows": int(x_train.size(0)),
            "test_windows": int(x_test.size(0)),
            "steps_per_epoch": steps_per_epoch,
            "batches_total": steps_per_epoch * epochs,
        },
    }


def run_experiment_exploration(config_path: Path | None = None) -> dict:
    """Ejecuta los 8 experimentos y guarda JSON + informe HTML."""
    base_config = load_config(config_path)
    exp_cfg = base_config["experiment_exploration"]
    epochs = exp_cfg["epochs_per_run"]
    cache_root = package_path(exp_cfg["cache_root"])
    results_path = package_path(exp_cfg["results_path"])
    report_path = package_path(exp_cfg["report_path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Exploración de experimentos · dispositivo={} · {} épocas/exp", device, epochs)

    results: list[dict] = []
    by_config_key: dict[tuple, dict] = {}
    for spec in EXPERIMENT_SPECS:
        config = _merge_config(base_config, spec["overrides"])
        key = _experiment_config_key(config)
        cache_dir = cache_root / spec["id"]
        if key in by_config_key:
            source = by_config_key[key]
            _copy_experiment_artifacts(Path(source["tokenizer_cache_dir"]), cache_dir)
            _write_experiment_config(
                cache_dir / EXPERIMENT_CONFIG_NAME,
                spec,
                config,
                reused_from=source["id"],
            )
            reused = copy.deepcopy(source)
            reused["id"] = spec["id"]
            reused["name"] = spec["name"]
            reused["variable"] = spec["variable"]
            reused["research_question"] = spec["research_question"]
            reused["expected_insight"] = spec["expected_insight"]
            reused["reused_from"] = source["id"]
            reused["tokenizer_cache_dir"] = str(cache_dir)
            reused["weights_path"] = str(cache_dir / EXPERIMENT_WEIGHTS_NAME)
            reused["config_path"] = str(cache_dir / EXPERIMENT_CONFIG_NAME)
            results.append(reused)
            logger.info(
                "  → {} reutiliza entrenamiento de {} (misma config); "
                "artefactos copiados a {}",
                spec["id"],
                reused["reused_from"],
                cache_dir,
            )
            continue
        row = run_single_experiment(spec, base_config, epochs, device, cache_root)
        by_config_key[key] = row
        results.append(row)
        logger.info(
            "  → {} val_loss={:.4f} (mejor época {} val={:.4f})",
            spec["id"],
            row["val_loss"],
            row["best_epoch"]["epoch"],
            row["best_epoch"]["val_loss"],
        )

    best = min(results, key=lambda r: r["best_epoch"]["val_loss"])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "description": (
                "Ocho configuraciones etiquetadas, con entrenamiento real solo cuando la "
                "combinación (corpus BPE, vocab, window, n_blocks) es nueva. Cada id "
                "tiene su carpeta en data/experiments/ con BPE, model.pth y "
                "experiment_config.json (solo Alice ≠ Alice+HP; vocab 200 ≠ 300). Ejes: corpus "
                "(solo Alice vs Alice+4HP), window_size (64 vs 128), vocab BPE (200 vs 300), "
                "n_blocks (2 vs 4). Misma lr, batch y épocas; validación siempre en Alice."
            ),
            "epochs_per_run": epochs,
            "metric_selection": "Menor val_loss por época (checkpoint implícito en entrenar_epochs_causal).",
            "fixed_hyperparams": _config_snapshot(base_config),
        },
        "experiments": results,
        "best": best,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Resultados JSON: {}", results_path)

    generate_experiment_html(payload, report_path)
    logger.info("Informe HTML: {}", report_path)

    _print_summary_table(results, best)
    return payload


def _print_summary_table(results: list[dict], best: dict) -> None:
    table = Table(title="Exploración de experimentos (8 runs)", show_lines=True)
    table.add_column("ID", style="cyan")
    table.add_column("Corpus train")
    table.add_column("Window", justify="right")
    table.add_column("Val loss", justify="right")
    table.add_column("Ventanas train", justify="right")
    table.add_column("Mejor época", justify="right")
    table.add_column("", justify="center")

    for row in sorted(results, key=lambda r: r["best_epoch"]["val_loss"]):
        is_best = row["id"] == best["id"]
        table.add_row(
            row["id"],
            row["config"]["corpus_train"],
            str(row["config"]["window_size"]),
            f"{row['best_epoch']['val_loss']:.4f}",
            str(row["dataset_stats"]["train_windows"]),
            str(row["best_epoch"]["epoch"]),
            "★" if is_best else "",
            style="bold green" if is_best else "",
        )
    Console().print(table)
