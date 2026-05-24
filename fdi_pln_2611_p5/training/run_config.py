"""Save and backfill training configuration next to model artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from fdi_pln_2611_p5.config import DEFAULT_CONFIG_PATH, PACKAGE_DIR, load_config, package_path

TRAINING_CONFIG_NAME = "training_config.json"
CONFIG_COPY_NAME = "config.json"
EXPERIMENT_CONFIG_NAME = "experiment_config.json"


def resolve_config_path(config_path: Path | None) -> Path:
    """Return the config file path used for a training run."""
    return Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH


def save_reproducibility_artifacts(
    output_dir: Path,
    config: dict,
    *,
    run_type: str,
    config_path: Path | None = None,
    extra: dict | None = None,
    write_training_config: bool = True,
    training_config_filename: str | None = None,
) -> Path:
    """Copy ``config.json`` and optionally write ``training_config.json`` under ``output_dir``.

    Args:
        output_dir: Directory that holds weights, metrics, or experiment caches.
        config: Full configuration dictionary used for the run.
        run_type: Label such as ``causal``, ``ner``, ``grid_search``, or ``experiment``.
        config_path: Source JSON file path (defaults to package ``config.json``).
        extra: Optional metadata merged into ``training_config.json``.
        write_training_config: When ``False``, only copies ``config.json`` (e.g. experiment suite).
        training_config_filename: Override default ``training_config.json`` basename.

    Returns:
        Path to ``config.json`` when written, else ``training_config.json``.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = resolve_config_path(config_path)
    copy_dest: Path | None = None
    if output_dir != PACKAGE_DIR.resolve():
        copy_dest = output_dir / CONFIG_COPY_NAME
        copy_dest.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Copia de config guardada en {}", copy_dest)

    if write_training_config:
        payload: dict = {
            "run_type": run_type,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "config_source": str(source),
            "config": deepcopy(config),
        }
        if extra:
            payload.update(extra)
        training_path = output_dir / (training_config_filename or TRAINING_CONFIG_NAME)
        training_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Config de entrenamiento guardada en {}", training_path)
        return training_path

    if copy_dest is None:
        raise ValueError("No se escribió ningún archivo de configuración")
    return copy_dest


def _backfill_grid_search(cache_dir: Path, base_config: dict) -> None:
    """Backfill grid-search config using ``grid_search_results.json`` when present."""
    results_path = cache_dir / "grid_search_results.json"
    if not results_path.is_file():
        results_path = package_path("data/grid_search_results.json")
    config = deepcopy(base_config)
    extra: dict = {
        "backfilled": True,
        "note": "Mejor lr/batch inferidos de grid_search_results.json.",
    }
    if results_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        best = payload.get("best", {})
        if best:
            config["training"] = {
                **config["training"],
                "learning_rate": best["learning_rate"],
                "batch_size": best["batch_size"],
            }
            extra["grid_search_best"] = best
            extra["grid_search_results_path"] = str(results_path)
    save_reproducibility_artifacts(
        cache_dir,
        config,
        run_type="grid_search",
        extra=extra,
    )


def backfill_missing_training_configs() -> list[Path]:
    """Write config files for artifact directories that lack them.

    Returns:
        Paths of newly written ``training_config.json`` or ``experiment_config.json`` files.
    """
    from fdi_pln_2611_p5.training.experiment_exploration import (
        EXPERIMENT_SPECS,
        _merge_config,
        _write_experiment_config,
    )

    base_config = load_config()
    written: list[Path] = []
    spec_by_id = {spec["id"]: spec for spec in EXPERIMENT_SPECS}

    def _write(path: Path) -> None:
        if path not in written:
            written.append(path)

    # Standalone weight files at package root.
    for weights_name in ("p5_causal_2611.pth", "p5_ner_2611.pth"):
        weights = package_path(weights_name)
        if not weights.is_file():
            continue
        out_dir = weights.parent
        training_name = f"{weights.stem}_training_config.json"
        if (out_dir / training_name).is_file():
            continue
        run_type = "ner" if "ner" in weights_name else "causal"
        save_reproducibility_artifacts(
            out_dir,
            base_config,
            run_type=run_type,
            training_config_filename=training_name,
            extra={
                "backfilled": True,
                "weights_path": str(weights),
            },
        )
        _write(out_dir / training_name)

    experiment_roots = [
        package_path("data/experiments"),
        package_path("experiments"),
    ]
    for root in experiment_roots:
        if not root.is_dir():
            continue
        for cache_dir in sorted(root.iterdir()):
            if not cache_dir.is_dir():
                continue
            exp_id = cache_dir.name
            exp_config = cache_dir / EXPERIMENT_CONFIG_NAME
            training_config = cache_dir / TRAINING_CONFIG_NAME

            if exp_id in spec_by_id and not exp_config.is_file():
                spec = spec_by_id[exp_id]
                config = _merge_config(base_config, spec["overrides"])
                exp_cfg = config.get("experiment_exploration", {})
                config = {
                    **config,
                    "experiment_exploration": {
                        **exp_cfg,
                        "epochs_per_run": exp_cfg.get("epochs_per_run", 5),
                    },
                }
                _write_experiment_config(exp_config, spec, config)
                (cache_dir / CONFIG_COPY_NAME).write_text(
                    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                logger.info("Backfill experiment_config en {}", exp_config)
                _write(exp_config)

            if exp_config.is_file() and not (cache_dir / CONFIG_COPY_NAME).is_file():
                payload = json.loads(exp_config.read_text(encoding="utf-8"))
                merged = payload.get("config", base_config)
                (cache_dir / CONFIG_COPY_NAME).write_text(
                    json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                logger.info("Backfill config.json en {}", cache_dir / CONFIG_COPY_NAME)

            if training_config.is_file():
                continue

            if exp_id == "grid_search":
                _backfill_grid_search(cache_dir, base_config)
                if (cache_dir / TRAINING_CONFIG_NAME).is_file():
                    _write(cache_dir / TRAINING_CONFIG_NAME)
                continue

            if exp_id in ("causal_final", "ner"):
                run_type = "ner" if exp_id == "ner" else "causal"
                config = deepcopy(base_config)
                extra: dict = {"backfilled": True, "note": "Reconstruido desde config.json del paquete."}
                if run_type == "causal":
                    history = cache_dir / "causal_history.csv"
                    if history.is_file():
                        extra["causal_history_csv"] = history.name
                else:
                    history = cache_dir / "ner_history.csv"
                    if history.is_file():
                        extra["ner_history_csv"] = history.name
                save_reproducibility_artifacts(
                    cache_dir, config, run_type=run_type, extra=extra
                )
                _write(cache_dir / TRAINING_CONFIG_NAME)
                continue

            if exp_id in spec_by_id:
                continue

    return written
