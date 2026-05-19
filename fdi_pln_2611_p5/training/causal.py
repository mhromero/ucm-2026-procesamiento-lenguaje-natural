from __future__ import annotations

import json
import random
from csv import DictWriter
from pathlib import Path

import torch
from loguru import logger

from fdi_pln_2611_p5.BPETokenizer import BPETokenizer
from fdi_pln_2611_p5.checkpoints import save_causal_checkpoint
from fdi_pln_2611_p5.config import load_config, package_path
from fdi_pln_2611_p5.corpus import concatenar_archivos_txt
from fdi_pln_2611_p5.LLM import LLM
from fdi_pln_2611_p5.training.grid_search import run_grid_search
from fdi_pln_2611_p5.training.utils import entrenar_epochs_causal


def _prepare_tokenizer_and_tokens(
    config: dict,
) -> tuple[BPETokenizer, list[int], list[int]]:
    corpus_cfg = config["corpus"]
    tokenizer_cfg = config["tokenizer"]

    alice_textos = concatenar_archivos_txt(package_path(corpus_cfg["data_dir"])).lower()
    extra_train_textos = concatenar_archivos_txt(
        package_path(corpus_cfg["extra_data_dir"])
    ).lower()
    textos_train = alice_textos + "\n" + extra_train_textos

    tokenizer_path = package_path(tokenizer_cfg["cache_path"])
    train_tokens_path = package_path(tokenizer_cfg["train_tokens_cache_path"])
    test_tokens_path = package_path(tokenizer_cfg["test_tokens_cache_path"])

    if tokenizer_cfg["use_cache"] and tokenizer_path.exists():
        tokenizer = BPETokenizer.load(str(tokenizer_path))
    else:
        tokenizer = BPETokenizer(textos_train, vocab_size=tokenizer_cfg["vocab_size"])
        tokenizer.save(str(tokenizer_path))

    if (
        tokenizer_cfg["use_cache"]
        and train_tokens_path.exists()
        and test_tokens_path.exists()
    ):
        train_tokenizado = json.loads(train_tokens_path.read_text(encoding="utf-8"))
        test_tokenizado = json.loads(test_tokens_path.read_text(encoding="utf-8"))
    else:
        train_tokenizado = tokenizer.encode(textos_train)
        train_tokens_path.write_text(json.dumps(train_tokenizado), encoding="utf-8")
        test_tokenizado = tokenizer.encode(alice_textos)
        test_tokens_path.write_text(json.dumps(test_tokenizado), encoding="utf-8")

    return tokenizer, train_tokenizado, test_tokenizado


def train_tokenizer(config_path: Path | None = None) -> BPETokenizer:
    """Entrena y guarda el tokenizador BPE; sobreescribe la caché existente."""
    config = load_config(config_path)
    corpus_cfg = config["corpus"]
    tokenizer_cfg = config["tokenizer"]

    alice_textos = concatenar_archivos_txt(package_path(corpus_cfg["data_dir"])).lower()
    extra_textos = concatenar_archivos_txt(package_path(corpus_cfg["extra_data_dir"])).lower()
    texto_completo = alice_textos + "\n" + extra_textos

    tokenizer = BPETokenizer(texto_completo, vocab_size=tokenizer_cfg["vocab_size"])
    tokenizer_path = package_path(tokenizer_cfg["cache_path"])
    tokenizer.save(str(tokenizer_path))
    logger.info("Tokenizador BPE guardado en {} (vocab_size={})", tokenizer_path, len(tokenizer.tok2id))
    return tokenizer


def build_model(config: dict, tokenizer: BPETokenizer) -> LLM:
    model_cfg = config["model"]
    return LLM(
        tokenizer=tokenizer,
        d_model=model_cfg["d_model"],
        n_blocks=model_cfg["n_blocks"],
        n_heads=model_cfg["n_heads"],
        window_size=model_cfg["window_size"],
        dropout=model_cfg["dropout"],
    )


def train_causal(
    weights_path: Path,
    config_path: Path | None = None,
    grid_search: bool = False,
) -> dict:
    config = load_config(config_path)
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Semilla: {}", seed)

    tokenizer, train_tokens, test_tokens = _prepare_tokenizer_and_tokens(config)
    model_cfg = config["model"]
    train_cfg = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: {}", device)

    model = build_model(config, tokenizer)
    model.to(device)
    x_train, y_train = model.build_windows(train_tokens)
    x_test, y_test = model.build_windows(test_tokens)

    model_config = {
        "d_model": model_cfg["d_model"],
        "n_blocks": model_cfg["n_blocks"],
        "n_heads": model_cfg["n_heads"],
        "window_size": model_cfg["window_size"],
        "dropout": model_cfg["dropout"],
    }
    tokenizer_path = package_path(config["tokenizer"]["cache_path"])

    if grid_search:
        best = run_grid_search(
            config=config,
            build_model=lambda: build_model(config, tokenizer),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            device=device,
        )
        train_cfg = {**train_cfg, **best["params"]}
        model = build_model(config, tokenizer)
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    train_loss, test_loss = entrenar_epochs_causal(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_test,
        y_val=y_test,
        optimizer=optimizer,
        device=device,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
    )
    logger.info("train_loss={:.4f} test_loss={:.4f}", train_loss, test_loss)

    save_causal_checkpoint(
        weights_path,
        model,
        model_config,
        tokenizer_path,
        extra={
            "train_loss": train_loss,
            "test_loss": test_loss,
            "hyperparams": train_cfg,
        },
    )

    metrics_path = package_path(config["metrics"]["loss_csv_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(
            handle,
            fieldnames=["train_loss", "test_loss", "learning_rate", "batch_size"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "learning_rate": train_cfg["learning_rate"],
                "batch_size": train_cfg["batch_size"],
            }
        )

    sample = model.generate(
        prompt=config["generation"]["prompt"],
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
    )
    logger.info("Muestra generada:\n{}", sample)
    return {"train_loss": train_loss, "test_loss": test_loss, "sample": sample}
