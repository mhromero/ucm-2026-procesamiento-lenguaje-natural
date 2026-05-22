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
from fdi_pln_2611_p5.corpus import build_extra_train_corpus, concatenar_archivos_txt
from fdi_pln_2611_p5.LLM import LLM
from fdi_pln_2611_p5.training.grid_search import run_grid_search
from fdi_pln_2611_p5.training.utils import entrenar_epochs_causal


def prepare_tokenizer_and_tokens(
    config: dict,
    cache_dir: Path | None = None,
) -> tuple[BPETokenizer, list[int], list[int]]:
    """Prepara BPE y tokens. Si ``cache_dir`` se indica, usa cachés aisladas (p. ej. por experimento)."""
    corpus_cfg = config["corpus"]
    tokenizer_cfg = config["tokenizer"]

    alice_textos = concatenar_archivos_txt(package_path(corpus_cfg["data_dir"])).lower()
    max_books = corpus_cfg.get("extra_max_books")
    extra_train_textos = build_extra_train_corpus(corpus_cfg)
    if max_books == 0:
        logger.info("Corpus entrenamiento: solo Alice (sin Harry Potter)")
    elif max_books == 4:
        logger.info(
            "Corpus entrenamiento: Alice + 4 primeros libros HP ({} caracteres HP)",
            len(extra_train_textos),
        )
    elif extra_train_textos:
        logger.info(
            "Corpus entrenamiento: Alice + HP ({} caracteres extra)",
            len(extra_train_textos),
        )
    textos_train = (
        alice_textos if not extra_train_textos else alice_textos + "\n" + extra_train_textos
    )

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = cache_dir / "bpe_tokenizer.json"
        train_tokens_path = cache_dir / "train_tokens.json"
        test_tokens_path = cache_dir / "test_tokens.json"
    else:
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
    extra_textos = build_extra_train_corpus(corpus_cfg)
    texto_completo = (
        alice_textos if not extra_textos else alice_textos + "\n" + extra_textos
    )

    tokenizer = BPETokenizer(texto_completo, vocab_size=tokenizer_cfg["vocab_size"])
    tokenizer_path = package_path(tokenizer_cfg["cache_path"])
    tokenizer.save(str(tokenizer_path))
    logger.info(
        "Tokenizador BPE guardado en {} (vocab_size={})",
        tokenizer_path,
        len(tokenizer.tok2id),
    )
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
    """Entrena el LLM causal sobre Alice + Harry Potter y guarda el checkpoint.

    Si grid_search=True ejecuta primero 9 combinaciones lr×batch y usa la mejor
    configuración para el entrenamiento final.
    """
    config = load_config(config_path)
    corpus_cfg = config["corpus"]
    max_books = corpus_cfg.get("extra_max_books")
    if max_books == 0:
        logger.info("Corpus entrenamiento: solo Alice (sin Harry Potter)")
    elif max_books == 4:
        logger.info("Corpus entrenamiento: Alice + 4 primeros libros HP")
    elif max_books is None:
        logger.info("Corpus extra: todos los libros HP (extra_max_books no definido)")
    else:
        logger.info("Corpus extra: usar primeros {} libros HP", max_books)

    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Semilla: {}", seed)

    tokenizer, train_tokens, test_tokens = prepare_tokenizer_and_tokens(config)
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
    train_loss, test_loss, history = entrenar_epochs_causal(
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

    history_path = package_path(config["metrics"]["causal_history_csv_path"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    sample = model.generate(
        prompt=config["generation"]["prompt"],
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
    )
    logger.info("Muestra generada:\n{}", sample)
    return {"train_loss": train_loss, "test_loss": test_loss, "sample": sample}
