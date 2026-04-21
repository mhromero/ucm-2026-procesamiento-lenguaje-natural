import json
import math
from pathlib import Path

import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from p5.BPETokenizer import BPETokenizer
from p5.LLM import LLM


def concatenar_textos_data(data_dir: str | Path = "p5/data") -> str:
    data_path = Path(data_dir)
    textos = []

    for archivo in sorted(data_path.glob("*.txt")):
        textos.append(archivo.read_text(encoding="utf-8"))

    return "\n".join(textos)


def cargar_config(path: str | Path = "p5/config.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    for start in range(0, x.size(0), batch_size):
        end = start + batch_size
        yield x[start:end], y[start:end]


def guardar_checkpoint(
    checkpoint_path: Path,
    model: LLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    last_loss: float,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "last_loss": last_loss,
    }
    torch.save(payload, checkpoint_path)


def cargar_checkpoint(
    checkpoint_path: Path,
    model: LLM,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["epoch"]), float(payload.get("last_loss", 0.0))


def mover_optimizador_a_dispositivo(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def guardar_salida_epoch(
    output_path: Path,
    epoch: int,
    loss: float,
    generated_text: str,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        history = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        history = []

    history.append(
        {
            "epoch": epoch,
            "loss": loss,
            "generated_text": generated_text,
        }
    )
    output_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def __main__():
    print("=== Inicio del pipeline LLM ===")
    print("Cargando configuración...")
    config = cargar_config()
    corpus_cfg = config["corpus"]
    tokenizer_cfg = config["tokenizer"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    checkpoint_cfg = config["checkpoint"]
    epoch_output_cfg = config["epoch_output"]
    gen_cfg = config["generation"]
    print("Configuración cargada correctamente.")

    print(f"Leyendo corpus desde: {corpus_cfg['data_dir']}")
    textos = concatenar_textos_data(corpus_cfg["data_dir"])
    print(f"Corpus cargado. Caracteres totales: {len(textos)}")
    tokenizer_path = Path(tokenizer_cfg["cache_path"])
    tokens_path = Path(tokenizer_cfg["tokens_cache_path"])

    if tokenizer_cfg["use_cache"] and tokenizer_path.exists():
        print(f"Cargando tokenizador desde {tokenizer_path}...")
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        print("Tokenizador cargado desde caché.")
    else:
        print("Entrenando tokenizador...")
        tokenizer = BPETokenizer(
            textos,
            vocab_size=tokenizer_cfg["vocab_size"],
        )
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizador guardado en {tokenizer_path}")

    if tokenizer_cfg["use_cache"] and tokens_path.exists():
        print(f"Cargando tokens desde {tokens_path}...")
        texto_tokenizado = json.loads(tokens_path.read_text(encoding="utf-8"))
        print("Tokens cargados desde caché.")
    else:
        print("Generando tokens del corpus...")
        texto_tokenizado = tokenizer.encode(textos)
        tokens_path.write_text(json.dumps(texto_tokenizado), encoding="utf-8")
        print(f"Tokens guardados en {tokens_path}")

    print(f"Total de ids tokenizados: {len(texto_tokenizado)}")
    print(f"Tamano del vocabulario: {len(tokenizer.get_tokens())}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")
    if device.type == "cuda":
        print(f"GPU activa: {torch.cuda.get_device_name(0)}")
    print("Construyendo modelo...")

    model = LLM(
        tokenizer=tokenizer,
        d_model=model_cfg["d_model"],
        n_blocks=model_cfg["n_blocks"],
        n_heads=model_cfg["n_heads"],
        window_size=model_cfg["window_size"],
        dropout=model_cfg["dropout"],
    )
    model.to(device)
    print("Modelo inicializado.")

    print("Construyendo ventanas deslizantes para entrenamiento...")
    x, y = model.build_windows(texto_tokenizado)
    print(f"Ventanas para entrenamiento: {x.size(0)}")
    print(f"Tamaño de ventana (D): {model_cfg['window_size']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    start_epoch = 0
    checkpoint_path = Path(checkpoint_cfg["path"])
    if checkpoint_cfg["resume_if_exists"] and checkpoint_path.exists():
        print(f"Cargando checkpoint desde {checkpoint_path}...")
        start_epoch, last_loss = cargar_checkpoint(checkpoint_path, model, optimizer)
        mover_optimizador_a_dispositivo(optimizer, device)
        print(
            f"Checkpoint cargado. Reanudando desde epoch {start_epoch + 1} "
            f"(ultima loss: {last_loss:.4f})"
        )

    steps_per_epoch = math.ceil(x.size(0) / train_cfg["batch_size"])
    remaining_epochs = max(train_cfg["epochs"] - start_epoch, 0)
    total_steps = steps_per_epoch * remaining_epochs
    print(
        f"Iniciando entrenamiento: epochs={train_cfg['epochs']}, "
        f"batch_size={train_cfg['batch_size']}, lr={train_cfg['learning_rate']}"
    )

    with Progress(
        TextColumn("[bold green]Entrenando LLM"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("train", total=total_steps)
        for epoch in range(start_epoch, train_cfg["epochs"]):
            epoch_loss = 0.0
            steps = 0
            for x_batch, y_batch in iter_batches(x, y, train_cfg["batch_size"]):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = model.train_step(x_batch, y_batch, optimizer)
                epoch_loss += loss
                steps += 1
                progress.advance(task)
                progress.update(
                    task,
                    description=f"Entrenando LLM (epoch {epoch + 1}/{train_cfg['epochs']})",
                )

            avg_loss = epoch_loss / max(steps, 1)
            print(f"Epoch {epoch + 1}/{train_cfg['epochs']} - loss: {avg_loss:.4f}")
            sample_text = model.generate(
                prompt=gen_cfg["prompt"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
            )
            guardar_salida_epoch(
                output_path=Path(epoch_output_cfg["path"]),
                epoch=epoch + 1,
                loss=avg_loss,
                generated_text=sample_text,
            )
            print(
                f"Salida de epoch {epoch + 1} guardada en {epoch_output_cfg['path']}"
            )
            if (epoch + 1) % checkpoint_cfg["save_every_epochs"] == 0:
                guardar_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    last_loss=avg_loss,
                )
                print(f"Checkpoint guardado en epoch {epoch + 1}: {checkpoint_path}")

    if train_cfg["epochs"] > 0 and checkpoint_cfg["save_at_end"]:
        final_loss = avg_loss if "avg_loss" in locals() else 0.0
        guardar_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=train_cfg["epochs"],
            last_loss=final_loss,
        )
        print(f"Checkpoint final guardado: {checkpoint_path}")

    print("Entrenamiento finalizado.")
    print("Generando texto de ejemplo...")
    generated_text = model.generate(
        prompt=gen_cfg["prompt"],
        max_new_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"],
    )
    print("\n=== Texto generado ===")
    print(generated_text)
    print("=== Fin del pipeline LLM ===")

if __name__ == "__main__":
    __main__()
