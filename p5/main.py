import json
import math
from csv import DictWriter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from p5.BPETokenizer import BPETokenizer
from p5.LLM import LLM


def concatenar_archivos_txt(data_dir: str | Path) -> str:
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
    train_loss: float,
    test_loss: float,
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
            "train_loss": train_loss,
            "test_loss": test_loss,
            "generated_text": generated_text,
        }
    )
    output_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@torch.no_grad()
def evaluar_loss(
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


def guardar_losses_csv(output_path: Path, rows: list[dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=["epoch", "train_loss", "test_loss"])
        writer.writeheader()
        writer.writerows(rows)


def guardar_grafica_losses(output_path: Path, rows: list[dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    test_losses = [row["test_loss"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="train_loss")
    plt.plot(epochs, test_losses, marker="o", label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolucion de train/test loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


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
    metrics_cfg = config["metrics"]
    gen_cfg = config["generation"]
    print("Configuración cargada correctamente.")

    print(f"Leyendo corpus principal (Alice) desde: {corpus_cfg['data_dir']}")
    alice_textos = concatenar_archivos_txt(corpus_cfg["data_dir"])
    print(f"Corpus Alice cargado. Caracteres totales: {len(alice_textos)}")
    print(f"Leyendo corpus extra (train-only) desde: {corpus_cfg['extra_data_dir']}")
    extra_train_textos = concatenar_archivos_txt(corpus_cfg["extra_data_dir"]).lower()
    print(
        f"Corpus extra cargado (solo train). Caracteres totales: {len(extra_train_textos)}"
    )
    textos_train = alice_textos + "\n" + extra_train_textos
    tokenizer_path = Path(tokenizer_cfg["cache_path"])
    train_tokens_path = Path(tokenizer_cfg["train_tokens_cache_path"])
    test_tokens_path = Path(tokenizer_cfg["test_tokens_cache_path"])

    if tokenizer_cfg["use_cache"] and tokenizer_path.exists():
        print(f"Cargando tokenizador desde {tokenizer_path}...")
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        print("Tokenizador cargado desde caché.")
    else:
        print("Entrenando tokenizador...")
        tokenizer = BPETokenizer(
            textos_train,
            vocab_size=tokenizer_cfg["vocab_size"],
        )
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizador guardado en {tokenizer_path}")

    if (
        tokenizer_cfg["use_cache"]
        and train_tokens_path.exists()
        and test_tokens_path.exists()
    ):
        print(f"Cargando tokens train desde {train_tokens_path}...")
        train_tokenizado = json.loads(train_tokens_path.read_text(encoding="utf-8"))
        print(f"Cargando tokens test desde {test_tokens_path}...")
        test_tokenizado = json.loads(test_tokens_path.read_text(encoding="utf-8"))
        print("Tokens train/test cargados desde caché.")
    else:
        print("Generando tokens de train (Alice + Harry Potter)...")
        train_tokenizado = tokenizer.encode(textos_train)
        train_tokens_path.write_text(json.dumps(train_tokenizado), encoding="utf-8")
        print(f"Tokens train guardados en {train_tokens_path}")
        print("Generando tokens de test (solo Alice)...")
        test_tokenizado = tokenizer.encode(alice_textos)
        test_tokens_path.write_text(json.dumps(test_tokenizado), encoding="utf-8")
        print(f"Tokens test guardados en {test_tokens_path}")

    print(f"Total de ids tokenizados (train): {len(train_tokenizado)}")
    print(f"Total de ids tokenizados (test/Alice): {len(test_tokenizado)}")
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
    x_train, y_train = model.build_windows(train_tokenizado)
    x_test, y_test = model.build_windows(test_tokenizado)
    print(f"Ventanas para entrenamiento (Alice + HP): {x_train.size(0)}")
    print(f"Ventanas para test (solo Alice): {x_test.size(0)}")
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

    steps_per_epoch = math.ceil(x_train.size(0) / train_cfg["batch_size"])
    remaining_epochs = max(train_cfg["epochs"] - start_epoch, 0)
    total_steps = steps_per_epoch * remaining_epochs
    loss_rows = []
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
            for x_batch, y_batch in iter_batches(x_train, y_train, train_cfg["batch_size"]):
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

            train_loss = epoch_loss / max(steps, 1)
            test_loss = evaluar_loss(
                model=model,
                x_data=x_test,
                y_data=y_test,
                batch_size=train_cfg["batch_size"],
                device=device,
            )
            print(
                f"Epoch {epoch + 1}/{train_cfg['epochs']} - "
                f"train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f}"
            )
            loss_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                }
            )
            guardar_losses_csv(Path(metrics_cfg["loss_csv_path"]), loss_rows)
            sample_text = model.generate(
                prompt=gen_cfg["prompt"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
            )
            guardar_salida_epoch(
                output_path=Path(epoch_output_cfg["path"]),
                epoch=epoch + 1,
                train_loss=train_loss,
                test_loss=test_loss,
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
                    last_loss=train_loss,
                )
                print(f"Checkpoint guardado en epoch {epoch + 1}: {checkpoint_path}")

    if train_cfg["epochs"] > 0 and checkpoint_cfg["save_at_end"]:
        final_loss = train_loss if "train_loss" in locals() else 0.0
        guardar_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=train_cfg["epochs"],
            last_loss=final_loss,
        )
        print(f"Checkpoint final guardado: {checkpoint_path}")
    guardar_grafica_losses(Path(metrics_cfg["loss_plot_path"]), loss_rows)
    print(f"CSV de losses guardado en: {metrics_cfg['loss_csv_path']}")
    print(f"Grafica de losses guardada en: {metrics_cfg['loss_plot_path']}")

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
