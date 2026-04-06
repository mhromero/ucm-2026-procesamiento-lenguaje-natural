from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

DATA_DIR = Path(__file__).parent / "data"
HTML_PATH = DATA_DIR / "2000-h.htm"
PARRAFOS_PATH = DATA_DIR / "parrafos_index.json"
INDICE_PATH = DATA_DIR / "vocabulario_index.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_IDS_PATH = DATA_DIR / "embeddings_ids.npy"
console = Console()


def _info(msg: str) -> None:
    console.print(f"[cyan]{msg}[/]")


def _ok(msg: str) -> None:
    console.print(f"[green]{msg}[/]")


def _warn(msg: str) -> None:
    console.print(f"[yellow]{msg}[/]")


def _error(msg: str) -> None:
    console.print(f"[bold red]{msg}[/]")


def _resolver_modelo_rag(modelo_solicitado: str) -> str:
    from .busqueda_rag import DEFAULT_MODEL

    modelo = modelo_solicitado.strip()
    if not modelo:
        _warn(
            f"Aviso: --rag-model vacío; se usará el modelo por defecto '{DEFAULT_MODEL}'."
        )
        return DEFAULT_MODEL
    try:
        import ollama

        ollama.show(model=modelo)
        return modelo
    except Exception as e:
        _warn(
            f"Aviso: no se pudo usar el modelo RAG '{modelo}' ({e}). "
            f"Se usará '{DEFAULT_MODEL}'."
        )
        return DEFAULT_MODEL


def _regenerar_indices() -> bool:
    from .indexar_parrafos import index_html

    if not HTML_PATH.exists():
        _error(f"Error: no se encontró el corpus fuente en '{HTML_PATH}'.")
        _warn("No se pueden regenerar los índices sin ese archivo.")
        return False

    try:
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]Regenerando índices...[/]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("index", total=None)
            paragraphs, vocabulary = index_html(HTML_PATH)
            progress.update(task_id, completed=1)
        PARRAFOS_PATH.write_text(
            json.dumps(paragraphs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        INDICE_PATH.write_text(
            json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _ok(
            f"Índices generados: {len(paragraphs)} párrafos, {len(vocabulary)} términos."
        )
        return True
    except Exception as e:
        _error(f"Error al regenerar índices: {e}")
        return False


def _regenerar_embeddings() -> bool:
    """Devuelve True si los embeddings se generaron correctamente, False si ollama no está disponible."""
    from .busqueda_semantica import calcular_embeddings_con_progreso, guardar_embeddings

    try:
        parrafos = json.loads(PARRAFOS_PATH.read_text(encoding="utf-8"))
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]Regenerando embeddings...[/]"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("embeddings", total=len(parrafos))

            def _on_progress(processed: int, total: int) -> None:
                progress.update(task_id, total=total, completed=processed)

            embeddings, ids = calcular_embeddings_con_progreso(parrafos, _on_progress)

        guardar_embeddings(
            str(EMBEDDINGS_PATH), str(EMBEDDINGS_IDS_PATH), embeddings, ids
        )
        _ok(f"Embeddings generados: {len(ids)} chunks.")
        return True
    except Exception as e:
        _warn(f"Aviso: no se pudieron generar los embeddings ({e}).")
        _warn("La búsqueda semántica y el RAG no estarán disponibles.")
        _warn("Asegúrate de que ollama está en ejecución: ollama serve")
        return False


def _run(
    rag_model: str = typer.Option(
        "llama3.2",
        "--rag-model",
        help="Modelo de ollama para el modo RAG.",
    ),
    regenerate: bool = typer.Option(
        False,
        "--regenerate",
        help="Regenera índices y embeddings.",
    ),
    top_k_rag: int = typer.Option(
        5,
        "--top-k-rag",
        help="Número de chunks por búsqueda (clásica y semántica) usados por RAG.",
    ),
    top_k_semantica: int = typer.Option(
        20,
        "--top-k-semantica",
        help="Número de resultados devueltos por la búsqueda semántica.",
    ),
) -> None:
    if top_k_rag < 1:
        typer.secho(
            "Error: --top-k-rag debe ser un entero mayor o igual que 1.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    if top_k_semantica < 1:
        typer.secho(
            "Error: --top-k-semantica debe ser un entero mayor o igual que 1.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    rag_model_resuelto = _resolver_modelo_rag(rag_model)

    tiene_indices = PARRAFOS_PATH.exists() and INDICE_PATH.exists()
    tiene_embeddings = EMBEDDINGS_PATH.exists() and EMBEDDINGS_IDS_PATH.exists()

    if regenerate:
        if not _regenerar_indices():
            raise typer.Exit(code=1)
        embeddings_disponibles = _regenerar_embeddings()
    else:
        if not tiene_indices:
            # Si faltan índices, regeneramos todo para mantener consistencia.
            if not _regenerar_indices():
                raise typer.Exit(code=1)
            embeddings_disponibles = _regenerar_embeddings()
        elif not tiene_embeddings:
            # Si solo faltan embeddings, regeneramos embeddings.
            embeddings_disponibles = _regenerar_embeddings()
        else:
            # Si está todo, no hacemos nada.
            embeddings_disponibles = True

    if not embeddings_disponibles:
        typer.secho(
            "Aviso: embeddings no disponibles. Usa --regenerate para generarlos. "
            "La búsqueda semántica y RAG no estarán disponibles.",
            fg=typer.colors.YELLOW,
        )

    from .buscador_textual import Buscador

    app = Buscador(
        str(INDICE_PATH),
        str(PARRAFOS_PATH),
        str(EMBEDDINGS_PATH) if embeddings_disponibles else None,
        str(EMBEDDINGS_IDS_PATH) if embeddings_disponibles else None,
        rag_model_resuelto,
        top_k_rag,
        top_k_semantica,
    )
    app.run()


def main() -> None:
    typer.run(_run)


if __name__ == "__main__":
    main()
