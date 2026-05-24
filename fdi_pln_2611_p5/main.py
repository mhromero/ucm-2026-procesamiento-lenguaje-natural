"""CLI entry point for Practice 5: causal LM and NER on Alice in Wonderland."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fdi_pln_2611_p5.annotations.merge_annotators import merge_annotations
from fdi_pln_2611_p5.config import load_config, package_path
from fdi_pln_2611_p5.inference import extract_entities_from_file, generate_text
from fdi_pln_2611_p5.paths import PathNotFoundError, require_file
from fdi_pln_2611_p5.training.causal import train_causal, train_tokenizer
from fdi_pln_2611_p5.training.experiment_exploration import run_experiment_exploration
from fdi_pln_2611_p5.training.ner_train import train_ner

app = typer.Typer(
    help="Practice 5: causal language model and NER on Alice in Wonderland."
)

_EXAMPLE_NER_PATH = package_path("data/sample_ner_test.txt")
_STDERR_CONSOLE = Console(stderr=True)
_STDOUT_CONSOLE = Console()


def _echo_path_not_found(exc: FileNotFoundError) -> None:
    """Print a styled path-not-found message on stderr."""
    if isinstance(exc, PathNotFoundError):
        title = "[bold red]File not found[/]"
        if exc.kind == "directory":
            title = "[bold red]Directory not found[/]"
        body = f"[bold]{exc.label}[/]\n[dim]→[/] [cyan]{exc.path}[/]"
    else:
        title = "[bold red]File not found[/]"
        lines = str(exc).splitlines()
        if len(lines) == 1:
            body = lines[0]
        else:
            body = lines[0] + "\n" + "\n".join(
                f"[dim]→[/] {line.strip()}" for line in lines[1:]
            )
    _STDERR_CONSOLE.print(
        Panel(body, title=title, border_style="red", padding=(0, 1))
    )


def _cli_file_not_found(fn):
    """Convert ``FileNotFoundError`` into a clean CLI exit (code 1)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as exc:
            _echo_path_not_found(exc)
            raise typer.Exit(code=1) from exc

    return wrapper


def _echo_example_notice(
    reason: str, value: str, *, max_display: int | None = 80
) -> None:
    """Print a styled stderr notice when default example input is used."""
    if max_display is None or len(value) <= max_display:
        display = value
    else:
        display = f"{value[: max_display - 1]}…"
    _STDERR_CONSOLE.print(
        Panel(
            f"[dim]{reason}[/]\n[bold italic #fbbf24]{display}[/]",
            title="[bold #818cf8]ℹ  Usando ejemplo[/]",
            border_style="#6366f1",
            padding=(0, 1),
        )
    )


def _echo_generation_output(text: str) -> None:
    """Print generated text in a styled panel."""
    _STDOUT_CONSOLE.print(
        Panel(
            text,
            title="[bold #818cf8]Generación[/]",
            border_style="#6366f1",
            padding=(1, 2),
        )
    )


def _echo_ner_entities(entities: list[dict]) -> None:
    """Print NER predictions as a styled table."""
    table = Table(
        title="Entidades detectadas",
        title_style="bold #818cf8",
        border_style="#6366f1",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Tipo", style="bold #a5b4fc", min_width=6)
    table.add_column("Texto", style="#fef3c7")
    for entity in entities:
        table.add_row(entity["type"], entity["text"])
    _STDOUT_CONSOLE.print(table)


def _echo_no_entities() -> None:
    """Print a styled message when NER finds no entities."""
    _STDOUT_CONSOLE.print(
        Panel(
            "[dim]No se encontraron entidades con el umbral actual.[/]",
            title="[bold #818cf8]NER[/]",
            border_style="#6366f1",
            padding=(0, 1),
        )
    )


@app.command("train-tokenizer")
@_cli_file_not_found
def cmd_train_tokenizer(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to the JSON configuration file."),
    ] = None,
):
    """Train the BPE tokenizer on the full corpus and write it to disk."""
    train_tokenizer(config_path=config)


@app.command("train-causal")
@_cli_file_not_found
def cmd_train_causal(
    weights: Annotated[
        Path,
        typer.Option("--weights", help="Output path for causal model weights (.pth)."),
    ] = package_path("p5_causal_2611.pth"),
    config: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to the JSON configuration file."),
    ] = None,
    grid_search: Annotated[
        bool,
        typer.Option(
            "--grid-search",
            help="Run a 3×3 learning-rate × batch-size grid search before final training.",
        ),
    ] = False,
    tokenizer: Annotated[
        Optional[Path],
        typer.Option(
            "--tokenizer",
            help="Path to a pre-trained BPE tokenizer (bpe_tokenizer.json).",
        ),
    ] = None,
):
    """Train (or retrain) the causal language model."""
    train_causal(
        weights,
        config_path=config,
        grid_search=grid_search,
        tokenizer_path=tokenizer,
    )


@app.command("train-ner")
@_cli_file_not_found
def cmd_train_ner(
    weights: Annotated[
        Path,
        typer.Option("--weights", help="Output path for NER model weights (.pth)."),
    ] = package_path("p5_ner_2611.pth"),
    causal_weights: Annotated[
        Path,
        typer.Option(
            "--causal-weights", help="Path to pretrained causal backbone weights."
        ),
    ] = package_path("p5_causal_2611.pth"),
    annotations: Annotated[
        Path,
        typer.Option(
            "--annotations", help="Merged annotation JSON for NER fine-tuning."
        ),
    ] = package_path("data/annotations/merged.json"),
    config: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to the JSON configuration file."),
    ] = None,
    tokenizer: Annotated[
        Optional[Path],
        typer.Option(
            "--tokenizer",
            help="Path to BPE tokenizer JSON (default: path stored in causal checkpoint).",
        ),
    ] = None,
):
    """Fine-tune the NER head on top of the causal backbone."""
    train_ner(
        weights,
        causal_weights,
        annotations,
        config_path=config,
        tokenizer_path=tokenizer,
    )


@app.command("inference-generate")
@_cli_file_not_found
def cmd_generate(
    weights: Annotated[
        Path,
        typer.Option(
            "--weights", help="Path to causal model weights (.pth)."
        ),
    ] = package_path("p5_causal_2611.pth"),
    prompt: Annotated[
        Optional[str],
        typer.Option(
            "--prompt",
            "-p",
            help="Initial prompt text (if omitted, uses the example from config.json).",
        ),
    ] = None,
    max_new_tokens: Annotated[
        int, typer.Option("--max-new-tokens", help="Maximum tokens to generate.")
    ] = 100,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature.")
    ] = 1.0,
    tokenizer: Annotated[
        Optional[Path],
        typer.Option(
            "--tokenizer",
            help="Path to BPE tokenizer JSON (default: from checkpoint / config).",
        ),
    ] = None,
):
    """Generate text continuation from a prompt."""
    if prompt is None:
        prompt = load_config()["generation"]["prompt"]
        _echo_example_notice("Sin --prompt · valor de config.json (generation.prompt)", prompt)
    text = generate_text(
        weights, prompt, max_new_tokens, temperature, tokenizer_path=tokenizer
    )
    _echo_generation_output(text)


@app.command("inference-ner")
@_cli_file_not_found
def cmd_ner(
    text_file: Annotated[
        Optional[Path],
        typer.Argument(help="UTF-8 text file to run NER on."),
    ] = None,
    weights: Annotated[
        Path,
        typer.Option("--weights", help="Path to NER model weights (.pth)."),
    ] = package_path("p5_ner_2611.pth"),
    tokenizer: Annotated[
        Optional[Path],
        typer.Option(
            "--tokenizer",
            help="Path to BPE tokenizer JSON (default: from checkpoint / config).",
        ),
    ] = None,
):
    """Print named entities detected in a text file."""
    using_example = text_file is None
    if using_example:
        text_file = _EXAMPLE_NER_PATH
    text_file = require_file(text_file, label="text file")
    if using_example:
        _echo_example_notice(
            f"Sin fichero · usando ejemplo {_EXAMPLE_NER_PATH.name}",
            text_file.read_text(encoding="utf-8").strip(),
            max_display=None,
        )

    entities = extract_entities_from_file(
        weights, text_file, tokenizer_path=tokenizer
    )
    if not entities:
        _echo_no_entities()
        raise typer.Exit(code=0)
    _echo_ner_entities(entities)


@app.command("prepare-annotations")
@_cli_file_not_found
def cmd_prepare_annotations(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for generated JSON templates."),
    ] = package_path("data/alice_jsons"),
    n_json: Annotated[
        int,
        typer.Option(
            "--n-json", help="Number of json_XX.json files (one per annotator)."
        ),
    ] = 14,
    frases_por_json: Annotated[
        int, typer.Option("--frases-por-json", help="Sentences per JSON file.")
    ] = 5,
    anotadores_por_frase: Annotated[
        int,
        typer.Option(
            "--anotadores-por-frase",
            help="How many JSON files share each sentence (for Cohen's kappa).",
        ),
    ] = 2,
    n_frases: Annotated[
        Optional[int],
        typer.Option(
            "--n-frases",
            help="Unique sentences in the pool; default (n_json × frases_por_json) / annotators.",
        ),
    ] = None,
    min_palabras: Annotated[
        int,
        typer.Option("--min-palabras", help="Minimum word count per sentence."),
    ] = 20,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 46,
):
    """Create empty word-level JSON annotation templates from long Alice sentences."""
    from fdi_pln_2611_p5.annotations.json_templates import (
        crear_jsons_anotacion,
        tokenizar_palabras,
    )

    info = crear_jsons_anotacion(
        archivo_entrada=package_path("data/corpus/alice_in_wonderland.txt"),
        directorio_salida=output_dir,
        tokenizar=tokenizar_palabras,
        granularidad="palabra",
        n_frases=n_frases,
        n_json=n_json,
        frases_por_json=frases_por_json,
        anotadores_por_frase=anotadores_por_frase,
        min_palabras=min_palabras,
        seed=seed,
    )
    typer.echo(
        f"Templates written to {info['directorio']}: {info['n_json']} JSON × "
        f"{info['frases_por_json']} sentences ({info['n_frases']} unique, "
        f"{info['anotadores_por_frase']} annotators/sentence, "
        f"≥{info['min_palabras']} words, seed={seed})"
    )


@app.command("merge-annotations")
@_cli_file_not_found
def cmd_merge_annotations(
    json_dir: Annotated[
        Path,
        typer.Option(
            "--json-dir",
            help="Directory with json_XX.json files (and frases_seleccionadas.json).",
        ),
    ] = package_path("data/alice_jsons"),
    output: Annotated[
        Path, typer.Option("--output", help="Merged dataset output path.")
    ] = package_path("data/annotations/merged.json"),
):
    """Merge annotator JSON files from one directory and write merged.json."""
    bundle = merge_annotations(json_dir, output)
    typer.echo(
        f"Merged {bundle.report['n_frases']} sentences → {output} "
        f"(κ={bundle.report['mean_cohen_kappa']:.3f})"
    )


@app.command("run-experiments")
@_cli_file_not_found
def cmd_run_experiments(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to the JSON configuration file."),
    ] = None,
):
    """Run eight causal-LM ablations (corpus, window, vocab, depth) and write an HTML report."""
    run_experiment_exploration(config_path=config)


def main() -> None:
    """Invoke the Typer CLI application."""
    app()


if __name__ == "__main__":
    main()
