"""CLI entry point for Practice 5: causal LM and NER on Alice in Wonderland."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from fdi_pln_2611_p5.annotations.merge_annotators import merge_annotations
from fdi_pln_2611_p5.config import package_path
from fdi_pln_2611_p5.inference import (
    extract_entities_from_file,
    extract_entities_from_text,
    generate_text,
)
from fdi_pln_2611_p5.training.causal import train_causal, train_tokenizer
from fdi_pln_2611_p5.training.experiment_exploration import run_experiment_exploration
from fdi_pln_2611_p5.training.ner_train import train_ner

app = typer.Typer(
    help="Practice 5: causal language model and NER on Alice in Wonderland."
)


@app.command("train-tokenizer")
def cmd_train_tokenizer(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to the JSON configuration file."),
    ] = None,
):
    """Train the BPE tokenizer on the full corpus and write it to disk."""
    train_tokenizer(config_path=config)


@app.command("train-causal")
def cmd_train_causal(
    weights: Annotated[
        Path, typer.Option("--weights", help="Output path for causal model weights (.pth).")
    ],
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
):
    """Train (or retrain) the causal language model."""
    train_causal(weights, config_path=config, grid_search=grid_search)


@app.command("train-ner")
def cmd_train_ner(
    weights: Annotated[
        Path, typer.Option("--weights", help="Output path for NER model weights (.pth).")
    ],
    causal_weights: Annotated[
        Path,
        typer.Option("--causal-weights", help="Path to pretrained causal backbone weights."),
    ],
    annotations: Annotated[
        Path,
        typer.Option("--annotations", help="Merged annotation JSON for NER fine-tuning."),
    ] = package_path("data/annotations/merged.json"),
    config: Annotated[
        Optional[Path], typer.Option("--config", help="Path to the JSON configuration file.")
    ] = None,
):
    """Fine-tune the NER head on top of the causal backbone."""
    train_ner(weights, causal_weights, annotations, config_path=config)


@app.command("inference-generate")
def cmd_generate(
    weights: Annotated[
        Path, typer.Option("--weights", help="Path to causal model weights (.pth).")
    ],
    prompt: Annotated[
        str, typer.Option("--prompt", "-p", help="Initial prompt text.")
    ] = "Alice",
    max_new_tokens: Annotated[
        int, typer.Option("--max-new-tokens", help="Maximum tokens to generate.")
    ] = 100,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature.")
    ] = 1.0,
):
    """Generate text continuation from a prompt."""
    text = generate_text(weights, prompt, max_new_tokens, temperature)
    typer.echo(text)


@app.command("inference-ner")
def cmd_ner(
    weights: Annotated[
        Path, typer.Option("--weights", help="Path to NER model weights (.pth).")
    ],
    text_file: Annotated[
        Optional[Path], typer.Argument(help="Text file to run NER on.")
    ] = None,
    text: Annotated[
        Optional[str],
        typer.Option("--text", "-t", help="Raw text to run NER on (alternative to a file)."),
    ] = None,
):
    """Print named entities detected in a file or inline text."""
    if text_file is not None and text is not None:
        typer.echo("Error: use --text OR a file path, not both.", err=True)
        raise typer.Exit(code=1)
    if text_file is None and text is None:
        typer.echo("Error: provide a file path or use --text.", err=True)
        raise typer.Exit(code=1)

    entities = (
        extract_entities_from_file(weights, text_file)
        if text_file is not None
        else extract_entities_from_text(weights, text)
    )
    if not entities:
        typer.echo("No entities found.")
        raise typer.Exit(code=0)
    for entity in entities:
        typer.echo(f"{entity['type']}\t{entity['text']}")


@app.command("prepare-annotations")
def cmd_prepare_annotations(
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Directory for generated JSON templates.")
    ] = package_path("data/alice_jsons"),
    n_json: Annotated[
        int,
        typer.Option("--n-json", help="Number of json_XX.json files (one per annotator)."),
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
