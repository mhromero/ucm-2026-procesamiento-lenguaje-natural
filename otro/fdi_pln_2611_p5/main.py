from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from fdi_pln_2611_p5.annotations.etiquetados import merge_etiquetados
from fdi_pln_2611_p5.annotations.merge import merge_annotations
from fdi_pln_2611_p5.annotations.report import generate_annotation_report
from fdi_pln_2611_p5.config import package_path
from fdi_pln_2611_p5.inference import (
    extract_entities_from_file,
    extract_entities_from_text,
    generate_text,
)
from fdi_pln_2611_p5.training.causal import train_causal, train_tokenizer
from fdi_pln_2611_p5.training.experiment_exploration import run_experiment_exploration
from fdi_pln_2611_p5.training.ner_train import train_ner

app = typer.Typer(help="Práctica 5: LM causal + NER sobre Alice in Wonderland.")


@app.command("train-tokenizer")
def cmd_train_tokenizer(
    config: Annotated[
        Optional[Path], typer.Option("--config", help="Fichero de configuración JSON.")
    ] = None,
):
    """Entrena el tokenizador BPE sobre el corpus completo y lo guarda en disco."""
    train_tokenizer(config_path=config)


@app.command("train-causal")
def cmd_train_causal(
    weights: Annotated[Path, typer.Option("--weights", help="Ruta del .pth causal.")],
    config: Annotated[
        Optional[Path], typer.Option("--config", help="Fichero de configuración JSON.")
    ] = None,
    grid_search: Annotated[
        bool, typer.Option("--grid-search", help="Explorar 9 combinaciones lr×batch.")
    ] = False,
):
    """Entrena (o reentrena) el modelo de lenguaje causal."""
    train_causal(weights, config_path=config, grid_search=grid_search)


@app.command("train-ner")
def cmd_train_ner(
    weights: Annotated[Path, typer.Option("--weights", help="Ruta del .pth NER.")],
    causal_weights: Annotated[
        Path, typer.Option("--causal-weights", help="Pesos del backbone causal.")
    ],
    annotations: Annotated[
        Path,
        typer.Option(
            "--annotations",
            help="JSON fusionado de anotaciones.",
        ),
    ] = package_path("data/annotations/merged.json"),
    config: Annotated[Optional[Path], typer.Option("--config")] = None,
):
    """Entrena el cabezal NER sobre el backbone causal."""
    train_ner(weights, causal_weights, annotations, config_path=config)


@app.command("generate")
def cmd_generate(
    weights: Annotated[Path, typer.Option("--weights", help="Pesos causales (.pth).")],
    prompt: Annotated[
        str, typer.Option("--prompt", "-p", help="Texto inicial.")
    ] = "Alice",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 100,
    temperature: Annotated[float, typer.Option("--temperature")] = 1.0,
):
    """Genera texto a partir de un prompt."""
    text = generate_text(weights, prompt, max_new_tokens, temperature)
    typer.echo(text)


@app.command("ner")
def cmd_ner(
    weights: Annotated[Path, typer.Option("--weights", help="Pesos NER (.pth).")],
    text_file: Annotated[
        Optional[Path], typer.Argument(help="Fichero de texto a etiquetar.")
    ] = None,
    text: Annotated[
        Optional[str], typer.Option("--text", "-t", help="Texto directo a etiquetar.")
    ] = None,
):
    """Lista entidades nombradas detectadas en un fichero o en texto directo."""
    if text_file is not None and text is not None:
        typer.echo("Error: usa --text O un fichero, no los dos a la vez.", err=True)
        raise typer.Exit(code=1)
    if text_file is None and text is None:
        typer.echo("Error: proporciona un fichero o usa --text.", err=True)
        raise typer.Exit(code=1)

    entities = (
        extract_entities_from_file(weights, text_file)
        if text_file is not None
        else extract_entities_from_text(weights, text)
    )
    if not entities:
        typer.echo("No se encontraron entidades.")
        raise typer.Exit(code=0)
    for entity in entities:
        typer.echo(f"{entity['type']}\t{entity['text']}")


@app.command("generate-templates-word")
def cmd_generate_templates_word(
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Directorio de salida.")
    ] = package_path("data/alice_jsons_palabra"),
    n_frases: Annotated[
        Optional[int],
        typer.Option(
            "--n-frases", help="Si se omite, se deduce de n-json y frases/json."
        ),
    ] = 50,
    n_json: Annotated[int, typer.Option("--n-json")] = 25,
    frases_por_json: Annotated[int, typer.Option("--frases-por-json")] = 4,
    min_palabras: Annotated[int, typer.Option("--min-palabras")] = 0,
    seed: Annotated[int, typer.Option("--seed", help="Semilla aleatoria.")] = 44,
):
    """Genera plantillas JSON con una etiqueta por palabra."""
    from fdi_pln_2611_p5.annotations.templates import (
        crear_jsons_anotacion,
        tokenizar_palabras,
    )

    info = crear_jsons_anotacion(
        archivo_entrada=package_path("data/alice_in_wonderland.txt"),
        directorio_salida=output_dir,
        tokenizar=tokenizar_palabras,
        granularidad="palabra",
        n_frases=n_frases,
        n_json=n_json,
        frases_por_json=frases_por_json,
        min_palabras=min_palabras,
        seed=seed,
    )
    typer.echo(
        f"Plantillas (palabra) en {info['directorio']} ({info['n_frases']} frases, seed={seed})"
    )


@app.command("generate-templates-6frases")
def cmd_generate_templates_6frases(
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Directorio de salida.")
    ] = package_path("data/alice_jsons_6frases"),
    n_json: Annotated[int, typer.Option("--n-json")] = 13,
    min_palabras: Annotated[int, typer.Option("--min-palabras")] = 20,
    seed: Annotated[int, typer.Option("--seed", help="Semilla aleatoria.")] = 46,
):
    """6 frases por JSON, solo frases largas; no modifica data/alice_jsons/."""
    from fdi_pln_2611_p5.annotations.templates import (
        crear_jsons_anotacion,
        tokenizar_palabras,
    )

    info = crear_jsons_anotacion(
        archivo_entrada=package_path("data/alice_in_wonderland.txt"),
        directorio_salida=output_dir,
        tokenizar=tokenizar_palabras,
        granularidad="palabra",
        n_frases=None,
        n_json=n_json,
        frases_por_json=6,
        min_palabras=min_palabras,
        seed=seed,
    )
    typer.echo(
        f"Plantillas en {info['directorio']}: {info['n_json']} JSON × "
        f"{info['frases_por_json']} frases ({info['n_frases']} frases únicas, "
        f"≥{info['min_palabras']} palabras, seed={seed})"
    )


@app.command("generate-templates-1json-9frases")
def cmd_generate_templates_1json_9frases(
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Directorio de salida.")
    ] = package_path("data/alice_jsons_1json_9frases"),
    min_palabras: Annotated[int, typer.Option("--min-palabras")] = 20,
    seed: Annotated[int, typer.Option("--seed", help="Semilla aleatoria.")] = 46,
):
    """Un JSON: 6 frases de json_01 (6frases) + 3 extra del mismo pool."""
    from fdi_pln_2611_p5.annotations.templates import (
        crear_jsons_anotacion,
        seleccionar_frases_6frases_mas_extra,
        tokenizar_palabras,
    )

    entrada = package_path("data/alice_in_wonderland.txt")
    frases = seleccionar_frases_6frases_mas_extra(
        entrada,
        min_palabras=min_palabras,
        seed=seed,
        frases_extra=3,
        seed_extra=seed + 1,
    )
    info = crear_jsons_anotacion(
        archivo_entrada=entrada,
        directorio_salida=output_dir,
        tokenizar=tokenizar_palabras,
        granularidad="palabra",
        n_frases=len(frases),
        n_json=1,
        frases_por_json=9,
        min_palabras=min_palabras,
        seed=seed,
        frases_fijas=frases,
    )
    typer.echo(
        f"Plantilla en {info['directorio']}: {info['n_json']} JSON "
        f"(6 de json_01 6frases + 3 extra, ≥{info['min_palabras']} palabras, seed={seed})"
    )


@app.command("generate-templates-token")
def cmd_generate_templates_token(
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Directorio de salida.")
    ] = package_path("data/alice_jsons_token"),
    n_frases: Annotated[int, typer.Option("--n-frases")] = 50,
    n_json: Annotated[int, typer.Option("--n-json")] = 25,
    frases_por_json: Annotated[int, typer.Option("--frases-por-json")] = 4,
    seed: Annotated[int, typer.Option("--seed", help="Semilla aleatoria.")] = 44,
):
    """Genera plantillas JSON con una etiqueta por subpalabra BPE."""
    from fdi_pln_2611_p5.annotations.templates import crear_jsons_anotacion
    from fdi_pln_2611_p5.BPETokenizer import BPETokenizer

    tokenizer_path = package_path("data/bpe_tokenizer.json")
    if not tokenizer_path.exists():
        raise typer.BadParameter(
            f"No existe {tokenizer_path}. Ejecuta antes train-causal para crear el BPE."
        )
    tokenizer = BPETokenizer.load(str(tokenizer_path))
    crear_jsons_anotacion(
        archivo_entrada=package_path("data/alice_in_wonderland.txt"),
        directorio_salida=output_dir,
        tokenizar=lambda text: tokenizer.decode_tokens(tokenizer.encode(text)),
        granularidad="token",
        n_frases=n_frases,
        n_json=n_json,
        frases_por_json=frases_por_json,
        seed=seed,
    )
    typer.echo(f"Plantillas (token BPE) en {output_dir} (seed={seed})")


@app.command("merge-annotations")
def cmd_merge_annotations(
    json_dir: Annotated[
        Path, typer.Option("--json-dir", help="Directorio con json_XX.json.")
    ] = package_path("data/alice_jsons"),
    assignments: Annotated[
        Path, typer.Option("--assignments", help="Fichero de asignaciones.")
    ] = package_path("data/alice_jsons/asignaciones.json"),
    output: Annotated[
        Path, typer.Option("--output", help="Salida fusionada.")
    ] = package_path("data/annotations/merged.json"),
):
    """Fusiona anotaciones de varios anotadores y calcula acuerdo."""
    merge_annotations(json_dir, assignments, output)


@app.command("merge-etiquetados")
def cmd_merge_etiquetados(
    etiquetados_dir: Annotated[Path, typer.Option("--etiquetados-dir")] = package_path(
        "data/etiquetados"
    ),
    parte1_assignments: Annotated[
        Path, typer.Option("--parte1-assignments")
    ] = package_path("data/asignaciones/alice_jsons_parte1/asignaciones.json"),
    parte2_assignments: Annotated[
        Path, typer.Option("--parte2-assignments")
    ] = package_path("data/asignaciones/alice_jsons_parte2/asignaciones.json"),
    lote_9frases_assignments: Annotated[
        Path | None,
        typer.Option(
            "--lote-9frases-assignments",
            help="Asignaciones json_14/json_15 (9 frases compartidas).",
        ),
    ] = package_path("data/asignaciones/alice_jsons_1json_9frases/asignaciones.json"),
    sin_lote_9frases: Annotated[
        bool,
        typer.Option(
            "--sin-lote-9frases",
            help="No fusionar json_14/json_15 aunque existan asignaciones.",
        ),
    ] = False,
    output: Annotated[Path, typer.Option("--output")] = package_path(
        "data/annotations/merged.json"
    ),
):
    """Fusiona data/etiquetados (p1=parte1, p2=parte2; dos JSON por frase)."""
    lote_9 = None if sin_lote_9frases else lote_9frases_assignments
    bundle = merge_etiquetados(
        etiquetados_root=etiquetados_dir,
        parte1_assignments=parte1_assignments,
        parte2_assignments=parte2_assignments,
        lote_9frases_assignments=lote_9,
        output_path=output,
    )
    typer.echo(
        f"Fusionadas {bundle.report['n_frases']} frases → {output} "
        f"(κ={bundle.report['mean_cohen_kappa']:.3f})"
    )


@app.command("annotation-report")
def cmd_annotation_report(
    etiquetados_dir: Annotated[Path, typer.Option("--etiquetados-dir")] = package_path(
        "data/etiquetados"
    ),
    output_html: Annotated[Path, typer.Option("--output-html")] = package_path(
        "data/annotations/informe_etiquetado.html"
    ),
    merged_json: Annotated[Path, typer.Option("--merged-json")] = package_path(
        "data/annotations/merged.json"
    ),
    lote_9frases_assignments: Annotated[
        Path | None,
        typer.Option("--lote-9frases-assignments"),
    ] = package_path("data/asignaciones/alice_jsons_1json_9frases/asignaciones.json"),
    sin_lote_9frases: Annotated[
        bool, typer.Option("--sin-lote-9frases")
    ] = False,
    skip_merge: Annotated[
        bool, typer.Option("--skip-merge", help="Usar merged.json existente.")
    ] = False,
):
    """Genera informe HTML con métricas y gráficos del etiquetado."""
    lote_9 = None if sin_lote_9frases else lote_9frases_assignments
    bundle = None
    if not skip_merge:
        bundle = merge_etiquetados(
            etiquetados_root=etiquetados_dir,
            output_path=merged_json,
            lote_9frases_assignments=lote_9,
        )
    path = generate_annotation_report(
        bundle=bundle,
        etiquetados_root=etiquetados_dir,
        output_html=output_html,
        merged_json=merged_json,
        lote_9frases_assignments=lote_9,
    )
    typer.echo(f"Informe guardado en {path}")


@app.command("run-experiments")
def cmd_run_experiments(
    config: Annotated[
        Optional[Path], typer.Option("--config", help="Fichero de configuración JSON.")
    ] = None,
):
    """Exploración: 8 experimentos (corpus, ventana, vocab, profundidad) + informe HTML."""
    run_experiment_exploration(config_path=config)


@app.command("experiment-report")
def cmd_experiment_report(
    results_json: Annotated[
        Path,
        typer.Option("--results-json", help="JSON de resultados de experimentos."),
    ] = package_path("data/experiment_results.json"),
    output_html: Annotated[
        Path, typer.Option("--output-html", help="Informe HTML de salida.")
    ] = package_path("data/informe_experimentos.html"),
):
    """Regenera el informe HTML a partir de experiment_results.json."""
    from fdi_pln_2611_p5.training.experiment_report import (
        generate_experiment_html_from_json,
    )

    if not results_json.exists():
        typer.echo(
            f"No existe {results_json}. Ejecuta primero run-experiments.",
            err=True,
        )
        raise typer.Exit(code=1)
    path = generate_experiment_html_from_json(results_json, output_html)
    typer.echo(f"Informe guardado en {path}")


@app.command("grid-search-report")
def cmd_grid_search_report(
    results_json: Annotated[
        Path, typer.Option("--results-json", help="JSON de resultados del grid search.")
    ] = package_path("data/grid_search_results.json"),
    output_html: Annotated[
        Path, typer.Option("--output-html", help="Ruta del informe HTML de salida.")
    ] = package_path("data/informe_grid_search.html"),
):
    """Genera (o regenera) el informe HTML de exploración de hiperparámetros."""
    from fdi_pln_2611_p5.training.grid_search_report import generate_grid_search_html

    if not results_json.exists():
        typer.echo(
            f"No existe {results_json}. Ejecuta primero train-causal --grid-search.",
            err=True,
        )
        raise typer.Exit(code=1)
    path = generate_grid_search_html(results_json, output_html)
    typer.echo(f"Informe guardado en {path}")


def main():
    app()


if __name__ == "__main__":
    main()
