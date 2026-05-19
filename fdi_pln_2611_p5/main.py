from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from fdi_pln_2611_p5.annotations.etiquetados import merge_etiquetados
from fdi_pln_2611_p5.annotations.merge import merge_annotations
from fdi_pln_2611_p5.annotations.report import generate_annotation_report
from fdi_pln_2611_p5.config import DEFAULT_CONFIG_PATH, package_path
from fdi_pln_2611_p5.inference import extract_entities_from_file, generate_text
from fdi_pln_2611_p5.training.causal import train_causal, train_tokenizer
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
    text_file: Annotated[Path, typer.Argument(help="Fichero de texto a etiquetar.")],
):
    """Lista entidades nombradas detectadas en un fichero."""
    entities = extract_entities_from_file(weights, text_file)
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
    ] = package_path("data/alice_jsons_parte1/asignaciones.json"),
    parte2_assignments: Annotated[
        Path, typer.Option("--parte2-assignments")
    ] = package_path("data/alice_jsons_parte2/asignaciones.json"),
    output: Annotated[Path, typer.Option("--output")] = package_path(
        "data/annotations/merged.json"
    ),
):
    """Fusiona data/etiquetados (p1=parte1, p2=parte2; dos JSON por frase)."""
    bundle = merge_etiquetados(
        etiquetados_root=etiquetados_dir,
        parte1_assignments=parte1_assignments,
        parte2_assignments=parte2_assignments,
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
    skip_merge: Annotated[
        bool, typer.Option("--skip-merge", help="Usar merged.json existente.")
    ] = False,
):
    """Genera informe HTML con métricas y gráficos del etiquetado."""
    bundle = None
    if not skip_merge:
        bundle = merge_etiquetados(
            etiquetados_root=etiquetados_dir,
            output_path=merged_json,
        )
    path = generate_annotation_report(
        bundle=bundle,
        etiquetados_root=etiquetados_dir,
        output_html=output_html,
        merged_json=merged_json,
    )
    typer.echo(f"Informe guardado en {path}")


def main():
    app()


if __name__ == "__main__":
    main()
