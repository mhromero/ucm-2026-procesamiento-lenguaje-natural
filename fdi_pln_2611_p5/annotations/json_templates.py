"""Generate blank annotation JSON files from corpus text.

Builds per-annotator JSON templates by sampling sentences, assigning them
across files, and tokenizing text at word or BPE granularity for the NER
labeling workflow.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from pathlib import Path

from fdi_pln_2611_p5.model.lm_causal.bpe_tokenizer import BPETokenizer


def extraer_frases(texto: str) -> list[str]:
    """Split text into sentences on period boundaries.

    Args:
        texto: Raw input text.

    Returns:
        Non-empty sentence strings with surrounding whitespace stripped.
    """
    partes = texto.strip().split(".")
    return [frase.strip() for frase in partes if frase.strip()]


def contar_palabras(frase: str) -> int:
    """Count word tokens in a sentence.

    Args:
        frase: Sentence text.

    Returns:
        Number of Unicode word matches.
    """
    return len(re.findall(r"\w+", frase, flags=re.UNICODE))


def filtrar_frases_por_longitud(
    frases: list[str],
    min_palabras: int,
) -> list[str]:
    """Keep sentences that meet a minimum word count.

    Args:
        frases: Candidate sentences.
        min_palabras: Minimum number of words required.

    Returns:
        Sentences with at least ``min_palabras`` words.
    """
    return [frase for frase in frases if contar_palabras(frase) >= min_palabras]


def tokenizar_palabras(texto: str) -> list[str]:
    """Tokenize text into one unit per word, space, or punctuation mark.

    Args:
        texto: Input text.

    Returns:
        Sequence of word, whitespace, and punctuation tokens.
    """
    return re.findall(r"\s+|\w+|[^\w\s]", texto, flags=re.UNICODE)


def tokenizar_bpe(texto: str, tokenizer: BPETokenizer) -> list[str]:
    """Tokenize text into BPE subword units used by the NER model.

    Args:
        texto: Input text (lowercased before encoding).
        tokenizer: BPE tokenizer shared with the NER pipeline.

    Returns:
        Decoded BPE token strings.
    """
    token_ids = tokenizer.encode(texto.lower())
    return tokenizer.decode_tokens(token_ids)


def generar_asignaciones(
    n_frases: int,
    n_json: int,
    frases_por_json: int,
    anotadores_por_frase: int = 2,
) -> list[list[int]]:
    """Assign sentence indices to annotation JSON files.

    Each sentence must appear in exactly ``anotadores_por_frase`` JSON files.
    When multiple JSON files are used, the last file may receive extra capacity
    to satisfy the duplication requirement.

    Args:
        n_frases: Number of distinct sentences to annotate.
        n_json: Number of annotation JSON files to produce.
        frases_por_json: Target sentence count per JSON file.
        anotadores_por_frase: How many annotators label each sentence.

    Returns:
        One list of sentence indices per JSON file.

    Raises:
        ValueError: If slot counts are inconsistent with the duplication plan.
        RuntimeError: If a valid assignment cannot be completed.
    """
    if n_json == 1:
        if frases_por_json != n_frases:
            raise ValueError(
                f"Con un solo JSON, frases_por_json ({frases_por_json}) "
                f"debe coincidir con n_frases ({n_frases})."
            )
        return [list(range(n_frases))]

    total_slots = n_json * frases_por_json
    required_slots = n_frases * anotadores_por_frase

    capacidades = [frases_por_json] * n_json
    if total_slots < required_slots:
        capacidades[-1] += required_slots - total_slots
    elif total_slots > required_slots:
        raise ValueError(
            "Hay mas huecos que frases duplicadas. Ajusta parametros para evitar relleno artificial."
        )

    apariciones_restantes = {i: anotadores_por_frase for i in range(n_frases)}
    asignaciones = [[] for _ in range(n_json)]

    for idx_json in range(n_json):
        capacidad = capacidades[idx_json]
        for _ in range(capacidad):
            candidatas = [i for i, n in apariciones_restantes.items() if n > 0]
            if not candidatas:
                break
            # Prefer sentences that still need the most annotator copies.
            elegida = max(candidatas, key=lambda i: apariciones_restantes[i])
            asignaciones[idx_json].append(elegida)
            apariciones_restantes[elegida] -= 1

    if any(v != 0 for v in apariciones_restantes.values()):
        raise RuntimeError(
            f"No se pudo completar la asignacion: cada frase debe aparecer en "
            f"{anotadores_por_frase} JSON exactos."
        )

    return asignaciones


def records_from_units(units: list[str]) -> list[dict]:
    """Build blank annotation records from token units.

    Args:
        units: Token strings to annotate.

    Returns:
        Records with ``clave`` set to each unit and empty ``valor`` labels.
    """
    return [{"clave": unit, "valor": ""} for unit in units]


def resolver_n_frases(
    n_frases: int | None,
    n_json: int,
    frases_por_json: int,
    anotadores_por_frase: int = 2,
) -> int:
    """Resolve the number of sentences to sample when not explicitly given.

    Args:
        n_frases: Explicit sentence count, or ``None`` to infer from slots.
        n_json: Number of annotation JSON files.
        frases_por_json: Sentence slots per JSON file.
        anotadores_por_frase: Annotators per sentence.

    Returns:
        Number of distinct sentences to select.

    Raises:
        ValueError: If total slots are not divisible by annotators per sentence.
    """
    if n_frases is not None:
        return n_frases
    if n_json == 1:
        return frases_por_json
    total_slots = n_json * frases_por_json
    if total_slots % anotadores_por_frase != 0:
        raise ValueError(
            f"n_json * frases_por_json ({total_slots}) debe ser divisible por "
            f"anotadores_por_frase ({anotadores_por_frase})."
        )
    return total_slots // anotadores_por_frase


def crear_jsons_anotacion(
    archivo_entrada: Path,
    directorio_salida: Path,
    tokenizar: Callable[[str], list[str]],
    granularidad: str,
    n_frases: int | None = None,
    n_json: int = 14,
    frases_por_json: int = 5,
    anotadores_por_frase: int = 2,
    min_palabras: int = 20,
    seed: int = 46,
    frases_fijas: list[str] | None = None,
) -> dict:
    """Create blank annotation JSON files and assignment metadata.

    Args:
        archivo_entrada: Source corpus text file (ignored when ``frases_fijas``
            is provided).
        directorio_salida: Directory for JSON templates and metadata files.
        tokenizar: Callable that splits joined sentence text into units.
        granularidad: Tokenization mode label stored in metadata (e.g.
            ``"palabra"`` or ``"bpe"``).
        n_frases: Number of sentences to sample; inferred when ``None``.
        n_json: Number of annotation JSON files to write.
        frases_por_json: Target sentences per JSON file.
        anotadores_por_frase: Annotators assigned to each sentence.
        min_palabras: Minimum words per eligible sentence (0 disables filtering).
        seed: Random seed for sentence sampling.
        frases_fijas: Fixed sentence list; bypasses corpus sampling.

    Returns:
        Summary dict with output directory and assignment parameters.

    Raises:
        ValueError: If sampling constraints or fixed sentences are inconsistent.
    """
    if frases_fijas is not None:
        frases_seleccionadas = list(frases_fijas)
        n_frases = len(frases_seleccionadas)
        if n_json == 1 and frases_por_json != n_frases:
            raise ValueError(
                f"frases_fijas tiene {n_frases} frases pero frases_por_json={frases_por_json}"
            )
    else:
        random.seed(seed)
        texto = archivo_entrada.read_text(encoding="utf-8")
        frases = extraer_frases(texto)
        if min_palabras > 0:
            frases = filtrar_frases_por_longitud(frases, min_palabras)

        n_frases = resolver_n_frases(
            n_frases, n_json, frases_por_json, anotadores_por_frase
        )
        if len(frases) < n_frases:
            raise ValueError(
                f"Solo hay {len(frases)} frases elegibles (min_palabras={min_palabras}); "
                f"se necesitan {n_frases}."
            )

        frases_seleccionadas = random.sample(frases, n_frases)
    asignaciones = generar_asignaciones(
        n_frases, n_json, frases_por_json, anotadores_por_frase
    )
    directorio_salida.mkdir(parents=True, exist_ok=True)

    for i, indices in enumerate(asignaciones, start=1):
        frases_json = [frases_seleccionadas[idx] for idx in indices]
        texto_json = " ".join(frases_json)
        units = tokenizar(texto_json)
        payload = records_from_units(units)
        salida = directorio_salida / f"json_{i:02d}.json"
        salida.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    directorio_salida.joinpath("frases_seleccionadas.json").write_text(
        json.dumps(frases_seleccionadas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    directorio_salida.joinpath("asignaciones.json").write_text(
        json.dumps(
            {
                "granularidad": granularidad,
                "min_palabras": min_palabras,
                "frases_por_json": frases_por_json,
                "anotadores_por_frase": anotadores_por_frase,
                "n_json": n_json,
                "seed": seed,
                "frases": frases_seleccionadas,
                "asignaciones": asignaciones,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "directorio": str(directorio_salida),
        "n_frases": n_frases,
        "n_json": n_json,
        "frases_por_json": frases_por_json,
        "anotadores_por_frase": anotadores_por_frase,
        "min_palabras": min_palabras,
    }
