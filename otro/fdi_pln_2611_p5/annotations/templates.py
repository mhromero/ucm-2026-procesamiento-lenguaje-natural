from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from pathlib import Path

from fdi_pln_2611_p5.BPETokenizer import BPETokenizer


def extraer_frases(texto: str) -> list[str]:
    partes = texto.strip().split(".")
    return [frase.strip() for frase in partes if frase.strip()]


def contar_palabras(frase: str) -> int:
    return len(re.findall(r"\w+", frase, flags=re.UNICODE))


def filtrar_frases_por_longitud(
    frases: list[str],
    min_palabras: int,
) -> list[str]:
    return [frase for frase in frases if contar_palabras(frase) >= min_palabras]


def tokenizar_palabras(texto: str) -> list[str]:
    """Una unidad por palabra, espacio o signo de puntuación."""
    return re.findall(r"\s+|\w+|[^\w\s]", texto, flags=re.UNICODE)


def tokenizar_bpe(texto: str, tokenizer: BPETokenizer) -> list[str]:
    """Una unidad por subpalabra BPE (misma que usa el modelo NER)."""
    token_ids = tokenizer.encode(texto.lower())
    return tokenizer.decode_tokens(token_ids)


def generar_asignaciones(
    n_frases: int,
    n_json: int,
    frases_por_json: int,
) -> list[list[int]]:
    if n_json == 1:
        if frases_por_json != n_frases:
            raise ValueError(
                f"Con un solo JSON, frases_por_json ({frases_por_json}) "
                f"debe coincidir con n_frases ({n_frases})."
            )
        return [list(range(n_frases))]

    total_slots = n_json * frases_por_json
    required_slots = n_frases * 2

    capacidades = [frases_por_json] * n_json
    if total_slots < required_slots:
        capacidades[-1] += required_slots - total_slots
    elif total_slots > required_slots:
        raise ValueError(
            "Hay mas huecos que frases duplicadas. Ajusta parametros para evitar relleno artificial."
        )

    apariciones_restantes = {i: 2 for i in range(n_frases)}
    asignaciones = [[] for _ in range(n_json)]

    for idx_json in range(n_json):
        capacidad = capacidades[idx_json]
        for _ in range(capacidad):
            candidatas = [i for i, n in apariciones_restantes.items() if n > 0]
            if not candidatas:
                break
            elegida = max(candidatas, key=lambda i: apariciones_restantes[i])
            asignaciones[idx_json].append(elegida)
            apariciones_restantes[elegida] -= 1

    if any(v != 0 for v in apariciones_restantes.values()):
        raise RuntimeError(
            "No se pudo completar la asignacion de frases en 2 JSONs exactos."
        )

    return asignaciones


def records_from_units(units: list[str]) -> list[dict]:
    return [{"clave": unit, "valor": ""} for unit in units]


def seleccionar_frases_6frases_mas_extra(
    archivo_entrada: Path,
    *,
    min_palabras: int = 20,
    seed: int = 46,
    n_json: int = 13,
    frases_por_json: int = 6,
    indice_json: int = 0,
    frases_extra: int = 3,
    seed_extra: int | None = None,
) -> list[str]:
    """6 frases de un JSON del esquema 6frases + N adicionales del mismo pool."""
    random.seed(seed)
    texto = archivo_entrada.read_text(encoding="utf-8")
    frases = extraer_frases(texto)
    if min_palabras > 0:
        frases = filtrar_frases_por_longitud(frases, min_palabras)

    n_frases_pool = resolver_n_frases(None, n_json, frases_por_json)
    if len(frases) < n_frases_pool + frases_extra:
        raise ValueError(
            f"Solo hay {len(frases)} frases elegibles; se necesitan al menos "
            f"{n_frases_pool + frases_extra}."
        )

    pool = random.sample(frases, n_frases_pool)
    asignaciones = generar_asignaciones(n_frases_pool, n_json, frases_por_json)
    if not (0 <= indice_json < len(asignaciones)):
        raise ValueError(f"indice_json debe estar entre 0 y {len(asignaciones) - 1}")

    base = [pool[i] for i in asignaciones[indice_json]]
    restantes = [f for f in pool if f not in base]
    if len(restantes) < frases_extra:
        raise ValueError(
            f"Solo quedan {len(restantes)} frases en el pool tras las {len(base)} base; "
            f"se piden {frases_extra} extra."
        )

    random.seed(seed_extra if seed_extra is not None else seed + 1)
    extra = random.sample(restantes, frases_extra)
    return base + extra


def resolver_n_frases(n_frases: int | None, n_json: int, frases_por_json: int) -> int:
    if n_frases is not None:
        return n_frases
    if n_json == 1:
        return frases_por_json
    if (n_json * frases_por_json) % 2 != 0:
        raise ValueError(
            "n_json * frases_por_json debe ser par para que cada frase aparezca en 2 JSON."
        )
    return (n_json * frases_por_json) // 2


def crear_jsons_anotacion(
    archivo_entrada: Path,
    directorio_salida: Path,
    tokenizar: Callable[[str], list[str]],
    granularidad: str,
    n_frases: int | None = 50,
    n_json: int = 25,
    frases_por_json: int = 4,
    min_palabras: int = 0,
    seed: int = 44,
    frases_fijas: list[str] | None = None,
) -> dict:
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

        n_frases = resolver_n_frases(n_frases, n_json, frases_por_json)
        if len(frases) < n_frases:
            raise ValueError(
                f"Solo hay {len(frases)} frases elegibles (min_palabras={min_palabras}); "
                f"se necesitan {n_frases}."
            )

        frases_seleccionadas = random.sample(frases, n_frases)
    asignaciones = generar_asignaciones(n_frases, n_json, frases_por_json)
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
        "min_palabras": min_palabras,
    }
