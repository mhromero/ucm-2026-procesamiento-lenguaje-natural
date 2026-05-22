"""Plantilla única: 6 frases de json_01 (6frases) + 3 más del mismo pool."""

from pathlib import Path

from fdi_pln_2611_p5.annotations.templates import (
    crear_jsons_anotacion,
    seleccionar_frases_6frases_mas_extra,
    tokenizar_palabras,
)

FRASES_POR_JSON = 9
MIN_PALABRAS = 20
SEED = 46
SEED_EXTRA = 47
INDICE_JSON_6FRASES = 0

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    entrada = base / "data" / "alice_in_wonderland.txt"
    frases = seleccionar_frases_6frases_mas_extra(
        entrada,
        min_palabras=MIN_PALABRAS,
        seed=SEED,
        indice_json=INDICE_JSON_6FRASES,
        frases_extra=3,
        seed_extra=SEED_EXTRA,
    )
    info = crear_jsons_anotacion(
        archivo_entrada=entrada,
        directorio_salida=base / "data" / "alice_jsons_1json_9frases",
        tokenizar=tokenizar_palabras,
        granularidad="palabra",
        n_frases=len(frases),
        n_json=1,
        frases_por_json=FRASES_POR_JSON,
        min_palabras=MIN_PALABRAS,
        seed=SEED,
        frases_fijas=frases,
    )
    print(
        f"Generados {info['n_json']} JSON en {info['directorio']} "
        f"(6 frases de json_01 6frases + 3 extra, "
        f"min {info['min_palabras']} palabras)"
    )
