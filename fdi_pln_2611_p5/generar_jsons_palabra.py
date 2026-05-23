"""Genera plantillas de anotación con una etiqueta por palabra (o espacio/puntuación)."""

from pathlib import Path

from fdi_pln_2611_p5.annotations.templates import (
    crear_jsons_anotacion,
    tokenizar_palabras,
)

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    crear_jsons_anotacion(
        archivo_entrada=base / "data" / "corpus" / "alice_in_wonderland.txt",
        directorio_salida=base / "data" / "alice_jsons_palabra",
        tokenizar=tokenizar_palabras,
        granularidad="palabra",
    )
