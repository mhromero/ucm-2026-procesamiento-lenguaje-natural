"""Compatibilidad: delega en generar_jsons_palabra."""

from pathlib import Path

from fdi_pln_2611_p5.annotations.templates import (
    crear_jsons_anotacion,
    tokenizar_palabras,
)

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    crear_jsons_anotacion(
        archivo_entrada=base / "data" / "alice_in_wonderland.txt",
        directorio_salida=base / "data" / "alice_jsons",
        tokenizar=tokenizar_palabras,
        granularidad="palabra",
    )
