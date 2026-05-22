"""Genera plantillas de anotación con una etiqueta por subpalabra BPE."""

from pathlib import Path

from fdi_pln_2611_p5.annotations.templates import crear_jsons_anotacion
from fdi_pln_2611_p5.BPETokenizer import BPETokenizer
from fdi_pln_2611_p5.config import package_path


def main():
    base = Path(__file__).resolve().parent
    tokenizer_path = package_path("data/bpe_tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"No existe {tokenizer_path}. Entrena antes el tokenizador "
            "(p. ej. con train-causal) o copia bpe_tokenizer.json."
        )
    tokenizer = BPETokenizer.load(str(tokenizer_path))

    crear_jsons_anotacion(
        archivo_entrada=base / "data" / "alice_in_wonderland.txt",
        directorio_salida=base / "data" / "alice_jsons_token",
        tokenizar=lambda text: tokenizer.decode_tokens(tokenizer.encode(text)),
        granularidad="token",
    )


if __name__ == "__main__":
    main()
