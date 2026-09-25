# Practice 3 — PLNCG26

Standalone command-line utility for encoding, decoding, and detecting PLNCG26 files.

## Dependency

- Python 3.12+
- [`typer`](https://typer.tiangolo.com/)

## Usage

The script is `fdi-pln-2611-p3.py`:

```bash
uv run fdi-pln-2611-p3.py decode <FILE>
uv run fdi-pln-2611-p3.py encode <FILE>
uv run fdi-pln-2611-p3.py detect <FILE>
```

The binary files in this folder (`largo.bin`, `noticia.bin`, and `principal.bin`) are inputs to be decrypted by the script. `prueba_utf8.txt` is a bundled text fixture.
