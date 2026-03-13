#!/usr/bin/env python3
# /// script
# dependencies = [
#   "typer",
# ]
# ///
"""
Codificación/decodificación PLNCG26 <-> UTF-8.

Uso: uv run fdi-pln-2611-p3.py <decode|encode|detect> <FICHERO>
Salida: stdout (texto UTF-8 en decode, bytes PLNCG26 en encode, probabilidad en detect).

- decode: PLNCG26 (bytes) -> UTF-8 (texto)
- encode: UTF-8 (texto) -> PLNCG26 (bytes)
- detect: probabilidad [0,1] de que el fichero contenga texto en PLNCG26
"""
import sys
from enum import Enum
from pathlib import Path

import typer

BASE = 20               # a=20, b=21, ..., z=45
ALPHABET_SIZE = 31      # 26 letras + 5 símbolos especiales de PLNCG26
SPACE = 0x0B            # 11: espacio
NEWLINE = 0x0A          # 10: salto de línea
ACCENT_IDX = 30         # índice lógico del símbolo "acento" dentro del alfabeto
ACCENT_BYTE = 50        # (30 + 20) índice 30 = acento sobre vocal anterior
DIGIT_BASE = 60         # 60–69: dígitos '0'–'9'
DIERESIS = 0x33         # 51: u -> ü
ENYE = 0x34             # 52: ñ/Ñ
MAYUSCULA = 0x35        # 53: mayúscula del carácter anterior
PUNTO = 0x46            # 70: .
COMA = 0x47             # 71: ,
DOS_PUNTOS = 73         # 73: :
GUION_LARGO = 78        # 78 78 ... 78 78 = —algo—
COMILLAS = 80           # 80...80 = "texto"
HEADER = 100            # 100 = #, 100 100 = ##, etc.
ASTERISCO = 101         # 101: *
PAREN_OPEN = 0x51       # 81: (
PAREN_CLOSE = 0x52      # 82: )
PUNTO_COMA = 0x48       # 72: ;


class QuoteStyle(str, Enum):
    """Estilo de comillas de salida al decodificar."""

    ESPANOLAS = "espanolas"
    INGLESAS = "inglesas"


ACCENT_MAP = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}
REVERSE_ACCENT = {v: k for k, v in ACCENT_MAP.items()} | {
    "Á": "a",
    "É": "e",
    "Í": "i",
    "Ó": "o",
    "Ú": "u",
}

# Mapeos directos byte <-> carácter sin lógica adicional
BYTE_TO_CHAR = {
    NEWLINE: "\n",
    SPACE: " ",
    COMA: ",",
    DOS_PUNTOS: ":",
    HEADER: "#",
    ASTERISCO: "*",
    PUNTO: ".",
    PAREN_OPEN: "(",
    PAREN_CLOSE: ")",
    PUNTO_COMA: ";",
}

CHAR_TO_BYTE = {v: k for k, v in BYTE_TO_CHAR.items()} | {
    '"': COMILLAS,
    "«": COMILLAS,
    "»": COMILLAS,
    "—": None,  # se trata aparte
}

# Conjunto de bytes válidos en PLNCG26 (para detección)
VALID_PLNCG26_BYTES = frozenset({
    NEWLINE,
    SPACE,
    *range(BASE, BASE + 26),  # a-z
    ACCENT_BYTE,
    DIERESIS,
    ENYE,
    MAYUSCULA,
    *range(DIGIT_BASE, DIGIT_BASE + 10),  # 0-9
    PUNTO,
    COMA,
    PUNTO_COMA,
    DOS_PUNTOS,
    GUION_LARGO,
    COMILLAS,
    PAREN_OPEN,
    PAREN_CLOSE,
    HEADER,
    ASTERISCO,
})

VOWEL_BYTES = {BASE + (ord(v) - ord("a")) for v in "aeiou"}     # Vocales a, e, i, o, u
LETTER_N_BYTE = BASE + (ord("n") - ord("a"))                    # Letra n
MODIFIER_BYTES = {ACCENT_BYTE, DIERESIS, ENYE, MAYUSCULA}       # Símbolos que modifican al anterior

app = typer.Typer(help="Codificación/decodificación PLNCG26 <-> UTF-8.")


def _sym_index(byte: int) -> int:
    """Índice de símbolo [0,30] a partir del byte."""
    return (byte - BASE) % ALPHABET_SIZE


def decode_bytes(data: bytes, quote_style: QuoteStyle = QuoteStyle.ESPANOLAS) -> str:
    """Decodifica una secuencia de bytes PLNCG26 a texto UTF-8 plano.

    La función realiza una pasada sobre los bytes interpretando:
    - letras base (a–z) a partir de su índice de símbolo;
    - espacios, saltos de línea y signos de puntuación (.,;:()#*…);
    - modificadores sobre el carácter anterior (mayúsculas, acentos, diéresis, ñ);
    - guiones largos como pares de bytes GUION_LARGO (78 78 -> "—");
    - comillas según el `quote_style` indicado:
      * `ESPANOLAS`: alterna « y » al encontrar COMILLAS;
      * `INGLESAS`: usa comillas dobles estándar (").

    Los bytes de acento se representan inicialmente como marcadores internos y se
    aplican en una segunda pasada sobre las vocales decodificadas (incluyendo la
    conservación de mayúsculas/minúsculas). Cualquier acento "huérfano" se ignora
    para producir una salida legible.
    """
    data_list = list(data)
    tmp_chars: list[str] = []
    count_78 = 0
    in_spanish_quotes = False

    for i, b in enumerate(data_list):
        if b == GUION_LARGO:
            count_78 += 1
            if count_78 == 2:
                tmp_chars.append("—")
                count_78 = 0
            continue
        else:
            count_78 = 0
        if b == COMILLAS:
            if quote_style is QuoteStyle.ESPANOLAS:
                quote_char = "«" if not in_spanish_quotes else "»"
                in_spanish_quotes = not in_spanish_quotes
                tmp_chars.append(quote_char)
            else:
                tmp_chars.append('"')
            continue

        direct_char = BYTE_TO_CHAR.get(b)
        if direct_char is not None:
            tmp_chars.append(direct_char)
        elif b == DIERESIS:
            if tmp_chars and tmp_chars[-1].lower() == "u":
                tmp_chars[-1] = "ü" if tmp_chars[-1].islower() else "Ü"
            else:
                tmp_chars.append("a")
        elif DIGIT_BASE <= b <= DIGIT_BASE + 9:
            tmp_chars.append(chr(ord("0") + (b - DIGIT_BASE)))
        elif b == MAYUSCULA:
            if tmp_chars and tmp_chars[-1] not in " \n":
                tmp_chars[-1] = tmp_chars[-1].upper()
        elif b == ENYE:
            if tmp_chars and tmp_chars[-1].lower() == "n":
                tmp_chars[-1] = "ñ" if tmp_chars[-1].islower() else "Ñ"
            elif not tmp_chars or tmp_chars[-1] in " \n":
                tmp_chars.append("Ñ")
            else:
                tmp_chars.append("b")
        else:
            s = _sym_index(b)
            if s == ACCENT_IDX:
                tmp_chars.append(("\x00", b))  # Marcador acento + byte para round-trip
            else:
                ch = chr(ord("a") + (s % 26))
                tmp_chars.append(ch)

    out: list[str] = []
    for item in tmp_chars:
        if isinstance(item, tuple):
            if out and out[-1].lower() in ACCENT_MAP:
                base = out[-1].lower()
                ac = ACCENT_MAP[base]
                out[-1] = ac.upper() if out[-1].isupper() else ac
            else:
                # Acento huérfano (sin vocal anterior): se omite para salida legible
                pass
        else:
            out.append(item)

    return "".join(out)


def encode_to_bytes(text: str) -> bytes:
    """Codifica texto UTF-8 a una secuencia de bytes en PLNCG26.

    A partir de una cadena Unicode genera bytes PLNCG26 aplicando las reglas:
    - letras a–z -> índices base (BASE + posición en el alfabeto);
    - letras A–Z -> letra minúscula + modificador MAYUSCULA;
    - vocales acentuadas (á, é, í, ó, ú / Á, É, Í, Ó, Ú) -> vocal base
      (+ MAYUSCULA si procede) + ACCENT_BYTE;
    - ü / Ü -> u (+ MAYUSCULA si procede) + DIERESIS;
    - ñ / Ñ -> n (+ MAYUSCULA si procede) + ENYE;
    - dígitos 0–9 -> rango de bytes 60–69;
    - guion largo — -> dos bytes GUION_LARGO (78 78);
    - comillas «», " y la puntuación soportada -> sus códigos PLNCG26 directos.

    Para cualquier carácter Unicode no mapeado explícitamente, se proyecta al
    alfabeto mediante su versión en minúscula si es letra, o a "a" en caso
    contrario, de forma que siempre se obtiene alguna secuencia de bytes válida.
    """
    result: list[int] = []
    prev: str | None = None

    for ch in text:
        direct_byte = CHAR_TO_BYTE.get(ch)
        if direct_byte is not None:
            result.append(direct_byte)
        elif ch == "—":
            result.append(GUION_LARGO)
            result.append(GUION_LARGO)
        elif ch in REVERSE_ACCENT:
            base = REVERSE_ACCENT[ch]
            result.append(BASE + ord(base) - ord("a"))
            if ch.isupper():
                result.append(MAYUSCULA)
            result.append(ACCENT_BYTE)
        elif ch == "ü":
            result.append(BASE + ord("u") - ord("a"))
            result.append(DIERESIS)
        elif ch == "Ü":
            result.append(BASE + ord("u") - ord("a"))
            result.append(MAYUSCULA)
            result.append(DIERESIS)
        elif ch == "ñ":
            if prev and prev.lower() == "n":
                result.append(ENYE)
            else:
                result.append(BASE + ord("n") - ord("a"))
                result.append(ENYE)
        elif ch == "Ñ":
            # Siempre n + MAYUSCULA + ENYE para round-trip consistente con .bin
            result.append(BASE + ord("n") - ord("a"))
            result.append(MAYUSCULA)
            result.append(ENYE)
        elif "a" <= ch <= "z":
            result.append(BASE + ord(ch) - ord("a"))
        elif "A" <= ch <= "Z":
            result.append(BASE + ord(ch.lower()) - ord("a"))
            result.append(MAYUSCULA)
        elif "0" <= ch <= "9":
            result.append(DIGIT_BASE + ord(ch) - ord("0"))
        else:
            idx = ord(ch.lower()) - ord("a") if ch.isalpha() else 0
            result.append(BASE + (idx % 26))
        prev = ch

    return bytes(result)


def _fatal_error(message: str) -> None:
    """Muestra un mensaje de error por stderr y termina con código 1."""
    typer.secho(f"Error: {message}", err=True, fg="red", bold=True)
    raise typer.Exit(code=1)


@app.command()
def decode(
    fichero: Path = typer.Argument(
        ..., help="Fichero en PLNCG26 (bytes) a decodificar."
    ),
    comillas: QuoteStyle = typer.Option(
        QuoteStyle.ESPANOLAS,
        "--comillas",
        help="Tipo de comillas de salida: espanolas o inglesas.",
        case_sensitive=False,
    ),
) -> None:
    """Decodifica PLNCG26 a UTF-8 y escribe en stdout."""
    try:
        data: bytes = fichero.read_bytes()
    except OSError as exc:
        _fatal_error(f"No se pudo leer el fichero '{fichero}': {exc}")

    salida = decode_bytes(data, quote_style=comillas)

    try:
        print(salida, end="")
    except OSError as exc:
        _fatal_error(f"No se pudo escribir en stdout: {exc}")


@app.command()
def encode(
    fichero: Path = typer.Argument(..., help="Fichero en UTF-8 a codificar a PLNCG26."),
) -> None:
    """Codifica UTF-8 a PLNCG26 (bytes) y escribe en stdout."""
    try:
        texto = fichero.read_text(encoding="utf-8")
    except OSError as exc:
        _fatal_error(f"No se pudo leer el fichero '{fichero}': {exc}")

    data = encode_to_bytes(texto)

    try:
        sys.stdout.buffer.write(data)
    except OSError as exc:
        _fatal_error(f"No se pudo escribir en stdout (bytes): {exc}")


def plncg26_probability(data: bytes) -> float:
    """Calcula la probabilidad [0, 1] de que data sea texto en PLNCG26.

    La heurística combina tres factores:
    - porcentaje de bytes válidos de PLNCG26 presentes en el fichero;
    - uso correcto de modificadores (acentos, diéresis, ñ, mayúsculas) en contexto;
    - diversidad de símbolos válidos distintos usados con respecto al plano PLNCG26.
    """
    if not data:
        return 0.0
    total = len(data)
    valid_bytes = 0
    modifier_total = 0
    valid_modifiers = 0
    distinct_valid: set[int] = set()
    prev: int | None = None
    prev_prev: int | None = None

    for b in data:
        if b not in VALID_PLNCG26_BYTES:
            prev_prev = prev
            prev = b
            continue

        is_valid = True
        if b in MODIFIER_BYTES:
            modifier_total += 1
            if prev is None:
                is_valid = False
            elif b == ACCENT_BYTE:
                # Vocal acentuada: vocal + [MAYUSCULA] + ACCENT_BYTE
                is_valid = prev in VOWEL_BYTES or (
                    prev == MAYUSCULA and prev_prev in VOWEL_BYTES
                )
            elif b == DIERESIS:
                # ü / Ü: u + [MAYUSCULA] + DIERESIS
                u_byte = BASE + (ord("u") - ord("a"))
                is_valid = prev == u_byte or (
                    prev == MAYUSCULA and prev_prev == u_byte
                )
            elif b == ENYE:
                # ñ / Ñ: n + ENYE o n + MAYUSCULA + ENYE
                is_valid = prev == LETTER_N_BYTE or (
                    prev == MAYUSCULA and prev_prev == LETTER_N_BYTE
                )
            elif b == MAYUSCULA:
                # Modificador de mayúsculas no tiene sentido tras espacio/salto de línea
                is_valid = prev not in (SPACE, NEWLINE)

        if is_valid:
            valid_bytes += 1
            distinct_valid.add(b)
            if b in MODIFIER_BYTES:
                valid_modifiers += 1

        prev_prev = prev
        prev = b

    factor_chars = valid_bytes / total
    factor_modifiers = valid_modifiers / modifier_total if modifier_total else 1.0
    factor_diversity = len(distinct_valid) / len(VALID_PLNCG26_BYTES)

    return 0.5 * factor_chars + 0.3 * factor_modifiers + 0.2 * factor_diversity


@app.command()
def detect(
    fichero: Path = typer.Argument(
        ..., help="Fichero del que calcular la probabilidad de ser PLNCG26."
    ),
) -> None:
    """Calcula la probabilidad de que el fichero contenga texto en PLNCG26 (salida en [0, 1])."""
    try:
        data = fichero.read_bytes()
    except OSError as exc:
        _fatal_error(f"No se pudo leer el fichero '{fichero}': {exc}")

    prob = plncg26_probability(data)

    try:
        print(f"{prob:.3f}")
    except OSError as exc:
        _fatal_error(f"No se pudo escribir en stdout: {exc}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
