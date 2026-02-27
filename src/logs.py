"""
Salida formateada por consola: colores ANSI, separadores y prefijos por tipo.

Distingue [BOT], [LLM], [ERROR], [CARTA ESTADO], etc., para seguir el flujo.
"""

import json
from typing import Any

# Estilos ANSI
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"


def print_section(title: str) -> None:
    """Imprime una sección destacada con separadores (_____)."""
    line = "_" * 70
    print(f"\n{DIM}{line}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{DIM}{line}{RESET}")


def print_kv(label: str, value: Any, color: str = CYAN) -> None:
    """Imprime una línea etiquetada con [BOT] y el color indicado."""
    print(f"{color}{BOLD}[BOT]{RESET} {BOLD}{label}:{RESET} {value}")


def print_status_letter(text: str) -> None:
    """Imprime la carta de estado con etiqueta [CARTA ESTADO]."""
    print(f"{MAGENTA}{BOLD}[CARTA ESTADO]{RESET}\n{text}")


def print_raw_letter(content: Any) -> None:
    """Imprime el contenido crudo de una carta con etiqueta [CARTA CRUDA]."""
    print(
        f"{MAGENTA}{BOLD}[CARTA CRUDA]{RESET}\n"
        f"{json.dumps(content, ensure_ascii=False, indent=2)}"
    )


def print_llm(analysis: Any) -> None:
    """Imprime el análisis del LLM con etiqueta [LLM]."""
    print(
        f"{CYAN}{BOLD}[LLM]{RESET} {json.dumps(analysis, ensure_ascii=False, indent=2)}"
    )


def print_error(msg: str) -> None:
    """Imprime un mensaje de error con etiqueta [ERROR]."""
    print(f"{RED}{BOLD}[ERROR]{RESET} {msg}")


def print_bot(msg: str, *, success: bool = False, warning: bool = False) -> None:
    """Imprime un mensaje del bot [BOT]; success o warning cambian el color."""
    color = GREEN if success else (YELLOW if warning else RESET)
    print(f"{color}{BOLD}[BOT]{RESET} {msg}")


def print_bot_dim(msg: str) -> None:
    """Imprime un mensaje secundario del bot en tono apagado."""
    print(f"{DIM}{msg}{RESET}")


def print_mailbox(mailbox: Any) -> None:
    """Imprime el contenido del buzón como JSON formateado."""
    print(json.dumps(mailbox, ensure_ascii=False, indent=2))


def print_decision(title: str, data: Any) -> None:
    """Imprime una decisión con título y JSON con etiqueta [DECISIÓN]."""
    print(f"{CYAN}{BOLD}[DECISIÓN] {title}{RESET}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def print_llm_response(text: str) -> None:
    """Imprime la respuesta cruda del LLM con etiqueta [LLM]."""
    print(f"{CYAN}{BOLD}[LLM]{RESET} {text}")


def print_retry(msg: str) -> None:
    """Imprime mensaje de reintento/advertencia con etiqueta [BOT] en amarillo."""
    print(f"{YELLOW}{BOLD}[BOT]{RESET} {msg}")
