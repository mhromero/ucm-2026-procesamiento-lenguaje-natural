"""
Bot de intercambio de recursos entre agentes usando LLM (Ollama).

Sistema multi-agente que negocia recursos mediante cartas y paquetes:
- Lee estado y buzón desde Butler
- Interpreta cartas con un LLM local
- Acepta/rechaza ofertas y confirma envíos automáticamente
"""

__all__ = ["main"]


def __getattr__(name: str):
    if name == "main":
        from .app import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
