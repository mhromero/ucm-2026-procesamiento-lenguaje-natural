"""
Generación y análisis de cartas: prompts para Ollama y carta de estado.
"""

import json
from typing import Any, Dict

from .config import GOLD_RESOURCE_NAME
from .ollama_client import ollama

# JSON Schema para forzar la forma del análisis de cartas (Ollama format).
ANALIZAR_CARTA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tipo": {
            "type": "string",
            "enum": ["oferta", "confirmacion", "otro"],
        },
        "oferta": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "pide": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "recursos_recibidos": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
    },
    "required": ["tipo", "oferta", "pide", "recursos_recibidos"],
    "additionalProperties": False,
}


def build_status_letter(
    alias: str,
    inventario: Dict[str, int],
    objetivo: Dict[str, int],
    needs: Dict[str, int],
    surplus: Dict[str, int],
) -> str:
    """
    Carta preescrita que enviamos a todos los jugadores con lo que tenemos,
    lo que necesitamos y lo que podemos ofrecer (sin oro).
    """
    return f"""
Necesito:
{json.dumps(needs, ensure_ascii=False, indent=2)}

Ofrezco:
{json.dumps(surplus, ensure_ascii=False, indent=2)}

Si te interesa intercambiar, por favor propón un trato indicando:
- qué recursos me ofreces y cuántas unidades
- qué recursos quieres a cambio y cuántas unidades
- si me has enviado ya recursos (confirmación de envío)
""".strip()


def build_simple_offer_letter(
    recurso_necesario: str,
    recurso_sobrante: str,
) -> str:
    """
    Genera una 'mini carta' muy simple proponiendo un intercambio 1 a 1:
    queremos 1 unidad de `recurso_necesario` y ofrecemos 1 unidad de
    `recurso_sobrante` a cambio.

    Ejemplo: "Te propongo intercambiar 1 piedra por 1 tela."
    """
    return (
        f"Te propongo intercambiar 1 {recurso_necesario} que necesito "
        f"por 1 {recurso_sobrante} que te ofrezco."
    )


def build_trade_confirmation_letter(
    recursos_enviados: Dict[str, int],
    recursos_esperados: Dict[str, int],
) -> str:
    """
    Carta prefabricada para confirmar que hemos aceptado una oferta:
    indicamos qué recursos hemos enviado y cuáles esperamos recibir.
    """
    return f"""
He aceptado tu oferta.

Te he enviado los recursos que pedías:
{json.dumps(recursos_enviados, ensure_ascii=False, indent=2)}

Espero recibir a cambio los recursos que ofrecías:
{json.dumps(recursos_esperados, ensure_ascii=False, indent=2)}
""".strip()


def analizar_carta(
    carta_dict: Dict[str, Any],
    needs: Dict[str, Any],
    surplus: Dict[str, int],
) -> Dict[str, Any]:
    """
    Usa Ollama para interpretar una carta y devolver un JSON con
    tipo (oferta|confirmacion|otro), oferta, pide, recursos_recibidos.
    """
    prompt = f"""
Eres un asistente que ayuda a interpretar cartas de intercambio de recursos
entre agentes en un juego.

Tu tarea es LEER la carta y devolver un JSON estructurado con esta forma:

{{
  "tipo": "oferta" | "confirmacion" | "otro",
  "oferta": {{
    "recurso": cantidad entero
  }},
  "pide": {{
    "recurso": cantidad entero
  }},
  "recursos_recibidos": {{
    "recurso": cantidad entero
  }}
}}

Donde:
- "tipo" = "oferta" si la carta propone un intercambio (yo te doy X, tú me das Y).
- "tipo" = "confirmacion" si la carta dice que ya nos han enviado recursos.
- "tipo" = "otro" si no encaja claramente en ninguno de los casos.
- "oferta" describe lo que EL OTRO agente nos ofrece.
- "pide" describe lo que EL OTRO agente quiere que le enviemos.
- "recursos_recibidos" son los recursos que el agente afirma que YA nos ha enviado.

IMPORTANTE:
- Devuelve SIEMPRE un JSON VÁLIDO, sin texto adicional.
- Si algún campo no está claro en la carta, devuélvelo como un objeto vacío {{}}.

OFRECEMOS:
{json.dumps(surplus, ensure_ascii=False, indent=2)}

NECESITAMOS:
{json.dumps(needs, ensure_ascii=False, indent=2)}

CARTA RECIBIDA (como JSON bruto de la API):
{json.dumps(carta_dict, ensure_ascii=False, indent=2)}
"""
    respuesta = ollama(prompt)
    print(respuesta)

    try:
        data = json.loads(respuesta)
        if not isinstance(data, dict):
            raise ValueError("Respuesta no es un dict")
        return data
    except (json.JSONDecodeError, ValueError):
        print("ERROR: Ollama no devolvió JSON válido al analizar carta")
        print(respuesta)
        return {"tipo": "otro", "oferta": {}, "pide": {}, "recursos_recibidos": {}}
