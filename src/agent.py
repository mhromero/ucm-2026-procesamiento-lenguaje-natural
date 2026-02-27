"""
Lógica del agente: análisis de cartas y ofertas con LLM.
"""

import json
from typing import Any, Dict

import requests

from .config import GOLD_RESOURCE_NAME
from .ollama_client import ollama


def _gold_only(needs: Dict[str, Any], surplus: Dict[str, int], inventory: Dict[str, int]) -> bool:
    """True si no hemos alcanzado objetivo, no tenemos surplus y tenemos al menos 1 oro."""
    if len(needs) == 0 or surplus:
        return False
    return inventory.get(GOLD_RESOURCE_NAME, 0) >= 1


def parse_letter(
    letter_data: Dict[str, Any],
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
{json.dumps(letter_data, ensure_ascii=False, indent=2)}
"""
    try:
        response = ollama(prompt)
    except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
        print("ERROR: Timeout al analizar carta; se usa fallback.")
        return {"tipo": "otro", "oferta": {}, "pide": {}, "recursos_recibidos": {}}

    print(response)

    try:
        data = json.loads(response)
        if not isinstance(data, dict):
            raise ValueError("Respuesta no es un dict")
        return data
    except (json.JSONDecodeError, ValueError):
        print("ERROR: Ollama no devolvió JSON válido al analizar carta")
        print(response)
        return {"tipo": "otro", "oferta": {}, "pide": {}, "recursos_recibidos": {}}


def analyze_offer(
    offer: Dict[str, Any],
    needs: Dict[str, Any],
    surplus: Dict[str, int],
    inventory: Dict[str, int],
) -> Dict[str, Any]:
    """
    Usa Ollama para decidir si aceptar o rechazar una oferta.
    Devuelve un JSON con decision (aceptada|rechazada), oferta y pide.
    Si needs está vacío (objetivo cumplido), acepta ofertas que nos den oro a cambio de surplus.
    Si solo tenemos oro (surplus vacío, tenemos needs y >= 1 oro), acepta ofertas de 1 recurso a cambio de 1 oro.
    """
    objective_met = len(needs) == 0
    gold_only = _gold_only(needs, surplus, inventory)

    if objective_met:
        rules = f"""
- Ya hemos cumplido el objetivo de recursos. Ahora queremos MAXIMIZAR ORO.
- "decision" = "aceptada" si:
    a) nos ofrecen {GOLD_RESOURCE_NAME} (oro)
    b) nos piden recursos que tenemos en surplus (PODEMOS OFRECER)
    c) podemos dar las cantidades que pide
    d) es razonable (p. ej. 1:1 o menos recursos que recibimos de oro)
- "decision" = "rechazada" si no nos ofrecen oro o piden algo que no tenemos en surplus.
"""
    elif gold_only:
        rules = f"""
- Solo tenemos {GOLD_RESOURCE_NAME} para ofrecer y no hemos alcanzado el objetivo.
- "decision" = "aceptada" si:
    a) nos ofrecen 1 unidad de CUALQUIER recurso (no necesariamente de NECESITAMOS)
    b) nos piden exactamente 1 {GOLD_RESOURCE_NAME} (y tenemos al menos 1)
- "decision" = "rechazada" si no nos ofrecen ningún recurso o no piden 1 oro.
"""
    else:
        rules = """
- "decision" = "aceptada" si se cumplen TODAS las condiciones siguientes:
    a) los recursos que ofrece los necesitamos para cumplir el objetivo
    b) no pide recursos que necesitemos para nuestro objetivo
    c) podemos dar los recursos que pide
    d) se envían como máximo el mismo número de recursos que se reciben, salvo que con la oferta completemos el objetivo al 100%

- "decision" = "rechazada" si no se cumple alguna de las condiciones anteriores
"""
    prompt = f"""
Eres un asistente que toma decisiones sobre ofertas recibidas.

Tu tarea es LEER la oferta y devolver un JSON estructurado con esta forma:

{{
  "decision": "aceptada" | "rechazada",
  "oferta": {{
    "recurso": cantidad entero
  }},
  "pide": {{
    "recurso": cantidad entero
  }},

}}

Donde:
{rules}
- "oferta" describe lo que EL OTRO agente nos ofrece.
- "pide" describe lo que EL OTRO agente quiere que le enviemos.
- No hace falta aceptar la oferta al completo, se puede aceptar parcialmente solo los recursos que nos interesen y que cumplan las condiciones anteriores.
IMPORTANTE:
- Devuelve SIEMPRE un JSON VÁLIDO, sin texto adicional.

NECESITAMOS:
{json.dumps(needs, ensure_ascii=False, indent=2)}

PODEMOS OFRECER:
{json.dumps(surplus, ensure_ascii=False, indent=2)}

OFERTA:
{json.dumps(offer, ensure_ascii=False, indent=2)}

"""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = ollama(prompt)
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
            if attempt < max_attempts - 1:
                print(f"Intento {attempt + 1}/{max_attempts}: Timeout con Ollama, reintentando...")
            else:
                print("ERROR: Timeout con Ollama al analizar oferta (tras 3 intentos)")
                return {"decision": "rechazada", "oferta": {}, "pide": {}}
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                raise ValueError("Respuesta no es un dict")
            return data
        except (json.JSONDecodeError, ValueError):
            if attempt < max_attempts - 1:
                print(f"Intento {attempt + 1}/{max_attempts}: JSON inválido, reintentando...")
            else:
                print("ERROR: Ollama no devolvió JSON válido al analizar oferta (tras 3 intentos)")
                print(response)
                return {"decision": "rechazada", "oferta": {}, "pide": {}}
