import json
from typing import Any, Dict

from . import api
from .config import GOLD_RESOURCE_NAME
from .letters import build_trade_confirmation_letter
from .ollama_client import ollama

# JSON Schema para forzar la forma de la decisión de oferta (Ollama format).
ANALIZAR_OFERTA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["aceptada", "rechazada"],
        },
        "oferta": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "pide": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
    },
    "required": ["decision", "oferta", "pide"],
    "additionalProperties": False,
}


def analizar_oferta(
    oferta: Dict[str, Any],
    needs: Dict[str, Any],
    surplus: Dict[str, int],
) -> Dict[str, Any]:
    """
    Usa Ollama para decidir si aceptar o rechazar una oferta.
    Devuelve un JSON con decision (aceptada|rechazada), oferta y pide.
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
- "decision" = "aceptada" si se cumplen TODAS las condiciones siguientes:
    a) los recursos que ofrece los necesitamos para cumplir el objetivo
    b) no pide recursos que necesitemos para nuestro objetivo
    c) podemos dar los recursos que pide
    d) se envían como máximo el mismo número de recursos que se reciben, salvo que con la oferta completemos el objetivo al 100%

- "decision" = "rechazada" si no se cumple alguna de las condiciones anteriores
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
{json.dumps(oferta, ensure_ascii=False, indent=2)}

"""
    respuesta = ollama(prompt, format=ANALIZAR_OFERTA_JSON_SCHEMA)
    try:
        data = json.loads(respuesta)
        if not isinstance(data, dict):
            raise ValueError("Respuesta no es un dict")
        return data
    except (json.JSONDecodeError, ValueError):
        print("ERROR: Ollama no devolvió JSON válido al analizar oferta")
        print(respuesta)
        return {"decision": "rechazada", "oferta": {}, "pide": {}}


def process_offer(
    analisis: Dict[str, Any],
    needs: Dict[str, Any],
    surplus: Dict[str, int],
    inventario: Dict[str, int],
) -> Dict[str, Any]:
    """
    Procesa una oferta: decide si se acepta y comprueba todas las condiciones
    (no enviar oro, no enviar lo que necesitamos, tener stock suficiente).

    Estructura devuelta:
    {
      "aceptada": bool,
      "motivo": str,
      "oferta": Dict[str, int],
      "pide": Dict[str, int],
      "recursos_a_enviar": Dict[str, int] | {}
    }
    """
    oferta = analisis.get("oferta") or {}
    pide = analisis.get("pide") or {}

    if not oferta or not pide:
        return {
            "aceptada": False,
            "motivo": "Oferta sin datos claros (oferta/pide vacíos).",
            "oferta": oferta,
            "pide": pide,
            "recursos_a_enviar": {},
        }

    decision = analizar_oferta(analisis, needs, surplus)

    if decision.get("decision") != "aceptada":
        return {
            "aceptada": False,
            "motivo": "Oferta rechazada según analizar_oferta.",
            "oferta": decision.get("oferta") or oferta,
            "pide": decision.get("pide") or pide,
            "recursos_a_enviar": {},
        }

    oferta_decidida = decision.get("oferta") or oferta
    pide_decidido = decision.get("pide") or pide

    try:
        recursos_a_enviar = {k: int(v) for k, v in (pide_decidido or {}).items()}
    except (TypeError, ValueError):
        return {
            "aceptada": False,
            "motivo": "No se pudo interpretar las cantidades de recursos a enviar.",
            "oferta": oferta_decidida,
            "pide": pide_decidido,
            "recursos_a_enviar": {},
        }

    # Comprobaciones de condiciones antes de aceptar
    if (
        GOLD_RESOURCE_NAME in recursos_a_enviar
        and recursos_a_enviar.get(GOLD_RESOURCE_NAME, 0) > 0
    ):
        return {
            "aceptada": False,
            "motivo": "No enviamos oro.",
            "oferta": oferta_decidida,
            "pide": pide_decidido,
            "recursos_a_enviar": {},
        }

    for recurso, cant in recursos_a_enviar.items():
        if needs.get(recurso, 0) > 0:
            return {
                "aceptada": False,
                "motivo": f"No enviamos recurso que necesitamos para el objetivo: {recurso}.",
                "oferta": oferta_decidida,
                "pide": pide_decidido,
                "recursos_a_enviar": {},
            }
        if inventario.get(recurso, 0) < cant:
            return {
                "aceptada": False,
                "motivo": f"No tenemos suficientes '{recurso}' (tenemos {inventario.get(recurso, 0)}, piden {cant}).",
                "oferta": oferta_decidida,
                "pide": pide_decidido,
                "recursos_a_enviar": {},
            }

    return {
        "aceptada": True,
        "motivo": "Oferta aceptada.",
        "oferta": oferta_decidida,
        "pide": pide_decidido,
        "recursos_a_enviar": recursos_a_enviar,
    }


def process_confirmation(
    analisis: Dict[str, Any],
    inventario: Dict[str, int],
    needs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Procesa una confirmación de envío: extrae qué nos han enviado y qué piden
    a cambio, y comprueba las condiciones (no enviar oro, no enviar lo que
    necesitamos, tener stock suficiente) antes de autorizar el envío.

    Estructura devuelta:
    {
      "tiene_recursos_recibidos": bool,
      "es_regalo": bool,
      "puede_enviar": bool,
      "motivo": str,
      "recursos_recibidos": Dict[str, int],
      "pide": Dict[str, int],
      "recursos_a_enviar": Dict[str, int] | {}
    }
    """
    recursos_recibidos = analisis.get("recursos_recibidos") or {}
    pide = analisis.get("pide") or {}

    if not recursos_recibidos:
        return {
            "tiene_recursos_recibidos": False,
            "es_regalo": False,
            "puede_enviar": False,
            "motivo": "Confirmación sin recursos recibidos claros.",
            "recursos_recibidos": {},
            "pide": pide,
            "recursos_a_enviar": {},
        }

    if not pide:
        return {
            "tiene_recursos_recibidos": True,
            "es_regalo": True,
            "puede_enviar": False,
            "motivo": "Confirmación sin recursos a enviar a cambio, se asume regalo.",
            "recursos_recibidos": recursos_recibidos,
            "pide": {},
            "recursos_a_enviar": {},
        }

    try:
        recursos_a_enviar = {k: int(v) for k, v in pide.items()}
    except (TypeError, ValueError):
        return {
            "tiene_recursos_recibidos": True,
            "es_regalo": False,
            "puede_enviar": False,
            "motivo": "No se pudo interpretar las cantidades de recursos a enviar.",
            "recursos_recibidos": recursos_recibidos,
            "pide": pide,
            "recursos_a_enviar": {},
        }

    # Comprobaciones de condiciones antes de autorizar el envío
    if (
        GOLD_RESOURCE_NAME in recursos_a_enviar
        and recursos_a_enviar.get(GOLD_RESOURCE_NAME, 0) > 0
    ):
        return {
            "tiene_recursos_recibidos": True,
            "es_regalo": False,
            "puede_enviar": False,
            "motivo": "No enviamos oro en confirmación.",
            "recursos_recibidos": recursos_recibidos,
            "pide": pide,
            "recursos_a_enviar": {},
        }

    for recurso, cant in recursos_a_enviar.items():
        if needs.get(recurso, 0) > 0:
            return {
                "tiene_recursos_recibidos": True,
                "es_regalo": False,
                "puede_enviar": False,
                "motivo": f"No enviamos recurso que necesitamos para el objetivo: {recurso}.",
                "recursos_recibidos": recursos_recibidos,
                "pide": pide,
                "recursos_a_enviar": {},
            }
        if inventario.get(recurso, 0) < cant:
            return {
                "tiene_recursos_recibidos": True,
                "es_regalo": False,
                "puede_enviar": False,
                "motivo": f"No tenemos suficientes '{recurso}' para enviar (tenemos {inventario.get(recurso, 0)}, piden {cant}).",
                "recursos_recibidos": recursos_recibidos,
                "pide": pide,
                "recursos_a_enviar": {},
            }

    return {
        "tiene_recursos_recibidos": True,
        "es_regalo": False,
        "puede_enviar": True,
        "motivo": "Confirmación con recursos a enviar a cambio.",
        "recursos_recibidos": recursos_recibidos,
        "pide": pide,
        "recursos_a_enviar": recursos_a_enviar,
    }


def handle_offer(
    remitente: str,
    analisis: Dict[str, Any],
    needs: Dict[str, Any],
    surplus: Dict[str, int],
    inventario: Dict[str, int],
) -> bool:
    """
    Procesa una oferta: decide, comprueba condiciones, envía paquete y carta
    de confirmación si se acepta. Devuelve True si nuestros recursos cambiaron.
    """
    resultado = process_offer(analisis, needs, surplus, inventario)
    print("Decisión sobre la oferta:")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))

    if not resultado.get("aceptada"):
        print(f"Oferta rechazada: {resultado.get('motivo')}")
        return False

    oferta = resultado.get("oferta") or {}
    recursos_a_enviar = resultado.get("recursos_a_enviar") or {}
    if not recursos_a_enviar:
        print(
            "Oferta aceptada pero sin recursos a enviar (resultado vacío), no se realiza envío."
        )
        return False

    try:
        print(f"Aceptando oferta de {remitente}. Enviando paquete: {recursos_a_enviar}")
        api.send_package(remitente, recursos_a_enviar)
    except Exception as e:
        print(f"ERROR enviando paquete de oferta a {remitente}: {e}")
        return False

    try:
        carta_confirmacion = build_trade_confirmation_letter(
            recursos_enviados=recursos_a_enviar,
            recursos_esperados=oferta,
        )
        print(f"→ Enviando carta de confirmación de oferta aceptada a {remitente}...")
        api.send_letter(
            remitente, "Confirmación de oferta aceptada", carta_confirmacion
        )
    except Exception as e:
        print(f"ERROR enviando carta de confirmación a {remitente}: {e}")

    return True


def handle_confirmation(
    remitente: str,
    analisis: Dict[str, Any],
    inventario: Dict[str, int],
    needs: Dict[str, Any],
) -> bool:
    """
    Procesa una confirmación: decide, comprueba condiciones, envía paquete
    y carta de confirmación si aplica. Devuelve True si nuestros recursos cambiaron.
    """
    resultado = process_confirmation(analisis, inventario, needs)
    print("Decisión sobre la confirmación:")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))

    if not resultado.get("tiene_recursos_recibidos"):
        print(f"No se procesan recursos: {resultado.get('motivo')}")
        return False

    recursos_recibidos = resultado.get("recursos_recibidos") or {}
    recursos_a_enviar = resultado.get("recursos_a_enviar") or {}

    if resultado.get("es_regalo"):
        print(
            "Se interpreta la confirmación como regalo, no se envían recursos a cambio."
        )
        return True

    if not resultado.get("puede_enviar") or not recursos_a_enviar:
        print(
            f"No se envía paquete de confirmación: {resultado.get('motivo', 'sin recursos a enviar')}."
        )
        return False

    try:
        print(
            f"Confirmación correcta de {remitente}. Enviando paquete de vuelta: {recursos_a_enviar}"
        )
        api.send_package(remitente, recursos_a_enviar)
    except Exception as e:
        print(f"ERROR enviando paquete de confirmación a {remitente}: {e}")
        return False

    try:
        carta_confirmacion = build_trade_confirmation_letter(
            recursos_enviados=recursos_a_enviar,
            recursos_esperados=recursos_recibidos,
        )
        print(f"→ Enviando carta de confirmación de envío de recursos a {remitente}...")
        api.send_letter(
            remitente, "Confirmación de envío de recursos", carta_confirmacion
        )
    except Exception as e:
        print(
            f"ERROR enviando carta de confirmación (confirmación recibida) a {remitente}: {e}"
        )

    return True
