"""
Composición de cartas: ofertas simples, surplus→oro, confirmaciones.

Genera el texto de cada tipo de carta y gestiona el broadcast a los agentes.
"""

import json
import random
from typing import Any

from . import api
from . import logs
from .config import GOLD_RESOURCE_NAME
from .game_state import State


def build_status_letter(
    alias: str,
    inventory: dict[str, int],
    target: dict[str, int],
    needs: dict[str, int],
    surplus: dict[str, int],
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
    needed_resource: str,
    surplus_resource: str,
) -> str:
    """
    Genera una 'mini carta' muy simple proponiendo un intercambio 1 a 1:
    queremos 1 unidad de `recurso_necesario` y ofrecemos 1 unidad de
    `recurso_sobrante` a cambio.

    Ejemplo: "Te propongo intercambiar 1 piedra por 1 tela."
    """
    return (
        f"Te ofrezo 1 {surplus_resource} y tu me das "
        f"1 {needed_resource}."
    )


def build_surplus_for_gold_letter(surplus_resource: str, gold_name: str) -> str:
    """
    Genera una mini carta ofreciendo 1 unidad de recurso sobrante a cambio de 1 oro.
    Se usa cuando ya hemos alcanzado el objetivo y queremos maximizar oro.
    """
    return (
        f"Ya he cumplido mi objetivo de recursos. "
        f"Te ofrezco 1 {surplus_resource} a cambio de 1 {gold_name}."
    )


def build_gold_for_any_letter(needs: dict[str, int], gold_name: str) -> str:
    """
    Genera una carta ofreciendo 1 oro a cambio de 1 unidad de un recurso concreto que necesitemos.
    Se usa cuando solo tenemos oro y no hemos alcanzado el objetivo.
    """
    if needs:
        target_resource = random.choice(list(needs.keys()))
        return (
            f"No tengo otros recursos para intercambiar. Necesito: {json.dumps(needs, ensure_ascii=False)}. "
            f"Te ofrezco 1 {gold_name} a cambio de 1 unidad de {target_resource}."
        )

    # Fallback si por alguna razón no hay needs: cualquier recurso
    return (
        f"No tengo otros recursos para intercambiar. "
        f"Te ofrezco 1 {gold_name} a cambio de 1 unidad de cualquier recurso."
    )


def build_trade_confirmation_letter(
    resources_sent: dict[str, int],
    resources_expected: dict[str, int],
) -> str:
    """
    Carta prefabricada para confirmar que hemos aceptado una oferta:
    indicamos qué recursos hemos enviado y cuáles esperamos recibir.
    """
    return f"""
He aceptado tu oferta.

Te he enviado los recursos que pedías:
{json.dumps(resources_sent, ensure_ascii=False, indent=2)}

Espero recibir a cambio los recursos que ofrecías:
{json.dumps(resources_expected, ensure_ascii=False, indent=2)}
""".strip()


def _iter_other_people(people: list[Any], my_alias: str):
    """Itera sobre (alias, person) de agentes distintos a uno mismo."""
    for p in people:
        alias = p.get("alias") or p.get("Alias") if isinstance(p, dict) else p
        if alias and alias != my_alias:
            yield alias, p


def broadcast_offers(
    people: list[Any],
    state: State,
    offers_per_person: int,
    reason: str = "",
) -> None:
    """
    Envía ofertas a todos los otros agentes según el estado actual.
    reason: contexto para logs (ej. "5 cartas analizadas. ").
    offers_per_person: número de ofertas aleatorias por persona en el caso normal.
    """
    prefix = reason or ""

    if state.has_reached_objective() and state.surplus:
        surplus_list = list(state.surplus.keys())
        msg = f"{prefix}Objetivo alcanzado. Enviando una oferta surplus→oro aleatoria por persona."
        logs.print_bot(msg.strip(), success=True)
        for alias, _ in _iter_other_people(people, state.alias):
            surplus_resource = random.choice(surplus_list)
            body = build_surplus_for_gold_letter(surplus_resource, GOLD_RESOURCE_NAME)
            subject = f"Oferta: 1 {surplus_resource} por 1 {GOLD_RESOURCE_NAME}"
            try:
                logs.print_kv("Enviando oferta surplus→oro a", f"{alias} -> {subject}", color=logs.GREEN)
                api.send_letter(alias, subject, body)
            except Exception as e:
                logs.print_error(f"al enviar oferta surplus→oro a {alias}: {e}")

    elif state.only_has_gold_to_trade():
        msg = f"{prefix}Solo tenemos oro. Enviando oferta 1 oro por cualquier recurso."
        logs.print_bot(msg.strip(), success=True)
        body = build_gold_for_any_letter(state.needs, GOLD_RESOURCE_NAME)
        subject = f"Oferta: 1 {GOLD_RESOURCE_NAME} por 1 recurso que necesite"
        for alias, _ in _iter_other_people(people, state.alias):
            try:
                logs.print_kv("Enviando oferta oro por recurso a", f"{alias} -> {subject}", color=logs.GREEN)
                api.send_letter(alias, subject, body)
            except Exception as e:
                logs.print_error(f"al enviar oferta oro por recurso a {alias}: {e}")

    elif state.needs and state.surplus:
        offer_pairs = [(n, s) for n in state.needs.keys() for s in state.surplus.keys()]
        if offer_pairs:
            msg = f"{prefix}Enviando {offers_per_person} ofertas aleatorias por persona."
            logs.print_bot(msg.strip(), success=True)
            k = min(offers_per_person, len(offer_pairs))
            for alias, _ in _iter_other_people(people, state.alias):
                chosen = random.choices(offer_pairs, k=k)
                for needed_resource, surplus_resource in chosen:
                    body = build_simple_offer_letter(
                        needed_resource=needed_resource,
                        surplus_resource=surplus_resource,
                    )
                    subject = f"Oferta: 1 {needed_resource} por 1 {surplus_resource}"
                    try:
                        logs.print_kv("Enviando mini oferta a", f"{alias} -> {subject}", color=logs.GREEN)
                        api.send_letter(alias, subject, body)
                    except Exception as e:
                        logs.print_error(f"al enviar mini oferta a {alias}: {e}")
