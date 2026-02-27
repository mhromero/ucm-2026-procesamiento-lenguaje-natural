"""
Flujo principal del bot: ciclo de negociación con Butler.

1. Obtiene estado (/info) y lista de agentes (/gente).
2. Envía ofertas según necesidades y excedentes.
3. Lee buzón, interpreta cartas con LLM y actúa (acepta/rechaza).
4. Si cambian los recursos, reenvía ofertas actualizadas.
"""

import json
import time
from typing import Any

import requests

from . import api
from .config import ALIAS, LETTERS_BEFORE_REBROADCAST, OFFERS_PER_PERSON, REMITENTE_SISTEMA, SINGLE_PLAYER_MODE
from .game_state import State
from .agent import parse_letter
from .letters import broadcast_offers
from . import logs
from .logs import (
    print_section,
    print_kv,
    print_status_letter,
    print_raw_letter,
    print_llm,
    print_error,
    print_bot,
    print_bot_dim,
    print_mailbox,
)
from .trader import handle_confirmation, handle_offer


def _process_letter(letter_id: str, content: dict[str, Any], state: State) -> str | None:
    """
    Interpreta una carta con el LLM y actúa según su tipo (oferta o confirmación).

    Retorna el tipo de carta procesada, o None si se ignora.
    La eliminación del buzón la realiza quien llama.
    """
    sender = content.get("remi", "??")

    if sender == state.alias:
        return None

    print_section(f"CARTA RECIBIDA de {sender}")
    print_kv("ID", letter_id)
    print_kv("Remitente", sender)
    print_kv("Asunto", content.get("asunto", ""))
    print_kv("Fecha", content.get("fecha", ""))
    print_raw_letter(content)

    analysis = parse_letter(content, state.needs, state.surplus)
    print_section("ANÁLISIS LLM DE LA CARTA")
    print_llm(analysis)

    letter_type = analysis.get("tipo", "otro")
    state.update()

    if letter_type == "oferta":
        sender_val = content.get("remi")
        if not sender_val:
            print_bot("Oferta sin remitente claro, se ignora.", warning=True)
        else:
            print_kv("Acción", f"Gestionando OFERTA de {sender_val}", color=logs.GREEN)
            handle_offer(sender_val, analysis, state.needs, state.surplus, state.inventory)
    elif letter_type == "confirmacion":
        sender_val = content.get("remi")
        if not sender_val:
            print_bot("Confirmación sin remitente claro, se ignora.", warning=True)
        else:
            print_kv("Acción", f"Gestionando CONFIRMACIÓN de {sender_val}", color=logs.GREEN)
            handle_confirmation(sender_val, analysis, state.inventory, state.needs, state.surplus)

    return letter_type


def main() -> None:
    """
    Flujo principal del bot:

    1. Registrar alias y obtener estado (inventario, objetivo, buzón) desde Butler.
    2. Enviar ofertas a los otros agentes según necesidades y excedentes.
    3. Procesar cartas del buzón: interpretar con LLM y ejecutar aceptar/rechazar.
    4. Esperar, refrescar buzón y repetir; reenviar ofertas periódicamente.
    """
    print_section("INICIO DEL BOT")

    # Registrar alias en Butler (en modo monopuesto el servidor lo asigna, no se llama POST /alias)
    #if ALIAS and not SINGLE_PLAYER_MODE:
    if ALIAS:
        try:
            print_kv("Alias configurado", ALIAS)
            api.set_alias(ALIAS)
        except Exception as e:
            print_error(f"No se pudo configurar el alias '{ALIAS}': {e}")

    # Cargar estado inicial: inventario, objetivo y buzón desde Butler
    print_kv("Acción", "Obteniendo nuestros recursos (/info)")
    state = State(alias="", inventory={}, target={}, needs={}, surplus={}, mailbox={})
    while True:
        try:
            state.update()
            break
        except requests.exceptions.RequestException as e:
            print_error(f"Error leyendo /info al arrancar: {e}")
            print_bot("Reintentando en 5 s...", warning=True)
            time.sleep(5)

    print_section("ESTADO INICIAL")
    print_kv("Alias", state.alias)
    print_kv("Inventario inicial", json.dumps(state.inventory, ensure_ascii=False))
    print_kv("Objetivo de recursos", json.dumps(state.target, ensure_ascii=False))

    print_section("AGENTES")
    print_kv("Acción", "Obteniendo agentes (/gente)")
    people = api.remove_myself({"Alias": state.alias}, api.get_people())
    print_kv("Otros agentes", [p.get("alias", p) for p in people])

    print_section("NECESIDADES Y EXCEDENTES")
    print_kv("Necesitamos", json.dumps(state.needs, ensure_ascii=False))
    print_kv(
        "Podemos ofrecer (sin oro, se filtra internamente al enviar)",
        json.dumps(state.surplus, ensure_ascii=False),
    )

    # Enviar ofertas según estado: si objetivo cumplido → surplus→oro; si no → intercambios necesidad↔excedente
    print_section("CARTAS DE OFERTA A ENVIAR")
    broadcast_offers(people, state, offers_per_person=OFFERS_PER_PERSON)

    # Bucle principal: procesar cartas, actuar, y reenviar ofertas periódicamente
    print_section("BUZÓN INICIAL")
    print_kv("Acción", "Leyendo cartas del buzón")
    print_kv("Cartas", len(state.mailbox))

    letters_processed = 0
    empty_mailbox_cycles = 0
    empty_mailbox_rebroadcast_every = 5
    while True:
        rebroadcasted_this_cycle = False
        # Ordenar cartas por fecha para procesar las más antiguas primero
        sorted_letters = sorted(
            state.mailbox.items(),
            key=lambda item: item[1].get("fecha", ""),
        )
        had_letters = len(sorted_letters) > 0

        for letter_id, content in sorted_letters:
            # Ignorar nuestras propias cartas (eco) y las del sistema
            if content.get("remi") == state.alias:
                api.delete_letter(letter_id)
                continue
            if content.get("remi") == REMITENTE_SISTEMA:
                print_bot_dim(f"[BOT] Ignorando carta de {REMITENTE_SISTEMA} (no se procesa)")
                api.delete_letter(letter_id)
                continue

            letter_type = _process_letter(letter_id, content, state)
            if letter_type is not None:
                letters_processed += 1
            print_bot_dim(f"[BOT] Eliminando carta del buzón (id={letter_id})")
            api.delete_letter(letter_id)

        # Reenviar ofertas cada N cartas procesadas para mantener visibilidad
        if letters_processed >= LETTERS_BEFORE_REBROADCAST:
            letters_processed = 0
            reason = f"{LETTERS_BEFORE_REBROADCAST} cartas analizadas. "
            broadcast_offers(people, state, offers_per_person=OFFERS_PER_PERSON, reason=reason)
            rebroadcasted_this_cycle = True

        if had_letters:
            empty_mailbox_cycles = 0
        else:
            empty_mailbox_cycles += 1
            # Reenviar cada N ciclos consecutivos de buzón vacío para reactivar negociación.
            if (
                empty_mailbox_cycles >= empty_mailbox_rebroadcast_every
                and not rebroadcasted_this_cycle
            ):
                reason = f"Buzón vacío durante {empty_mailbox_cycles} revisiones. "
                broadcast_offers(people, state, offers_per_person=OFFERS_PER_PERSON, reason=reason)
                empty_mailbox_cycles = 0

        # Esperar y refrescar buzón
        print_section("BUZÓN VACÍO")
        print_bot(
            "Sin cartas en buzón. Esperando 5 s y releyendo buzón...",
            warning=True,
        )
        time.sleep(5)
        try:
            state.update()
        except requests.exceptions.RequestException as e:
            print_error(f"Error refrescando estado: {e}")
            continue
        print_mailbox(state.mailbox)
