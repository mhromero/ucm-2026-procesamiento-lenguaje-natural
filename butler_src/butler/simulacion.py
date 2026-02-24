# === PLN BUTLER - SIMULACIÓN ===
#
# Lógica de simulación del juego:
# - Estado global de puestos
# - Funciones de gestión de recursos y correo
# - Tasks periódicas (balanceo y logging)

import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from random import choice, randint, sample
from uuid import uuid4

from butler.models import Carta, InfoPuesto, Inventario
from butler.settings import settings

# ==== Estado global ====

# Mapa de IP a info del puesto
puestos: defaultdict[str, InfoPuesto] = defaultdict(lambda: crear_puesto())


# ==== Persistencia ====


def save():
    """Persiste el estado de los puestos a JSON (si persistencia está configurado)."""
    if not getattr(settings.server, "persistencia", None):
        return
    path = Path(settings.server.persistencia)
    data = {ip: puesto.model_dump() for ip, puesto in puestos.items()}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load():
    """Carga el estado de los puestos desde JSON, si existe y persistencia está configurado."""
    if not getattr(settings.server, "persistencia", None):
        return
    path = Path(settings.server.persistencia)
    if not path.exists():
        return
    data = json.loads(path.read_text())
    for ip, puesto_data in data.items():
        puestos[ip] = InfoPuesto.model_validate(puesto_data)


# ==== Funciones auxiliares ====

nombres_recursos = set(r.nombre for r in settings.recursos)


def generar_alias() -> str:
    """Genera un nuevo alias aleatorio único"""
    alias = None
    while alias is None or buscar_alias(alias):
        nombre = choice(settings.simulacion.nombres)
        adjetivo = choice(settings.simulacion.adjetivos)
        alias = f"{nombre} {adjetivo}"
    return alias


def crear_puesto() -> InfoPuesto:
    """Inicializa un nuevo InfoPuesto aleatoriamente"""
    puesto = InfoPuesto()

    if settings.server.buzon:
        puesto.Buzon = {}

    puesto.Alias = generar_alias()

    # Objetivos iniciales aleatorios
    for r in sample(list(nombres_recursos), settings.simulacion.num_objetivos):
        puesto.Objetivo[r] = randint(
            settings.simulacion.min_objetivo, settings.simulacion.max_objetivo
        )

    # Asignamos (posiblemente) un poco de uno de los objetivos
    puesto.Recursos[choice(list(puesto.Objetivo.keys()))] = randint(
        settings.simulacion.min_cantidad, settings.simulacion.max_cantidad
    )

    # Y otros recursos no-objetivo
    for r in sample(
        list(nombres_recursos - set(puesto.Objetivo.keys())),
        settings.simulacion.num_recursos_extra,
    ):
        puesto.Recursos[r] = randint(
            settings.simulacion.min_cantidad, settings.simulacion.max_cantidad
        )

    # Damos oro en funcion de los recursos recibidos
    total_recursos = sum(puesto.Recursos.values())
    puesto.Recursos["oro"] = settings.simulacion.oro_inicial - total_recursos

    return puesto


def buscar_alias(nombre: str) -> str | None:
    """Busca un alias entre todos los puestos y retorna su IP"""
    for ip, puesto in puestos.items():
        if nombre == puesto.Alias:
            return ip
    return None


def entregar_carta(carta: Carta, ip_destino: str | None = None) -> str:
    """
    Entrega una carta al buzón de un puesto.
    Si ip_destino es None, busca el destinatario por alias en carta.dest.
    Si ip_destino está presente, sobrescribe carta.dest con el primer alias del puesto.
    Retorna el ID de la carta entregada.
    """
    if ip_destino is None:
        ip_destino = buscar_alias(carta.dest)
    else:
        carta.dest = puestos[ip_destino].Alias
    if ip_destino is None:
        raise KeyError
    carta.id = str(uuid4())
    carta.fecha = datetime.now().isoformat()
    puestos[ip_destino].Buzon[carta.id] = carta
    return carta.id


def entregar_paquete(
    paquete: Inventario, ip_origen: str | None, ip_destino: str
) -> None:
    """
    Transfiere recursos de un puesto a otro.
    Si ip_origen es None, los recursos vienen del Sistema (no se restan de ningún puesto).
    Si ip_origen está presente, se restan del puesto origen.
    Lanza ValueError si no hay suficientes recursos en el origen.
    Envía una carta al destinatario notificando la recepción.
    """
    if ip_origen is not None:
        # Restar recursos del origen (lanza ValueError si no hay suficientes)
        puestos[ip_origen].Recursos -= paquete
        asunto = "Paquete recibido"
    else:
        asunto = "Recursos generados"

    puestos[ip_destino].Recursos += paquete

    if settings.server.buzon:
        entregar_carta(
            Carta(
                remi=settings.simulacion.remitente_sistema,
                asunto=asunto,
                cuerpo=f"Has recibido:\n{paquete}",
            ),
            ip_destino,
        )
    # TODO: Considerar alternativas cuando no hay buzón (logging, eventos, etc.)


def calcular_porcentaje_objetivo(puesto: InfoPuesto) -> float:
    """
    Calcula el porcentaje de recursos objetivos conseguidos.
    Retorna un valor entre 0 y 100.
    """
    if not puesto.Objetivo:
        return 0.0

    total_conseguido = 0
    total_objetivo = 0

    for recurso, cantidad_objetivo in puesto.Objetivo.items():
        cantidad_actual = puesto.Recursos[recurso]
        total_conseguido += min(cantidad_actual, cantidad_objetivo)
        total_objetivo += cantidad_objetivo

    return (total_conseguido / total_objetivo * 100) if total_objetivo > 0 else 0


# ==== Tasks periódicas ====


async def log_progreso():
    """Task periódica que imprime el progreso de cada puesto"""
    while True:
        await asyncio.sleep(settings.server.log_interval)

        if not puestos:
            print("No hay puestos registrados todavía")
            continue

        print("\n" + "=" * 60)
        print("PROGRESO DE PUESTOS")
        print("=" * 60)

        for ip, puesto in puestos.items():
            porcentaje = calcular_porcentaje_objetivo(puesto)
            print(f"{puesto.Alias}: {porcentaje:.1f}%")

        # Calcular totales de recursos (excluyendo oro)
        total_disponible: dict[str, int] = defaultdict(int)
        total_necesitado: dict[str, int] = defaultdict(int)
        total_conseguido: dict[str, int] = defaultdict(int)
        oros: list[int] = []

        for puesto in puestos.values():
            oros.append(puesto.Recursos["oro"])
            for recurso, cantidad in puesto.Recursos.items():
                if recurso != "oro":
                    total_disponible[recurso] += cantidad
            for recurso, objetivo in puesto.Objetivo.items():
                total_necesitado[recurso] += objetivo
                total_conseguido[recurso] += min(puesto.Recursos[recurso], objetivo)

        # Mostrar recursos
        print("-" * 60)
        print("RECURSOS (disponible / conseguido / necesitado)")
        for recurso in sorted(set(total_disponible) | set(total_necesitado)):
            disp = total_disponible[recurso]
            cons = total_conseguido[recurso]
            nece = total_necesitado[recurso]
            print(f"  {recurso}: {disp} / {cons} / {nece}")

        # Mostrar oro
        if oros:
            print("-" * 60)
            print(
                f"ORO: total={sum(oros)}, min={min(oros)}, media={sum(oros) / len(oros):.1f}, max={max(oros)}"
            )

        print("=" * 60 + "\n")


async def balancear_recursos():
    """Task periódica que verifica y balancea recursos en el sistema"""
    while True:
        await asyncio.sleep(settings.simulacion.tick)

        # Calcular total de recursos actuales y objetivos por tipo
        total_recursos = defaultdict(int)
        total_objetivos = defaultdict(int)

        for puesto in puestos.values():
            for recurso, cantidad in puesto.Recursos.items():
                total_recursos[recurso] += cantidad
            for recurso, cantidad in puesto.Objetivo.items():
                total_objetivos[recurso] += cantidad

        # Verificar cada recurso con margen
        recursos_faltantes = {}
        for recurso, objetivo in total_objetivos.items():
            objetivo_con_margen = objetivo * settings.simulacion.margen
            actual = total_recursos.get(recurso, 0)
            if actual < objetivo_con_margen:
                faltante = objetivo_con_margen - actual
                recursos_faltantes[recurso] = int(
                    faltante * settings.simulacion.balanceo_porcentaje
                )

        # Regalar recursos faltantes agrupados por destinatario
        if recursos_faltantes and puestos:
            ips_puestos = list(puestos.keys())
            # Acumular recursos por destinatario usando Inventario
            paquetes: dict[str, Inventario] = defaultdict(Inventario)

            for recurso, cantidad in recursos_faltantes.items():
                # Repartir la cantidad entre puestos aleatorios
                for _ in range(cantidad):
                    ip_afortunado = choice(ips_puestos)
                    paquetes[ip_afortunado][recurso] += 1

            # Enviar un paquete por destinatario (ip_origen=None indica que viene del Sistema)
            for ip_destino, paquete in paquetes.items():
                entregar_paquete(paquete, ip_origen=None, ip_destino=ip_destino)
