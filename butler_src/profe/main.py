import asyncio
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from operator import itemgetter
from time import time

import httpx
import typer
from loguru import logger

from butler.models import Carta, Inventario

from .llm import nueva_carta, responder_carta

BUTLER_ADDRESS = os.getenv("FDI_PLN__BUTLER_ADDRESS", "127.0.0.1:7719")


@dataclass
class AgentState:
    alias: str

    sobra: Inventario = field(default_factory=Inventario)
    falta: Inventario = field(default_factory=Inventario)
    entrada: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=10))

    otra_gente: list[str] = field(default_factory=list)
    convs: defaultdict[str, list] = field(default_factory=lambda: defaultdict(list))
    visto_ultima_vez: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    client: httpx.AsyncClient = field(default_factory=httpx.AsyncClient)

    async def get(self, endpoint):
        return await self.client.get(
            f"http://{BUTLER_ADDRESS}/{endpoint}", params={"agente": self.alias}
        )

    async def post(self, endpoint, datum={}, **kwargs):
        try:
            data = datum.dict()
        except AttributeError:
            data = datum
        data.update(kwargs)
        return await self.client.post(
            f"http://{BUTLER_ADDRESS}/{endpoint}",
            params={"agente": self.alias},
            json=data,
        )

    async def delete(self, endpoint):
        return await self.client.delete(
            f"http://{BUTLER_ADDRESS}/{endpoint}", params={"agente": self.alias}
        )

    def __repr__(self):
        return f"Agente {self.alias}\nSobra: {self.sobra}\nFalta: {self.falta}"


async def registrar_en_butler(agente):
    r = await agente.post("alias/" + agente.alias)
    if r.status_code == 200:
        logger.info("Registrado en butler")
        return
    if r.status_code == 403:
        logger.info("Retomando conexion con butler")
        return
    logger.debug(r.text)
    raise httpx.NetworkError


# interval in seconds
async def tarea_infinita(coro, interval, jitter, *args, **kwargs):
    while True:
        try:
            next_run = time() + interval + random.uniform(-jitter, jitter)
            await coro(*args, **kwargs)
            await asyncio.sleep(next_run - time())
        except asyncio.CancelledError:
            break
        except httpx.RemoteProtocolError:
            pass  # butler a veces congestionado


async def informarse(agente):
    logger.debug("Obteniendo info de butler")
    gentes = await agente.get("gente")
    agente.otra_gente = [
        g["alias"] for g in gentes.json() if g["alias"] != agente.alias
    ]
    r = await agente.get("info")
    data = r.json()
    logger.debug(data)
    recursos = Inventario._from_dict(data["Recursos"])
    objetivo = Inventario._from_dict(data["Objetivo"])
    agente.sobra = recursos << objetivo
    agente.falta = objetivo << recursos
    logger.info(agente)
    for c in sorted(data["Buzon"].values(), key=itemgetter("fecha")):
        carta = Carta(**c)
        logger.trace(carta)
        await agente.entrada.put(carta)
        await agente.delete("mail/" + carta.id)


async def leer_carta(agente):
    carta = await agente.entrada.get()
    logger.debug(carta)
    if carta.remi == "Sistema":  # descartamos notificaciones
        return
    agente.visto_ultima_vez[carta.remi] = time()
    conversacion = agente.convs[carta.remi]
    if len(conversacion) > 5:
        conversacion.pop(0)
    await asyncio.sleep(random.uniform(1, 3))
    if mensaje := await responder_carta(agente, carta, conversacion):
        respuesta = Carta(
            remi=agente.alias,
            dest=carta.remi,
            asunto=carta.asunto
            if carta.asunto.startswith("RE:")
            else f"RE: {carta.asunto}",
            cuerpo=mensaje,
        )
        fmt = respuesta.formatear()
        logger.info(fmt)
        await agente.post("carta", respuesta)


async def abrir_conversacion(agente):
    if agente.entrada.qsize() > 0 or len(agente.otra_gente) == 0:
        return
    quien = random.choice(agente.otra_gente)
    # anti-spam
    conversacion = agente.convs[quien]
    if len(conversacion) > 0:
        if (time() - agente.visto_ultima_vez[quien]) > 15:
            conversacion[:] = []
        return
    if mensaje := await nueva_carta(agente, quien):
        carta = Carta(remi=agente.alias, dest=quien, asunto="Hola!", cuerpo=mensaje)
        fmt = carta.formatear()
        conversacion.append(fmt)
        logger.info(fmt)
        await agente.post("carta", carta)


async def agent_loop(agente):
    try:
        logger.info("Conectando con butler...")
        await registrar_en_butler(agente)
        logger.info("Iniciando tareas de trabajo")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(tarea_infinita(informarse, 6, 1, agente))
            tg.create_task(tarea_infinita(leer_carta, 3, 2, agente))
            tg.create_task(tarea_infinita(abrir_conversacion, 15, 10, agente))
    except* httpx.NetworkError:
        logger.debug("Imposible conectar con butler, reintentando")


def main(alias: str, log_level: str = typer.Option("INFO", help="Nivel de log")):
    logger.remove()
    logger.add(typer.get_text_stream("stderr"), level=log_level.upper())
    logger.info(f"Iniciando agente {alias}")
    agente = AgentState(alias=alias)
    try:
        asyncio.run(tarea_infinita(agent_loop, 5, 0, agente))
    except* KeyboardInterrupt:
        logger.info("Recibido Ctrl-C")
    logger.info(f"Finalizando agente {alias}")


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
