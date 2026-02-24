import os
from string import Template

from loguru import logger
from ollama import AsyncClient
from pydantic import BaseModel, Field

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = "ministral-3:8b"

ollama = AsyncClient(host=OLLAMA_HOST)

### PROMPTS

sistema = Template("""Eres un agente comercial para negocios de materias básicas.
Tu nombre es ${alias}.
Tu objetivo es intercambiar materiales con otros agentes,
para llegar a conseguir los materiales necesarios.
Tu manera de interactuar es mediante cartas.
Envías cartas a otros agentes,
y respondes a las cartas que te envían.
Eres generoso, siempre estás dispuesto a hacer un trato.
Usas un lenguaje claro, sencillo, directo y no muy formal.

NO explicas por qué dices o respondes con una carta concreta,
simplemente lo haces. Eres terso y sintético.
Sólo haces los tratos de uno en uno.
Cuando un trato es aceptable, lo confirmas
pero no vuelves a preguntar.
Una vez finalizada una conversación, no respondes más.

En general, intercambias recursos 1 a 1, es decir,
siempre estás dispuesto a dar 1 madera por 1 piedra,
a dar 1 oro por cualquier recurso,
o a dar cualquier recurso a cambio de 1 oro.
Haces los negocios de un recurso por vez,
no negocias varios recursos a la vez.
Eres generoso, y envías los recursos en cuanto 
ambas partes están de acuerdo en qué se cambia,
cuánto se da y cuanto se recibe.
NO envías recursos hasta que el trato está hecho.
Buscas con interés recibir los recursos que te faltan.
Si te envían recursos por un trato y tú no has enviado aún,
envías tu parte sin problema.
NUNCA pides que te envíen de vuelta, confías en los demás.
En este mundo los recursos siempre se reciben inmediatamente,
puedes asumir que se han recibido.
NUNCA envias recursos que te hacen falta.
Para enviar recursos **es obligatorio** usar la herramienta `acuerdo`.

Si tienes más de 10 de oro,
ofreces y aceptas recursos por 2 de oro cada uno.

Si no te falta nada, entregas los recursos a cambio de oro.

Tienes los siguientes recursos disponibles:
---
${sobra}
---

Necesitas conseguir los siguientes:
---
${falta}
---
""")

respuesta = Template("""
Si la conversación ha llegado a su fin,
por ejemplo porque no hay nada que responder,
el interlocutor es grosero,
o la carta recibida simplemente es una confirmación,
usa la herramienta `noresponder`.
Si se ha alcanzado un acuerdo,
o vas a enviar recursos,
usa la herramienta `acuerdo`.
""")

hola = Template("""Escribe un mensaje a ${quien} para iniciar una conversación: """)


### TOOLS

herramientas = {}


def make_tool(model: type[BaseModel]) -> dict:
    nombre = model.__name__.lower()
    herramientas[nombre] = model
    return {
        "type": "function",
        "function": {
            "name": nombre,
            "description": model.__doc__,
            "parameters": model.model_json_schema(),
        },
    }


class NoResponder(BaseModel):
    """Da por finalizada una conversación"""

    motivo: str = Field(description="Motivo por el que se ha acabado la conversación")


tool_parar = make_tool(NoResponder)


class Acuerdo(BaseModel):
    """Finaliza un acuerdo entre las partes, estableciendo el resultado de la negociación"""

    comentario: str = Field(description="Mensaje para el destinatario del trato")
    envio_primero: bool = Field(
        description="true si es el primer envio, false si el destinatario ya ha enviado sus recursos"
    )
    envio_recurso: str = Field(description="Nombre del recurso a enviar")
    envio_numero: int = Field(description="Cantidad del recurso a enviar")
    recibo_recurso: str = Field(description="Nombre del recurso a recibir")
    recibo_numero: int = Field(description="Cantidad del recurso a recibir")


tool_acuerdo = make_tool(Acuerdo)


async def chat(agente, mensaje, datos, contexto=[], *tools):
    msgs = [
        {
            "role": "system",
            "content": sistema.substitute(
                {"alias": agente.alias, "sobra": agente.sobra, "falta": agente.falta}
            ),
        },
        *[{"role": "user", "content": c} for c in contexto],
        {
            "role": "user",
            "content": mensaje.substitute(datos),
        },
    ]
    response = await ollama.chat(model=OLLAMA_MODEL, messages=msgs, tools=tools)
    logger.trace(response)
    if tools := response.message.tool_calls:
        return [
            herramientas[t.function.name].model_validate(t.function.arguments)
            for t in tools
        ]
    return response.message.content


async def responder_carta(agente, carta, conversacion):
    conversacion.append(carta.formatear())
    match r := await chat(
        agente,
        respuesta,
        {},
        conversacion,
        tool_parar,
        tool_acuerdo,
    ):
        case [Acuerdo() as t, *_]:
            conversacion[:] = []
            try:
                agente.sobra[t.envio_recurso] -= t.envio_numero
                await agente.post(
                    "paquete/" + carta.remi, {t.envio_recurso: t.envio_numero}
                )
                logger.warning(
                    f"Enviado {t.envio_numero} de {t.envio_recurso} a {carta.remi}"
                )
            except ValueError:
                return None
            return t.comentario if t.envio_primero else None
        case [NoResponder(), *_]:
            conversacion[:] = []
            return None
        case [*_]:
            logger.error(r"Recibida una respuesta inesperada del llm: {r}")
            return None
        case _:
            conversacion.append(r)
            return r


async def nueva_carta(agente, quien):
    return await chat(agente, hola, {"quien": quien})
