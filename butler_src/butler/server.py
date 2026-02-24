# === PLN BUTLER - SERVIDOR ===
#
# Servidor FastAPI con endpoints para:
# - Gestión de alias
# - Envío y recepción de cartas
# - Transferencia de paquetes (recursos)
# - Consulta de información del puesto

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from butler.models import Carta, InfoPuesto, Inventario
from butler.settings import settings
from butler.simulacion import (
    balancear_recursos,
    buscar_alias,
    calcular_porcentaje_objetivo,
    entregar_carta,
    entregar_paquete,
    load,
    log_progreso,
    puestos,
    save,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida del servidor: arranque y parada"""
    # Startup - cargar estado persistido si existe
    load()
    # Iniciar tasks de balanceo y log
    task_balanceo = asyncio.create_task(balancear_recursos())
    task_log = asyncio.create_task(log_progreso())
    yield
    # Shutdown - guardar estado y cancelar tasks
    save()
    task_balanceo.cancel()
    task_log.cancel()
    await asyncio.gather(task_balanceo, task_log, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


def get_puesto_id(request: Request, agente: str | None) -> str:
    """
    Retorna el identificador del puesto según el modo configurado.

    - Modo multipuesto: usa la IP del cliente
    - Modo monopuesto: usa el parámetro 'agente' (obligatorio)
    """
    if settings.server.modo_monopuesto:
        if not agente:
            raise HTTPException(
                status_code=400,
                detail="En modo monopuesto, el parámetro 'agente' es obligatorio",
            )
        return agente
    else:
        # Modo multipuesto: usar IP (ignorar 'agente' si se proporciona)
        return request.client.host if request.client else "unknown"


# ==== Endpoints ====

q_agente = Query(
    None,
    description="nombre del agente en modo monopuesto",
    include_in_schema=settings.server.modo_monopuesto,
)


@app.get("/info")
async def get_info(request: Request, agente: str | None = q_agente) -> InfoPuesto:
    """Retorna la información del puesto del cliente"""
    puesto_id = get_puesto_id(request, agente)
    return puestos[puesto_id]


@app.post("/alias/{nombre}")
async def set_alias(
    nombre: str, request: Request, agente: str | None = q_agente
) -> dict:
    """Establece el alias del puesto del cliente"""
    puesto_id = get_puesto_id(request, agente)

    if buscar_alias(nombre):
        raise HTTPException(status_code=403, detail="No se puede repetir alias")

    puestos[puesto_id].Alias = nombre
    return {"status": "ok", "alias": nombre}


if settings.server.buzon:

    @app.post("/carta")
    async def send_carta(carta: Carta, request: Request) -> dict:
        """Envía una carta a un destinatario"""
        try:
            carta_id = entregar_carta(carta)
            return {"status": "ok", "id": carta_id}
        except (KeyError, TypeError):
            raise HTTPException(status_code=404, detail="Destinatario no encontrado")

    @app.delete("/mail/{uid}")
    async def delete_mail(
        uid: str, request: Request, agente: str | None = q_agente
    ) -> dict:
        """Elimina un correo del buzón del cliente"""
        puesto_id = get_puesto_id(request, agente)

        if uid in puestos[puesto_id].Buzon:
            del puestos[puesto_id].Buzon[uid]
            return {"status": "ok"}
        else:
            raise HTTPException(status_code=404, detail="Mail no encontrado")


@app.get("/gente")
async def get_gente() -> list[dict]:
    """Retorna la lista de todos los alias disponibles"""
    if settings.server.modo_monopuesto:
        return [{"alias": puesto.Alias} for puesto in puestos.values()]
    else:
        return [{"alias": puesto.Alias, "ip": ip} for ip, puesto in puestos.items()]


@app.post("/paquete/{dest}")
async def send_paquete(
    paquete: Inventario, dest: str, request: Request, agente: str | None = q_agente
) -> dict:
    """Envía un paquete de recursos a un destinatario"""
    puesto_id = get_puesto_id(request, agente)
    if (dest_id := buscar_alias(dest)) is None:
        raise HTTPException(status_code=404, detail="Destinatario no encontrado")
    try:
        entregar_paquete(paquete, puesto_id, dest_id)
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    """Dashboard visual para proyector con estado de los puestos"""
    from collections import defaultdict

    # Paleta de colores distinguibles
    COLORES = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#9b59b6",
        "#f39c12",
        "#1abc9c",
        "#e91e63",
        "#00bcd4",
        "#8bc34a",
        "#ff5722",
        "#673ab7",
        "#009688",
        "#ffeb3b",
        "#795548",
        "#607d8b",
    ]

    def color_para_alias(alias: str) -> str:
        """Asigna un color consistente basado en el hash del alias"""
        return COLORES[hash(alias) % len(COLORES)]

    def texto_contraste(color_hex: str) -> str:
        """Retorna '#000' o '#fff' según la luminosidad del color"""
        r, g, b = (
            int(color_hex[1:3], 16),
            int(color_hex[3:5], 16),
            int(color_hex[5:7], 16),
        )
        luminosidad = 0.299 * r + 0.587 * g + 0.114 * b
        return "#000" if luminosidad > 140 else "#fff"

    # Calcular progreso de cada puesto
    progreso = [
        (puesto.Alias, calcular_porcentaje_objetivo(puesto))
        for puesto in puestos.values()
    ]

    # Agrupar por posición en el camino (redondeado a múltiplos de 5)
    posiciones: dict[int, list[str]] = defaultdict(list)
    for alias, pct in progreso:
        pos = int(pct // 5) * 5
        posiciones[pos].append(alias)

    # Calcular estadísticas de recursos
    total_disponible: dict[str, int] = defaultdict(int)
    total_conseguido: dict[str, int] = defaultdict(int)
    total_oro = 0

    for puesto in puestos.values():
        total_oro += puesto.Recursos["oro"]
        for recurso, cantidad in puesto.Recursos.items():
            if recurso != "oro":
                total_disponible[recurso] += cantidad
        for recurso, objetivo in puesto.Objetivo.items():
            total_conseguido[recurso] += min(puesto.Recursos[recurso], objetivo)

    # Generar HTML de recursos (solo libres = disponible - conseguido)
    recursos_html = " ".join(
        f'<span class="recurso"><b>{recurso}</b> {total_disponible[recurso] - total_conseguido[recurso]}</span>'
        for recurso in sorted(total_disponible.keys())
    )

    # Generar marcadores del camino (solo primera palabra del alias, con color)
    marcadores_html = ""
    for pos in range(0, 105, 5):
        nombres = posiciones.get(pos, [])
        columna = "".join(
            f'<div class="nombre" style="background:{(c := color_para_alias(n))};color:{texto_contraste(c)}">{n.split()[0]}</div>'
            for n in nombres
        )
        clase_tick = "tick-major" if pos % 25 == 0 else "tick"
        label = f'<span class="tick-label">{pos}</span>' if pos % 25 == 0 else ""
        marcadores_html += f'''
        <div class="marcador" style="left:{pos}%">
            <div class="nombres-col">{columna}</div>
            <div class="{clase_tick}"></div>
            {label}
        </div>'''

    # Lista de gente (con color)
    gente_html = " ".join(
        f'<span class="persona" style="background:{(c := color_para_alias(p.Alias))};color:{texto_contraste(c)}">{p.Alias}</span>'
        for p in puestos.values()
    )

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{settings.server.dashboard_refresh}">
<title>Butler Dashboard</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    font-family: system-ui, sans-serif;
    background: #111;
    color: #fff;
    min-height: 100vh;
    padding: 2rem 3rem;
}}

h1 {{
    text-align: center;
    font-size: 3rem;
    font-weight: 300;
    margin-bottom: 1.5rem;
    color: #fff;
}}

h1 .pre {{
    color: #888;
    font-weight: 400;
}}

h1 .name {{
    color: #3498db;
    font-weight: 600;
}}

.section {{
    margin-bottom: 2rem;
}}

.section-title {{
    font-size: 1.2rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #888;
    margin-bottom: 0.8rem;
}}

/* Gente */
.gente {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
    justify-content: center;
}}

.persona {{
    font-size: 1.6rem;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-weight: 500;
}}

/* Camino */
.camino-wrap {{
    background: #1a1a1a;
    border-radius: 8px;
    padding: 1.5rem 2rem 1rem;
}}

.camino {{
    position: relative;
    height: 180px;
    margin: 0 2rem;
}}

.camino-linea {{
    position: absolute;
    bottom: 50px;
    left: 0;
    right: 0;
    height: 12px;
    background: linear-gradient(90deg, #e74c3c 0%, #f1c40f 50%, #2ecc71 100%);
    border-radius: 6px;
}}

.marcador {{
    position: absolute;
    bottom: 0;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.nombres-col {{
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 10px;
    gap: 4px;
}}

.nombre {{
    font-size: 1.3rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    white-space: nowrap;
}}

.tick {{
    width: 2px;
    height: 15px;
    background: #444;
    margin-top: 6px;
}}

.tick-major {{
    width: 4px;
    height: 25px;
    background: #fff;
    margin-top: 6px;
}}

.tick-label {{
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    margin-top: 6px;
}}

/* Footer: recursos y oro */
.footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 2rem;
}}

.recursos {{
    display: flex;
    flex-wrap: wrap;
    gap: 2.5rem;
}}

.recurso {{
    font-size: 2.5rem;
    color: #ccc;
}}

.recurso b {{
    color: #f1c40f;
    margin-right: 0.3rem;
}}

.oro {{
    font-size: 2.5rem;
    color: #f39c12;
    font-weight: 700;
}}

.empty {{
    color: #555;
    font-size: 1.5rem;
    text-align: center;
    padding: 2rem;
}}
</style>
</head><body>

<h1><span class="pre">FDI-PLN:</span> <span class="name">Butler</span></h1>

<div class="section">
    <div class="section-title">Participantes</div>
    <div class="gente">
        {gente_html if puestos else '<span class="empty">Esperando conexiones...</span>'}
    </div>
</div>

<div class="section">
    <div class="section-title">Progreso</div>
    <div class="camino-wrap">
        <div class="camino">
            <div class="camino-linea"></div>
            {marcadores_html}
        </div>
    </div>
</div>

<div class="section">
    <div class="section-title">Recursos libres</div>
    <div class="footer">
        <div class="recursos">
            {recursos_html if recursos_html else '<span class="empty">—</span>'}
        </div>
        <div class="oro">⬡ {total_oro}</div>
    </div>
</div>

</body></html>"""


# ==== Punto de entrada ====


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.server.host, port=settings.server.port)
