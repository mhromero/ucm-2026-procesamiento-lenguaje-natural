# === PLN BUTLER - CLI ===
#
# Interfaz de línea de comandos usando click

import click


@click.group()
def cli():
    """Butler - Servidor de coordinación para prácticas de PLN"""
    pass


@cli.command()
@click.option("--host", "-h", default=None, help="Host del servidor")
@click.option("--port", "-p", default=None, type=int, help="Puerto del servidor")
@click.option(
    "--monopuesto/--multipuesto",
    default=None,
    help="Modo monopuesto (identifica puestos por parámetro agente)",
)
@click.option(
    "--log-interval", default=None, type=int, help="Segundos entre logs de progreso"
)
@click.option(
    "--dashboard-refresh",
    default=None,
    type=int,
    help="Segundos entre refrescos del dashboard",
)
@click.option(
    "--buzon/--no-buzon",
    default=None,
    help="Habilitar sistema de buzón para cartas",
)
@click.option(
    "--persistencia",
    default=None,
    type=click.Path(),
    help="Ruta del archivo JSON para persistir/cargar estado de la simulación",
)
def server(host, port, monopuesto, log_interval, dashboard_refresh, buzon, persistencia):
    """Lanza el servidor Butler"""
    import uvicorn

    from butler.settings import settings

    # Sobrescribir settings con opciones CLI si se proporcionan
    if monopuesto is not None:
        settings.set("server.modo_monopuesto", monopuesto)
    if log_interval is not None:
        settings.set("server.log_interval", log_interval)
    if dashboard_refresh is not None:
        settings.set("server.dashboard_refresh", dashboard_refresh)
    if buzon is not None:
        settings.set("server.buzon", buzon)
    if persistencia is not None:
        settings.set("server.persistencia", persistencia)

    from butler.server import app

    uvicorn.run(
        app,
        host=host or settings.server.host,
        port=port or settings.server.port,
    )


@cli.command()
def config():
    """Imprime la configuración actual en formato TOML"""
    from butler.settings import USER_CONFIG_DIR, USER_CONFIG_FILE, settings

    # Comentario con paths de configuración
    print(f"""\
# Configuración de Butler
# Para sobrescribir estos valores, copia las líneas que quieras cambiar a:
#   - {USER_CONFIG_FILE}
#   - config.local.toml (en el directorio de trabajo)
# También puedes usar variables de entorno con prefijo NLP_BUTLER_
# (ej: NLP_BUTLER_SERVER__PORT=8080)
""")

    # Imprimir settings en formato TOML
    _print_toml(settings.as_dict())


def _print_toml(d: dict, prefix: str = "") -> None:
    """Imprime un diccionario en formato TOML"""
    for key, value in d.items():
        match value:
            case dict():
                print(f"[{prefix}{key}]")
                for k, v in value.items():
                    if not isinstance(v, (dict, list)):
                        print(f"{k} = {_toml_value(v)}")
                print()
                # Recursivo para subdicts anidados
                for k, v in value.items():
                    if isinstance(v, dict):
                        _print_toml({k: v}, prefix=f"{prefix}{key}.")
                    elif isinstance(v, list) and v and isinstance(v[0], dict):
                        for item in v:
                            print(f"[[{prefix}{key}.{k}]]")
                            for ik, iv in item.items():
                                print(f"{ik} = {_toml_value(iv)}")
                            print()
            case list() if value and isinstance(value[0], dict):
                for item in value:
                    print(f"[[{prefix}{key}]]")
                    for k, v in item.items():
                        print(f"{k} = {_toml_value(v)}")
                    print()
            case _:
                print(f"{key} = {_toml_value(value)}")


def _toml_value(v) -> str:
    """Convierte un valor Python a formato TOML"""
    match v:
        case str():
            return f'"{v}"'
        case bool():
            return "true" if v else "false"
        case list():
            return "[" + ", ".join(_toml_value(i) for i in v) + "]"
        case _:
            return str(v)


if __name__ == "__main__":
    cli()
