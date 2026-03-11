import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Input, Static


def normalizar(texto: str) -> str:
    """
    Normaliza texto eliminando tildes y pasando a minúsculas.
    """
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


def cargar_json(ruta: str) -> Any:
    """
    Carga un archivo JSON desde disco.
    """
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


class Resultado(Static):
    """
    Widget que representa un resultado de búsqueda.
    """

    def __init__(self, parrafo: dict[str, Any]) -> None:
        """
        Inicializa el resultado con el contenido del párrafo.
        """
        super().__init__()
        self.parrafo = parrafo

    def render(self) -> str:
        """
        Renderiza el párrafo y sus metadatos.
        """
        texto = self.parrafo.get("text", self.parrafo.get("texto", ""))
        parrafo_id = self.parrafo.get("index", self.parrafo.get("id"))
        headings = self.parrafo.get("headings")
        if isinstance(headings, list) and headings:
            meta_str = " > ".join(str(h) for h in headings)
        else:
            meta = self.parrafo.get("metadatos", {})
            meta_str = " | ".join(f"{k}:{v}" for k, v in meta.items()) if meta else ""

        return f"[b cyan]Párrafo {parrafo_id}[/]\n[dim]{meta_str}[/]\n\n{texto}"


class Buscador(App):
    """
    Aplicación Textual que busca palabras en un índice invertido.
    """

    CSS = """
    Screen { layout: vertical; }

    #busqueda { margin: 1; }

    #resultados {
        margin: 1;
        border: round #666;
        padding: 1;
    }

    Resultado {
        margin-bottom: 1;
        border: round #444;
        padding: 1;
    }
    """

    def __init__(self, indice_path: str, parrafos_path: str) -> None:
        """
        Inicializa la app cargando índice y párrafos.
        """
        super().__init__()

        self.indice: dict[str, list[int]] = cargar_json(indice_path)
        lista_parrafos: list[dict[str, Any]] = cargar_json(parrafos_path)
        self.parrafos: dict[int, dict[str, Any]] = {}
        for p in lista_parrafos:
            pid = p.get("index", p.get("id"))
            if pid is None:
                continue
            self.parrafos[int(pid)] = p

    def compose(self) -> ComposeResult:
        """
        Construye la interfaz.
        """
        yield Header()
        yield Input(placeholder="Buscar palabra...", id="busqueda")
        yield Static("Introduce un término.", id="estado")
        yield VerticalScroll(id="resultados")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Ejecuta la búsqueda cuando el usuario pulsa Enter.
        """
        termino = normalizar(event.value.strip())

        estado = self.query_one("#estado", Static)
        contenedor = self.query_one("#resultados", VerticalScroll)

        contenedor.remove_children()

        if not termino:
            estado.update("Escribe algo.")
            return

        ids = self.indice.get(termino)

        if not ids:
            estado.update("No se encontraron resultados.")
            return

        estado.update(f"{len(ids)} resultado(s). Índices: {', '.join(str(i) for i in ids)}")

        for pid in ids[:2]:
            parrafo = self.parrafos.get(pid)
            if parrafo:
                contenedor.mount(Resultado(parrafo))


def main() -> None:
    """
    Punto de entrada del buscador.
    """
    if len(sys.argv) != 3:
        print("Uso: python buscador.py indice.json parrafos.json")
        sys.exit(1)

    indice = Path(sys.argv[1])
    parrafos = Path(sys.argv[2])

    app = Buscador(str(indice), str(parrafos))
    app.run()


if __name__ == "__main__":
    main()
