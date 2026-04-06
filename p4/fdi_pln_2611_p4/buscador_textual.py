from __future__ import annotations

from typing import Any, Literal

import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, Footer, Header, Input, Static

from .busqueda_clasica import buscar_frase, cargar_json, destacar
from .busqueda_rag import buscar_rag
from .busqueda_semantica import buscar_semantica, cargar_embeddings

ModoBusqueda = Literal["clasica", "semantica", "rag"]


class Buscador(App):
    """Aplicación TUI para búsqueda clásica, semántica y RAG."""

    RAG_MAX_LINEAS_PAGINA = 28
    TITLE = "Explorador del Quijote - Busqueda Clasica, Semantica y RAG"

    BINDINGS = [
        Binding("ctrl+1", "modo_clasica", "Clásica"),
        Binding("ctrl+2", "modo_semantica", "Semántica"),
        Binding("ctrl+3", "modo_rag", "RAG"),
        Binding("left,p", "anterior", "Anterior página"),
        Binding("right,n", "siguiente", "Siguiente página"),
        Binding("slash", "enfocar_busqueda", "Buscar"),
    ]

    CSS = """
    Screen { layout: vertical; }
    #barra_modo {
        margin: 1 1 0 1;
        height: auto;
        align-vertical: bottom;
    }
    #modo_actual {
        margin-left: 1;
        color: $text-muted;
        content-align: left bottom;
        height: 3;
        width: auto;
    }
    #barra_modo Button {
        margin-right: 1;
        min-width: 10;
        padding: 0 1;
    }
    #busqueda { margin: 1; }
    #estado { margin: 0 1; color: $text-muted; }
    #resultado {
        margin: 1;
        border: round #666;
        padding: 1;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(
        self,
        indice_path: str,
        parrafos_path: str,
        embeddings_path: str | None,
        embeddings_ids_path: str | None,
        rag_model: str,
        top_k_rag: int,
        top_k_semantica: int,
    ) -> None:
        """Inicializa índices, chunks, embeddings y estado de navegación."""
        super().__init__()
        self.indice: dict[str, list] = cargar_json(indice_path)
        lista_parrafos: list[dict[str, Any]] = cargar_json(parrafos_path)
        self.parrafos: dict[int, dict[str, Any]] = {
            int(p["index"]): p for p in lista_parrafos if "index" in p
        }

        self._embeddings_disponibles = embeddings_path is not None
        self._rag_model = rag_model
        self._top_k_rag = top_k_rag
        self._top_k_semantica = top_k_semantica
        if self._embeddings_disponibles:
            self.embeddings, self.embeddings_ids = cargar_embeddings(
                embeddings_path, embeddings_ids_path
            )
        else:
            self.embeddings = None
            self.embeddings_ids = None

        self._modo: ModoBusqueda = "clasica"
        self._resultados_clasica: list[tuple[int, float, int]] = []
        self._resultados_semantica: list[tuple[int, float]] = []
        self._claves: set[str] = set()
        self._pos: int = 0
        self._query: str = ""
        self._rag_paginas: list[str] = []
        self._rag_pos: int = 0

    def compose(self) -> ComposeResult:
        """Construye los widgets principales de la interfaz."""
        yield Header()
        with Horizontal(id="barra_modo"):
            yield Button("Clásica", id="btn-clasica", variant="primary")
            yield Button("Semántica", id="btn-semantica")
            yield Button("RAG", id="btn-rag")
            yield Static("", id="modo_actual")
        yield Input(placeholder="Buscar palabra o frase...", id="busqueda")
        yield Static("Realice una consulta.", id="estado")
        yield Static("", id="resultado")
        yield Footer()

    def _set_modo(self, modo: ModoBusqueda) -> None:
        """Cambia de modo y reinicia estado y resultados."""
        self._modo = modo
        self._resultados_clasica = []
        self._resultados_semantica = []
        self._claves = set()
        self._pos = 0
        self._query = ""
        self._rag_paginas = []
        self._rag_pos = 0
        self.query_one("#busqueda", Input).value = ""
        self._actualizar_botones_modo()
        self._actualizar_placeholder_busqueda()
        self._mostrar()

    def on_mount(self) -> None:
        """Sincroniza botones y placeholders al iniciar la app."""
        self._actualizar_botones_modo()
        self._actualizar_placeholder_busqueda()

    def _actualizar_placeholder_busqueda(self) -> None:
        """Adapta el placeholder según el modo de búsqueda actual."""
        input_busqueda = self.query_one("#busqueda", Input)
        if self._modo == "rag":
            input_busqueda.placeholder = (
                "Buscar palabra o frase o realice una pregunta..."
            )
        else:
            input_busqueda.placeholder = "Buscar palabra o frase..."

    def _actualizar_botones_modo(self) -> None:
        """Marca visualmente el modo activo en los botones."""
        btn_clasica = self.query_one("#btn-clasica", Button)
        btn_semantica = self.query_one("#btn-semantica", Button)
        btn_rag = self.query_one("#btn-rag", Button)
        modo_actual = self.query_one("#modo_actual", Static)
        btn_clasica.variant = "primary" if self._modo == "clasica" else "default"
        btn_semantica.variant = "primary" if self._modo == "semantica" else "default"
        btn_rag.variant = "primary" if self._modo == "rag" else "default"
        modo_actual.update(
            f"Modo actual: [b]{self._modo_label()}[/]  [dim](click en los botones o comando)[/]"
        )

    def action_modo_clasica(self) -> None:
        """Atajo para cambiar al modo clásico."""
        self._set_modo("clasica")

    def action_modo_semantica(self) -> None:
        """Atajo para cambiar al modo semántico."""
        self._set_modo("semantica")

    def action_modo_rag(self) -> None:
        """Atajo para cambiar al modo RAG."""
        self._set_modo("rag")

    def action_enfocar_busqueda(self) -> None:
        """Mueve el foco al input de búsqueda."""
        self.query_one("#busqueda", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Gestiona clicks en botones de selección de modo."""
        if event.button.id == "btn-clasica":
            self._set_modo("clasica")
        elif event.button.id == "btn-semantica":
            self._set_modo("semantica")
        elif event.button.id == "btn-rag":
            self._set_modo("rag")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Ejecuta la búsqueda al pulsar Enter según el modo activo."""
        self._query = event.value.strip()
        self._pos = 0
        self._rag_paginas = []
        self._rag_pos = 0
        self.query_one("#busqueda", Input).blur()

        if self._modo == "clasica":
            self._resultados_clasica, self._claves = buscar_frase(
                self._query, self.indice
            )
            self._resultados_semantica = []
        elif self._modo == "semantica":
            if not self._embeddings_disponibles:
                self._mostrar()
                return
            self._resultados_semantica = buscar_semantica(
                self._query,
                self.embeddings,
                self.embeddings_ids,
                top_k=self._top_k_semantica,
            )
            self._resultados_clasica = []
            self._claves = set()
        elif self._modo == "rag":
            if not self._embeddings_disponibles:
                self._mostrar()
                return
            self._resultados_clasica, self._claves = buscar_frase(
                self._query, self.indice
            )
            self._resultados_semantica = buscar_semantica(
                self._query,
                self.embeddings,
                self.embeddings_ids,
                top_k=self._top_k_semantica,
            )
            self._mostrar_rag_cargando()
            self._lanzar_rag(self._query)
            return

        self._mostrar()

    def _mostrar_rag_cargando(self) -> None:
        """Muestra estado de carga mientras se consulta al LLM."""
        estado = self.query_one("#estado", Static)
        contenedor = self.query_one("#resultado", Static)
        estado.update(f"Consultando al LLM ({self._rag_model})...")
        contenedor.update("[dim]Generando respuesta, por favor espera...[/]")

    def _paginar_texto(self, texto: str) -> list[str]:
        """Divide una respuesta larga en páginas de líneas fijas."""
        lineas = texto.splitlines()
        if not lineas:
            return [texto]
        paginas = []
        for i in range(0, len(lineas), self.RAG_MAX_LINEAS_PAGINA):
            paginas.append("\n".join(lineas[i : i + self.RAG_MAX_LINEAS_PAGINA]))
        return paginas or [texto]

    def _lanzar_rag(self, query: str) -> None:
        """Lanza la consulta RAG en un hilo para no bloquear la TUI."""

        def _worker() -> None:
            respuesta = buscar_rag(
                query,
                self.parrafos,
                self._resultados_clasica,
                self._resultados_semantica,
                self._rag_model,
                self._top_k_rag,
            )
            self.call_from_thread(self._mostrar_respuesta_rag, respuesta)

        threading.Thread(target=_worker, daemon=True).start()

    def _mostrar_respuesta_rag(self, respuesta: str) -> None:
        """Almacena la respuesta RAG paginada y la muestra desde la primera página."""
        self._rag_paginas = self._paginar_texto(respuesta)
        self._rag_pos = 0
        self._mostrar()

    def _n_paginas_rag(self) -> int:
        """Devuelve el número total de páginas de la respuesta RAG."""
        return len(self._rag_paginas)

    def _mostrar_pagina_rag(self) -> None:
        """Renderiza la página RAG actual con estado de navegación."""
        estado = self.query_one("#estado", Static)
        contenedor = self.query_one("#resultado", Static)
        if not self._rag_paginas:
            estado.update("Introduce una pregunta y pulsa Enter.")
            contenedor.update("[yellow]Introduce una pregunta y pulsa Enter.[/]")
            return
        n = self._n_paginas_rag()
        estado.update(
            f"Consulta: '{self._query}'  [dim]página: {self._rag_pos + 1}/{n}  (← → navegar)[/]"
        )
        contenedor.update(self._rag_paginas[self._rag_pos])

    def _n_resultados(self) -> int:
        """Devuelve el número de resultados del modo actual no-RAG."""
        if self._modo == "clasica":
            return len(self._resultados_clasica)
        if self._modo == "semantica":
            return len(self._resultados_semantica)
        return 0

    def action_siguiente(self) -> None:
        """Avanza a la siguiente página o resultado."""
        if self._modo == "rag":
            if self._rag_pos < self._n_paginas_rag() - 1:
                self._rag_pos += 1
                self._mostrar()
            return
        if self._pos < self._n_resultados() - 1:
            self._pos += 1
            self._mostrar()

    def action_anterior(self) -> None:
        """Retrocede a la página o resultado anterior."""
        if self._modo == "rag":
            if self._rag_pos > 0:
                self._rag_pos -= 1
                self._mostrar()
            return
        if self._pos > 0:
            self._pos -= 1
            self._mostrar()

    def _modo_label(self) -> str:
        """Devuelve la etiqueta legible del modo actual."""
        if self._modo == "clasica":
            return "Clásica"
        if self._modo == "semantica":
            return "Semántica"
        return "RAG"

    def _normalizar_heading_visual(self, heading: str) -> str:
        """Normaliza encabezados largos para una visualización más compacta."""
        h = heading.strip()
        h_lower = h.lower()
        if "primera parte del ingenioso caballero don quijote de la mancha" in h_lower:
            return "Primera Parte"
        if "el ingenioso hidalgo don quijote de la mancha" in h_lower:
            return "Primera Parte"
        if "segunda parte del ingenioso caballero don quijote de la mancha" in h_lower:
            return "Segunda Parte"
        return h

    def _render_chunk(self, chunk: dict[str, Any], texto_renderizado: str) -> str:
        """Compone la representación visual de un chunk con metadatos."""
        headings = chunk.get("headings", [])
        headings_vis = [self._normalizar_heading_visual(h) for h in headings]
        meta_str = " > ".join(headings_vis) if headings_vis else ""
        cid = chunk.get("index", "?")

        sent_start = chunk.get("sent_start", 0)
        sent_end = chunk.get("sent_end", 0)
        n_total = chunk.get("n_sents_total", 1)
        # Mostramos puntos suspensivos si el chunk no cubre el inicio o fin del texto.
        prefijo_texto = "[dim]…[/] " if sent_start > 0 else ""
        sufijo_texto = " [dim]…[/]" if sent_end < n_total - 1 else ""

        return (
            f"[b cyan]Chunk {cid}[/]\n[dim]{meta_str}[/]\n\n"
            + prefijo_texto
            + texto_renderizado
            + sufijo_texto
        )

    def _mostrar(self) -> None:
        """Renderiza el estado principal de la interfaz según modo y resultados."""
        estado = self.query_one("#estado", Static)
        contenedor = self.query_one("#resultado", Static)

        if not self._query:
            estado.update("Realice una consulta.")
            contenedor.update("")
            return

        if self._modo in ("semantica", "rag") and not self._embeddings_disponibles:
            estado.update("No hay embeddings disponibles.")
            contenedor.update(
                "[red]ollama no está disponible.[/]\n"
                "Asegúrate de que ollama está en ejecución y vuelve a arrancar la aplicación:\n"
                "  ollama serve\n"
                "  ollama pull nomic-embed-text"
            )
            return

        if self._modo == "rag":
            self._mostrar_pagina_rag()
            return

        n = self._n_resultados()
        if n == 0:
            estado.update(f"Sin resultados para '{self._query}'.")
            contenedor.update("")
            return

        if self._modo == "clasica":
            pid, tfidf, n_matched = self._resultados_clasica[self._pos]
            n_terminos = len(self._claves)
            estado.update(
                f"[b]{self._pos + 1}[/] / {n}  "
                f"[dim]términos: {n_matched}/{n_terminos}  "
                f"[b cyan]TF-IDF: {tfidf:.4f}[/]  (← → navegar)[/]"
            )
            parrafo = self.parrafos.get(pid)
            if parrafo is None:
                contenedor.update("[red]Chunk no encontrado.[/]")
                return
            contenedor.update(
                self._render_chunk(
                    parrafo, destacar(parrafo.get("text", ""), self._claves)
                )
            )

        elif self._modo == "semantica":
            pid, score = self._resultados_semantica[self._pos]
            estado.update(
                f"[b]{self._pos + 1}[/] / {n}  "
                f"[b cyan]similitud: {score:.4f}[/]  (← → navegar)"
            )
            parrafo = self.parrafos.get(pid)
            if parrafo is None:
                contenedor.update("[red]Chunk no encontrado.[/]")
                return
            contenedor.update(self._render_chunk(parrafo, parrafo.get("text", "")))
