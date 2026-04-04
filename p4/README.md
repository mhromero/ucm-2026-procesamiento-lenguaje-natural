# FDI PLN 2611 — Práctica 4 — Grupo 11

## Integrantes

- María Romero Huertas
- Javier Martín Fuentes

## Descripción

Aplicación de terminal para búsqueda de información sobre un corpus en español (*Don Quijote de la Mancha*). Implementa tres modos de búsqueda:

1. **Clásica**: búsqueda por lema con eliminación de stopwords y ranking TF-IDF.
2. **Semántica**: búsqueda por similitud coseno usando embeddings generados con `nomic-embed-text` (ollama).
3. **RAG**: combina ambas búsquedas para construir un contexto y genera una respuesta con un LLM (`llama3.2`) via ollama.

## Estructura del repositorio

```
fdi_pln_2611_p4/
├── data/
│   ├── 2000-h.htm             # Corpus fuente (Don Quijote, Project Gutenberg)
│   ├── parrafos_index.json    # Chunks extraídos del corpus
│   ├── vocabulario_index.json # Índice invertido con TF-IDF
│   ├── embeddings.npy         # Embeddings de los chunks (generado en primer arranque)
│   └── embeddings_ids.npy     # IDs de los chunks con embedding
├── main.py                    # Entry point
├── buscador_textual.py        # Interfaz TUI (Textual)
├── busqueda_clasica.py        # Búsqueda por lema + TF-IDF
├── busqueda_semantica.py      # Búsqueda por embeddings (ollama)
├── busqueda_rag.py            # RAG (ollama)
└── indexar_parrafos.py        # Extracción y chunking del corpus
```

## Modelos

| Modelo | Uso | Instalación |
|---|---|---|
| `es_core_news_sm` (spacy) | Lematización y segmentación de frases | Incluido en dependencias |
| `nomic-embed-text` (ollama) | Embeddings para búsqueda semántica | `ollama pull nomic-embed-text` |
| `llama3.2` (ollama) | Generación de respuestas RAG | `ollama pull llama3.2` |

## Requisitos previos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [ollama](https://ollama.com) instalado y en ejecución

## Instalación y ejecución

```bash
# 1. Instalar dependencias
uv sync

# 2. Arrancar ollama (en una terminal aparte)
ollama serve

# 3. Descargar los modelos de ollama
ollama pull nomic-embed-text
ollama pull llama3.2

# 4. Ejecutar la aplicación
uv run fdi-pln-2611-p4
```

El primer arranque generará automáticamente los embeddings (~3400 chunks, puede tardar varios minutos). Los arranques posteriores los cargan directamente desde disco.

## Uso de la interfaz

| Tecla | Acción |
|---|---|
| `Ctrl+1` | Modo búsqueda clásica |
| `Ctrl+2` | Modo búsqueda semántica |
| `Ctrl+3` | Modo RAG |
| `/` | Enfocar caja de búsqueda |
| `←` / `→` | Navegar entre resultados |
| `Enter` | Lanzar búsqueda |
| `q` | Salir |

## Regenerar índices manualmente

Si se quiere forzar la regeneración de los índices o embeddings, basta con borrar los archivos correspondientes de `fdi_pln_2611_p4/data/` y volver a ejecutar la aplicación.