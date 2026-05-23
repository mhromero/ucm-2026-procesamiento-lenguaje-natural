# Práctica 5 — LM causal y NER (Alice in Wonderland)

**Asignatura**: Procesamiento del Lenguaje Natural — UCM 2024/25  
**Grupo**: 2611

## Integrantes

- María Romero Huertas
- Javier Martín Fuentes

---

## Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Instalación

```bash
uv sync
```

O desde el wheel:

```bash
pip install fdi_pln_2611_p5-1.0-py3-none-any.whl
```

---

## Comandos

### Entrenamiento

| Comando | Descripción |
|---------|-------------|
| `train-tokenizer` | Entrena el tokenizador BPE y lo guarda en `data/bpe_tokenizer.json` |
| `train-causal --weights <ruta.pth>` | Entrena el LM causal y guarda los pesos |
| `train-causal --weights <ruta.pth> --grid-search` | Grid search lr×batch (9 combinaciones) + entrenamiento final |
| `train-ner --weights <ruta.pth> --causal-weights <ruta.pth>` | Fine-tuning del cabezal NER sobre el backbone causal |

### Inferencia

| Comando | Descripción |
|---------|-------------|
| `generate --weights <ruta.pth> --prompt <texto>` | Genera texto a partir de un prompt |
| `ner --weights <ruta.pth> <fichero.txt>` | Lista entidades nombradas en un fichero |
| `ner --weights <ruta.pth> --text <texto>` | Lista entidades nombradas en texto directo |

### Informes y exploración

| Comando | Descripción |
|---------|-------------|
| `run-experiments` | Lanza 8 experimentos (corpus, ventana, vocab, profundidad) y genera informe HTML |
| `experiment-report` | Regenera `informes/informe_experimentos.html` desde `data/experiment_results.json` |
| `grid-search-report` | Regenera `informes/informe_grid_search.html` desde `data/grid_search_results.json` |

### Anotación

| Comando | Descripción |
|---------|-------------|
| `merge-etiquetados` | Fusiona `data/etiquetados/` (parte1 + parte2) en `data/annotations/merged.json` |
| `annotation-report` | Genera informe HTML con métricas y gráficos del etiquetado |
| `merge-annotations` | Fusiona JSON de anotadores y calcula κ de Cohen |
| `generate-templates-word` | Genera plantillas JSON con una etiqueta por palabra |
| `generate-templates-6frases` | Genera plantillas de 6 frases largas por JSON |
| `generate-templates-1json-9frases` | Genera un JSON con 9 frases (6 de parte1 + 3 extra) |
| `generate-templates-token` | Genera plantillas con una etiqueta por subpalabra BPE |

---

## Ejemplos de uso

```bash
# Entrenar el tokenizador (también se ejecuta automáticamente dentro de train-causal)
uv run fdi-pln-2611-p5 train-tokenizer

# Entrenar el LM causal
uv run fdi-pln-2611-p5 train-causal --weights p5_causal_2611.pth

# Grid search de hiperparámetros + entrenamiento final
uv run fdi-pln-2611-p5 train-causal --weights p5_causal_2611.pth --grid-search

# Entrenar NER
uv run fdi-pln-2611-p5 train-ner --weights p5_ner_2611.pth --causal-weights p5_causal_2611.pth

# Generar texto
uv run fdi-pln-2611-p5 generate --weights p5_causal_2611.pth --prompt "Alice"
uv run fdi-pln-2611-p5 generate --weights p5_causal_2611.pth --prompt "The Queen" --max-new-tokens 200 --temperature 0.8

# NER desde fichero
uv run fdi-pln-2611-p5 ner --weights p5_ner_2611.pth fdi_pln_2611_p5/data/corpus/alice_in_wonderland.txt

# NER desde texto directo
uv run fdi-pln-2611-p5 ner --weights p5_ner_2611.pth --text "Alice met the Queen of Hearts"

# Exploración de hiperparámetros y arquitectura
uv run fdi-pln-2611-p5 run-experiments

# Regenerar informes HTML
uv run fdi-pln-2611-p5 experiment-report
uv run fdi-pln-2611-p5 grid-search-report

# Fusionar etiquetados e informe de anotación
uv run fdi-pln-2611-p5 merge-etiquetados
uv run fdi-pln-2611-p5 annotation-report --skip-merge
```

---

## Hiperparámetros

### Tokenizador BPE

| Parámetro | Valor |
|-----------|-------|
| `vocab_size` | 300 |

### Arquitectura del modelo

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `d_model` | 128 | Dimensión del espacio de representación |
| `n_blocks` | 4 | Número de bloques Transformer |
| `n_heads` | 4 | Cabezas de atención multi-head |
| `window_size` | 128 | Longitud máxima de secuencia |
| `dropout` | 0.1 | Tasa de dropout |

### Entrenamiento causal

| Parámetro | Valor |
|-----------|-------|
| `epochs` | 10 |
| `batch_size` | 64 |
| `learning_rate` | 0.0003 |
| `seed` | 42 |

### Grid search

| Parámetro | Valores explorados |
|-----------|-------------------|
| `learning_rate` | 0.0001, 0.0003, 0.001 |
| `batch_size` | 32, 64, 128 |
| `epochs_por_run` | 3 |

### Entrenamiento NER

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `epochs` | 15 | Épocas de fine-tuning |
| `batch_size` | 16 | Tamaño de lote |
| `learning_rate` | 0.0003 | Tasa de aprendizaje |
| `val_ratio` | 0.2 | Fracción de validación (estratificada) |
| `backbone_lr_factor` | 0.1 | Factor de lr para el backbone (fine-tuning suave) |

---

## Etiquetas NER

| Etiqueta | Significado |
|----------|-------------|
| `o` | Fuera de entidad |
| `pi` | Inicio de persona |
| `pc` | Continuación de persona |
| `li` | Inicio de lugar |
| `lc` | Continuación de lugar |

---

## Corpus

| Uso | Datos |
|-----|-------|
| Entrenamiento causal | Alice in Wonderland + Through the Looking-Glass + hasta 4 libros de Harry Potter |
| Validación causal | Alice in Wonderland |
| Fine-tuning NER | Frases de Alice in Wonderland anotadas manualmente |

Los textos de entrenamiento se encuentran en `fdi_pln_2611_p5/data/corpus/`.

---

## Informes

| Fichero | Contenido |
|---------|-----------|
| `informes/informe_experimentos.html` | Comparativa de 8 experimentos (corpus, ventana, vocab, profundidad) |
| `informes/informe_grid_search.html` | Grid search lr × batch_size |
| `informes/ner_report.html` | Métricas y matriz de confusión del entrenamiento NER |
| `informes/informe_etiquetado.html` | Acuerdo entre anotadores (κ de Cohen) |
| `informe_2611.md` | Informe general con observaciones y conclusiones |

---

## Entrega

```
fdi_pln_2611_p5-1.0-py3-none-any.whl   (uv build)
p5_causal_2611.pth
p5_ner_2611.pth
informe_2611.pdf
```

---

## Formato

```bash
uv run ruff format --check fdi_pln_2611_p5/
```
