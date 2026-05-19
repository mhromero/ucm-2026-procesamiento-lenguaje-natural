# Práctica 5 — LM causal y NER (Alice in Wonderland)

**Asignatura**: Procesamiento del Lenguaje Natural — UCM 2024/25  
**Grupo**: 2611

## Integrantes

- María Romero Huertas
- Javier Martín Fuentes

## Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Instalación

```bash
uv sync
```

O desde el wheel:

```bash
pip install dist/fdi_pln_2611_p5-1.0-py3-none-any.whl
```

## Comandos

| Comando | Descripción |
|---------|-------------|
| `train-tokenizer` | Entrena el tokenizador BPE |
| `train-causal` | Entrena el LM causal (`--weights`, opcional `--grid-search`) |
| `train-ner` | Entrena NER sobre el backbone (`--weights`, `--causal-weights`) |
| `generate` | Genera texto (`--weights`, `--prompt`) |
| `ner` | Lista entidades en un fichero o texto directo (`--weights`) |
| `merge-etiquetados` | Fusiona `data/etiquetados/` (p1/parte1 + p2/parte2) |
| `annotation-report` | Informe HTML con métricas y gráficos de anotación |
| `grid-search-report` | Regenera el informe HTML de exploración de hiperparámetros |
| `merge-annotations` | Fusiona JSON de anotadores y calcula κ de Cohen |

### Ejemplos

```bash
# Entrenar tokenizador (opcional, también se entrena dentro de train-causal)
uv run fdi-pln-2611-p5 train-tokenizer

# Exploración de hiperparámetros (9 combinaciones) + entrenamiento final
uv run fdi-pln-2611-p5 train-causal --weights p5_causal_2611.pth --grid-search

# Entrenar NER (requiere anotaciones fusionadas con etiquetas)
uv run fdi-pln-2611-p5 train-ner --weights p5_ner_2611.pth --causal-weights p5_causal_2611.pth

# Inferencia — generación de texto
uv run fdi-pln-2611-p5 generate --weights p5_causal_2611.pth --prompt "Alice"

# Inferencia NER — desde fichero
uv run fdi-pln-2611-p5 ner --weights p5_ner_2611.pth fdi_pln_2611_p5/data/alice_in_wonderland.txt

# Inferencia NER — texto directo
uv run fdi-pln-2611-p5 ner --weights p5_ner_2611.pth --text "Alice met the Queen of Hearts"

# Fusionar etiquetados + informe
uv run fdi-pln-2611-p5 merge-etiquetados
uv run fdi-pln-2611-p5 annotation-report --skip-merge
```

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
| `epochs` | 15 | Épocas de ajuste fino |
| `batch_size` | 16 | Tamaño de lote |
| `learning_rate` | 0.0003 | Tasa de aprendizaje |
| `val_ratio` | 0.2 | Fracción de validación (estratificada por presencia de entidades) |

## Etiquetas NER

| Etiqueta | Significado |
|----------|-------------|
| `o` | Fuera de entidad |
| `pi` | Inicio de persona |
| `pc` | Continuación de persona |
| `li` | Inicio de lugar |
| `lc` | Continuación de lugar |

## Etiquetado

Hay dos granularidades de plantilla:

| Nivel | Comando | Salida | Qué se etiqueta |
|-------|---------|--------|-----------------|
| **Palabra** | `generate-templates-word` | `data/alice_jsons_palabra/` | Cada palabra, espacio o signo |
| **Token BPE** | `generate-templates-token` | `data/alice_jsons_token/` | Cada subpalabra del tokenizador |

```bash
# Por palabra (no requiere BPE previo)
uv run fdi-pln-2611-p5 generate-templates-word

# Por token BPE (requiere data/bpe_tokenizer.json)
uv run fdi-pln-2611-p5 generate-templates-token
```

## Corpus

- **Entrenamiento**: Alice in Wonderland + Harry Potter y la Piedra Filosofal (ambos en minúsculas)
- **Test/validación causal**: Alice in Wonderland
- **Anotaciones NER**: frases de Alice in Wonderland etiquetadas manualmente

## Entrega

```
fdi_pln_2611_p5-1.0-py3-none-any.whl   (uv build)
p5_causal_2611.pth
p5_ner_2611.pth
informe_2611.html
```

## Formato

```bash
uv run ruff format --check fdi_pln_2611_p5/
```
