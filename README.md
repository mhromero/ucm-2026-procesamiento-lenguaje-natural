# fdi-pln-2611 - Práctica 1

## Integrantes
- Javier Martin Fuentes
- Maria Romero Huertas

## Requisitos
- Python 3.12
- uv
- Butler (`fdi-pln-butler`)
- Ollama (opcional, para la parte LLM)

## Ejecucion

### 1) Lanzar Butler
```bash
fdi-pln-butler server
```

### 2) Ejecutar agente
```bash
cd ~/fdi-pln-2611
FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 uv run fdi-pln-2611-p1
```

## Variables de entorno
- `FDI_PLN__BUTLER_ADDRESS`: URL de Butler.
- `FDI_PLN__ALIAS`: alias del agente.
- `FDI_PLN__MODEL` : modelo de Ollama.
- `FDI_PLN__OLLAMA_URL`: endpoint de Ollama.

## Calidad y formato
```bash
uv format --check
```

## Empaquetado (wheel)
```bash
uv build
```

## Resumen de funcionamiento
- Obtiene estado del puesto (`/info`) y lista de agentes (`/gente`).
- Calcula necesidades y excedentes respecto al objetivo.
- El agente lee y procesa buzon para negociar.
- Envía paquetes de recursos con `/paquete/{dest}` cuando aplica.
