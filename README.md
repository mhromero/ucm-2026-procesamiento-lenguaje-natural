# fdi-pln-2611 - Práctica 1

Bot de negociación de recursos entre agentes usando LLM (Ollama). Interpreta cartas, acepta/rechaza ofertas y envía paquetes automáticamente.

## Integrantes
- Javier Martin Fuentes
- Maria Romero Huertas

## Arquitectura

```
src/
├── app.py          # Flujo principal: ciclo de negociación con Butler
├── api.py          # Cliente HTTP de Butler (info, gente, cartas, paquetes)
├── agent.py        # Interpretación de cartas y decisiones con LLM (Ollama)
├── ollama_client.py # Cliente HTTP para Ollama
├── trader.py       # Evaluación de ofertas/confirmaciones y envío de paquetes
├── letters.py       # Composición de cartas (ofertas, confirmaciones)
├── game_state.py   # Estado del puesto: inventario, objetivo, needs/surplus
├── logs.py         # Salida formateada por consola
└── config.py       # Configuración (env > CLI > config.json)
```

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
1. Obtiene estado del puesto (`/info`) y lista de agentes (`/gente`).
2. Calcula necesidades y excedentes respecto al objetivo.
3. Envía ofertas según el estado (necesidad↔excedente o surplus→oro).
4. Lee buzón, interpreta cartas con Ollama y actúa (acepta/rechaza).
5. Envía paquetes con `/paquete/{dest}` y cartas de confirmación cuando aplica.
