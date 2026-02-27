# fdi-pln-2611 - Práctica 1

## Índice
- [Descripción breve](#descripción-breve)
- [Integrantes](#integrantes)
- [Requisitos](#requisitos)
- [Instrucciones de ejecución](#instrucciones-de-ejecución)
- [Funcionamiento y lógica del bot](#funcionamiento-y-lógica-del-bot)
- [Estructura del código](#estructura-del-código)
- [Pruebas con cartas de prueba](#pruebas-con-cartas-de-prueba)
- [Variables de entorno y opciones CLI](#variables-de-entorno-y-opciones-cli)
- [Calidad y empaquetado](#calidad-y-empaquetado)

## Descripción breve
Bot de negociación de recursos entre agentes para Butler.  
El agente analiza cartas con LLM (Ollama), decide si aceptar/rechazar ofertas según su estado (necesidades, excedentes e inventario), envía paquetes y confirma intercambios por carta.

## Integrantes
- Javier Martín Fuentes
- María Romero Huertas

## Requisitos
- Python 3.12
- `uv`
- Butler (`fdi-pln-butler`)
- Ollama (necesario para el flujo con LLM)

## Instrucciones de ejecución
### 1) Lanzar Butler
```bash
fdi-pln-butler server
--buzon --monopuesto # para probar en local
```

### 2) Lanzar Ollama
```bash
ollama serve
```

### 3) Verificar/descargar modelo
```bash
ollama list
ollama pull <nombre-del-modelo>
```

### 4) Ejecutar agente
```bash
cd ~/fdi-pln-2611
FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 uv run fdi-pln-2611-p1
```

### 5) Ejecutar dos agentes (modo monopuesto)
Terminal A:
```bash
cd ~/fdi-pln-2611
FDI_PLN__MODO_MONOPUESTO=true FDI_PLN__ALIAS=ag001 FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 uv run fdi-pln-2611-p1
```

Terminal B:
```bash
cd ~/fdi-pln-2611
FDI_PLN__MODO_MONOPUESTO=true FDI_PLN__ALIAS=ag002 FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 uv run fdi-pln-2611-p1
```

## Funcionamiento y lógica del bot
### Flujo general
1. Arranca y obtiene estado inicial desde `/info` (alias, inventario, objetivo y buzón).
2. Obtiene agentes desde `/gente` y elimina su propio alias de la lista destino.
3. Calcula:
- `needs`: recursos que faltan para completar objetivo.
- `surplus`: recursos sobrantes (excluyendo oro).
4. Envía ofertas iniciales a otros agentes.
5. Entra en bucle:
- procesa cartas por fecha (antiguas primero),
- interpreta cada carta con LLM,
- decide aceptar/rechazar,
- envía paquete y carta de confirmación cuando procede,
- elimina carta procesada del buzón,
- reintenta lectura de buzón cada 5 segundos.

### Tipos de oferta que genera
- `propuesta intercambio`: intercambio simple 1:1 (surplus por recurso necesario).
- `Oferta: 1 <surplus> por 1 oro`: cuando ya cumplió objetivo y busca oro.
- `Oferta: 1 oro por 1 recurso que necesite`: cuando solo puede negociar con oro.

### Procesado de cartas entrantes
1. Se parsea la respuesta de Ollama incluso si trae texto adicional (se extrae JSON válido).
2. Se normaliza el esquema para convertir variantes a formato canónico:
- `{"recurso":"queso"}` -> `{"queso":1}`
- `{"recurso":"queso","cantidad":2}` -> `{"queso":2}`
3. Si una carta no trae cantidades numéricas explícitas, se evita inflar cantidades y se normaliza a 1 por recurso detectado.
4. Si la carta es de tipo `oferta`:
- se vuelve a evaluar con LLM en `analyze_offer`,
- la decisión final se acota a recursos/cantidades presentes en la oferta original (sin inventar más).
5. Si la carta es de tipo `confirmacion`, se evalúa si procede devolver recursos.

### Reglas de aceptación/rechazo (resumen)
- Rechaza ofertas incompletas (`oferta` o `pide` vacío).
- No envía recursos que necesita para su propio objetivo.
- No envía más de lo que tiene en inventario.
- El oro tiene restricciones específicas.
- Si una oferta es aceptada:
1. envía paquete (`/paquete/{dest}`),
2. envía carta de confirmación.

### Política de rebroadcast
- Reenvía ofertas al procesar `LETTERS_BEFORE_REBROADCAST` cartas.
- Si el buzón permanece vacío, también rebroadcast cada 5 revisiones vacías consecutivas (con polling cada 5 s).

## Estructura del código
```text
src/
├── app.py            # Flujo principal del bot y bucle de buzón
├── api.py            # Cliente HTTP Butler (/info, /gente, /carta, /mail, /paquete)
├── agent.py          # Parseo/decisión con LLM + normalización/validación de salida
├── trader.py         # Reglas de aceptación/rechazo y ejecución de intercambios
├── letters.py        # Plantillas de cartas y broadcast de ofertas
├── game_state.py     # Estado local: alias, inventario, objetivo, needs/surplus
├── ollama_client.py  # Cliente HTTP de Ollama
├── logs.py           # Salida por consola
├── config.py         # Configuración (env > CLI > config.json)
└── config.json       # Valores por defecto del proyecto

scripts/
└── send_test_letter.sh  # Script para inyectar cartas de prueba
```

## Pruebas con cartas de prueba
Puedes inyectar cartas manuales sin levantar un tercer agente con:
`scripts/send_test_letter.sh`

Ayuda:
```bash
scripts/send_test_letter.sh --help
```

Ejemplo básico:
```bash
scripts/send_test_letter.sh \
  --dest ag002 \
  --from ag001 \
  --subject "propuesta intercambio" \
  --body "Te ofrezo 1 arroz y tu me das 1 trigo."
```

## Variables de entorno y opciones CLI
Variables principales:
- `FDI_PLN__BUTLER_ADDRESS`: URL base de Butler.
- `FDI_PLN__ALIAS`: alias del agente.
- `FDI_PLN__MODO_MONOPUESTO`: `true/false`.
- `FDI_PLN__MODEL`: modelo de Ollama.
- `FDI_PLN__OLLAMA_URL`: endpoint de Ollama.

Opciones CLI:
```bash
uv run fdi-pln-2611-p1 --help
```

## Calidad y empaquetado
Formato/calidad:
```bash
uv format --check
```

Construcción de wheel:
```bash
uv build
```
