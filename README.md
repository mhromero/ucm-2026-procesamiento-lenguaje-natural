# Práctica 1 - Los agentes de Butler
<code>**Procesamiento del Lenguaje Natural**</code>

## Índice
- [Descripción breve](#descripción-breve)
- [Integrantes](#integrantes)
- [Requisitos](#requisitos)
- [Instrucciones de ejecución](#instrucciones-de-ejecución)
- [Funcionamiento y lógica del bot](#funcionamiento-y-lógica-del-bot)
- [Estructura del código](#estructura-del-código)
- [Pruebas con cartas de prueba](#pruebas-con-cartas-de-prueba)
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
Arranca el servidor Butler con el buzón de cartas habilitado:

```bash
fdi-pln-butler server --buzon
```

Para pruebas en local (modo monopuesto), añade la bandera `--monopuesto`:

```bash
fdi-pln-butler server --buzon --monopuesto
```

> **Nota:** En modo monopuesto el servidor asigna el alias automáticamente; no es necesario pasarlo por variables de entorno salvo que quieras forzar uno concreto.

### 2) Lanzar Ollama
El bot usa Ollama para interpretar cartas con el LLM. Asegúrate de que el servidor está en ejecución:

```bash
ollama serve
```

### 3) Verificar o descargar el modelo
Comprueba que tienes disponible un modelo compatible (por defecto: `qwen3-vl:8b`):

```bash
ollama list
ollama pull qwen3-vl:8b   # o el modelo que uses en config.json
```

Si usas otro modelo, configura la variable `FDI_PLN__MODEL` o la opción `--model`.

### 4) Ejecutar el agente
Desde el directorio del proyecto:

```bash
cd ~/fdi-pln-2611
FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 uv run fdi-pln-2611-p1
```

El puerto `7719` es el que Butler usa por defecto. Si Butler corre en otra URL, ajusta `FDI_PLN__BUTLER_ADDRESS` en consecuencia.

### 5) Ejecutar dos agentes (modo monopuesto)
Para que sea modo monopuesto, Butler debe estar lanzado con el flag `--monopuesto` (véase paso 1). Ejecuta cada agente en una terminal distinta, asignando alias distintos:

**Terminal A:**
```bash
cd ~/fdi-pln-2611
FDI_PLN__ALIAS=ag001 FDI_PLN__MODO_MONOPUESTO=true FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 uv run fdi-pln-2611-p1
```

**Terminal B:**
```bash
cd ~/fdi-pln-2611
FDI_PLN__ALIAS=ag002 FDI_PLN__MODO_MONOPUESTO=true FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 uv run fdi-pln-2611-p1
```

Si Butler corre en otra URL, ajusta `FDI_PLN__BUTLER_ADDRESS` en consecuencia.

---

### Variables de entorno
La configuración sigue la prioridad: **variables de entorno > argumentos CLI > config.json**. Estas son las variables principales:

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `FDI_PLN__BUTLER_ADDRESS` | URL base de Butler | `http://127.0.0.1:7719` |
| `FDI_PLN__ALIAS` | Alias del agente (modo monopuesto) | `ag001` |
| `FDI_PLN__MODO_MONOPUESTO` | Activar modo monopuesto | `true` / `1` / `yes` |
| `FDI_PLN__MODEL` | Modelo de Ollama | `qwen3-vl:8b` |
| `FDI_PLN__OLLAMA_URL` | Endpoint de la API de Ollama | `http://localhost:11434/api/generate` |
| `FDI_PLN__GOLD_RESOURCE_NAME` | Nombre del recurso oro | `oro` |
| `FDI_PLN__MAILBOX_ENDPOINT` | Endpoint del buzón | `/buzon` |
| `FDI_PLN__LETTER_ENDPOINT` | Endpoint de cartas | `/carta` |
| `FDI_PLN__PACKAGE_ENDPOINT` | Endpoint de paquetes | `/paquete` |
| `FDI_PLN__LETTERS_BEFORE_REBROADCAST` | Cartas procesadas antes de reenviar ofertas | `5` |
| `FDI_PLN__OFFERS_PER_PERSON` | Ofertas aleatorias por destinatario | `2` |
| `FDI_PLN__REMITENTE_SISTEMA` | Remitente de cartas del sistema (se ignoran) | `Sistema` |
| `FDI_PLN__MAX_INTENTOS_OFERTA` | Intentos máximos al analizar una oferta con Ollama | `3` |

### Opciones CLI
Puedes sobreescribir cualquier valor con argumentos en la línea de comandos:

```bash
uv run fdi-pln-2611-p1 --help
```

**Opciones principales:**

| Opción | Variable de entorno | Descripción |
|--------|---------------------|-------------|
| `--api-base TEXT` | `FDI_PLN__BUTLER_ADDRESS` | URL base de Butler (ej. `http://127.0.0.1:7719`) |
| `--ollama-url TEXT` | `FDI_PLN__OLLAMA_URL` | URL de la API de Ollama |
| `--model TEXT` | `FDI_PLN__MODEL` | Modelo de Ollama (ej. `mistral:7b`) |
| `--alias TEXT` | `FDI_PLN__ALIAS` | Alias del agente |
| `--modo-monopuesto` / `--no-modo-monopuesto` | `FDI_PLN__MODO_MONOPUESTO` | Activar/desactivar modo monopuesto |
| `--letters-before-rebroadcast INT` | `FDI_PLN__LETTERS_BEFORE_REBROADCAST` | Cartas antes de reenviar ofertas |
| `--offers-per-person INT` | `FDI_PLN__OFFERS_PER_PERSON` | Ofertas por persona |
| `--gold-resource-name TEXT` | `FDI_PLN__GOLD_RESOURCE_NAME` | Nombre del recurso oro |
| `--remitente-sistema TEXT` | `FDI_PLN__REMITENTE_SISTEMA` | Remitente de cartas del sistema |
| `--max-intentos-oferta INT` | `FDI_PLN__MAX_INTENTOS_OFERTA` | Intentos máximos al analizar ofertas |

**Ejemplo con CLI:**
```bash
uv run fdi-pln-2611-p1 --api-base http://127.0.0.1:7719 --alias ag001 --model mistral:7b
```

También puedes modificar los valores por defecto en `config.json`, que se aplican cuando no se especifica una variable de entorno ni una opción CLI correspondiente.

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
└── send_test_letter.sh  # Script para inyectar cartas de prueba
```

## Pruebas con cartas de prueba
Puedes inyectar cartas manuales sin levantar un tercer agente con:
`src/send_test_letter.sh`

Ayuda:
```bash
src/send_test_letter.sh --help
```

Ejemplo básico:
```bash
src/send_test_letter.sh \
  --dest ag002 \
  --from ag001 \
  --subject "propuesta intercambio" \
  --body "Te ofrezo 1 arroz y tu me das 1 trigo."
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
