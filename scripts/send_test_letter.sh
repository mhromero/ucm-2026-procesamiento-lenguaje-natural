#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Uso:
  scripts/send_test_letter.sh --dest ALIAS [opciones]

Opciones:
  --dest ALIAS          Alias del destinatario (obligatorio)
  --subject TEXTO       Asunto de la carta
  --body TEXTO          Cuerpo de la carta
  --from ALIAS          Remitente (default: tester_bot)
  --base URL            URL base Butler (default: http://127.0.0.1:7719)
  --id UUID             ID manual (default: uuidgen)
  --date ISO8601        Fecha manual UTC (default: ahora)
  -h, --help            Mostrar ayuda

Ejemplo:
  scripts/send_test_letter.sh \
    --dest agente_a \
    --subject "Oferta: 1 queso por 1 aceite" \
    --body "Te propongo intercambiar 1 queso que necesito por 1 aceite que te ofrezco."
USAGE
}

BASE="http://127.0.0.1:7719"
DEST=""
FROM_ALIAS="tester_bot"
SUBJECT="Oferta: 1 queso por 1 aceite"
BODY="Te propongo intercambiar 1 queso que necesito por 1 aceite que te ofrezco."
LETTER_ID=""
LETTER_DATE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="${2:-}"
      shift 2
      ;;
    --subject)
      SUBJECT="${2:-}"
      shift 2
      ;;
    --body)
      BODY="${2:-}"
      shift 2
      ;;
    --from)
      FROM_ALIAS="${2:-}"
      shift 2
      ;;
    --base)
      BASE="${2:-}"
      shift 2
      ;;
    --id)
      LETTER_ID="${2:-}"
      shift 2
      ;;
    --date)
      LETTER_DATE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: opción no reconocida: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DEST" ]]; then
  echo "Error: --dest es obligatorio" >&2
  usage
  exit 1
fi

if [[ -z "$LETTER_ID" ]]; then
  if command -v uuidgen >/dev/null 2>&1; then
    LETTER_ID="$(uuidgen)"
  else
    LETTER_ID="manual-$(date +%s)"
  fi
fi

if [[ -z "$LETTER_DATE" ]]; then
  LETTER_DATE="$(date -u +%Y-%m-%dT%H:%M:%S)"
fi

json_escape() {
  local input="${1:-}"
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$input"
}

REMI_JSON="$(json_escape "$FROM_ALIAS")"
DEST_JSON="$(json_escape "$DEST")"
SUBJECT_JSON="$(json_escape "$SUBJECT")"
BODY_JSON="$(json_escape "$BODY")"
ID_JSON="$(json_escape "$LETTER_ID")"
DATE_JSON="$(json_escape "$LETTER_DATE")"

PAYLOAD=$(cat <<JSON
{
  "remi": $REMI_JSON,
  "dest": $DEST_JSON,
  "asunto": $SUBJECT_JSON,
  "cuerpo": $BODY_JSON,
  "id": $ID_JSON,
  "fecha": $DATE_JSON
}
JSON
)

echo "Enviando carta a '$DEST' via $BASE/carta?agente=$DEST ..."
curl -sS -X POST "$BASE/carta?agente=$DEST" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"
echo

echo "OK -> remi=$FROM_ALIAS dest=$DEST asunto=$SUBJECT id=$LETTER_ID fecha=$LETTER_DATE"
