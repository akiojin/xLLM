#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

XLLM_BIN="${XLLM_BIN:-$ROOT_DIR/build/xllm}"
HOST="${XLLM_E2E_HOST:-127.0.0.1}"
PORT="${XLLM_E2E_PORT:-32769}"
WORKDIR="${XLLM_E2E_WORKDIR:-$ROOT_DIR/tmp/e2e-real-models}"
MODELS_DIR="${XLLM_E2E_MODELS_DIR:-$WORKDIR/models}"
LOG_DIR="$WORKDIR/logs"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] missing command: $1" >&2
    exit 1
  fi
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[ERROR] required env var not set: $name" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd jq
require_cmd python3

require_env XLLM_E2E_TEXT_MODEL_REF
require_env XLLM_E2E_VISION_MODEL_REF
require_env XLLM_E2E_IMAGE_MODEL_REF
require_env XLLM_E2E_ASR_MODEL_REF
require_env XLLM_E2E_TTS_MODEL

if [[ ! -x "$XLLM_BIN" ]]; then
  if [[ -x "$ROOT_DIR/build/Release/xllm" ]]; then
    XLLM_BIN="$ROOT_DIR/build/Release/xllm"
  elif [[ -x "$ROOT_DIR/build/Debug/xllm" ]]; then
    XLLM_BIN="$ROOT_DIR/build/Debug/xllm"
  fi
fi

if [[ ! -x "$XLLM_BIN" ]]; then
  echo "[ERROR] xllm binary not found: $XLLM_BIN" >&2
  echo "        Build it with: cmake -S . -B build && cmake --build build" >&2
  exit 1
fi

export XLLM_ALLOW_SAFETENSORS_NO_METADATA=1
export XLLM_ALLOW_BIN_MODELS=1

mkdir -p "$MODELS_DIR" "$LOG_DIR"

server_log="$LOG_DIR/xllm.log"
server_pid=""

cleanup() {
  set +e
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

repo_from_ref() {
  python3 - "$1" <<'PY'
import sys, urllib.parse
ref = sys.argv[1]
if '://' in ref:
    path = urllib.parse.urlparse(ref).path
else:
    path = ref
path = path.split('?', 1)[0].split('#', 1)[0]
path = path.strip('/')
parts = [p for p in path.split('/') if p]
if len(parts) < 2:
    sys.exit(1)
print(parts[0] + '/' + parts[1])
PY
}

sanitize_model_id() {
  python3 - "$1" <<'PY'
import sys
value = sys.argv[1]
if '..' in value or '\x00' in value:
    print('_latest')
    sys.exit(0)
out = []
for c in value:
    if ('a' <= c <= 'z') or ('0' <= c <= '9') or c in '-_.':
        out.append(c)
    elif ('A' <= c <= 'Z'):
        out.append(c.lower())
    elif c in '/\\':
        out.append('/')
    else:
        out.append('_')
out = ''.join(out).strip('/')
if out in ('', '.', '..'):
    out = '_latest'
print(out)
PY
}

pick_model_file() {
  local model_dir="$1"
  shift
  python3 - "$model_dir" "$@" <<'PY'
import os, sys
model_dir = sys.argv[1]
exts = sys.argv[2:]
exclude = {"config.json", "tokenizer.json"}
candidates = []
for root, _, files in os.walk(model_dir):
    for name in files:
        if name in exclude:
            continue
        if name.endswith(".safetensors.index.json"):
            continue
        if not any(name.endswith(ext) for ext in exts):
            continue
        path = os.path.join(root, name)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        candidates.append((size, path))
if not candidates:
    sys.exit(1)
candidates.sort(reverse=True)
print(candidates[0][1])
PY
}

wait_for_ready() {
  local timeout="${1:-300}"
  local elapsed=0
  while (( elapsed < timeout )); do
    if curl -fsS "http://$HOST:$PORT/api/startup" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    ((elapsed++))
  done
  return 1
}

echo "[INFO] Starting xllm server on $HOST:$PORT (logs: $server_log)"
XLLM_PORT="$PORT" \
XLLM_MODELS_DIR="$MODELS_DIR" \
XLLM_LOG_LEVEL="${XLLM_LOG_LEVEL:-info}" \
"$XLLM_BIN" serve --host "$HOST" --port "$PORT" >"$server_log" 2>&1 &
server_pid=$!

echo "[INFO] Waiting for server readiness..."
if ! wait_for_ready 600; then
  echo "[ERROR] server did not become ready (see $server_log)" >&2
  exit 1
fi

export LLMLB_HOST="$HOST"
export XLLM_PORT="$PORT"

pull_model() {
  local ref="$1"
  echo "[INFO] Pulling model: $ref"
  "$XLLM_BIN" pull "$ref"
}

text_repo="$(repo_from_ref "$XLLM_E2E_TEXT_MODEL_REF")"
vision_repo="$(repo_from_ref "$XLLM_E2E_VISION_MODEL_REF")"
image_repo="$(repo_from_ref "$XLLM_E2E_IMAGE_MODEL_REF")"
asr_repo="$(repo_from_ref "$XLLM_E2E_ASR_MODEL_REF")"

pull_model "$XLLM_E2E_TEXT_MODEL_REF"
pull_model "$XLLM_E2E_VISION_MODEL_REF"
pull_model "$XLLM_E2E_IMAGE_MODEL_REF"
pull_model "$XLLM_E2E_ASR_MODEL_REF"

text_model="$text_repo"
vision_model="$vision_repo"

image_dir="$MODELS_DIR/$(sanitize_model_id "$image_repo")"
asr_dir="$MODELS_DIR/$(sanitize_model_id "$asr_repo")"

image_model_path="$(pick_model_file "$image_dir" .safetensors .ckpt)"
asr_model_path=""
if asr_model_path="$(pick_model_file "$asr_dir" .bin .gguf)"; then
  :
else
  echo "[ERROR] ASR model file not found in $asr_dir" >&2
  exit 1
fi

echo "[INFO] Text model: $text_model"
echo "[INFO] Vision model: $vision_model"
echo "[INFO] Image model path: $image_model_path"
echo "[INFO] ASR model path: $asr_model_path"

if [[ "$XLLM_E2E_TTS_MODEL" != "vibevoice" && "$XLLM_E2E_TTS_MODEL" != *"VibeVoice"* ]]; then
  echo "[ERROR] TTS E2E currently requires VibeVoice (set XLLM_E2E_TTS_MODEL=vibevoice)" >&2
  exit 1
fi
require_env XLLM_VIBEVOICE_RUNNER

echo "[INFO] Running text generation..."
text_body=$(jq -n --arg model "$text_model" '{model:$model,messages:[{role:"user",content:"hello"}] }')
text_code=$(curl -sS -o "$WORKDIR/text.json" -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$text_body" \
  "http://$HOST:$PORT/v1/chat/completions")
if [[ "$text_code" != "200" ]]; then
  echo "[ERROR] text generation failed (HTTP $text_code)" >&2
  cat "$WORKDIR/text.json" >&2
  exit 1
fi
jq -e '.choices[0].message.content' "$WORKDIR/text.json" >/dev/null

echo "[INFO] Running vision chat..."
png_data="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
vision_body=$(jq -n --arg model "$vision_model" --arg url "$png_data" '{model:$model,messages:[{role:"user",content:[{type:"text",text:"What is in this image?"},{type:"image_url",image_url:{url:$url}}]}] }')
vision_code=$(curl -sS -o "$WORKDIR/vision.json" -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$vision_body" \
  "http://$HOST:$PORT/v1/chat/completions")
if [[ "$vision_code" != "200" ]]; then
  echo "[ERROR] vision chat failed (HTTP $vision_code)" >&2
  cat "$WORKDIR/vision.json" >&2
  exit 1
fi
jq -e '.choices[0].message.content' "$WORKDIR/vision.json" >/dev/null

echo "[INFO] Running image generation..."
img_body=$(jq -n --arg model "$image_model_path" '{model:$model,prompt:"a tiny cat",size:"256x256",steps:2,response_format:"b64_json"}')
img_code=$(curl -sS -o "$WORKDIR/image.json" -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$img_body" \
  "http://$HOST:$PORT/v1/images/generations")
if [[ "$img_code" != "200" ]]; then
  echo "[ERROR] image generation failed (HTTP $img_code)" >&2
  cat "$WORKDIR/image.json" >&2
  exit 1
fi
jq -e '.data[0].b64_json | length > 0' "$WORKDIR/image.json" >/dev/null

echo "[INFO] Running ASR transcription..."
audio_path="$WORKDIR/asr_sample.wav"
python3 - <<'PY' "$audio_path"
import sys, wave, math, struct
path = sys.argv[1]
sample_rate = 16000
duration = 1.0
freq = 440.0
num_samples = int(sample_rate * duration)
with wave.open(path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    for i in range(num_samples):
        val = int(0.2 * 32767.0 * math.sin(2 * math.pi * freq * i / sample_rate))
        wf.writeframes(struct.pack('<h', val))
PY

asr_code=$(curl -sS -o "$WORKDIR/asr.json" -w "%{http_code}" \
  -F "file=@$audio_path" \
  -F "model=$asr_model_path" \
  -F "response_format=json" \
  "http://$HOST:$PORT/v1/audio/transcriptions")
if [[ "$asr_code" != "200" ]]; then
  echo "[ERROR] ASR failed (HTTP $asr_code)" >&2
  cat "$WORKDIR/asr.json" >&2
  exit 1
fi
jq -e 'has("text")' "$WORKDIR/asr.json" >/dev/null

echo "[INFO] Running TTS (VibeVoice)..."
tts_body=$(jq -n --arg model "$XLLM_E2E_TTS_MODEL" '{model:$model,input:"hello",response_format:"wav"}')
tts_code=$(curl -sS -o "$WORKDIR/tts.wav" -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$tts_body" \
  "http://$HOST:$PORT/v1/audio/speech")
if [[ "$tts_code" != "200" ]]; then
  echo "[ERROR] TTS failed (HTTP $tts_code)" >&2
  cat "$WORKDIR/tts.wav" >&2
  exit 1
fi
if [[ ! -s "$WORKDIR/tts.wav" ]]; then
  echo "[ERROR] TTS output is empty" >&2
  exit 1
fi

echo "[INFO] Cleaning up models..."
"$XLLM_BIN" rm "$text_repo" || true
"$XLLM_BIN" rm "$vision_repo" || true
"$XLLM_BIN" rm "$image_repo" || true
"$XLLM_BIN" rm "$asr_repo" || true

echo "[INFO] E2E real-model tests completed successfully"
