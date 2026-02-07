#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

XLLM_BIN="${XLLM_BIN:-$ROOT_DIR/build/xllm}"
HOST="${XLLM_E2E_HOST:-127.0.0.1}"
PORT="${XLLM_E2E_PORT:-32769}"
WORKDIR="${XLLM_E2E_WORKDIR:-$ROOT_DIR/tmp/e2e-real-models}"
MODELS_DIR="${XLLM_E2E_MODELS_DIR:-$WORKDIR/models}"
LOG_DIR="$WORKDIR/logs"
E2E_TIMEOUT="${XLLM_E2E_TIMEOUT:-600}"
IMAGE_STEPS="${XLLM_E2E_IMAGE_STEPS:-4}"
IMAGE_SIZE="${XLLM_E2E_IMAGE_SIZE:-256x256}"
ASR_MODEL_FILE="${XLLM_E2E_ASR_MODEL_FILE:-}"
IMAGE_MODEL_FILE="${XLLM_E2E_IMAGE_MODEL_FILE:-}"
STREAMING_TESTS="${XLLM_E2E_STREAMING:-1}"

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
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

log_tail() {
  if [[ -f "$server_log" ]]; then
    echo "[INFO] --- xllm log tail ---" >&2
    tail -n 200 "$server_log" >&2 || true
    echo "[INFO] --- end log tail ---" >&2
  fi
}

ensure_port_free() {
  if curl -fsS "http://$HOST:$PORT/api/startup" >/dev/null 2>&1; then
    echo "[ERROR] Port $PORT already has an xllm server running" >&2
    exit 1
  fi
}

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

curl_json() {
  local method="$1"
  local url="$2"
  local body="$3"
  local out="$4"
  local code=""
  code=$(curl -sS -o "$out" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -X "$method" \
    -d "$body" \
    "$url")
  echo "$code"
}

assert_png() {
  local b64="$1"
  python3 - <<'PY' "$b64"
import base64, sys
data = base64.b64decode(sys.argv[1])
if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
    raise SystemExit("invalid png")
print("ok")
PY
}

assert_wav() {
  local path="$1"
  python3 - <<'PY' "$path"
import sys, wave
path = sys.argv[1]
with wave.open(path, 'rb') as wf:
    if wf.getnframes() <= 0:
        raise SystemExit("empty wav")
    if wf.getframerate() <= 0:
        raise SystemExit("invalid sample rate")
print("ok")
PY
}

ensure_port_free

echo "[INFO] Starting xllm server on $HOST:$PORT (logs: $server_log)"
XLLM_PORT="$PORT" \
XLLM_MODELS_DIR="$MODELS_DIR" \
XLLM_LOG_LEVEL="${XLLM_LOG_LEVEL:-info}" \
"$XLLM_BIN" serve --host "$HOST" --port "$PORT" >"$server_log" 2>&1 &
server_pid=$!

echo "[INFO] Waiting for server readiness..."
if ! wait_for_ready "$E2E_TIMEOUT"; then
  echo "[ERROR] server did not become ready (see $server_log)" >&2
  log_tail
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

if [[ -n "$IMAGE_MODEL_FILE" ]]; then
  image_model_path="$image_dir/$IMAGE_MODEL_FILE"
  if [[ ! -f "$image_model_path" ]]; then
    echo "[ERROR] IMAGE_MODEL_FILE not found: $image_model_path" >&2
    exit 1
  fi
else
  image_model_path="$(pick_model_file "$image_dir" .safetensors .ckpt)"
fi

if [[ -n "$ASR_MODEL_FILE" ]]; then
  asr_model_path="$asr_dir/$ASR_MODEL_FILE"
  if [[ ! -f "$asr_model_path" ]]; then
    echo "[ERROR] ASR_MODEL_FILE not found: $asr_model_path" >&2
    exit 1
  fi
else
  asr_model_path=""
  if asr_model_path="$(pick_model_file "$asr_dir" .bin .gguf)"; then
    :
  else
    echo "[ERROR] ASR model file not found in $asr_dir" >&2
    exit 1
  fi
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
text_code=$(curl_json POST "http://$HOST:$PORT/v1/chat/completions" "$text_body" "$WORKDIR/text.json")
if [[ "$text_code" != "200" ]]; then
  echo "[ERROR] text generation failed (HTTP $text_code)" >&2
  cat "$WORKDIR/text.json" >&2
  log_tail
  exit 1
fi
jq -e '.choices[0].message.content | length > 0' "$WORKDIR/text.json" >/dev/null
if ! jq -e '.usage.total_tokens >= 0' "$WORKDIR/text.json" >/dev/null; then
  echo "[ERROR] text generation missing usage.total_tokens" >&2
  cat "$WORKDIR/text.json" >&2
  log_tail
  exit 1
fi

if [[ "$STREAMING_TESTS" == "1" ]]; then
  echo "[INFO] Running text streaming..."
  stream_body=$(jq -n --arg model "$text_model" '{model:$model,messages:[{role:"user",content:"stream"}],stream:true }')
  stream_code=$(curl -sS -o "$WORKDIR/text_stream.txt" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "$stream_body" \
    "http://$HOST:$PORT/v1/chat/completions")
  if [[ "$stream_code" != "200" ]]; then
    echo "[ERROR] text streaming failed (HTTP $stream_code)" >&2
    cat "$WORKDIR/text_stream.txt" >&2
    log_tail
    exit 1
  fi
  grep -q "data:" "$WORKDIR/text_stream.txt"
  grep -q "\\[DONE\\]" "$WORKDIR/text_stream.txt"
fi

echo "[INFO] Running vision chat..."
png_data="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
vision_body=$(jq -n --arg model "$vision_model" --arg url "$png_data" '{model:$model,messages:[{role:"user",content:[{type:"text",text:"What is in this image?"},{type:"image_url",image_url:{url:$url}}]}] }')
vision_code=$(curl_json POST "http://$HOST:$PORT/v1/chat/completions" "$vision_body" "$WORKDIR/vision.json")
if [[ "$vision_code" != "200" ]]; then
  echo "[ERROR] vision chat failed (HTTP $vision_code)" >&2
  cat "$WORKDIR/vision.json" >&2
  log_tail
  exit 1
fi
jq -e '.choices[0].message.content | length > 0' "$WORKDIR/vision.json" >/dev/null

if [[ "$STREAMING_TESTS" == "1" ]]; then
  echo "[INFO] Running responses streaming..."
  resp_stream_body=$(jq -n --arg model "$text_model" '{model:$model,input:"stream",stream:true }')
  resp_stream_code=$(curl -sS -o "$WORKDIR/response_stream.txt" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "$resp_stream_body" \
    "http://$HOST:$PORT/v1/responses")
  if [[ "$resp_stream_code" != "200" ]]; then
    echo "[ERROR] responses streaming failed (HTTP $resp_stream_code)" >&2
    cat "$WORKDIR/response_stream.txt" >&2
    log_tail
    exit 1
  fi
  grep -q "response.output_text.delta" "$WORKDIR/response_stream.txt"
  grep -q "response.completed" "$WORKDIR/response_stream.txt"
fi

echo "[INFO] Running image generation..."
img_body=$(jq -n --arg model "$image_model_path" --arg size "$IMAGE_SIZE" --argjson steps "$IMAGE_STEPS" \
  '{model:$model,prompt:"a tiny cat",size:$size,steps:$steps,response_format:"b64_json"}')
img_code=$(curl_json POST "http://$HOST:$PORT/v1/images/generations" "$img_body" "$WORKDIR/image.json")
if [[ "$img_code" != "200" ]]; then
  echo "[ERROR] image generation failed (HTTP $img_code)" >&2
  cat "$WORKDIR/image.json" >&2
  log_tail
  exit 1
fi
img_b64="$(jq -r '.data[0].b64_json' "$WORKDIR/image.json")"
if [[ -z "$img_b64" || "$img_b64" == "null" ]]; then
  echo "[ERROR] image generation returned empty b64_json" >&2
  exit 1
fi
assert_png "$img_b64"

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
  log_tail
  exit 1
fi
if ! jq -e '.text | length > 0' "$WORKDIR/asr.json" >/dev/null; then
  echo "[ERROR] ASR returned empty text" >&2
  cat "$WORKDIR/asr.json" >&2
  log_tail
  exit 1
fi

echo "[INFO] Running TTS (VibeVoice)..."
tts_body=$(jq -n --arg model "$XLLM_E2E_TTS_MODEL" '{model:$model,input:"hello",response_format:"wav"}')
tts_code=$(curl -sS -o "$WORKDIR/tts.wav" -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$tts_body" \
  "http://$HOST:$PORT/v1/audio/speech")
if [[ "$tts_code" != "200" ]]; then
  echo "[ERROR] TTS failed (HTTP $tts_code)" >&2
  cat "$WORKDIR/tts.wav" >&2
  log_tail
  exit 1
fi
if [[ ! -s "$WORKDIR/tts.wav" ]]; then
  echo "[ERROR] TTS output is empty" >&2
  exit 1
fi
assert_wav "$WORKDIR/tts.wav"

echo "[INFO] Cleaning up models..."
"$XLLM_BIN" rm "$text_repo" || true
"$XLLM_BIN" rm "$vision_repo" || true
"$XLLM_BIN" rm "$image_repo" || true
"$XLLM_BIN" rm "$asr_repo" || true

echo "[INFO] E2E real-model tests completed successfully"
