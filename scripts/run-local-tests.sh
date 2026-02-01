#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${XLLM_BUILD_DIR:-$ROOT_DIR/build}"
BUILD_TYPE="${XLLM_BUILD_TYPE:-Release}"
DEFAULT_PORTABLE="ON"
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  DEFAULT_PORTABLE="OFF"
fi
CMAKE_FLAGS="${XLLM_CMAKE_FLAGS:--DBUILD_TESTS=ON -DPORTABLE_BUILD=$DEFAULT_PORTABLE}"
RUN_E2E="${XLLM_RUN_E2E:-0}"

echo "[INFO] Configuring build in $BUILD_DIR"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" $CMAKE_FLAGS

echo "[INFO] Building ($BUILD_TYPE)"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE"

echo "[INFO] Running C++ tests"
ctest --output-on-failure --timeout 300 --verbose --test-dir "$BUILD_DIR"

if [[ "$RUN_E2E" == "1" ]]; then
  echo "[INFO] Running real-model E2E"
  "$ROOT_DIR/tests/e2e/real_models/run.sh"
else
  if [[ -n "${XLLM_E2E_TEXT_MODEL_REF:-}" && -n "${XLLM_E2E_VISION_MODEL_REF:-}" && -n "${XLLM_E2E_IMAGE_MODEL_REF:-}" && -n "${XLLM_E2E_ASR_MODEL_REF:-}" && -n "${XLLM_E2E_TTS_MODEL:-}" && -n "${XLLM_VIBEVOICE_RUNNER:-}" ]]; then
    echo "[INFO] Running real-model E2E (env detected)"
    "$ROOT_DIR/tests/e2e/real_models/run.sh"
  else
    echo "[INFO] Skipping real-model E2E (set XLLM_RUN_E2E=1 or required envs)"
  fi
fi
