# Development Guide

Steps for working on xLLM locally.

## Prerequisites

- CMake + C++20 compiler
- Git submodules
- Docker (optional)
- pnpm (optional, for markdownlint)

## Setup

```bash
git clone https://github.com/akiojin/xLLM.git
cd xLLM
git submodule update --init --recursive
```

## Build & Test

```bash
# Configure
cmake -S . -B build -DBUILD_TESTS=ON -DPORTABLE_BUILD=ON

# Build
cmake --build build --config Release

# Test
ctest --output-on-failure --timeout 300 --verbose
```

One-shot local run (build + tests, optional E2E):

```bash
scripts/run-local-tests.sh
```

Run with E2E:

```bash
XLLM_RUN_E2E=1 scripts/run-local-tests.sh
```

## Run

```bash
./build/xllm serve
```

## TDD Expectations

1. Write a failing test (contract/integration first, then unit).
2. Implement the minimum to make it pass.
3. Refactor with tests green.

## Required model-family tests
For gpt/nemotron/qwen/glm model families, verification is mandatory before merge.
Use the model verification suite or explicit E2E coverage and record results in the PR.

- Model verification: `.specify/scripts/model-verification/run-verification.sh --model <path> --format <gguf|safetensors> --capability TextGeneration --platform <platform>`
- Real-model E2E: `tests/e2e/real_models/run.sh` (see below)

## Real-model E2E (xLLM, all modalities)
Contract/integration tests run with test hooks (no real weights). For real-model
coverage across text, vision, image generation, ASR, and TTS:

```bash
cmake -S . -B build -DBUILD_TESTS=OFF -DPORTABLE_BUILD=ON \
  -DBUILD_WITH_WHISPER=ON -DBUILD_WITH_SD=ON -DBUILD_WITH_ONNX=OFF
cmake --build build --config Release
tests/e2e/real_models/run.sh
```

Required environment:
- `HF_TOKEN` (Hugging Face auth token for gated models)
- `XLLM_E2E_TEXT_MODEL_REF`
- `XLLM_E2E_VISION_MODEL_REF`
- `XLLM_E2E_IMAGE_MODEL_REF`
- `XLLM_E2E_ASR_MODEL_REF`
- `XLLM_E2E_TTS_MODEL` (use `vibevoice`)
- `XLLM_VIBEVOICE_RUNNER` (path to the VibeVoice runner script)
Optional overrides:
- `XLLM_E2E_IMAGE_MODEL_FILE` (explicit filename under the pulled image model dir)
- `XLLM_E2E_ASR_MODEL_FILE` (explicit filename under the pulled ASR model dir)
- `XLLM_E2E_IMAGE_STEPS` (default: 4)
- `XLLM_E2E_IMAGE_SIZE` (default: 256x256)
- `XLLM_E2E_TIMEOUT` (startup timeout seconds, default: 600)
- `XLLM_E2E_STREAMING` (default: 1; set 0 to skip streaming checks)

Notes:
- VibeVoice TTS is macOS-only; run the real-model E2E on a macOS GPU/Metal host.
- The GitHub Actions workflow `E2E Real Models` consumes the same env vars (set them as repo vars/secrets).
## Environment Variables

- `LLMLB_URL`: Optional. If set, xLLM registers itself to llmlb.
- `XLLM_PORT`, `XLLM_BIND_ADDRESS`
- `XLLM_MODELS_DIR`, `XLLM_CONFIG`, `XLLM_LOG_DIR`, `XLLM_LOG_LEVEL`, `XLLM_LOG_RETENTION_DAYS`
- `XLLM_ORIGIN_ALLOWLIST`, `XLLM_PGP_VERIFY`, `HF_TOKEN`
- `LLM_MODEL_IDLE_TIMEOUT`, `LLM_MAX_LOADED_MODELS`, `LLM_MAX_MEMORY_BYTES`

## Submodules

- `third_party/stable-diffusion.cpp` is pinned to a public fork for project-specific fixes.
- Upstream updates are synced manually on demand.
- Do not modify submodule contents directly. Use forks and update submodule pointers.
