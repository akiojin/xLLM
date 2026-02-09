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

Apple Silicon note:

- `PORTABLE_BUILD=ON` uses `-march=x86-64`. On arm64 macOS, set `XLLM_CMAKE_FLAGS` to disable it,
  or rely on the default in `scripts/run-local-tests.sh` which flips it to OFF.

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

- Model verification: `.specify/scripts/model-verification/run-verification.sh --model <path> --format <gguf|safetensors>`
  `--capability TextGeneration --platform`
- Real-model E2E: `tests/e2e/real_models/run.sh` (see below)

## Real-model E2E

Moved to `specs/SPEC-1dbf2acb/spec.md`.

## Environment Variables

- `LLMLB_URL`: Optional. If set, xLLM registers itself to llmlb.
- `XLLM_PORT`, `XLLM_BIND_ADDRESS`
- `XLLM_MODELS_DIR`, `XLLM_LOG_DIR`, `XLLM_LOG_LEVEL`, `XLLM_LOG_RETENTION_DAYS`
- `XLLM_ORIGIN_ALLOWLIST`, `XLLM_PGP_VERIFY`, `HF_TOKEN`
- `LLM_MODEL_IDLE_TIMEOUT`, `LLM_MAX_LOADED_MODELS`, `LLM_MAX_MEMORY_BYTES`

## Submodules

- `third_party/stable-diffusion.cpp` is pinned to a public fork for project-specific fixes.
- Upstream updates are synced manually on demand.
- Do not modify submodule contents directly. Use forks and update submodule pointers.
