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

## Run

```bash
./build/xllm serve
```

## TDD Expectations

1. Write a failing test (contract/integration first, then unit).
2. Implement the minimum to make it pass.
3. Refactor with tests green.

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
