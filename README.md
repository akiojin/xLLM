# xLLM

C++ inference engine based on llama.cpp and companion runtimes (whisper.cpp, stable-diffusion.cpp, safetensors.cpp).

This project can run standalone or register itself to llmlb as an endpoint.

## Quick Start

### Build

```bash
cmake -S . -B build -DBUILD_TESTS=ON -DPORTABLE_BUILD=ON
cmake --build build --config Release
```

### Run

```bash
./build/xllm serve
```

To register with llmlb automatically:

```bash
LLMLB_URL=http://127.0.0.1:32768 ./build/xllm serve
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLMLB_URL` | `http://127.0.0.1:32768` | Optional load balancer URL to register with |
| `XLLM_PORT` | `32769` | Listen port |
| `XLLM_BIND_ADDRESS` | `0.0.0.0` | Bind address |
| `XLLM_MODELS_DIR` | `~/.models` | Model storage directory |
| `XLLM_ORIGIN_ALLOWLIST` | `huggingface.co/*,cdn-lfs.huggingface.co/*` | Allowlist for direct downloads |
| `XLLM_CONFIG` | `~/.config.json` | Config file path |
| `XLLM_LOG_LEVEL` | `info` | Log level |
| `XLLM_LOG_DIR` | `~/.logs` | Log directory |
| `XLLM_LOG_RETENTION_DAYS` | `7` | Log retention (days) |
| `XLLM_PGP_VERIFY` | `false` | Verify HuggingFace PGP signatures when available |
| `HF_TOKEN` | (none) | HuggingFace API token for gated models |
| `LLM_MODEL_IDLE_TIMEOUT` | `300000` | Idle unload timeout (ms) |
| `LLM_MAX_LOADED_MODELS` | `0` | Max loaded models (0 = unlimited) |
| `LLM_MAX_MEMORY_BYTES` | `0` | Max memory bytes (0 = unlimited) |

## Docker

```bash
# Build (CPU)
docker build --build-arg CUDA=cpu -t xllm:latest .

# Run
docker run --rm -p 32769:32769   -e LLMLB_URL=http://host.docker.internal:32768   xllm:latest
```

## Project Structure

```
.
├── src
├── include
├── tests
├── engines
├── third_party
├── specs
└── docs
```

## Development

See `DEVELOPMENT.md`.

## License

MIT License
