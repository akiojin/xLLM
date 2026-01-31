# Reference Models (Source of Truth)

This file defines the concrete model IDs to be used for mandatory verification/tests
across xLLM specs. Update this list whenever required model families change.

Official test targets:
- Only the official repos listed below are valid for mandatory tests.
- Do not use third-party reuploads for tests.
- If an official repo provides GGUF, prefer that GGUF; otherwise use safetensors.

Testing rule:
- Model downloads for tests MUST use `xllm pull` (do not use direct downloads or custom scripts).
- After each test, delete the model with `xllm rm <MODEL>` to avoid leaving large artifacts.

## Text (safetensors)

Selection constraints (keep in sync with user requirements):
- Qwen must be Qwen3 series or later.
- Use Qwen3-Coder family for Qwen reference models.
- GLM reference includes GLM-4.7-Flash.
- Use official model repos (not third-party reuploads).

| Family | Tier | Hugging Face repo | Notes |
|--------|------|-------------------|-------|
| gpt-oss | ~20B | openai/gpt-oss-20b | Official safetensors (MoE + MXFP4) |
| gpt-oss | ~120B | openai/gpt-oss-120b | Official safetensors (MoE + MXFP4) |
| nemotron3 | ~30B | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | Official safetensors (MoE + Mamba-Transformer) |
| qwen3-coder | ~30B | Qwen/Qwen3-Coder-30B-A3B-Instruct | Qwen3-Coder (safetensors) |
| qwen3-coder | large | Qwen/Qwen3-Coder-480B-A35B-Instruct | Qwen3-Coder largest tier (no ~120B-class model published) |
| glm-4.7 | ~30B | zai-org/GLM-4.7-Flash | GLM-4.7-Flash (30B-A3B MoE) |
| glm-4.7 | large | zai-org/GLM-4.7 | GLM-4.7 main model (largest tier) |
| deepseek | TBD | deepseek-ai/DeepSeek-V3.2 | User-specified reference model |
| minimax | TBD | MiniMaxAI/MiniMax-M2.1 | User-specified reference model |
| llama | 8B | meta-llama/Llama-3.1-8B-Instruct | User-specified reference model |
