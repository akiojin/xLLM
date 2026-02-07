@claude

# Reference Models (Source of Truth)

This file defines the concrete model IDs to be used for mandatory verification/tests
across xLLM specs. Update this list whenever required model families change.

Official test targets:

- Only the official repos listed below are valid for mandatory tests.
- Do not use third-party reuploads for tests.
- If an official repo provides GGUF, prefer that GGUF; otherwise use safetensors.

Testing rule:

- Model downloads for tests MUST use `xllm pull` (server mode or `--direct`; no external scripts).
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
| qwen3-coder-next | ~80B | Qwen/Qwen3-Coder-Next | Qwen3-Coder-Next (safetensors, official) |
| qwen3-coder | large | Qwen/Qwen3-Coder-480B-A35B-Instruct | Qwen3-Coder largest tier (no ~120B-class model published) |
| glm-4.7 | ~30B | zai-org/GLM-4.7-Flash | GLM-4.7-Flash (30B-A3B MoE) |
| glm-4.7 | large | zai-org/GLM-4.7 | GLM-4.7 main model (largest tier) |
| deepseek | TBD | deepseek-ai/DeepSeek-V3.2 | User-specified reference model |
| minimax | TBD | MiniMaxAI/MiniMax-M2.1 | User-specified reference model |
| llama | 8B | meta-llama/Llama-3.1-8B-Instruct | User-specified reference model |

## Vision (image-text-to-text)

### Official reference models

| Role | Hugging Face repo | Notes |
|------|-------------------|-------|
| VLM | Qwen/Qwen2.5-VL-7B-Instruct | Official Qwen VL family |
| VLM | Qwen/Qwen2.5-VL-32B-Instruct | Official Qwen VL family |
| VLM | Qwen/Qwen3-VL-8B-Instruct | Official Qwen3 VL family |
| VLM | OpenGVLab/InternVL3-78B | Official InternVL family |
| VLM | google/gemma-3-27b-it | Official Google Gemma family |

### Popular models (official repos only, HF Most downloads, 2026-02-01)

| Rank | Hugging Face repo | Notes |
|------|-------------------|-------|
| 1 | Qwen/Qwen2.5-VL-3B-Instruct | Popular VL model |
| 2 | Qwen/Qwen2.5-VL-7B-Instruct | Popular VL model |
| 3 | deepseek-ai/DeepSeek-OCR | OCR-focused VLM |
| 4 | OpenGVLab/InternVL3-78B | Popular large VLM |
| 5 | openvla/openvla-7b | Popular openvla model |

Notes:

- xLLM vision requires GGUF + mmproj; most official repos ship transformers/safetensors and need conversion.

## Vision (image-to-text)

### Official reference models

| Role | Hugging Face repo | Notes |
|------|-------------------|-------|
| Caption | Salesforce/blip-image-captioning-base | Official BLIP captioning |
| Caption | Salesforce/blip-image-captioning-large | Official BLIP captioning |
| OCR | microsoft/trocr-large-printed | Official TrOCR |
| Doc | PaddlePaddle/UVDoc | Official PaddlePaddle doc model |
| Caption | Salesforce/blip2-opt-2.7b-coco | Official BLIP-2 captioning |

### Popular models (official repos only, HF Most downloads, 2026-02-01)

| Rank | Hugging Face repo | Notes |
|------|-------------------|-------|
| 1 | Salesforce/blip-image-captioning-base | Popular captioning |
| 2 | Salesforce/blip-image-captioning-large | Popular captioning |
| 3 | PaddlePaddle/UVDoc | Popular document model |
| 4 | microsoft/trocr-large-printed | Popular OCR model |
| 5 | PaddlePaddle/PP-LCNet_x1_0_doc_ori | Popular document model |

Notes:

- xLLM vision requires GGUF + mmproj; most official repos ship transformers/safetensors and need conversion.

## Image generation (text-to-image)

### Official reference models

| Role | Hugging Face repo | Notes |
|------|-------------------|-------|
| SDXL | stabilityai/stable-diffusion-xl-base-1.0 | Official SDXL base |
| SD | CompVis/stable-diffusion-v1-4 | Official SD v1 |
| Turbo | stabilityai/sd-turbo | Official SD Turbo |
| Turbo | stabilityai/sdxl-turbo | Official SDXL Turbo |
| SD | stable-diffusion-v1-5/stable-diffusion-v1-5 | Official SD v1.5 |

### Popular models (official repos only, HF Most downloads, 2026-02-01)

| Rank | Hugging Face repo | Notes |
|------|-------------------|-------|
| 1 | stabilityai/stable-diffusion-xl-base-1.0 | Popular SDXL base |
| 2 | stable-diffusion-v1-5/stable-diffusion-v1-5 | Popular SD v1.5 |
| 3 | CompVis/stable-diffusion-v1-4 | Popular SD v1 |
| 4 | stabilityai/sd-turbo | Popular SD Turbo |
| 5 | stabilityai/sdxl-turbo | Popular SDXL Turbo |

Notes:

- stable-diffusion.cpp supports `.safetensors` or `.ckpt` weights from official repos.

## ASR (automatic-speech-recognition)

### Official reference models

| Role | Hugging Face repo | Notes |
|------|-------------------|-------|
| Whisper | openai/whisper-large-v3 | Official Whisper |
| Whisper | openai/whisper-large-v3-turbo | Official Whisper Turbo |
| Whisper | openai/whisper-medium | Official Whisper |
| Whisper | openai/whisper-small | Official Whisper |
| Whisper | openai/whisper-base | Official Whisper |

### Popular models (official repos only, HF Most downloads, 2026-02-01)

| Rank | Hugging Face repo | Notes |
|------|-------------------|-------|
| 1 | openai/whisper-large-v3 | Popular Whisper |
| 2 | openai/whisper-large-v3-turbo | Popular Whisper Turbo |
| 3 | openai/whisper-small | Popular Whisper |
| 4 | openai/whisper-base | Popular Whisper |
| 5 | openai/whisper-medium | Popular Whisper |

Notes:

- xLLM ASR is whisper.cpp-based and expects `.bin` or `.gguf` (official repos require conversion).

## TTS (text-to-speech)

### Official reference models

| Role | Hugging Face repo | Notes |
|------|-------------------|-------|
| VibeVoice | microsoft/VibeVoice-1.5B | Official VibeVoice |
| VibeVoice | microsoft/VibeVoice-Realtime-0.5B | Official VibeVoice realtime |

### Popular models (official repos only, HF Most downloads, 2026-02-01)

| Rank | Hugging Face repo | Notes |
|------|-------------------|-------|
| 1 | microsoft/VibeVoice-1.5B | Popular VibeVoice |
| 2 | microsoft/VibeVoice-Realtime-0.5B | Popular VibeVoice realtime |

Notes:

- xLLM TTS currently supports VibeVoice only and requires `XLLM_VIBEVOICE_RUNNER`. Popular list is limited to supported models.
