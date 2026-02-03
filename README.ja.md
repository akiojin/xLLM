# xLLM

ggml をバックエンドとする統合 C++ 推論エンジン。
llama.cpp / whisper.cpp / stable-diffusion.cpp / safetensors.cpp を統合しています。

## 概要

xLLM は複数のモデルフォーマットとモダリティをサポートする単一の推論ランタイムです。

| エンジン | フォーマット | モダリティ |
|---------|-------------|-----------|
| **llama.cpp** | GGUF | テキスト生成 |
| **safetensors.cpp** | Safetensors | テキスト生成 |
| **whisper.cpp** | GGUF | 音声認識 |
| **stable-diffusion.cpp** | GGUF / Safetensors | 画像生成 |

全エンジンは **ggml** バックエンドを共有し、統一されたハードウェアサポートを提供します:

- **Metal** (Apple Silicon)
- **CUDA** (NVIDIA)
- **ROCm** (AMD)
- **Vulkan**
- **SYCL** (Intel)
- **CPU**

単体で動作し、必要に応じて llmlb のエンドポイントとして登録できます。

## クイックスタート

### ビルド

```bash
cmake -S . -B build -DBUILD_TESTS=ON -DPORTABLE_BUILD=ON
cmake --build build --config Release
```

### 実行

```bash
./build/xllm serve
```

llmlb に登録する場合:

```bash
LLMLB_URL=http://127.0.0.1:32768 ./build/xllm serve
```

## 環境変数

| 変数 | デフォルト | 説明 |
|---|---|---|
| `LLMLB_URL` | `http://127.0.0.1:32768` | （任意）ロードバランサー登録先 |
| `XLLM_PORT` | `32769` | 待受ポート |
| `XLLM_BIND_ADDRESS` | `0.0.0.0` | バインドアドレス |
| `XLLM_MODELS_DIR` | `~/.models` | モデル保存先 |
| `XLLM_ORIGIN_ALLOWLIST` | `huggingface.co/*,cdn-lfs.huggingface.co/*` | 直接ダウンロード許可リスト |
| `XLLM_CONFIG` | `~/.config.json` | 設定ファイル |
| `XLLM_LOG_LEVEL` | `info` | ログレベル |
| `XLLM_LOG_DIR` | `~/.logs` | ログディレクトリ |
| `XLLM_LOG_RETENTION_DAYS` | `7` | ログ保持日数 |
| `XLLM_PGP_VERIFY` | `false` | HuggingFaceのPGP署名検証 |
| `HF_TOKEN` | （なし） | HuggingFaceトークン |
| `LLM_MODEL_IDLE_TIMEOUT` | `300000` | アイドルアンロード(ms) |
| `LLM_MAX_LOADED_MODELS` | `0` | 同時ロード上限（0=無制限） |
| `LLM_MAX_MEMORY_BYTES` | `0` | メモリ上限（0=無制限） |

## Docker

```bash
# ビルド（CPU）
docker build --build-arg CUDA=cpu -t xllm:latest .

# 実行
docker run --rm -p 32769:32769   -e LLMLB_URL=http://host.docker.internal:32768   xllm:latest
```

## 構成

```text
.
├── src
├── include
├── tests
├── engines
├── third_party
├── specs
└── docs
```

## 開発ガイド

`DEVELOPMENT.md` を参照してください。

## ライセンス

MIT License
