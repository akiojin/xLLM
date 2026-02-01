# CLAUDE.md

このリポジトリは xLLM（ggml ベースの統合 C++ 推論エンジン）専用です。

## まず読む

- 目的: 複数のモデルフォーマット・モダリティを単一ランタイムで提供する
- ドキュメント: `README.md` → `DEVELOPMENT.md` → `specs/`
- 依存: `third_party/` はサブモジュール管理（直接編集禁止）

## アーキテクチャ

```text
xLLM
├── llama.cpp        (GGUF → テキスト生成)
├── safetensors.cpp  (Safetensors → テキスト生成)
├── whisper.cpp      (GGUF → 音声認識)
├── stable-diffusion.cpp (GGUF/Safetensors → 画像生成)
└── ggml             (共通バックエンド)
```

**ハードウェア対応** (ggml経由で全エンジン共通):

- Metal / CUDA / ROCm / Vulkan / SYCL / CPU

## ディレクトリ構成

```text
.
├── src           # メインソース
├── include       # ヘッダ
├── tests         # テスト
├── engines       # エンジン実装
├── third_party   # サブモジュール (llama.cpp, whisper.cpp, etc.)
├── specs         # 仕様書
├── docs          # ドキュメント
├── scripts       # ユーティリティスクリプト
└── templates     # テンプレート
```

## 開発ルール

- 実装前に仕様（`specs/`）を確認する
- TDD: RED → GREEN → REFACTOR
- 既存ファイルの改修を優先（不要な新規ファイル増殖は避ける）
- サブモジュールはフォーク運用。必要ならポインタ更新のみ

## ローカル検証

```bash
cmake -S . -B build -DBUILD_TESTS=ON -DPORTABLE_BUILD=ON
cmake --build build --config Release
ctest --output-on-failure --timeout 300 --verbose
```
