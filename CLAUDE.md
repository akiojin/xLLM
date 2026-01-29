# CLAUDE.md

このリポジトリは xLLM（C++推論エンジン）専用です。

## まず読む

- 目的: llama.cppベースのxLLMランタイムを単体で提供する
- ドキュメント: `README.md` → `DEVELOPMENT.md` → `specs/`
- 依存: `third_party/` はサブモジュール管理（直接編集禁止）

## ディレクトリ構成

```
.
├── src
├── include
├── tests
├── engines
├── third_party
├── specs
├── docs
├── scripts
└── templates
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
