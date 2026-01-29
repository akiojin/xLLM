#!/usr/bin/env bash
# check-compile.sh - C++ build check for xLLM
#
# Usage:
#   check-compile.sh

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

cmake -S . -B build -DPORTABLE_BUILD=ON
cmake --build build --config Release

echo "✅ C++ build succeeded"
