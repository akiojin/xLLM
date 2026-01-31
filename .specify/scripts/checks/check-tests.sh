#!/usr/bin/env bash
# check-tests.sh - C++ tests for xLLM
#
# Usage:
#   check-tests.sh

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

if [ ! -d "build" ]; then
  echo "⚠️  build directory not found, skipping ctest"
  exit 0
fi

if [ ! -f "build/CTestTestfile.cmake" ]; then
  echo "⚠️  CTest not configured in build/, skipping ctest"
  exit 0
fi

ctest --output-on-failure --timeout 300 --verbose -C Release
