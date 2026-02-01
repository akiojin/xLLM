#!/usr/bin/env bash

# check-markdown.sh - markdownlint (markdownlint-cli2) check
#
# Usage:
#   check-markdown.sh [--all]
#
# Defaults to linting staged Markdown files only.

set -euo pipefail

MODE="staged"
if [[ "${1:-}" == "--all" ]]; then
  MODE="all"
fi

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

run_markdownlint() {
  if command -v pnpm >/dev/null 2>&1; then
    pnpm dlx markdownlint-cli2 "$@"
    return
  fi
  if command -v npx >/dev/null 2>&1; then
    npx --no-install markdownlint-cli2 "$@"
    return
  fi
  echo "Error: pnpm or npx not found. Please install Node.js and pnpm/npm." >&2
  exit 2
}

if [[ "$MODE" == "all" ]]; then
  run_markdownlint "**/*.md" "!node_modules" "!.git" "!.github" "!.worktrees" "!third_party" "!build"
  exit 0
fi

FILES=()
while IFS= read -r file; do
  FILES+=("$file")
done < <(git diff --cached --name-only --diff-filter=ACM | rg -i '\.md$' || true)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ℹ️  No staged markdown files to lint"
  exit 0
fi

run_markdownlint "${FILES[@]}"
