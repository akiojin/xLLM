#!/usr/bin/env bash

# check-markdown.sh - markdownlint (markdownlint-cli2) check
#
# Usage:
#   check-markdown.sh [--all] [--range <range>]
#
# Defaults to linting staged Markdown files only.

set -euo pipefail

MODE="staged"
RANGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      MODE="all"
      shift
      ;;
    --range)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Error: --range requires a git range" >&2
        exit 2
      fi
      MODE="range"
      RANGE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

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
collect_files() {
  while IFS= read -r file; do
    case "$file" in
      third_party/*|.git/*|.github/*|.worktrees/*|build/*|node_modules/*)
        continue
        ;;
    esac
    if [[ ! -f "$file" ]]; then
      continue
    fi
    FILES+=("$file")
  done
}

if [[ "$MODE" == "range" ]]; then
  if [[ -z "$RANGE" ]]; then
    echo "Error: --range requires a git range" >&2
    exit 2
  fi
  collect_files < <(git diff --name-only --diff-filter=AM "$RANGE" | rg -i '\.md$' || true)
else
  collect_files < <(git diff --cached --name-only --diff-filter=ACM | rg -i '\.md$' || true)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ℹ️  No markdown files to lint"
  exit 0
fi

run_markdownlint "${FILES[@]}"
