#!/usr/bin/env bash
# Run inference via the interpreter (no compilation step).
#
# Fastest iteration for debugging or prompt engineering.
# ~100-200 tok/s on Apple M5 Pro for SmolLM2-135M.

set -euo pipefail

PROMPT="${PROMPT:-The meaning of life is}"
MAX_TOKENS="${MAX_TOKENS:-50}"

FORGE="${FORGE:-forge}"
if ! command -v "$FORGE" &> /dev/null; then
    FORGE="$(git rev-parse --show-toplevel)/target/release/forge"
    if [ ! -x "$FORGE" ]; then
        echo "Error: forge binary not found. Run 'cargo build --release' first." >&2
        exit 1
    fi
fi

CACHE="${HOME}/.cache/forgellm-examples"
mkdir -p "$CACHE"
GGUF="$CACHE/SmolLM2-135M-Instruct-Q8_0.gguf"
TOK="$CACHE/smollm2-tokenizer.json"

if [ ! -f "$GGUF" ]; then
    echo "Downloading SmolLM2-135M Q8_0 (~140 MB)..."
    curl -L --fail "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf" -o "$GGUF"
fi

if [ ! -f "$TOK" ]; then
    curl -L --fail "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json" -o "$TOK"
fi

echo ""
echo "=== Running interpreter ==="
"$FORGE" run --model "$GGUF" --tokenizer "$TOK" --prompt "$PROMPT" --num-tokens "$MAX_TOKENS"
