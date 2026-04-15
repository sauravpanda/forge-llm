#!/usr/bin/env bash
# Compile SmolLM2-135M to a Metal GPU binary for Apple Silicon.
#
# Generates Metal compute shaders with simdgroup reductions, packed_short4
# Q8_0 loads, fused QKV projections, and batched prefill.

set -euo pipefail

# macOS only
if [[ "$(uname)" != "Darwin" ]]; then
    echo "This example requires macOS with Apple Silicon. On Linux, use 01-compile-cpu.sh" >&2
    exit 1
fi

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

OUT=/tmp/forgellm-example-metal
echo ""
echo "Compiling to Metal target..."
rm -rf "$OUT"
"$FORGE" compile --model "$GGUF" --output "$OUT" --target metal

echo ""
echo "Exporting weights..."
"$FORGE" export-weights --model "$GGUF" --output "$OUT/weights.bin"
cp "$TOK" "$OUT/tokenizer.json"

echo ""
echo "Building generated Metal project (release)..."
(cd "$OUT" && cargo build --release 2>&1 | tail -5)

BIN="$OUT/target/release/$(basename "$OUT")"
echo ""
echo "=== Running Metal inference ==="
"$BIN" "$OUT/weights.bin" "$OUT/tokenizer.json" "$PROMPT" --max-tokens "$MAX_TOKENS"
