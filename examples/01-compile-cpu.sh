#!/usr/bin/env bash
# Compile SmolLM2-135M to a native CPU binary and run it.
#
# Uses NEON SIMD on aarch64, Apple Accelerate BLAS on macOS.
# Produces a self-contained binary with zero runtime dependencies.

set -euo pipefail

# Configurable
PROMPT="${PROMPT:-The meaning of life is}"
MAX_TOKENS="${MAX_TOKENS:-50}"

# Locate forge binary
FORGE="${FORGE:-forge}"
if ! command -v "$FORGE" &> /dev/null; then
    FORGE="$(git rev-parse --show-toplevel)/target/release/forge"
    if [ ! -x "$FORGE" ]; then
        echo "Error: forge binary not found. Run 'cargo build --release' first." >&2
        exit 1
    fi
fi

# Cache dir for models
CACHE="${HOME}/.cache/forgellm-examples"
mkdir -p "$CACHE"
GGUF="$CACHE/SmolLM2-135M-Instruct-Q8_0.gguf"
TOK="$CACHE/smollm2-tokenizer.json"

# Download model if needed (~140 MB)
if [ ! -f "$GGUF" ]; then
    echo "Downloading SmolLM2-135M Q8_0 (~140 MB)..."
    curl -L --fail "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf" -o "$GGUF"
fi

if [ ! -f "$TOK" ]; then
    echo "Downloading tokenizer..."
    curl -L --fail "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json" -o "$TOK"
fi

# Compile to CPU
OUT=/tmp/forgellm-example-cpu
echo ""
echo "Compiling to CPU target..."
rm -rf "$OUT"
"$FORGE" compile --model "$GGUF" --output "$OUT" --target cpu

echo ""
echo "Exporting weights..."
"$FORGE" export-weights --model "$GGUF" --output "$OUT/weights.bin"
cp "$TOK" "$OUT/tokenizer.json"

# Build the generated project
echo ""
echo "Building generated project (release)..."
(cd "$OUT" && cargo build --release 2>&1 | tail -5)

# Run
BIN="$OUT/target/release/$(basename "$OUT")"
echo ""
echo "=== Running inference ==="
"$BIN" "$OUT/weights.bin" "$OUT/tokenizer.json" "$PROMPT" --max-tokens "$MAX_TOKENS"
