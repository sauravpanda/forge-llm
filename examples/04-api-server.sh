#!/usr/bin/env bash
# Start an OpenAI-compatible API server for SmolLM2-135M.
#
# Compiles to Metal (or CPU), then starts a server with:
#   POST /v1/chat/completions (streaming + non-streaming)
#   GET  /v1/models
#   GET  /health
#
# Drop-in replacement for llama.cpp server or OpenAI for local development.

set -euo pipefail

PORT="${PORT:-8080}"
TARGET="${TARGET:-metal}"

if [[ "$TARGET" == "metal" && "$(uname)" != "Darwin" ]]; then
    echo "Metal target requires macOS. Set TARGET=cpu to use CPU instead." >&2
    exit 1
fi

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

OUT="/tmp/forgellm-example-server-$TARGET"
if [ ! -f "$OUT/target/release/$(basename $OUT)" ]; then
    echo ""
    echo "Compiling to $TARGET target..."
    rm -rf "$OUT"
    "$FORGE" compile --model "$GGUF" --output "$OUT" --target "$TARGET"
    "$FORGE" export-weights --model "$GGUF" --output "$OUT/weights.bin"
    cp "$TOK" "$OUT/tokenizer.json"
    (cd "$OUT" && cargo build --release 2>&1 | tail -3)
fi

BIN="$OUT/target/release/$(basename "$OUT")"
echo ""
echo "=== Starting ForgeLLM server on http://localhost:$PORT ==="
echo ""
echo "Try these curl commands in another terminal:"
echo ""
echo "  # Health check"
echo "  curl http://localhost:$PORT/health"
echo ""
echo "  # Non-streaming chat"
echo "  curl http://localhost:$PORT/v1/chat/completions \\"
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'"'"''
echo ""
echo "  # Streaming chat"
echo "  curl http://localhost:$PORT/v1/chat/completions \\"
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"messages": [{"role": "user", "content": "Hello!"}], "stream": true, "max_tokens": 50}'"'"''
echo ""
echo "Press Ctrl-C to stop."
echo ""

"$BIN" "$OUT/weights.bin" "$OUT/tokenizer.json" --serve --port "$PORT"
