#!/usr/bin/env bash
#
# 04-api-server.sh — Start an OpenAI-compatible HTTP API server.
#
# Exposes three endpoints against a ForgeLLM interpreter-backed model:
#   POST /v1/completions
#   POST /v1/chat/completions
#   GET  /v1/models
#   GET  /health
#
# Any OpenAI-compatible client can target `http://localhost:8080/v1`.
#
# Example client call (in another terminal):
#
#   curl http://localhost:8080/v1/completions \
#        -H 'Content-Type: application/json' \
#        -d '{"model":"smol","prompt":"The meaning of life is","max_tokens":50}'
#
#   curl http://localhost:8080/v1/chat/completions \
#        -H 'Content-Type: application/json' \
#        -d '{"model":"smol","messages":[{"role":"user","content":"Hi"}]}'
#
# The server runs in the foreground; Ctrl-C to stop.

set -euo pipefail

MODEL_DIR="${HOME}/.cache/forgellm-examples"
PORT="${PORT:-8080}"

GGUF="${MODEL_DIR}/SmolLM2-135M-Instruct-Q8_0.gguf"
TOK="${MODEL_DIR}/tokenizer.json"

mkdir -p "${MODEL_DIR}"

if [ ! -f "${GGUF}" ]; then
    echo "==> Downloading SmolLM2-135M Q8_0 (~145 MB)..."
    curl -L --fail \
        "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf" \
        -o "${GGUF}"
fi

if [ ! -f "${TOK}" ]; then
    echo "==> Downloading tokenizer.json..."
    curl -L --fail \
        "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json" \
        -o "${TOK}"
fi

echo "==> Starting server on http://localhost:${PORT}"
echo "    (Ctrl-C to stop)"
echo

forge serve \
    --model "${GGUF}" \
    --tokenizer "${TOK}" \
    --port "${PORT}"
