#!/usr/bin/env bash
#
# 03-run-interpreter.sh — Run a GGUF model directly via `forge run`.
#
# This skips the compile step entirely and uses the built-in interpreter.
# It is the fastest path from "have a GGUF file" to "generating text" and is
# ideal for experimentation, debugging, and quick comparisons between models.
#
# Performance is typically lower than an AOT-compiled binary (example 01)
# because the interpreter walks the IR graph at runtime. Use `forge compile`
# (examples 01/02) when you need maximum throughput.
#
# Expected output: a continuation of the prompt printed to stdout plus
# prefill/decode stats on stderr.

set -euo pipefail

MODEL_DIR="${HOME}/.cache/forgellm-examples"
PROMPT="${PROMPT:-The meaning of life is}"
MAX_TOKENS="${MAX_TOKENS:-50}"
TEMPERATURE="${TEMPERATURE:-0.7}"

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

echo "==> forge run"
forge run \
    --model "${GGUF}" \
    --tokenizer "${TOK}" \
    --prompt "${PROMPT}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}"
