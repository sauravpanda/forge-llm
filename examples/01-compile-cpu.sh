#!/usr/bin/env bash
#
# 01-compile-cpu.sh — AOT-compile SmolLM2-135M to a self-contained CPU binary.
#
# This is the canonical ForgeLLM workflow:
#   1. `forge compile` generates a Rust project from a GGUF model.
#   2. `forge export-weights` extracts dequantized weights into `weights.bin`.
#   3. `cargo build --release` produces a native binary with no runtime interpreter.
#   4. The binary runs inference directly.
#
# Expected output: a continuation of the prompt plus tokens/sec stats.

set -euo pipefail

MODEL_DIR="${HOME}/.cache/forgellm-examples"
OUT_DIR="${TMPDIR:-/tmp}/forgellm-example-cpu"
PROMPT="${PROMPT:-The meaning of life is}"
MAX_TOKENS="${MAX_TOKENS:-50}"

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

rm -rf "${OUT_DIR}"

echo "==> [1/4] forge compile --target cpu"
forge compile \
    --model "${GGUF}" \
    --output "${OUT_DIR}" \
    --target cpu

echo "==> [2/4] forge export-weights"
forge export-weights \
    --model "${GGUF}" \
    --output "${OUT_DIR}/weights.bin"

echo "==> [3/4] copying tokenizer"
cp "${TOK}" "${OUT_DIR}/tokenizer.json"

echo "==> [4/4] cargo build --release"
(cd "${OUT_DIR}" && cargo build --release)

BIN_NAME="$(basename "${OUT_DIR}")"
BIN="${OUT_DIR}/target/release/${BIN_NAME}"

echo
echo "==> Running compiled binary: ${BIN}"
echo
"${BIN}" \
    "${OUT_DIR}/weights.bin" \
    "${OUT_DIR}/tokenizer.json" \
    "${PROMPT}" \
    --max-tokens "${MAX_TOKENS}"
