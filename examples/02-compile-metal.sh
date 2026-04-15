#!/usr/bin/env bash
#
# 02-compile-metal.sh — AOT-compile SmolLM2-135M to an Apple Metal GPU binary.
#
# Requires macOS with Apple Silicon (M1/M2/M3/M4). The `metal` target generates
# a Rust project that links the Metal framework and dispatches compute shaders
# from `shaders/kernels.metal`.
#
# Expected output: GPU-accelerated text generation, typically faster than the
# CPU binary on Apple Silicon for small models.

set -euo pipefail

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: this example only works on macOS (detected: $(uname -s))" >&2
    exit 1
fi

MODEL_DIR="${HOME}/.cache/forgellm-examples"
OUT_DIR="${TMPDIR:-/tmp}/forgellm-example-metal"
PROMPT="${PROMPT:-The meaning of life is}"

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

echo "==> [1/4] forge compile --target metal"
forge compile \
    --model "${GGUF}" \
    --output "${OUT_DIR}" \
    --target metal

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
echo "==> Running Metal binary: ${BIN}"
echo
"${BIN}" \
    "${OUT_DIR}/weights.bin" \
    "${OUT_DIR}/tokenizer.json" \
    "${PROMPT}"
