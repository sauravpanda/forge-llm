#!/usr/bin/env bash
# ForgeLLM Benchmark Suite
# Usage: ./benchmarks/run.sh [--skip-download] [--aot-only] [--interp-only]
#
# Downloads SmolLM2-135M if not cached, then runs:
#   1. Interpreter benchmark (forge bench)
#   2. AOT compile + benchmark
#   3. Records results to benchmarks/results/<version>.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
AOT_DIR="/tmp/forgellm-bench-aot"

# Parse args
SKIP_DOWNLOAD=false
AOT_ONLY=false
INTERP_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --aot-only) AOT_ONLY=true ;;
        --interp-only) INTERP_ONLY=true ;;
    esac
done

# Get version from Cargo.toml
VERSION=$(grep '^version' "$PROJECT_DIR/Cargo.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')
echo "=== ForgeLLM Benchmark Suite v${VERSION} ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# --- Download model if needed ---
MODEL_ID="bartowski/SmolLM2-135M-Instruct-GGUF"
MODEL_FILE="SmolLM2-135M-Instruct-Q8_0.gguf"
TOKENIZER_ID="HuggingFaceTB/SmolLM2-135M-Instruct"

# Try to find cached model
GGUF_PATH=$(find ~/.cache/huggingface/hub/models--bartowski--SmolLM2-135M-Instruct-GGUF -name "$MODEL_FILE" 2>/dev/null | head -1)
TOK_PATH=$(find ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct -name "tokenizer.json" 2>/dev/null | head -1)

if [ -z "$GGUF_PATH" ] && [ "$SKIP_DOWNLOAD" = false ]; then
    echo "Downloading $MODEL_FILE..."
    GGUF_PATH=$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$MODEL_ID', '$MODEL_FILE'))")
fi

if [ -z "$TOK_PATH" ] && [ "$SKIP_DOWNLOAD" = false ]; then
    echo "Downloading tokenizer..."
    TOK_PATH=$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$TOKENIZER_ID', 'tokenizer.json'))")
fi

if [ -z "$GGUF_PATH" ] || [ -z "$TOK_PATH" ]; then
    echo "ERROR: Model or tokenizer not found. Run without --skip-download."
    exit 1
fi

echo "Model:     $GGUF_PATH"
echo "Tokenizer: $TOK_PATH"
echo ""

# --- Build release ---
echo "Building forge (release)..."
cd "$PROJECT_DIR"
cargo build --release -p forgellm-cli 2>&1 | tail -1
FORGE="$PROJECT_DIR/target/release/forge"
echo ""

# --- System info ---
ARCH=$(uname -m)
OS=$(uname -s)
CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || cat /proc/cpuinfo 2>/dev/null | grep "model name" | head -1 | cut -d: -f2 | xargs || echo "unknown")
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
echo "System: $OS $ARCH | $CPU | $CORES cores"
echo ""

# Prepare results
mkdir -p "$RESULTS_DIR"
RESULT_FILE="$RESULTS_DIR/${VERSION}.json"

# Initialize JSON
cat > "$RESULT_FILE" << ENDJSON
{
  "version": "$VERSION",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "system": {
    "os": "$OS",
    "arch": "$ARCH",
    "cpu": "$CPU",
    "cores": "$CORES"
  },
  "model": {
    "name": "SmolLM2-135M-Instruct",
    "quant": "Q8_0",
    "params": "135M"
  },
  "benchmarks": {
ENDJSON

COMMA=""

# --- Interpreter Benchmark ---
if [ "$AOT_ONLY" = false ]; then
    echo "=== Interpreter Benchmark ==="
    echo "Running: forge bench --model ... --num-tokens 64 --runs 3"
    INTERP_OUTPUT=$("$FORGE" bench \
        --model "$GGUF_PATH" \
        --tokenizer "$TOK_PATH" \
        --num-tokens 64 \
        --runs 3 \
        --prompt "The meaning of life is" 2>&1) || true
    echo "$INTERP_OUTPUT"

    # Extract tok/s from output (look for "Average:" or "Generate:" lines)
    INTERP_TOKS=$(echo "$INTERP_OUTPUT" | grep -oE '[0-9]+\.[0-9]+ tok/s' | tail -1 | grep -oE '[0-9]+\.[0-9]+')
    if [ -z "$INTERP_TOKS" ]; then
        INTERP_TOKS="0.0"
    fi
    echo ""
    echo "Interpreter: ${INTERP_TOKS} tok/s"
    echo ""

    cat >> "$RESULT_FILE" << ENDJSON
    "interpreter": {
      "generate_tok_s": $INTERP_TOKS,
      "num_tokens": 64,
      "runs": 3
    }
ENDJSON
    COMMA=","
fi

# --- AOT Benchmark ---
if [ "$INTERP_ONLY" = false ]; then
    echo "=== AOT Compilation Benchmark ==="

    # Clean previous AOT build
    rm -rf "$AOT_DIR"

    # Step 1: Compile
    echo "Step 1: forge compile..."
    COMPILE_START=$(date +%s)
    "$FORGE" compile \
        --model "$GGUF_PATH" \
        --output "$AOT_DIR" \
        --target cpu 2>&1
    echo ""

    # Step 2: Export weights
    echo "Step 2: Export weights..."
    "$FORGE" export-weights \
        --model "$GGUF_PATH" \
        --output "$AOT_DIR/weights.bin" 2>&1
    echo ""

    # Step 3: Copy tokenizer
    cp "$TOK_PATH" "$AOT_DIR/tokenizer.json"

    # Step 4: Build AOT binary
    echo "Step 3: cargo build --release (AOT binary)..."
    BUILD_START=$(date +%s)
    cd "$AOT_DIR"
    cargo build --release 2>&1 | tail -3
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
    echo "Build time: ${BUILD_TIME}s"
    echo ""

    # Step 5: Run AOT benchmark (3 runs)
    AOT_BIN="$AOT_DIR/target/release/$(ls "$AOT_DIR/target/release/" | grep -v '\.d$' | grep -v '\.dSYM' | grep -v 'deps' | grep -v 'build' | grep -v 'examples' | grep -v 'incremental' | grep -v '\.' | head -1)"

    echo "Step 4: Running AOT binary (3 runs, 64 tokens each)..."
    BEST_TOKS="0.0"
    TOTAL_TOKS="0.0"
    for run in 1 2 3; do
        echo "  Run $run/3..."
        AOT_OUTPUT=$("$AOT_BIN" "$AOT_DIR/weights.bin" "$AOT_DIR/tokenizer.json" "The meaning of life is" --max-tokens 64 --quiet --temp 0.7 --top-k 40 2>&1) || true
        RUN_TOKS=$(echo "$AOT_OUTPUT" | grep -oE '[0-9]+\.[0-9]+ tok/s' | tail -1 | grep -oE '[0-9]+\.[0-9]+')
        if [ -n "$RUN_TOKS" ]; then
            echo "    ${RUN_TOKS} tok/s"
            # Track best
            if [ "$(echo "$RUN_TOKS > $BEST_TOKS" | bc -l 2>/dev/null || python3 -c "print('1' if $RUN_TOKS > $BEST_TOKS else '0')")" = "1" ]; then
                BEST_TOKS="$RUN_TOKS"
            fi
            TOTAL_TOKS=$(python3 -c "print($TOTAL_TOKS + $RUN_TOKS)")
        fi
    done
    AVG_TOKS=$(python3 -c "print(round($TOTAL_TOKS / 3, 1))")
    echo ""
    echo "AOT: best=${BEST_TOKS} tok/s, avg=${AVG_TOKS} tok/s"
    echo "Build time: ${BUILD_TIME}s"

    COMPILE_END=$(date +%s)
    TOTAL_COMPILE_TIME=$((COMPILE_END - COMPILE_START))

    # Get binary size
    BIN_SIZE=$(ls -l "$AOT_BIN" | awk '{print $5}')
    BIN_SIZE_MB=$(python3 -c "print(round($BIN_SIZE / 1e6, 1))")

    cat >> "$RESULT_FILE" << ENDJSON
${COMMA}    "aot": {
      "generate_tok_s_best": $BEST_TOKS,
      "generate_tok_s_avg": $AVG_TOKS,
      "num_tokens": 64,
      "runs": 3,
      "build_time_s": $BUILD_TIME,
      "total_compile_time_s": $TOTAL_COMPILE_TIME,
      "binary_size_mb": $BIN_SIZE_MB
    }
ENDJSON

    cd "$PROJECT_DIR"
    echo ""
fi

# Close JSON
cat >> "$RESULT_FILE" << ENDJSON
  }
}
ENDJSON

echo ""
echo "=== Results saved to $RESULT_FILE ==="
cat "$RESULT_FILE"

# --- Append to HISTORY.md ---
HISTORY_FILE="$SCRIPT_DIR/HISTORY.md"
if [ ! -f "$HISTORY_FILE" ]; then
    cat > "$HISTORY_FILE" << 'ENDMD'
# ForgeLLM Benchmark History

Performance tracking across versions. All benchmarks run on SmolLM2-135M-Instruct Q8_0 (64 tokens).

| Version | Date | Interpreter (tok/s) | AOT (tok/s) | AOT Build (s) | Binary Size | System |
|---------|------|---------------------|-------------|---------------|-------------|--------|
ENDMD
fi

# Add row
DATE=$(date -u +%Y-%m-%d)
if [ "$AOT_ONLY" = false ] && [ "$INTERP_ONLY" = false ]; then
    echo "| v${VERSION} | ${DATE} | ${INTERP_TOKS} | ${AVG_TOKS} (best: ${BEST_TOKS}) | ${BUILD_TIME}s | ${BIN_SIZE_MB} MB | ${OS} ${ARCH} ${CORES}c |" >> "$HISTORY_FILE"
elif [ "$INTERP_ONLY" = true ]; then
    echo "| v${VERSION} | ${DATE} | ${INTERP_TOKS} | - | - | - | ${OS} ${ARCH} ${CORES}c |" >> "$HISTORY_FILE"
else
    echo "| v${VERSION} | ${DATE} | - | ${AVG_TOKS} (best: ${BEST_TOKS}) | ${BUILD_TIME}s | ${BIN_SIZE_MB} MB | ${OS} ${ARCH} ${CORES}c |" >> "$HISTORY_FILE"
fi

echo ""
echo "=== Benchmark History ==="
cat "$HISTORY_FILE"
