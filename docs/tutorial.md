# Compile Your First Model in 5 Minutes

This tutorial walks you through compiling SmolLM2-135M into a native Metal GPU binary and running inference on Apple Silicon.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3 (for downloading the model)

## Step 1: Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

Verify with `rustc --version` (1.75+ required).

## Step 2: Clone and Build ForgeLLM

```bash
git clone https://github.com/sauravpanda/forge-llm.git
cd forge-llm
cargo build --release
```

This builds the `forge` CLI at `target/release/forge`.

## Step 3: Download a Small Model

SmolLM2-135M is a 244 MB model -- small enough to download quickly, large enough to generate coherent text.

```bash
pip install huggingface-hub

# Download the Q8_0 quantized GGUF model
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('bartowski/SmolLM2-135M-Instruct-GGUF', 'SmolLM2-135M-Instruct-Q8_0.gguf')
print(path)
"

# Download the tokenizer
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('HuggingFaceTB/SmolLM2-135M-Instruct', 'tokenizer.json')
print(path)
"
```

Note the paths printed by each command. You'll use them below (referred to as `MODEL_PATH` and `TOKENIZER_PATH`).

## Step 4: Inspect the Model

```bash
./target/release/forge info MODEL_PATH
```

Expected output:
```
Architecture: Llama
Hidden size:  576
Intermediate: 1536
Layers:       30
Attn heads:   9
KV heads:     3
Head dim:     64
Vocab size:   49152
Max seq len:  8192
```

## Step 5: Compile to a Metal Binary

```bash
./target/release/forge compile \
  --model MODEL_PATH \
  --output ./smollm-metal \
  --target metal

./target/release/forge export-weights \
  --model MODEL_PATH \
  --output ./smollm-metal/weights.bin

cp TOKENIZER_PATH ./smollm-metal/tokenizer.json
```

This generates a complete Cargo project at `./smollm-metal/` with:
- Metal Shading Language compute shaders (matmul, attention, RMSNorm, RoPE, etc.)
- Rust host code using the `metal` crate for GPU dispatch
- All model dimensions baked in as compile-time constants

Now build it:

```bash
cd smollm-metal
cargo build --release
```

## Step 6: Run Inference

```bash
./target/release/smollm-metal weights.bin tokenizer.json "The meaning of life is"
```

You should see generated text streaming to the terminal, followed by timing stats (tokens/second).

Try different sampling parameters:

```bash
# Creative generation with temperature
./target/release/smollm-metal weights.bin tokenizer.json \
  "Write a haiku about Rust programming" \
  --temp 0.8 --top-p 0.9

# Deterministic generation with a fixed seed
./target/release/smollm-metal weights.bin tokenizer.json \
  "Explain quantum computing simply" \
  --seed 42 --max-tokens 100

# Interactive chat mode
./target/release/smollm-metal weights.bin tokenizer.json "" --interactive --temp 0.7
```

## Step 7: Start an API Server

The compiled binary includes a built-in OpenAI-compatible API server:

```bash
./target/release/smollm-metal weights.bin tokenizer.json --serve --port 8080
```

## Step 8: Query with curl

In another terminal:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello! What can you do?"}],
    "stream": true,
    "max_tokens": 128
  }'
```

The server responds with Server-Sent Events (SSE) in the OpenAI streaming format.

For non-streaming requests, set `"stream": false`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": false,
    "max_tokens": 64
  }'
```

## What Just Happened?

You compiled a language model into a native GPU binary. The binary:

1. Has **zero runtime interpretation** -- every transformer layer is a direct Metal compute shader dispatch
2. Has **all dimensions baked in** -- no dynamic shape checks, no generic kernels
3. Is **self-contained** -- just the binary + weights + tokenizer, no Python or framework dependencies
4. Uses **native Q8_0 GPU kernels** -- dequantizes on-the-fly during matmul, halving memory bandwidth
5. Runs the **entire forward pass in a single compute command encoder** -- no encoder transitions between layers

## Next Steps

- Try a larger model: Llama-3.2-1B or Llama-3.2-3B
- Compile for CPU instead: `--target cpu` (works on any platform with ARM NEON or x86)
- Embed weights in the binary: add `--embed-weights` for single-file deployment
- Read the [AOT Compilation Guide](src/guides/aot-compilation.md) for advanced options
- Read the [Architecture Overview](architecture.md) to understand the compiler pipeline
