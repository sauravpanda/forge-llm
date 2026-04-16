# ForgeLLM

[![CI](https://github.com/sauravpanda/forge-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravpanda/forge-llm/actions/workflows/ci.yml)

**Compile your LLMs, don't interpret them.**

ForgeLLM is a Rust-native ahead-of-time (AOT) ML compiler for language models (1M-7B parameters). It compiles GGUF models into optimized, self-contained binaries with native Metal GPU acceleration — no runtime interpreter, no Python dependencies, no dynamic dispatch.

**Faster than llama.cpp** on Apple Silicon.

[Documentation](https://sauravpanda.github.io/forge-llm/) | [Crates.io](https://crates.io/crates/forgellm-frontend) | [forgellm.dev](https://forgellm.dev) | [Blog: How we beat llama.cpp](blog/beating-llama-cpp.md)

## Performance

Benchmarks on Apple M5 Pro, 8-bit quantization, 64-token generation.

### Generation Speed (tok/s)

| Model | ForgeLLM Metal | MLX (8-bit) | llama.cpp (Q8_0) | vs MLX | vs llama.cpp |
|-------|---------------|-------------|-------------------|--------|-------------|
| SmolLM2-135M | **496 tok/s** | 414 tok/s | 481 tok/s | **1.20x** | **1.03x** |
| SmolLM2-360M | **289 tok/s** | 264 tok/s | 267 tok/s | **1.09x** | **1.08x** |
| Llama-3.2-1B | **178 tok/s** | 111 tok/s | 130 tok/s | **1.60x** | **1.37x** |
| Llama-3.2-3B | **70.4 tok/s** | 42.2 tok/s | 67.8 tok/s | **1.67x** | **1.04x** |

### Prefill Speed (tok/s, long prompt)

| Model | ForgeLLM Metal | MLX (8-bit) | llama.cpp (Q8_0) |
|-------|---------------|-------------|-------------------|
| SmolLM2-135M (~100 tok) | **4,900** | 1,507 | 2,812 |
| SmolLM2-135M (~1250 tok) | **23,300** | — | — |
| Llama-3.2-1B (~321 tok) | 2,040 | **2,718** | 556 |
| Llama-3.2-1B (~801 tok) | **3,390** | — | — |
| Llama-3.2-1B (~1501 tok) | **6,320** | — | — |
| Llama-3.2-1B (~3001 tok) | **~12,000** | — | — |
| Llama-3.2-3B (~401 tok) | **770** | — | — |
| Llama-3.2-3B (~1501 tok) | **2,390** | — | — |

Prefill uses hardware matrix-multiply via `simdgroup_matrix<float, 8, 8>`. The large-tile MMA kernel (`matmul_q8_mma32`, 32×32 tile) hits **~12.1 TFLOPS sustained** on Llama-3.2-1B at 1,501 tokens — ~93% of the M5 Pro FP32 peak. For 1B/3B a FP16-tile 4-simdgroup variant (`matmul_q8_mma32_h4`, 128 threads per TG, each simdgroup owning a 2×2 grid of 8×8 accumulators) doubles FLOP-per-simdgroup-load via A/B tile reuse.

### Deploy Size

| Model | Binary | Weights | Total |
|-------|--------|---------|-------|
| SmolLM2-135M | 3.7 MB | 244 MB | 248 MB |
| Llama-3.2-1B | 3.7 MB | 2.2 GB | 2.2 GB |
| Llama-3.2-3B | 3.7 MB | 4.6 GB | 4.9 GB |

Binary size is constant across all models. Compare: llama.cpp ~15 MB, MLX ~500 MB Python runtime.

**We beat MLX and llama.cpp on generation across all model sizes.** On prefill, we lead for short prompts on 135M and catch up at long contexts on 1B (~3x our previous number), using `simdgroup_matrix` hardware matrix-multiply tiles. MLX's Accelerate BLAS still edges us for mid-length 1B prefill (~325 tokens).

See [benchmarks/HISTORY.md](benchmarks/HISTORY.md) and [blog/beating-llama-cpp.md](blog/beating-llama-cpp.md) for details.

## Quick Start

### Install

```bash
# From crates.io (recommended)
cargo install forgellm-cli

# Or build from source
git clone https://github.com/sauravpanda/forge-llm.git
cd forge-llm && cargo build --release
```

### Metal GPU (Apple Silicon)

```bash
# Compile model to Metal binary
forge compile --model model.gguf --output ./my-model --target metal
forge export-weights --model model.gguf --output ./my-model/weights.bin
cp tokenizer.json ./my-model/

# Build and run
cd my-model && cargo build --release
./target/release/my-model weights.bin tokenizer.json "The meaning of life is"
```

### API Server

```bash
# Start OpenAI-compatible server
./target/release/my-model weights.bin tokenizer.json --serve --port 8080

# Query it
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

### CPU (cross-platform)

```bash
# Compile for CPU with NEON SIMD + Rayon parallelism
forge compile --model model.gguf --output ./my-model --target cpu --run
```

## Why ForgeLLM is faster

Every existing LLM inference engine — llama.cpp, vLLM, MLX — loads model weights at runtime and executes a generic inference loop. This is like shipping a Python interpreter when you could ship a compiled binary.

ForgeLLM compiles models into hardware-specific code:

| | llama.cpp (interpreter) | ForgeLLM (compiler) |
|---|---|---|
| **Dispatch** | Runtime graph build + plan + execute | Direct function calls, zero overhead |
| **Dimensions** | Dynamic (runtime checks) | Baked in at compile time |
| **GPU commands** | Multiple command encoders per layer | Single encoder for entire forward pass |
| **Projections** | Separate Q, K, V matmuls | Fused QKV in one dispatch |
| **Memory** | Runtime allocation | Static, pre-allocated buffers |
| **Quantization** | Dequant at load time | Native Q8_0/Q4_0 GPU kernels |
| **Output** | Shared library + runtime | Self-contained binary, deploy with `scp` |

## Compilation Targets

| Target | Command | Features |
|--------|---------|----------|
| **Metal GPU** | `--target metal` | Native MSL shaders, simdgroup reductions, Q8_0/Q4_0 kernels, API server |
| **CPU** | `--target cpu` | NEON sdot inline asm, Rayon parallelism, Apple AMX via Accelerate |
| **WASM** | `--target wasm` | SIMD128, wasm-bindgen exports, browser-ready |
| **wgpu/WGSL** | `--target gpu` | Cross-platform GPU via WebGPU |

## Supported Models

| Architecture | Models | Interpreter (`forge run`) | AOT Metal/CPU |
|-------------|--------|---------------------------|---------------|
| LlamaForCausalLM | SmolLM2 (135M, 360M, 1.7B), Llama 3.2 (1B, 3B), TinyLlama | ✅ Verified | ✅ Verified |
| Qwen2ForCausalLM | Qwen2.5 (0.5B–7B) | ✅ Verified | ✅ Verified (0.5B Q8_0 on CPU + Metal; fixes [#210](https://github.com/sauravpanda/forge-llm/issues/210)) |
| MistralForCausalLM | Mistral 7B (sliding-window attention) | ✅ Verified | ⚠️ Untested with v0.6.x MMA kernels |
| Phi3ForCausalLM | Phi-3 Mini | ✅ Verified | ⚠️ Untested with v0.6.x MMA kernels |
| GemmaForCausalLM | Gemma 2B, 7B | ✅ Verified | ⚠️ Untested with v0.6.x MMA kernels |
| StableLMForCausalLM | StableLM 1.6B, 3B | ✅ Verified | ⚠️ Untested with v0.6.x MMA kernels |

Supports GGUF quantization formats: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q2_K through Q8_K.
Also supports SafeTensors and LoRA adapter merging at compile time.

## Metal GPU Features

The Metal backend generates optimized Apple Silicon compute shaders:

- **Hardware matrix-multiply prefill** — `simdgroup_matrix<float, 8, 8>` MMA tiles dequantize Q8_0 into threadgroup memory and run 8×8×8 `simdgroup_multiply_accumulate` per tile, hitting ~8.7 TFLOPS sustained on 1B
- **Simdgroup cooperative matmul** — 32-lane SIMD reductions with shared memory vector caching (fast path for single-token decode)
- **Native Q8_0/Q4_0 kernels** — Dequantize on-the-fly during matmul, halving memory bandwidth
- **Fused projections** — QKV and gate+up concatenated into single matmul dispatches
- **Single compute encoder** — Entire forward pass in one encoder, zero transitions
- **Double-buffered prefill** — GPU overlaps with CPU encoding
- **`fast::` math** — Hardware-accelerated rsqrt/exp in normalization and attention
- **OpenAI-compatible API** — `--serve` mode with SSE streaming

## CLI Commands

```bash
# AOT compile to Metal GPU binary
forge compile --model model.gguf --output ./out --target metal

# AOT compile to CPU binary
forge compile --model model.gguf --output ./out --target cpu --run

# Export weights for compiled binary
forge export-weights --model model.gguf --output ./out/weights.bin

# Run interpreter (no compilation)
forge run --model model.gguf --tokenizer tokenizer.json --prompt "Hello"

# Interactive chat
forge chat --model model.gguf --tokenizer tokenizer.json

# Start API server (interpreter mode)
forge serve --model model.gguf --tokenizer tokenizer.json --port 8080

# Benchmark
forge bench --model model.gguf --tokenizer tokenizer.json --num-tokens 128

# Inspect model
forge info model.gguf

# ONNX export
forge export-onnx --model model.gguf --output model.onnx

# Speculative decoding
forge speculative --draft small.gguf --target-model large.gguf --output ./spec
```

## Architecture

```
GGUF/SafeTensors → Frontend → IR Graph → Optimizer → Codegen → Binary
                     parse      build      fuse       emit     compile
```

8 crates: `forgellm-frontend`, `forgellm-optimizer`, `forgellm-codegen-cpu`, `forgellm-codegen-wasm`, `forgellm-codegen-gpu`, `forgellm-codegen-metal`, `forgellm-runtime`, `forgellm-cli`

## Contributing

```bash
cargo test --workspace --exclude forgellm-python  # 258+ tests
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

## License

MIT
