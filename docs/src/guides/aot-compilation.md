# AOT Compilation Guide

ForgeLLM's ahead-of-time (AOT) compiler transforms a model into a standalone, optimized Rust binary. No Python, no runtime interpreter, no dynamic dispatch — just native code with all dimensions baked in at compile time.

## Quick Start

```bash
# Basic: generate a Cargo project
forge compile --model smollm-135m.Q8_0.gguf --output ./my-model

# Build and run in one step
forge compile --model smollm-135m.Q8_0.gguf --output ./my-model --run --prompt "Hello"

# Single-file binary with embedded weights
forge compile --model smollm-135m.Q8_0.gguf --output ./my-model --embed-weights --run
```

## What Gets Generated

The `forge compile` command produces a complete Cargo project:

```
my-model/
├── Cargo.toml          # Build config (LTO, codegen-units=1, rayon)
├── src/
│   ├── model.rs        # Kernels + forward function (all dims baked in)
│   └── main.rs         # Weight loader + tokenizer + generation loop
├── weights.bin         # (if --embed-weights) Exported model weights
└── tokenizer.json      # (if --embed-weights) Tokenizer data
```

### Generated Code Features

- **Shape-specialized matmul**: `matmul_vec_576x1536()` instead of generic `matmul(m, k, n)`
- **Zero-allocation forward pass**: All buffers are fixed-size stack arrays
- **Fused operators**: `silu_mul()` combines SiLU + elementwise multiply in one pass
- **In-place residual**: `residual_add()` avoids extra buffer copies
- **NEON SIMD**: ARM NEON intrinsics for dot products (with scalar fallback)
- **Rayon parallelism**: Large matmuls parallelized across CPU cores
- **Pre-allocated KV cache**: Sized to MAX_SEQ_LEN at startup

## Step-by-Step Walkthrough

### 1. Export weights separately

```bash
forge export-weights --model smollm-135m.Q8_0.gguf --output ./my-model/weights.bin
```

Weights are exported in a flat binary format (f32 little-endian) in a deterministic order: embed_tokens → per-layer weights → final_norm → lm_head.

### 2. Build the project

```bash
cd my-model
cargo build --release
```

The release build enables LTO and single codegen-unit for maximum optimization.

### 3. Run inference

```bash
# With external weight files
./target/release/my-model weights.bin /path/to/tokenizer.json "Tell me a story"

# With embedded weights (--embed-weights)
./target/release/my-model "Tell me a story"
```

## Embed Weights for Single-File Deployment

The `--embed-weights` flag bakes model weights and tokenizer directly into the binary using Rust's `include_bytes!` macro:

```bash
forge compile --model model.gguf --output ./my-model --embed-weights
cd my-model && cargo build --release
# Binary is now fully self-contained
cp target/release/my-model /usr/local/bin/
my-model "Hello, world!"
```

This is ideal for deployment — just copy the single binary. No external files needed.

## Performance Characteristics

| Feature | Interpreter | AOT Binary |
|---------|------------|------------|
| Dimensions | Runtime | Compile-time constants |
| Dispatch | Dynamic (match on Op) | Direct function calls |
| Buffers | Vec (heap) | Fixed-size arrays (stack) |
| Matmul | Generic m×k×n | Specialized per-shape |
| Multi-core | Single-threaded | Rayon parallel matmul |
| Startup | Load + build graph | Load weights only |
| Deployment | Needs forge runtime | Single binary |

## Cross-Compilation

Generate a project on one platform, build for another:

```bash
forge compile --model model.gguf --output ./my-model
cd my-model
cargo build --release --target x86_64-unknown-linux-gnu
```

The generated code uses `#[cfg(target_arch)]` for NEON vs scalar fallbacks, so it compiles correctly on any platform.

## Supported Architectures

All architectures supported by ForgeLLM work with AOT compilation:
- Llama (LLaMA 2/3, SmolLM, TinyLlama)
- Qwen2
- Mistral
- Phi-3
- Gemma
- StableLM
