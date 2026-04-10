# ForgeLLM

**Compile your LLMs, don't interpret them.**

ForgeLLM is a Rust-native ahead-of-time (AOT) ML compiler for small language models (1M-7B parameters). It takes a model definition and compiles it into an optimized, self-contained binary — no runtime interpreter, no Python dependencies, no dynamic dispatch.

[Documentation](https://sauravpanda.github.io/forge-llm/) | [Crates.io](https://crates.io/crates/forgellm-frontend) | [forgellm.dev](https://forgellm.dev)

## It Works

```
$ forge run --model SmolLM2-135M-Instruct-Q8_0.gguf \
            --tokenizer tokenizer.json \
            --prompt "The meaning of life is"

The meaning of life is a complex and multifaceted concept that has been
debated by philosophers, scientists, and theologians for centuries. At its
core, the question of what it means to be human...

Prefill: 5 tokens in 0.25s (19.7 tok/s)
Generate: 33 tokens in 1.53s (21.6 tok/s)
```

## Why ForgeLLM?

Every existing LLM inference engine loads model weights at runtime and executes a generic inference loop. This is like shipping a Python interpreter when you could ship a compiled binary.

ForgeLLM compiles models into hardware-specific code with:
- **Fused operations** baked into the binary
- **Shape-specialized kernels** tuned to exact weight dimensions
- **Compile-time quantization** — no quantization overhead at inference
- **Static memory planning** — zero allocations during inference
- **Single binary output** — deploy with `scp`

## Quick Start

```bash
# Build from source
git clone https://github.com/sauravpanda/forge-llm.git
cd forge-llm
cargo build --release

# Download a model
pip install huggingface-hub
python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('bartowski/SmolLM2-135M-Instruct-GGUF', 'SmolLM2-135M-Instruct-Q8_0.gguf'))"
python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('HuggingFaceTB/SmolLM2-135M-Instruct', 'tokenizer.json'))"

# Run inference
cargo run --release -- run \
  --model path/to/SmolLM2-135M-Instruct-Q8_0.gguf \
  --tokenizer path/to/tokenizer.json \
  --prompt "Hello, world"
```

## Supported Models

| Architecture | Models | Status |
|-------------|--------|--------|
| LlamaForCausalLM | SmolLM2 (135M, 360M, 1.7B), Llama 3.2 (1B, 3B), TinyLlama | Verified |
| Qwen2ForCausalLM | Qwen2.5 (0.5B-7B) | Verified |
| MistralForCausalLM | Mistral 7B | Supported |
| Phi3ForCausalLM | Phi-3 Mini | Supported |
| GemmaForCausalLM | Gemma 2B, 7B | Supported |
| StableLMForCausalLM | StableLM 1.6B, 3B | Supported |

Supports 12 GGUF quantization formats: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q2_K through Q8_K.

## Performance

Optimized kernels (unrolled matmul) on Apple Silicon:

| Model | Params | Generation |
|-------|--------|------------|
| SmolLM2-135M Q8_0 | 135M | **46.3 tok/s** |
| SmolLM2-360M Q8_0 | 360M | **17.5 tok/s** |
| Qwen2.5-0.5B Q8_0 | 494M | **12.0 tok/s** |

## AOT Compilation

The flagship feature: compile a GGUF model into a standalone, optimized binary.

```bash
# One-command compile and run
forge compile --model model.gguf --output ./my-model --run --prompt "Hello"

# Single-file binary with embedded weights
forge compile --model model.gguf --output ./my-model --embed-weights --run

# Cross-compile for Linux
forge compile --model model.gguf --output ./my-model --cross-target x86_64-unknown-linux-gnu
```

Generated binaries feature:
- **Shape-specialized matmul** — all dimensions baked in at compile time
- **Zero-allocation forward pass** — fixed-size stack arrays
- **Fused operators** — `silu_mul`, `residual_add` reduce memory passes
- **NEON SIMD + Rayon parallelism** — multi-core with ARM intrinsics
- **Precomputed RoPE** — frequency table computed once at startup
- **Full sampling** — temperature, top-k, top-p, repetition penalty
- **Interactive chat** — `--interactive` flag for REPL mode
- **EOS detection** — stops at end-of-sequence tokens
- **Memory-mapped weights** — fast startup via memmap2

## CLI Commands

```bash
# Run inference on a GGUF model
forge run --model model.gguf --tokenizer tokenizer.json --prompt "Hello"

# Interactive chat
forge chat --model model.gguf --tokenizer tokenizer.json

# Start OpenAI-compatible API server
forge serve --model model.gguf --tokenizer tokenizer.json --port 8080

# Benchmark performance
forge bench --model model.gguf --tokenizer tokenizer.json --num-tokens 128 --runs 3

# Inspect model architecture
forge info model.gguf

# AOT compile (see above for full options)
forge compile --model model.gguf --output ./out --run
```

## Architecture

```
GGUF/SafeTensors → Frontend (parse) → IR Graph → Optimizer → Codegen/Interpreter → Text
```

7 crates (all [published on crates.io](https://crates.io/search?q=forgellm)): `forgellm-frontend`, `forgellm-optimizer`, `forgellm-codegen-cpu`, `forgellm-codegen-wasm`, `forgellm-codegen-gpu`, `forgellm-runtime`, `forgellm-cli`

See the [full documentation](https://sauravpanda.github.io/forge-llm/) for architecture details.

## Contributing

```bash
cargo test --workspace            # Run tests (105+ tests)
cargo clippy --workspace -- -D warnings  # Lint
cargo fmt --all -- --check        # Format check
```

## License

MIT
