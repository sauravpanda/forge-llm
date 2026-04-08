# Forge

**Compile your LLMs, don't interpret them.**

Forge is a Rust-native ahead-of-time (AOT) ML compiler for small language models (1-7B parameters). It takes a model definition and compiles it into an optimized, self-contained binary — no runtime interpreter, no Python dependencies, no dynamic dispatch.

## Why Forge?

Every existing LLM inference engine loads model weights at runtime and executes a generic inference loop. This is like shipping a Python interpreter when you could ship a compiled binary.

Forge compiles models into hardware-specific code with:
- **Fused operations** baked into the binary
- **Shape-specialized kernels** tuned to exact weight dimensions
- **Compile-time quantization** — no quantization overhead at inference
- **Static memory planning** — zero allocations during inference
- **Single binary output** — deploy with `scp`

## Status

🚧 **Early development** — not yet ready for production use.

## Supported Targets

| Backend | Status |
|---------|--------|
| CPU (x86-64 AVX2/512) | Planned |
| CPU (ARM NEON / Apple Silicon) | Planned |
| WASM + SIMD128 | Planned |
| GPU via wgpu (Vulkan/Metal/DX12/WebGPU) | Planned |
| CUDA (optional) | Planned |

## Supported Models

| Architecture | Models | Priority |
|-------------|--------|----------|
| LlamaForCausalLM | Llama 3.2 (1B, 3B) | P0 |
| Qwen2ForCausalLM | Qwen2.5 (0.5B-7B) | P0 |
| MistralForCausalLM | Mistral 7B | P1 |
| Phi3ForCausalLM | Phi-3 Mini | P1 |

## Usage (Planned)

```bash
# Compile a model
forge compile --model meta-llama/Llama-3.2-1B-Instruct \
              --target cpu --quantize q4 --output llama-1b

# Run inference
forge run llama-1b --prompt "Hello, world"

# Benchmark
forge bench llama-1b --iterations 100

# Start API server
forge serve llama-1b --port 8080
```

## Building from Source

```bash
git clone https://github.com/sauravpanda/forge-llm.git
cd forge-llm
cargo build --release
```

## License

MIT
