# ForgeLLM

**Compile your LLMs, don't interpret them.**

ForgeLLM is a Rust-native ahead-of-time (AOT) ML compiler for small language models (1M-7B parameters). It takes a model definition and compiles it into an optimized, self-contained binary — no runtime interpreter, no Python dependencies, no dynamic dispatch.

## Why ForgeLLM?

Every existing LLM inference engine loads model weights at runtime and executes a generic inference loop with dynamic dispatch. This is like shipping a Python interpreter when you could ship a compiled binary.

ForgeLLM compiles models into hardware-specific code with:

- **Fused operations** baked into the binary
- **Shape-specialized kernels** tuned to exact weight dimensions
- **Compile-time quantization** — no quantization overhead at inference
- **Static memory planning** — zero allocations during inference
- **Single binary output** — deploy with `scp`

## Current Status

ForgeLLM is in active development. The reference interpreter is working and can run SmolLM2-135M at ~21 tok/s on Apple Silicon.

### What's Working

- GGUF model loading with dequantization (F32, F16, BF16, Q8_0, Q4_0, Q4_1)
- Full Llama-family inference (Llama, Qwen2, Mistral, SmolLM)
- Tokenizer integration (HuggingFace tokenizers)
- Greedy, top-k, top-p sampling with temperature
- CPU code generation (Rust source emission)
- Optimizer with fusion pattern detection

### Supported Models

| Architecture | Models | Status |
|-------------|--------|--------|
| LlamaForCausalLM | SmolLM2 (135M, 360M, 1.7B), Llama 3.2 (1B, 3B) | Working |
| Qwen2ForCausalLM | Qwen2.5 (0.5B-7B) | Working |
| MistralForCausalLM | Mistral 7B | Working |

## Quick Example

```bash
forge run \
  --model SmolLM2-135M-Instruct-Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The meaning of life is"
```

Output:
```
The meaning of life is a complex and multifaceted concept that has been
debated by philosophers, scientists, and theologians for centuries.
```
