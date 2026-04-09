# Architecture Overview

ForgeLLM is organized as a Rust workspace with 7 crates, each with a focused responsibility.

## Pipeline

```
Model File (GGUF/SafeTensors)
        │
        ▼
┌─────────────────┐
│  Frontend        │  Parse model format, extract config
│  (forgellm-      │  and tensor metadata
│   frontend)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  IR Graph        │  Build computation graph with typed
│  (forgellm-      │  tensor operations
│   frontend)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Optimizer       │  Fusion detection, dead code
│  (forgellm-      │  elimination, analysis passes
│   optimizer)     │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Codegen │ │Runtime │  Two execution paths:
│(CPU/   │ │(inter- │  1. AOT compile to Rust source
│ WASM/  │ │ preter)│  2. Direct interpretation
│ GPU)   │ │        │
└────────┘ └────────┘
```

## Crate Map

| Crate | Purpose |
|-------|---------|
| `forgellm-frontend` | GGUF/SafeTensors parsing, IR definition, graph building, weight loading |
| `forgellm-optimizer` | Fusion pattern detection, dead node elimination |
| `forgellm-codegen-cpu` | CPU Rust source code generation |
| `forgellm-codegen-wasm` | WASM target code generation (planned) |
| `forgellm-codegen-gpu` | GPU/WGSL code generation (planned) |
| `forgellm-runtime` | Reference interpreter, KV cache, sampling, tokenizer |
| `forgellm-cli` | CLI tool (`forge compile`, `forge run`, `forge info`) |

## Design Principles

1. **The IR is the central abstraction** — all frontends produce it, all backends consume it
2. **Zero allocations during inference** — static memory planning is the target
3. **AOT over JIT** — compile-time decisions, not runtime decisions
4. **Shape specialization** — every kernel knows its exact dimensions
5. **Progressive optimization** — reference interpreter first, then compiled, then SIMD
