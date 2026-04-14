# ForgeLLM Architecture

ForgeLLM is an ahead-of-time (AOT) compiler for transformer language models. It reads model files, builds a computation graph, optimizes it, and emits hardware-specific code as a standalone Rust project.

## Compiler Pipeline

```
GGUF/SafeTensors --> Frontend --> IR Graph --> Optimizer --> Codegen --> Binary
                      parse       build        fuse         emit       compile
```

**Input:** A GGUF or SafeTensors model file containing weights and hyperparameters.

**Output:** A self-contained Rust binary with all dimensions baked in as compile-time constants, zero dynamic dispatch, and static memory allocation.

---

## Stage 1: Frontend (forgellm-frontend)

**What it does:** Parses model files (GGUF, SafeTensors), extracts model configuration and weight tensors, and constructs the IR computation graph.

**Key types:**
- `GGUFFile` -- parsed GGUF container (header, metadata, tensor descriptors)
- `ModelConfig` -- hyperparameters (hidden_size, num_layers, num_heads, rope_theta, etc.)
- `Graph` -- the computation graph (DAG of typed tensor operations)
- `Node` -- a single operation with inputs, output shape, and dtype
- `Op` -- operation enum (MatMul, RMSNorm, RoPE, Attention, SiLU, Embedding, etc.)
- `DType` -- data type (F32, F16, BF16, Q8_0, Q4_0, Q4_1, etc.)
- `Shape` -- static dimension list (no dynamic sizes)

**Source files:**
- `gguf.rs` -- GGUF parser (header, metadata, tensor descriptors, dequantization)
- `safetensors.rs` -- SafeTensors parser
- `config.rs` -- architecture detection and ModelConfig extraction from metadata
- `ir.rs` -- IR type definitions (Graph, Node, Op, DType, Shape, TensorInfo)
- `graph_builder.rs` -- constructs the forward-pass graph from ModelConfig
- `weight_loader.rs` -- loads and dequantizes weight tensors, remaps GGUF names
- `lora.rs` -- LoRA adapter loading and compile-time merging
- `hub.rs` -- HuggingFace Hub model downloading
- `onnx_export.rs` -- ONNX graph export

**Example: what goes in and out**

In: `SmolLM2-135M-Instruct-Q8_0.gguf` (244 MB file)

Out: A `Graph` with ~450 nodes representing the full forward pass -- embedding lookup, 30 transformer layers (each with RMSNorm, Q/K/V projections, RoPE, attention, output projection, residual, FFN with SiLU gating), final norm, and logits projection. Plus a `ModelConfig` with `hidden_size=576, num_layers=30, num_heads=9, num_kv_heads=3, head_dim=64, intermediate_size=1536, vocab_size=49152`.

---

## Stage 2: Optimizer (forgellm-optimizer)

**What it does:** Analyzes the IR graph and identifies fusion opportunities where multiple operations can be merged into single kernels to reduce memory bandwidth.

**Key types:**
- `OptimizeConfig` -- controls which passes run (enable_fusion, enable_dce)
- `FusedOp` -- enum of fused patterns (RMSNormMatMul, GateUpSiLU)
- `FusionRecord` -- tracks which nodes were fused and their replacement
- `FusionResult` -- optimized graph + list of applied fusions

**Fusion patterns detected:**
- **RMSNorm + MatMul** -- when a norm output feeds a single matmul, fuse them to eliminate the intermediate tensor
- **GateUp + SiLU + Mul** -- the SiLU-gated FFN pattern (`gate = silu(x @ W_gate) * (x @ W_up)`) fuses four ops into one kernel

**Dead code elimination:** backward liveness analysis from the graph output removes unreachable nodes.

**Source files:**
- `passes.rs` -- pass orchestration (runs fusion + DCE in sequence)
- `fusion.rs` -- pattern matching for RMSNorm+MatMul and GateUp+SiLU patterns

**Example: what goes in and out**

In: A 450-node graph with separate RMSNorm, MatMul, SiLU, and Mul nodes.

Out: The same graph with fusion annotations identifying ~30 RMSNorm+MatMul pairs and ~30 GateUp+SiLU patterns (one per layer). Graph rewriting with fused ops is in progress.

---

## Stage 3: Code Generation

Four backend crates consume the IR graph and emit target-specific code. Each generates a complete Cargo project (or shader source) with all model dimensions as compile-time constants.

### forgellm-codegen-cpu

**What it does:** Generates a standalone Rust project with CPU kernels using ARM NEON SIMD intrinsics and Rayon parallelism.

**Key types:**
- `CodegenError` -- error during code generation
- `ProjectError` -- error creating the project structure

**What gets generated:**
- `Cargo.toml` -- with rayon, memmap2, tokenizers dependencies, LTO enabled
- `src/model.rs` -- shape-specialized kernels (`matmul_vec_576x1536()`), `Weights`/`LayerWeights`/`KVCache` structs, and a `forward()` function
- `src/main.rs` -- CLI with weight loading, tokenizer, generation loop, sampling, API server

**Source files:**
- `emit.rs` -- Rust source code emission (kernels, forward function, structs)
- `project.rs` -- Cargo project scaffolding
- `speculative.rs` -- speculative decoding runner generation

**Example: what goes in and out**

In: IR Graph for SmolLM2-135M.

Out: A Cargo project where `matmul` calls are `matmul_vec_576x1536()` with NEON `vdotq_s32` intrinsics, all buffers are fixed-size arrays, and large matmuls are parallelized with Rayon.

### forgellm-codegen-metal

**What it does:** Generates a Cargo project with Metal Shading Language (MSL) compute shaders for Apple Silicon GPUs. Uses the `metal` crate (metal-rs) for GPU dispatch.

**Key types:**
- `MetalCodegenError` -- error during Metal code generation

**What gets generated:**
- `Cargo.toml` -- with metal, objc, tokenizers, memmap2, half dependencies
- `shaders/kernels.metal` -- MSL compute kernels (matmul with simdgroup reductions, RMSNorm, RoPE, attention, SiLU, softmax, embedding, Q8_0/Q4_0 dequantization)
- `src/model.rs` -- MetalModel struct, compute pipeline setup, buffer management, forward pass
- `src/main.rs` -- CLI with weight loading, tokenizer, generation, API server with SSE streaming

**Key features:** simdgroup cooperative matmul, native quantized kernels (Q8_0/Q4_0 dequantize during matmul), fused QKV projections, single compute command encoder for entire forward pass, double-buffered prefill, `fast::` math functions.

**Example: what goes in and out**

In: IR Graph for Llama-3.2-1B.

Out: A Cargo project that loads weights into Metal GPU buffers via unified memory (zero-copy on Apple Silicon), dispatches MSL compute kernels, and achieves 178 tok/s generation on M5 Pro.

### forgellm-codegen-wasm

**What it does:** Generates a Cargo project targeting `wasm32-unknown-unknown` with WASM SIMD128 intrinsics and `wasm_bindgen` exports for browser integration.

**Key types:**
- `WasmCodegenError` -- error during WASM code generation

**What gets generated:**
- `Cargo.toml` -- targeting wasm32-unknown-unknown with wasm-bindgen
- `src/lib.rs` -- SIMD128-accelerated kernels + `WasmModel` export
- `pkg/model.js` -- JS glue layer for browser integration

### forgellm-codegen-gpu

**What it does:** Generates a Cargo project using wgpu with WGSL compute shaders for cross-platform GPU inference (Vulkan, Metal, DX12, WebGPU).

**Key types:**
- `GpuCodegenError` -- error during GPU code generation

**What gets generated:**
- `Cargo.toml` -- with wgpu, pollster, tokenizers dependencies
- `src/model.rs` -- GPU model with embedded WGSL compute shaders
- `src/main.rs` -- CLI with weight loading, tokenizer, GPU inference

---

## Stage 4: Runtime (forgellm-runtime)

**What it does:** Provides a reference interpreter that executes IR graphs directly (no compilation), plus shared components: KV cache, sampling strategies, tokenizer, and chat templating.

**Key types:**
- `KVCache` -- stores key/value projections across layers and sequence positions
- `SamplingConfig` -- temperature, top_k, top_p, repetition_penalty
- `Tokenizer` -- wraps HuggingFace tokenizers for BPE encoding/decoding
- `ChatTemplate` -- applies chat formatting (system/user/assistant turns)

**Source files:**
- `interpreter.rs` -- reference forward pass (f32, no SIMD, no fusion)
- `kernels.rs` -- pure Rust kernel implementations (matmul, rmsnorm, rope, softmax, silu, etc.)
- `kv_cache.rs` -- KV cache with per-layer append and advance
- `sampling.rs` -- greedy, temperature, top-k, top-p, repetition penalty
- `tokenizer.rs` -- tokenizer wrapper
- `chat.rs` -- chat template application

The interpreter serves two purposes: (1) validation baseline to verify model correctness, and (2) quick inference without compilation.

---

## Stage 5: CLI (forgellm-cli)

**What it does:** Orchestrates the entire pipeline through user-facing commands.

**Commands:**
- `forge compile` -- AOT compile a model to a Cargo project (CPU, Metal, WASM, GPU targets)
- `forge run` -- run inference using the interpreter (no compilation)
- `forge chat` -- interactive chat mode
- `forge serve` -- start an OpenAI-compatible API server (interpreter mode)
- `forge bench` -- benchmark token generation speed
- `forge info` -- inspect model metadata
- `forge export-weights` -- export weights as flat binary for AOT binaries
- `forge export-onnx` -- export computation graph as ONNX
- `forge speculative` -- set up speculative decoding with draft + target models

**Source files:**
- `main.rs` -- clap CLI definition and command dispatch

---

## Data Flow Summary

```
model.gguf
    |
    v
[GGUF Parser] --> GGUFFile { header, metadata, tensor_descriptors }
    |
    v
[Config Extraction] --> ModelConfig { hidden_size, num_layers, ... }
    |
    v
[Graph Builder] --> Graph { nodes: Vec<Node>, config: ModelConfig }
    |                  each Node = { op: Op, inputs: Vec<NodeId>, output: TensorInfo }
    v
[Optimizer] --> Graph (with fusion annotations, dead nodes removed)
    |
    v
[Codegen] --> Cargo project on disk
    |            src/model.rs  -- specialized kernels + forward()
    |            src/main.rs   -- CLI + generation loop
    |            shaders/      -- Metal/WGSL shaders (GPU targets)
    v
[cargo build --release] --> Self-contained binary
```

## Crate Dependency Graph

```
forgellm-cli
    |-- forgellm-frontend
    |-- forgellm-optimizer --> forgellm-frontend
    |-- forgellm-codegen-cpu --> forgellm-frontend
    |-- forgellm-codegen-metal --> forgellm-frontend
    |-- forgellm-codegen-wasm --> forgellm-frontend
    |-- forgellm-codegen-gpu --> forgellm-frontend
    |-- forgellm-runtime --> forgellm-frontend
```

The IR (`forgellm-frontend::ir`) is the central contract. All crates depend on the frontend for type definitions but not on each other.
