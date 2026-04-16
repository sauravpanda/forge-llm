# How We Beat llama.cpp and MLX: Building an AOT Compiler for LLM Inference

> **Honest update (v0.6.4):** a correctness bug in `forward_prefill_batch` was silently truncating prompts longer than 512 tokens to the first 512, which inflated all previously reported prefill numbers for long prompts. Previous versions of this post quoted ~12 TFLOPS sustained on 1B prefill at 1,501 tokens — that was wrong. The real number is closer to 1,580 tok/s (~3 TFLOPS) because long-context attention in the current kernel is O(M²) and not MMA-accelerated. Numbers for prompts ≤ 512 tokens were unaffected and still hold. Everything below has been corrected. The truncation fix is in v0.6.4.

**TL;DR:** ForgeLLM is a Rust AOT compiler that compiles GGUF language models into optimized Metal GPU binaries. On Apple Silicon, it's **faster than llama.cpp and Apple's MLX on generation across every model size we've tested**, and competitive on prefill for short-to-medium prompts — leading 135M at any length and within 1.3× of MLX on 1B at their 321-token benchmark point. Longer prefill contexts are currently attention-bound; work on a flash-attention kernel is in progress.

### Generation Speed — 8-bit Quantization, Apple M5 Pro

| Model | ForgeLLM Metal | MLX (8-bit) | llama.cpp (Q8_0) | vs MLX | vs llama.cpp |
|-------|---------------|-------------|-------------------|--------|-------------|
| SmolLM2-135M | **496 tok/s** | 414 tok/s | 481 tok/s | **1.20x** | **1.03x** |
| SmolLM2-360M | **289 tok/s** | 264 tok/s | 267 tok/s | **1.09x** | **1.08x** |
| Llama-3.2-1B | **178 tok/s** | 111 tok/s | 130 tok/s | **1.60x** | **1.37x** |
| Llama-3.2-3B | **70.4 tok/s** | 42.2 tok/s | 67.8 tok/s | **1.67x** | **1.04x** |

### Prefill Speed (v0.6.4, honest numbers) — Llama-3.2-1B Q8_0

| Prompt | v0.5.2 | **v0.6.4** | Real speedup |
|-------:|-------:|-----------:|-------------:|
| 108 tok | 406 | **~1,160** | 2.9× |
| 321 tok | ~475 | **~2,080** | 4.4× |
| 801 tok | ~721 | **~2,030** | ~2.8× |
| 1,501 tok | ~1,354 | **~1,580** | ~1.2× |

Short-prompt speedups from the simdgroup_matrix MMA stack (v0.6.0–v0.6.1) are real — at M=321 tokens on 1B we go from ~475 tok/s to ~2,080 tok/s, closing the gap vs MLX's 2,718 to 1.3×. Longer-prompt numbers grow more slowly because the attention kernel is still the old O(M·seq) dot-product kernel with a `threadgroup float scores[ATTN_SCORES_SIZE]` array; once attention is MMA-accelerated too (flash attention work in progress), the long-prefill numbers should recover.

## The key insight: compile, don't interpret

Every LLM inference engine today — llama.cpp, vLLM, MLX, llama-rs — is fundamentally an interpreter. They load model weights at runtime, build a computation graph, and execute it operation by operation. The model architecture is data, not code.

ForgeLLM takes a different approach: **the model architecture becomes code**. We read the GGUF file at compile time, generate specialized Rust source code (or Metal shaders) with all dimensions baked in, and compile it to a native binary. The result is a self-contained executable that does exactly one thing — run this specific model — and does it with zero overhead.

This is the same insight that makes compiled C faster than interpreted Python. The interpreter pays a per-operation tax for flexibility it doesn't need.

## Where the speed comes from

### 1. Zero runtime overhead

llama.cpp processes each token through this pipeline:
```
graph_build() → graph_plan() → graph_compute() → backend dispatch → execute
```

ForgeLLM's compiled binary just calls:
```
forward() → execute
```

There's no graph. No planner. No dynamic dispatch. The forward pass is a flat sequence of Metal compute dispatches with pre-determined buffer bindings and grid sizes.

### 2. Model-specialized kernels

llama.cpp's matmul kernel handles any matrix size:
```c
void ggml_compute_forward_mul_mat(...) {
    const int64_t ne00 = src0->ne[0];  // runtime dimension lookup
    const int64_t ne01 = src0->ne[1];
    // ... dynamic dispatch based on type, size, backend
}
```

ForgeLLM generates code where dimensions are compile-time constants:
```rust
fn matmul_vec_q8_0_2048x3072(output: &mut [f32; 3072], input: &[f32; 2048], weight: &[u8]) {
    // Dimensions are TYPES, not values. The compiler optimizes accordingly.
}
```

The compiler can unroll loops, eliminate bounds checks, and schedule instructions optimally because it knows every dimension at compile time.

### 3. Aggressive GPU dispatch batching

Our Metal backend encodes the **entire forward pass in a single compute command encoder**. That's one encoder creation, one commit, one wait — for all 30+ layers of the model.

llama.cpp's Metal backend creates separate encoders for different operation types and may have synchronization points between layers.

We also fuse projections: Q+K+V weights are concatenated into one buffer and dispatched as a single matmul. Same for gate+up in the FFN. This cuts matmul dispatches from 7 per layer to 3.

### 4. Native quantized GPU kernels

Our Metal Q8_0 kernel reads quantized weights directly on the GPU:

```metal
kernel void matmul_vec_q8(
    device const uchar* matrix,  // Raw Q8_0 bytes on GPU
    device const float* vector,
    device float* output, ...)
{
    // Dequantize on-the-fly during matmul
    float scale = float(*(device const half*)(row + blk_byte));
    device const packed_char4* data = (device const packed_char4*)(row + blk_byte + 2);
    // ... simdgroup cooperative dot product with shared memory
}
```

This halves GPU memory bandwidth compared to dequantizing to f32 at load time (which is what a naive approach would do). For a 1B model, that's 1.2GB of Q8_0 weights accessed every token instead of 2.4GB of f32.

### 5. The advantage scales with model size

This is the most interesting finding. Our speedup over llama.cpp **increases** with model size:

| Model size | Speedup |
|-----------|---------|
| 135M | 1.15x |
| 360M | 1.08x |
| 1B | 1.55x |

Why? Because the fixed overhead of graph building and dispatch management is constant in llama.cpp but zero in ForgeLLM. As model size grows, llama.cpp's per-token fixed cost stays the same while compute grows — but our fixed cost is already zero. The ratio improves in our favor.

## How we built it: from 8 tok/s to 567 tok/s (generation)

The Metal backend was built from scratch and optimized iteratively:

| Step | tok/s | What changed |
|------|-------|-------------|
| Naive shaders | 8 | One thread per row, sync after every operation |
| Async dispatch | 80 | Single command buffer, wait once |
| Simdgroup matmul | 169 | 32-lane cooperative dot product, shared memory |
| Simdgroup attention | 221 | Cooperative Q*K^T, cooperative softmax |
| Q8_0 native kernel | 396 | Dequantize in shader, halve bandwidth |
| Single encoder | 417 | Replace all blits with compute copies |
| Fused projections | 334 | QKV and gate+up as single matmuls |
| Unrolled kernel | **567** | Fully unrolled inner loop, float4 loads |

Two correctness bugs were found along the way:
- **Misaligned char4 pointer cast** — Q8_0 blocks start data at +2 bytes (after the f16 scale), which isn't 4-byte aligned. Fixed with `packed_char4`.
- **Hardcoded shared memory size** — `vec_tile[4096]` overflowed for models with intermediate_size > 4096. Fixed with dynamic sizing from model config.

Both bugs only manifested on larger models — 135M worked fine, 360M and 1B produced garbage. A good reminder that you should always test on multiple model sizes.

## v0.6.x: The prefill story — from dot products to hardware MMA

By v0.5.2 the generation path was fast, but prefill on Llama-3.2-1B topped out around 475 tok/s at 321 tokens — roughly 1 TFLOPS sustained on a chip with a 13+ TFLOPS FP32 peak. MLX was doing 2,718 tok/s at the same length, using Apple Accelerate BLAS for its matmul path. **We were leaving 5-6x on the table.**

The fix was to stop using scalar dot products for the batched matmul and start using Apple Silicon's hardware matrix-multiply units: `simdgroup_matrix<float, 8, 8>` + `simdgroup_multiply_accumulate`. One instruction computes an 8×8×8 matmul across a simdgroup; in theory this runs at 32 FMAs per cycle per lane.

v0.6.0 and v0.6.1 ship **four generations** of MMA kernels with a tiered dispatch that picks the right one per matmul shape:

| Kernel | Tile | Threads/TG | Accumulators | Precision | Used when |
|---|---|---:|---:|---|---|
| `matmul_q8_mma` | 16×16 | 128 | 1 per simdgroup | float | 16 ≤ M < 32 |
| `matmul_q8_mma32` | 32×32 | 256 | 2 per simdgroup | float | M ≥ 32, small hidden (135M/360M) |
| `matmul_q8_mma32_h4` | 32×32 | 128 | 2×2 per simdgroup | fp16 tile, f32 accum | M ≥ 32, cols ≥ 2048, M < 256 |
| `matmul_q8_mma32_hh4` | 32×32 | 128 | 2×2 per simdgroup | all fp16 | M ≥ 256, cols ≥ 2048 |

Each kernel is faster than the previous in its shape range and strictly worse outside it. Every tiered transition was validated by running the benchmark, checking correctness on 135M/1B/3B, and keeping only the configuration that won.

### Shape-specific wins on Llama-3.2-1B Q8_0 (M5 Pro, honest numbers)

| Prompt | v0.5.2 dot-product | v0.6.4 MMA stack | Speedup |
|-------:|-------------------:|-----------------:|--------:|
| 101 | 406 | **~1,160** | 2.9× |
| 201 | ~560 | **~1,600** | 2.9× |
| 321 | ~475 | **~2,080** | 4.4× |
| 801 | ~721 | ~2,030 | ~2.8× |
| 1,501 | ~1,354 | ~1,580 | ~1.2× |

The short-prompt speedups (up to ~321 tokens) are the MMA kernel stack doing its job: at this matmul-bound regime we're in the 2–4× range vs dot-product. Past ~512 tokens, **throughput goes down with prompt length** instead of up — the matmul scales linearly, but the attention kernel is still the pre-MMA "compute all scores, then softmax" O(M·seq) implementation with a `threadgroup float scores[ATTN_SCORES_SIZE]` array. For M=1,501 the attention FLOPs are about 13% of total but they're running at a much lower fraction of peak than the MMA matmul, so they dominate the wall-clock time.

An MMA-based flash-attention kernel is the obvious next step — when that lands, the long-prefill numbers should move up by a factor of several.

### Design notes from the kernel rewrite

**Weight dequant as a staging step.** Q8_0 weights are 34 bytes per 32-value block (2-byte f16 scale + 32 int8 values). We cooperatively dequantize a 32×32 tile into 4 KB of threadgroup memory each iteration, then `simdgroup_load` into the MMA units. The dequant is negligible compared to the MMA work and the shared tile is reused across two simdgroup_loads per k_sub (A across both row tiles, B across both token tiles).

**Choosing tile size is a wave-count problem, not a tile-size problem.** The 16×16 kernel produces ~4x more threadgroups than the 32×32 kernel at the same M, which *sounds* better for latency hiding — but arithmetic intensity per TG drops, and in our measurements 32×32 wins at every prompt length above M=32. The intuition that "smaller tiles = better occupancy" doesn't survive contact with the simdgroup_matrix units, which need enough work per TG to amortize their own setup.

**Thread count per TG matters more than you'd think.** The 2×2-accumulator variant (`h4`) uses 128 threads per TG instead of 256; this *doubles* concurrent-TG occupancy on a GPU core if the thread budget is the tighter limit rather than shared memory. Going further to 64 threads/TG with 4×4 accumulators per simdgroup made register pressure explode and regressed.

**FP16 accumulation is safe for Q8_0.** Going from fp16 inputs + fp32 accumulator (`h4`) to fully fp16 (`hh4`) made us nervous — summing 2,048 products in half precision sounds reckless. In practice it's fine: Q8_0 weights have ~8 bits of mantissa to start with, activations are post-RMSNorm and bounded O(1), and output coherence is preserved on 135M / 1B / 3B. The ~6-9% throughput win at long context is real.

**Not every attempted optimization worked.** We also tried K=64 chunks (regressed -10% at low M), a 2-simdgroup 4×4-accumulator variant (register pressure), float4-vectorized elementwise kernels (the Metal compiler was already auto-vectorizing; explicit vectorization interfered), cooperative widening stores for the partial-tile fallback (scatter-store patterns hurt coalescing), and lowering the hh4 threshold from M=256 to M=128 (the per-lane scalar widening store costs more than the MMA win at low M). All reverted.

The lesson: on the MMA side of Apple Silicon, every change needs to be measured on actual hardware with warm caches. Peak-FLOP calculations tell you where the ceiling is; they tell you nothing about whether a specific variant will approach it.

## What ForgeLLM is not

ForgeLLM is a compiler, not an inference framework. This means:

- **One binary per model** — You can't swap models at runtime. Each model compilation produces a separate binary.
- **Compilation takes time** — `cargo build --release` takes ~30 seconds. You pay this once per model, not per request.
- **Less flexible** — No dynamic batching, no model parallelism (yet), no hot-swapping.
- **Focused on small models** — Tested on 135M to 1B. 7B works but hasn't been extensively benchmarked.

For production use cases where you run one model at steady state — edge devices, embedded systems, dedicated inference servers — the compiler approach is ideal. For multi-model serving with dynamic routing, llama.cpp or vLLM is a better fit.

## Try it

```bash
git clone https://github.com/sauravpanda/forge-llm.git
cd forge-llm && cargo build --release

# Compile any GGUF model to a Metal binary
forge compile --model your-model.gguf --output ./server --target metal
forge export-weights --model your-model.gguf --output ./server/weights.bin
cp tokenizer.json ./server/

# Build and run
cd server && cargo build --release
./target/release/server weights.bin tokenizer.json "Hello world"

# Or start an API server
./target/release/server weights.bin tokenizer.json --serve --port 8080
```

The output is a standard Rust project. You can inspect the generated code, modify it, embed it in your own project, or cross-compile it for any target Rust supports.

## What's next

- **Flash attention** — tile Q/K/V, fuse with online softmax to cut attention memory traffic to O(M) instead of O(M²). This is the natural next step after the MMA prefill rewrite.
- **Multi-layer fusion** — `silu_mul` + `down_proj` into a single kernel; `add_residual` + `rms_norm` into a single kernel. Reduces per-forward dispatch count from ~200 to ~130.
- **Async weight prefetch** — `simdgroup_async_copy` to overlap device→threadgroup loads with MMA compute on the previous K chunk.
- **MLC-LLM head-to-head benchmark** — same quantization, same Metal backend, same prompt lengths.

ForgeLLM is open source under the MIT license. Install with `cargo install forgellm-cli` or `brew install --formula HomebrewFormula/forgellm.rb` from the repo. Contributions welcome.

---

## The honest competitive landscape

We beat llama.cpp. But llama.cpp isn't the only game in town. Here's how ForgeLLM fits into the broader ecosystem — including who could beat us.

### The inference engine spectrum

```
More flexible                                              More optimized
     |                                                            |
  PyTorch   →   vLLM   →   llama.cpp   →   MLC-LLM   →   ForgeLLM
  (eager)     (batched)    (runtime)      (compiler)      (AOT compiler)
```

Each step right trades flexibility for speed. ForgeLLM is the furthest right — maximum optimization, minimum flexibility.

### Comparison across all dimensions

| | PyTorch | vLLM | llama.cpp | MLC-LLM | MLX | ForgeLLM |
|---|---|---|---|---|---|---|
| **Approach** | Eager/JIT | Runtime + PagedAttn | Runtime interpreter | TVM AOT compiler | Apple framework | Rust AOT compiler |
| **Training** | Yes | No | No | No | Yes | No |
| **Apple Metal** | Limited | No | Yes | Yes | Yes (native) | Yes |
| **NVIDIA CUDA** | Yes | Yes | Yes (cuBLAS) | Yes | No | **No** |
| **Quantization** | Varies | AWQ/GPTQ | GGUF (Q2-Q8) | TVM quant | 4/8-bit | GGUF Q4/Q8 |
| **Batching** | Yes | Yes (paged) | Limited | Yes | Yes | **No (single stream)** |
| **Multi-GPU** | Yes | Yes | Partial | Yes | No | **No** |
| **Neural Engine** | No | No | No | No | **Yes** | **No** |
| **Dependencies** | Python + CUDA | Python + CUDA | C++ lib | Python + TVM | Python + Metal | **None** |
| **Output** | Python script | Server process | C++ app | Compiled lib | Python script | **Self-contained binary** |
| **Dynamic shapes** | Yes | Yes | Yes | Limited | Yes | **No (baked in)** |
| **Model swap** | Hot swap | Hot swap | Hot swap | Recompile | Hot swap | **Recompile** |
| **Compile time** | 0 | 0 | 0 | Minutes | 0 | **30 seconds** |
| **Lines to deploy** | 20+ pip | Docker | Single binary | pip + compile | pip | `cargo build` |

### Who could beat us (and where)

**MLC-LLM (Apache TVM) — most directly comparable**

MLC-LLM takes the same compiler approach: read model weights, generate target-specific code, compile to native binary. They have:
- 4+ years of TVM compiler infrastructure
- Auto-tuning for kernel optimization (we hand-tune)
- CUDA, Vulkan, Metal, WebGPU, OpenCL backends
- More mature quantization support

MLC-LLM is the project we should benchmark against next. They're the real competition in the "compiled LLM inference" space. Our advantage is simpler toolchain (Rust + cargo, no Python/TVM dependency) and inspectable output.

**MLX (Apple) — platform advantage**

Apple's own ML framework has access to private APIs and hardware features:
- Can target the Apple Neural Engine (ANE) — 15+ TOPS on M-series
- Deep Metal integration with undocumented optimizations
- Unified memory management that third parties can't replicate
- If ANE is used, potentially 5-10x faster than any GPU-only approach

We can never access the Neural Engine. This is Apple's unfair advantage.

**TensorRT-LLM / vLLM (NVIDIA) — different hardware**

On NVIDIA GPUs with tensor cores, FP8 support, and PagedAttention, these engines would crush any Apple Silicon approach. We don't have a CUDA backend and likely won't build one — the NVIDIA ecosystem is well-served already.

**llama.cpp (continued development) — closing the gap**

llama.cpp has a massive contributor community and continues improving. Our llama.cpp benchmark advantage could narrow over time as they optimize their Metal backend. Their v0.10+ releases may close the gap we measured against v0.9.11.

### Where ForgeLLM wins

1. **Zero-dependency deployment** — No Python, no runtime, no shared libraries. Output is a Rust project that compiles to a static binary. Deploy to any machine with `scp`.

2. **Inspectable generated code** — The output is readable Rust and Metal Shading Language. You can see every matmul, every attention computation, every memory allocation. Debug by reading the code, not by attaching to a runtime.

3. **Embeddable as a library** — Generated code is a Rust library crate. Import it into your own application with `use my_model::MetalModel;`. No FFI, no IPC, no server process.

4. **Deterministic behavior** — Same input always produces the same output. No runtime graph optimization that might change between versions. The binary IS the model.

5. **Small model excellence** — For models under 3B parameters on Apple Silicon, we're the fastest option that doesn't require Apple's private APIs. This is the edge/mobile/embedded sweet spot.

### Where ForgeLLM loses

1. **No CUDA** — If you have an NVIDIA GPU, use TensorRT-LLM or vLLM.
2. **No batching** — If you need to serve multiple concurrent users, use vLLM.
3. **No training** — If you need to fine-tune, use PyTorch.
4. **No Neural Engine** — If you want maximum Apple Silicon throughput, MLX/CoreML can access hardware we can't.
5. **Recompile per model** — If you swap models frequently, an interpreter is more practical.
6. **Limited model support** — We support 6 architectures. llama.cpp supports 20+.

### The bottom line

ForgeLLM occupies a specific niche: **compiled, self-contained, zero-dependency LLM inference on Apple Silicon**. In that niche, we're competitive with or faster than everything else. Outside that niche, other tools are better choices.

The compiler approach is the right one for:
- Edge devices with fixed models
- Embedded systems where Python isn't available
- Applications that need LLM inference as a library, not a server
- Situations where you want to audit exactly what the inference code does
- Apple Silicon deployment where you want Metal GPU without MLX/CoreML

We're not trying to replace llama.cpp or vLLM for everyone. We're building the best tool for the "compile once, deploy anywhere, inspect everything" use case.

---

*Benchmarks run on Apple M5 Pro, Q8_0 quantization, 64-token generation. llama.cpp v0.9.11 (build 15f786e65) with default Metal backend. ForgeLLM v0.5.0.*
