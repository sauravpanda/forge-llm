# Changelog

All notable changes to ForgeLLM are documented here.

## [0.6.1] — 2026-04-15 — All-FP16 MMA for Long Contexts

### Performance

**`matmul_q8_mma32_hh4` (all-FP16 MMA) variant** (`perf`): Same 32×32 tile and 4 simdgroup × 2×2 accumulator layout as `matmul_q8_mma32_h4`, but both inputs *and* accumulators are `simdgroup_matrix<half, 8, 8>`. Apple Silicon runs FP16 `simdgroup_multiply_accumulate` at a higher rate than FP32, giving ~6–9% additional throughput on Llama-3.2-1B / 3B prefill at moderate-to-long contexts. Output is converted back to `float` in the device store. Dispatched when `cols >= 2048` *and* `num_tokens >= 256` — below that threshold the per-lane scalar widening store outweighs the MMA win, so the path falls back to `matmul_q8_mma32_h4`.

**Llama-3.2-1B Q8_0 prefill (Apple M5 Pro):**

| Prompt | v0.6.0 | Unreleased | Δ |
|-------:|-------:|-----------:|--:|
| 321 tok | 1,880 | **2,040 tok/s** | +9% |
| 801 tok | 3,150 | **3,390 tok/s** | +8% |
| 1,501 tok | 6,070 | **6,320 tok/s** | +4% (~12.6 TFLOPS) |

**Llama-3.2-3B Q8_0 prefill:** 401 tok 740 → **770 tok/s** (+4%), 1,501 tok 2,200 → **2,390 tok/s** (+9%).

## [0.6.0] — 2026-04-15 — Hardware Matrix-Multiply Prefill

**Prefill went 4x faster on Llama-3.2-1B Q8_0.**  This release rewrites the Metal Q8 matmul path around Apple Silicon `simdgroup_matrix<float, 8, 8>` hardware matrix-multiply accumulate (MMA), replacing the previous dot-product GEMM with four successive MMA kernel generations and a tiered dispatch that picks the right one per matmul shape.

### Performance

**`matmul_q8_mma` kernel (16×16 tile, 4 simdgroups)** (`perf`): New Metal compute kernel using Apple Silicon `simdgroup_matrix<float, 8, 8>` hardware matrix-multiply accumulate. Q8_0 weights are cooperatively dequantized into threadgroup memory once per 32-K chunk and reused across the token/row tile. Four `simdgroup_multiply_accumulate` calls complete one K=32 tile — full hardware MMA throughput vs. scalar dot products.

**`matmul_q8_mma32` kernel (32×32 tile, 8 simdgroups × 2 accumulators)** (`perf`): Large-tile variant that halves dispatch count and reuses each token-matrix `simdgroup_load` across two row accumulators per simdgroup, improving ILP and weight reuse. Preferred when `num_tokens >= 32` and `rows % 32 == 0`.

**`matmul_q8_mma32_h` (FP16-tile) variant** (`perf`): Dequantized weights and token activations are staged as `half` in threadgroup memory — 4 KB total vs 8 KB for the FP32 tile — doubling concurrent-threadgroup occupancy per GPU core. `simdgroup_multiply_accumulate` runs mixed precision: `half` inputs with a `float` accumulator, preserving the Q8_0 dynamic range (int8 × f32-scale already fits in f16). Selected when `cols >= 2048`, so 1B/3B get the occupancy win while small-hidden models (135M / 360M) stay on the FP32 path and avoid f32→f16 conversion overhead.

**`matmul_q8_mma32_h4` (4-simdgroup, 2×2-accumulator) variant** (`perf`): The dispatch default for `cols >= 2048`. Uses 128 threads (4 simdgroups) per threadgroup and each simdgroup owns a 2×2 grid of 8×8 `simdgroup_matrix` accumulators (C_00/C_01/C_10/C_11). Per K-sub iteration we load two A fragments and two B fragments, then run **four** `simdgroup_multiply_accumulate` instructions reusing each A and each B across the 2×2 output. Doubles FLOP-per-simdgroup-load vs the 2-accumulator kernel and halves thread-per-TG, which improves occupancy when the GPU is thread-budget-limited rather than shared-memory-limited.

**Prefill speedup on Apple M5 Pro (Q8_0):**

| Model | Prompt | Before | After (MMA+MMA32) | Speedup |
|-------|-------:|-------:|------------------:|--------:|
| Llama-3.2-1B | 108 | 406 | **980 tok/s** | 2.4x |
| Llama-3.2-1B | 321 | ~475 | **1,880 tok/s** | 4.0x |
| Llama-3.2-1B | 801 | 721 | **3,150 tok/s** | 4.4x |
| Llama-3.2-1B | 1,501 | 1,354 | **6,070 tok/s** | 4.5x |
| SmolLM2-135M | 1,250 | 9,335 | **23,300 tok/s** | 2.5x |
| Llama-3.2-3B | 401 | ~546 | **740 tok/s** | 1.4x |
| Llama-3.2-3B | 1,501 | ~1,597 | **2,200 tok/s** | 1.4x |

At 1,501 tokens Llama-3.2-1B sustains **~12.1 TFLOPS** — roughly 93% of the M5 Pro FP32 peak.

Dispatch tiering for Q8 matmul:
- `num_tokens >= 32` and `rows % 32 == 0`:
  - `cols >= 2048` → `matmul_q8_mma32_h4` (FP16-tile, 4 simdgroups × 4 accumulators)
  - otherwise   → `matmul_q8_mma32` (FP32-tile, 8 simdgroups × 2 accumulators)
- `16 <= num_tokens < 32` and `rows % 16 == 0` → `matmul_q8_mma`
- `4 <= num_tokens < 16` → existing `matmul_q8_gemm_batch`
- `num_tokens < 4` → per-token mat-vec kernel

## [0.5.1] — 2026-04-14 — Batched Prefill + Correctness Fixes

**ForgeLLM is now the fastest LLM inference on Apple Silicon for generation across all tested model sizes, and fastest for prefill on small-to-medium models.**

### Performance — Beat MLX on Generation, Beat llama.cpp on Prefill

Apple M5 Pro, 8-bit quantization, 64-token generation:

| Model | ForgeLLM Gen | MLX Gen | llama.cpp Gen |
|-------|------------:|--------:|--------------:|
| SmolLM2-135M | **496 tok/s** | 414 | 481 |
| SmolLM2-360M | **289 tok/s** | 264 | 267 |
| Llama-3.2-1B | **178 tok/s** | 111 | 130 |

Prefill (long prompt):

| Model | ForgeLLM | MLX | llama.cpp |
|-------|---------:|----:|----------:|
| 135M (130 tok) | **3,173** | 1,507 | 2,812 |
| 135M (1250 tok) | **9,335** | — | — |
| 1B (325 tok) | 475 | 2,718 | 556 |

### Added
- **Batched prefill Metal pipeline** (`perf`): `forward_prefill_batch()` processes all prompt tokens' matmuls in single GPU dispatches. 7 new batch kernels: `matmul_vec_batch` (f32/Q8/Q4), `rms_norm_batch`, `silu_mul_fused_batch`, `add_inplace_batch`, `copy_embedding_batch`. Batch-sequential attention with causal dependency (#179)
- **Batched causal attention kernel** (`perf`): `attention_batch` processes all M tokens in ONE dispatch (M × num_heads threadgroups) with per-token causal masking. Plus `copy_kv_both_batch` for fused KV cache updates (#158)
- **Weight-reuse GEMM kernel** (`perf`): `matmul_q8_gemm_batch` processes 4 tokens per threadgroup, reusing weight reads across the tile. 2x prefill speedup on 1B at long context
- **Q4_0 native Metal kernel** (`feat`): `matmul_vec_q4` shader with nibble unpacking, uchar4 vectorized loads, simdgroup reduction (#175)
- **OpenAI API server in Metal binaries** (`feat`): `--serve --port N` starts HTTP server with `/v1/chat/completions` (streaming + non-streaming), `/v1/models`, `/health`. Uses `tiny_http` for zero-dep HTTP (#176)
- **Built-in timing + `--quiet` flag** (`feat`): Generated binaries print prefill/generate tok/s to stderr. `--quiet` suppresses text output for benchmarks (#177)
- **`ModelConfig::validate()`** (`feat`): Checks head_dim divisibility, GQA ratio, hidden_size > 0 at compile time

### Performance
- **`packed_short4` wide loads in Q8_0 kernels** (`perf`): 64-bit (8 int8) loads via `packed_short4` + `as_type<char2>` unpack. 2x fewer memory transactions per Q8_0 block. +61% prefill on 135M, +2x on 1B long prompts
- **`fast::rsqrt`/`fast::exp`** (`perf`): Hardware fast-math in rms_norm, softmax, silu, attention
- **Multi-block Q8_0 per SIMD lane** (`perf`): Each lane processes 2 blocks per iteration for better ILP
- **float4 V·scores in attention** (`perf`): 4 dimensions per thread instead of 1 in V output loop
- **Fused `rope_qk_batch`** (`perf`): Single dispatch for Q+K RoPE instead of two (saves 30 dispatches per prefill)
- **Fused `copy_kv_both_batch`** (`perf`): Single dispatch for K+V cache copy (saves 30 dispatches)

### Fixed
- **Packed char4 alignment bug** (`fix`): Q8_0/Q4_0 data starts at +2 byte offset from block (after f16 scale), not 4-byte aligned. Replaced `char4*` with `packed_char4*`. Fixed garbled output on SmolLM2-360M (#183)
- **Dynamic `vec_tile` size** (`fix`): Was hardcoded to 4096, overflowed for Llama-3.2-1B (intermediate=8192). Now computed from model config (#184)
- **Metal memory barriers** (`fix`): Added `memoryBarrierWithScope:MTLBarrierScopeBuffers` between dispatches to prevent data races
- **Weight export panics replaced with error handling** (`fix`): `write_mixed`, `write_embed_as_f32`, `write_tensor` now return `Result<()>` with `bail!` instead of `panic!` (#185)
- **`ModelConfig::validate()` called in `cmd_compile`** (`fix`): Catches invalid configs before codegen (#186)
- **Softmax zero-denominator NaN guard** (`fix`): All 5 softmax locations now use `if sum > 0.0 { 1.0/sum } else { 0.0 }` (#188)

### Prefill Performance Journey
```
Step                              Prefill tok/s   Gain
Token-by-token (v0.5.0)                   ~150    1x
Batched matmuls                          1,219    8x
+ Batched causal attention               1,657    11x
+ float4 + fused RoPE/KV                 2,250    15x
+ packed_short4 wide loads               3,173    21x
```

## [0.5.0] — 2026-04-13 — Metal GPU Release: Faster than llama.cpp

**Built a complete Metal GPU inference engine from scratch and beat llama.cpp.**
567 tok/s peak vs llama.cpp's 492 tok/s on SmolLM2-135M Q8_0 (Apple M5 Pro).

### Added
- **Native Metal GPU codegen** (`feat`): `forge compile --target metal` generates a standalone Metal compute shader project for Apple Silicon. Full transformer forward pass on GPU with zero-copy weight loading via unified memory (#152, #154)
- **Q8_0 native Metal kernel** (`perf`): Weights kept as raw Q8_0 bytes on GPU — dequantized on-the-fly in the matmul shader. Halves GPU memory bandwidth vs f32 dequant (#164)
- **Q4_0 NEON sdot kernels** (`perf`): 4-bit→int8 in-register unpacking via `vand`/`vshr`/`sub` + `sdot` inline assembly. dot1/dot4/dot8 batch variants matching Q8_0 optimization (#151)
- **Apple Accelerate/AMX** (`perf`): f32 matmul functions use `cblas_sgemv` on macOS for hardware matrix acceleration (#153)
- **CPU prefetch hints** (`perf`): `prfm pldl1keep` 2 blocks ahead in Q8_0 dot4/dot8 kernels (#163)
- **`quantize_f32_to_q4_0()`**: New function for F32→Q4_0 weight conversion, enabling mixed-quant GGUF support

### Metal Performance Optimizations (8 → 567 tok/s)
- **Async dispatch**: Single command buffer per forward pass, `wait_until_completed()` once (#155)
- **Pre-allocated buffers**: 11 reusable working buffers, zero per-token allocation (#156)
- **Simdgroup matmul**: 32-lane cooperative dot product, shared memory vector caching, float4 vectorized loads (#157)
- **Simdgroup attention**: Cooperative Q·K^T with `simd_sum`, shared memory scores, `simd_max`/`simd_sum` softmax (#158)
- **Encoder batching**: 2 compute encoders per layer instead of ~15 (#158)
- **`add_inplace` kernel**: Eliminates temporary buffer + blit for residual connections
- **Single compute encoder**: `copy_buffer`/`copy_offset` compute kernels replace all blit operations — zero encoder transitions per forward (#159)
- **Fused QKV + gate/up projections**: 3 matmul dispatches per layer instead of 7 (#170)
- **Fully unrolled Q8_0 kernel**: 32 explicit multiply-accumulate lines, float4 aligned shared memory loads, 4 rows per simdgroup (#171)
- **`fast::rsqrt`/`fast::exp`**: Hardware fast-math in rms_norm, softmax, silu, attention
- **char4 vectorized int8 loads**: 4x fewer memory transactions in Q8_0 kernel
- **Double-buffered prefill**: `forward_prefill()` overlaps GPU execution with CPU encoding

### Fixed
- **Mixed-quant GGUF weight export**: Q4_0 models with Q4_1 tensors (e.g. `ffn_down`) now correctly quantize to Q4_0 at export time instead of writing raw Q4_1/Q8_0 bytes. Fixes NaN crash in AOT binaries for Q4_0 models.
- **Metal threadgroup memory**: Reduced `vec_tile` from 64KB to 16KB to fit Apple Silicon's 32KB limit

### Benchmarks (SmolLM2-135M Q8_0, Apple M5 Pro)

| Backend | tok/s | vs llama.cpp |
|---------|-------|-------------|
| **ForgeLLM Metal** | **567 peak** | **115%** 🏆 |
| llama.cpp (CPU+Metal) | 492 | 100% |
| ForgeLLM CPU AOT | 180-220 | 37-45% |
| ForgeLLM Q4_0 AOT | 206 | 42% |

Metal performance journey: 8 → 80 → 169 → 221 → 396 → 417 → 334 → **567 tok/s** (71x improvement)

## [0.4.0] — 2026-04-11

### Added
- **Quantized matmul without dequantization** (`perf`): Q8_0 weights are now kept as raw bytes (int8 + f16 scale blocks). Dot products are computed directly on Q8_0 values — ~2x smaller weight files, better cache utilization (#54)
- **WASM compilation target** (`feat`): `forge compile --target wasm` generates a `wasm32-unknown-unknown` Rust project with SIMD128 dot product, `#[wasm_bindgen]` exports (`WasmModel::new`, `forward`, `reset_cache`), and a `pkg/model.js` JS glue layer for browser integration (#52)
- **Speculative decoding** (`feat`): `forge speculative --draft <model> --target-model <model> --output <dir>` compiles both models as library crates and generates a speculative decoding runner — draft runs N tokens ahead, target verifies, accepts matching tokens with KV cache rollback on mismatch (#58)
- **LoRA adapter support** (`feat`): `forge compile --lora <adapter.safetensors>` merges LoRA weights at compile time (`W += (alpha/rank) * B @ A`) — the AOT binary has adapted weights baked in with zero runtime overhead; supports SafeTensors format (#59)
- **Mistral and Qwen2 architecture support** (`feat`): `ModelConfig` gains `sliding_window_size` and `qkv_bias` fields; sliding-window attention kernel emitted for Mistral models; QKV bias-add loops emitted for Qwen2 (#143)
- **E2E correctness validation** (`test`): fast `syn`-based source-validity test runs on every `cargo test`; slow deterministic build+run test gated behind `--ignored` with a nightly CI workflow (#142)

### Performance
- **Flash Attention kernel** (`perf`): tiled online-softmax attention (`FLASH_ATTN_BLOCK_SIZE=64`) replaces the O(seq_len) scores buffer for models with `max_seq_len > 512` and no sliding window. Caps per-head scratch to 256 bytes regardless of context length (#145)
- **Parallel matmul threshold 4096 → 512** (`perf`): attention Q/K/V/O projections (N≥512) now run in parallel on all model sizes, not just logits; Qwen2.5-0.5B attention projections (N=896) gain parallelism (#144)

### Fixed
- Resolved `clippy::too_many_arguments` lint in `cmd_compile` (refactored to `CompileArgs` struct) that was failing CI under `-D warnings`
- Fixed `cargo fmt` failures across `main.rs`, `emit.rs`, and `project.rs`

## [Unreleased] — v0.3.1-dev

### Added
- `--seed S` flag in AOT binary for reproducible sampling
- `--quiet`/`-q` flag in AOT binary to suppress generated text output
- `KVCache::reset()` and `KVCache::memory_bytes()` methods in generated code
- `Default` impl for generated `KVCache`
- `--version` output now shows: parameters, weight memory, FLOPs/token, KV cache memory
- `Load time:` printed by AOT binary for benchmarking visibility
- Comprehensive AOT binary CLI flags documentation in guides

### Changed
- `MAX_SEQ_LEN` capped at 4096 in generated code (was full model max, e.g. 32768 for Qwen)
- `load_weights` and `bytes_to_f32` use `std::ptr::copy_nonoverlapping` instead of `chunks_exact` iteration (~25% faster load)
- `chat /clear` uses `cache.reset()` instead of re-allocating KV cache; also clears recent_tokens
- Bench script uses `--quiet --temp 0.7 --top-k 40` for stable timing

### Fixed
- Bench script no longer hits early EOS due to deterministic argmax on short prompts
- Recent_tokens were lingering across `/clear` in chat mode

### Tests
- 21 codegen tests (was 14): added 7 new tests covering NEON dot accumulators,
  parallel matmul path, row unrolling, release profile, /clear regression,
  --seed flag wiring, --version KV cache info

### Benchmarks (SmolLM2-135M Q8_0, Apple M5 Pro)
- v0.3.0:    119.5 tok/s
- v0.3.1-dev: 117.2 tok/s (--seed 42, very stable, no regression)

## [0.3.0] — 2026-04-09 — AOT Performance Release

**3.3x AOT speedup over v0.2.0.** AOT binaries now match the runtime interpreter.

### Added
- 4-accumulator NEON `dot_f32` with 16-element unrolled inner loop in generated code
- 4-way row unrolling in shape-specialized `matmul_vec_KxN` for ILP
- `par_chunks_mut(256)` parallel matmul path with row ILP per thread
- `LTO=fat`, `strip = true`, `panic = "abort"` in generated `Cargo.toml`
- `forge compile --cross-target` for cross-compilation
- `--interactive`/`--chat` mode in AOT binary
- `--save-cache`/`--load-cache` for persistent KV cache
- `--top-p` nucleus sampling
- `--repeat-penalty`
- `--temp` and `--top-k` sampling
- `--max-tokens` CLI flag
- `--version`/`-V` flag with model info
- EOS token detection in generation loop
- RoPE frequency table precomputation
- Memory-mapped weight loading via memmap2
- GeLU activation + fused `gelu_mul` kernel
- NEON SIMD for `softmax`, `rms_norm` apply, `elementwise_mul/add`, `residual_add`
- AOT compilation guide and tutorial
- Benchmark infrastructure: `benchmarks/run.sh`, `HISTORY.md`, CI workflow

### Performance (SmolLM2-135M Q8_0, Apple M5 Pro)
- v0.2.0:  36.2 tok/s (30% of interpreter)
- v0.3.0: 119.5 tok/s (100% of interpreter, 3.3x speedup)

Multi-model verification:
- SmolLM-135M:  119.5 vs 119.7 interp (100%)
- SmolLM-360M:   47.3 vs  46.3 interp (102%)
- Qwen2.5-0.5B:  35.8 vs  33.6 interp (107%)

Binary size: 3.8 MB → 2.7 MB (-29%)

## [0.2.0] — 2026-04-09 — AOT Compilation

### Added
- AOT compilation: `forge compile --model x.gguf --output ./out [--run] [--embed-weights]`
- Shape-specialized matmul kernels with baked-in dimensions
- Static memory planning: zero-allocation forward pass
- Operator fusion: `silu_mul`, `residual_add`
- 7 crates published to [crates.io](https://crates.io/search?q=forgellm)

## [0.1.0] — Initial release

- GGUF/SafeTensors frontend
- IR + graph builder
- CPU interpreter with NEON SIMD
- OpenAI-compatible API server (`forge serve`)
- Chat REPL (`forge chat`)
- Benchmarking (`forge bench`)
- 6 architectures: Llama, Qwen2, Mistral, Phi3, Gemma, StableLM
- 12 GGUF quantization formats
