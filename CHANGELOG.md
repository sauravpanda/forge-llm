# Changelog

All notable changes to ForgeLLM are documented here.

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
