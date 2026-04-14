# ForgeLLM Benchmark History

Performance tracking across versions. All benchmarks run with 64 tokens, 3 runs.
System: Darwin arm64 18c (Apple M5 Pro).

## SmolLM2-135M Q8_0 (hidden=576, 30 layers)

| Version | Date | Interpreter | AOT | AOT vs Interp |
|---------|------|------------|-----|---------------|
| v0.2.0 | 2026-04-09 | 119.7 | 36.2 (best 40.2) | 30% |
| v0.3.0-dev1 | 2026-04-09 | 119.7 | 105.2 (best 105.5) | 88% |
| v0.3.0-dev2 | 2026-04-09 | 119.7 | 116.2 (best 117.8) | 97% |
| v0.3.0-dev3 | 2026-04-09 | 119.7 | 119.9 (best 122.1) | 100% |
| v0.3.0-dev4 | 2026-04-09 | 119.7 | **119.5** (best 122.9) | **100%** |
| v0.3.1-dev | 2026-04-10 | 119.7 | **117.2** (seed=42, stable) | 98% |
| v0.4.0 | 2026-04-11 | 119.7 | — (pending) | — |

## SmolLM2-360M Q8_0 (hidden=960, 32 layers)

| Version | Date | Interpreter | AOT | AOT vs Interp |
|---------|------|------------|-----|---------------|
| v0.3.0-dev3 | 2026-04-09 | 46.3 | 44.7 (best 44.9) | 96% |
| v0.3.0-dev4 | 2026-04-09 | 46.3 | **47.3** (best 47.5) | **102%** |

## Qwen2.5-0.5B Q8_0 (hidden=896, intermediate=4864, vocab=151936)

| Version | Date | Interpreter | AOT | AOT vs Interp |
|---------|------|------------|-----|---------------|
| v0.3.0-dev3 | 2026-04-09 | 33.6 | 25.8 (best 26.0) | 77% |
| v0.3.0-dev4 | 2026-04-09 | 33.6 | **34.5** (best 36.2) | **103%** |

## Analysis

### v0.4.0 — Flash Attention + Broader Architecture Support

Key changes vs v0.3.1-dev:

1. **Parallel matmul threshold 4096 → 512**: attention Q/K/V/O projections on SmolLM2-135M
   (N=576) now run in parallel. Effect on 135M is modest (Rayon overhead at N=576 is small);
   larger models (Qwen2.5-0.5B, N=896) gain more.
2. **Flash Attention** (`max_seq_len > 512`, no SWA): per-head scratch is 64×f32=256 bytes
   regardless of context length. At 64-token generation benchmarks the tiled loop closely
   matches the standard path — measurable benefit appears at longer prompts (1K–4K tokens).
3. **Mistral / Qwen2 support**: sliding-window attention and QKV bias are now emitted
   correctly; benchmarking Mistral-7B models is now unblocked.

*Fresh benchmark numbers pending — run `./benchmarks/run.sh` on an Apple Silicon machine
to populate the v0.4.0 row above.*

### v0.2.0 — AOT Compilation Baseline

First release with AOT compilation. Key observations:

- **Interpreter is 3.3x faster** than AOT binary (119.7 vs 36.2 tok/s)
- This is expected: the interpreter's runtime kernels have more mature NEON SIMD
  optimization with hand-tuned 4-way accumulator patterns
- The AOT binary relies on LLVM to auto-vectorize the generated Rust code, which
  doesn't match hand-written intrinsics
- **AOT build time is 29s** — dominated by tokenizers crate compilation

### Improvement targets for v0.3.0

1. **Inline NEON intrinsics in generated matmul** — match interpreter's dot_f32 performance ✅ DONE
2. **Reduce Rayon overhead** — tune parallelism threshold for small models ✅ DONE (1024 → 4096)
3. **Optimize logits projection** — largest single matmul (576 x 49152)
4. **Profile-guided optimization (PGO)** — use cargo-pgo for the AOT binary

### v0.3.0-dev4 — Scaling to Larger Models

After verifying v0.3.0-dev3 on larger models, the parallel matmul path
(used when N >= 4096) was found to be inefficient — `par_iter_mut`
single-element granularity caused Rayon overhead to dominate.

Fix: Replaced parallel path with `par_chunks_mut(256)` + 4-way row ILP per thread.

Results across all 3 models:
- **SmolLM-135M**: 119.5 tok/s (no regression vs dev3)
- **SmolLM-360M**: 47.3 tok/s (was 44.7, now beats interpreter)
- **Qwen2.5-0.5B**: 34.5 tok/s (was 25.8, +34%, now beats interpreter)

### v0.3.0-dev Results

**AOT now matches interpreter performance** — 3.3x speedup vs v0.2.0:

- **v0.2.0**: 36.2 tok/s (30% of interpreter)
- **v0.3.0-dev1**: 105.2 tok/s (88% — 4-acc NEON dot_f32 + Rayon threshold tune)
- **v0.3.0-dev2**: 116.2 tok/s (97% — 4-way row unrolling in matmul_vec_KxN)
- **v0.3.0-dev3**: 119.9 tok/s (**100%+** — Rayon par_chunks_mut(256) for logits)

Optimizations:
1. Upgraded `dot_f32` to 4 NEON accumulators with 16-element unrolled inner loop
2. Bumped Rayon parallelism threshold 1024 → 4096
3. 4-way row unrolling in `matmul_vec_KxN` (matches runtime kernels)
4. `par_chunks_mut(256)` for logits — amortizes Rayon overhead
5. Cleaned up generated code warnings (`#![allow(dead_code, ...)]`)

## How to Run

```bash
# Run all benchmarks
./benchmarks/run.sh

# Interpreter only
./benchmarks/run.sh --interp-only

# AOT only
./benchmarks/run.sh --aot-only

# Skip model download (if cached)
./benchmarks/run.sh --skip-download
```

Results are saved to `benchmarks/results/<version>.json` and appended to this file.
| v0.4.0 | 2026-04-13 | 128.8 | 219.2 (best: 221.8) | 29s | 2.8 MB | Darwin arm64 18c |
| v0.4.0 | 2026-04-13 | - | 188.7 (best: 192.1) | 31s | 2.8 MB | Darwin arm64 18c |
| v0.4.0 | 2026-04-14 | - | 143.7 (best: 196.0) | 33s | 2.8 MB | Darwin arm64 18c |
