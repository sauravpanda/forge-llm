# ForgeLLM Benchmark History

Performance tracking across versions. All benchmarks run on SmolLM2-135M-Instruct Q8_0 (64 tokens, 3 runs).

## Results

| Version | Date | Interpreter (tok/s) | AOT (tok/s) | AOT Build (s) | Binary Size | System |
|---------|------|---------------------|-------------|---------------|-------------|--------|
| v0.2.0 | 2026-04-09 | 119.7 | 36.2 avg (best: 40.2) | 29s | 3.8 MB | Darwin arm64 18c (Apple M5 Pro) |

## Analysis

### v0.2.0 — AOT Compilation Baseline

First release with AOT compilation. Key observations:

- **Interpreter is 3.3x faster** than AOT binary (119.7 vs 36.2 tok/s)
- This is expected: the interpreter's runtime kernels have more mature NEON SIMD
  optimization with hand-tuned 4-way accumulator patterns
- The AOT binary relies on LLVM to auto-vectorize the generated Rust code, which
  doesn't match hand-written intrinsics
- **AOT build time is 29s** — dominated by tokenizers crate compilation

### Improvement targets for v0.3.0

1. **Inline NEON intrinsics in generated matmul** — match interpreter's dot_f32 performance
2. **Reduce Rayon overhead** — tune parallelism threshold for small models
3. **Optimize logits projection** — largest single matmul (576 x 49152)
4. **Profile-guided optimization (PGO)** — use cargo-pgo for the AOT binary

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
