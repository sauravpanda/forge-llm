# Performance Results

## SmolLM2-135M-Instruct (Q8_0)

Tested on Apple Silicon (M-series), reference interpreter (naive f32, no SIMD).

| Metric | Value |
|--------|-------|
| Model | SmolLM2-135M-Instruct |
| Quantization | Q8_0 |
| Parameters | 135M |
| Weight memory | 538 MB (dequantized to f32) |
| Weight load time | 0.1s |
| Prefill throughput | ~20 tok/s |
| Generation throughput | ~21.6 tok/s |

### Sample Output

```
Prompt: "The meaning of life is"

The meaning of life is a complex and multifaceted concept that has been
debated by philosophers, scientists, and theologians for centuries. At its
core, the question of what it means to be human...
```

## Performance Roadmap

The current numbers are from the reference interpreter — a naive, unoptimized f32 implementation. Planned optimizations:

| Optimization | Expected Speedup |
|-------------|-----------------|
| SIMD matmul (AVX2/NEON) | 3-5x |
| Operator fusion | 1.5-2x |
| Quantized inference (skip dequant) | 2-3x |
| Static memory planning | 1.2x |
| **Combined** | **10-30x** |

Target: **200+ tok/s** for SmolLM-135M on Apple Silicon.
