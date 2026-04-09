# Performance Results

## Verified Models

All benchmarks on Apple Silicon (M-series), optimized kernels (unrolled matmul).

### SmolLM2-135M-Instruct (Q8_0)

| Metric | Value |
|--------|-------|
| Parameters | 135M |
| Architecture | Llama |
| Quantization | Q8_0 |
| Weight memory | 538 MB |
| Avg prefill | 46.0 tok/s |
| Avg generate | **46.3 tok/s** |

### SmolLM2-360M-Instruct (Q8_0)

| Metric | Value |
|--------|-------|
| Parameters | 360M |
| Architecture | Llama |
| Quantization | Q8_0 |
| Weight memory | 1,447 MB |
| Avg generate | **17.5 tok/s** |

### Qwen2.5-0.5B-Instruct (Q8_0)

| Metric | Value |
|--------|-------|
| Parameters | 494M |
| Architecture | Qwen2 |
| Quantization | Q8_0 |
| Weight memory | 2,521 MB |
| Avg generate | **12.0 tok/s** |

## Sample Output

```
$ forge run --model SmolLM2-135M-Instruct-Q8_0.gguf \
            --tokenizer tokenizer.json \
            --prompt "The meaning of life is"

The meaning of life is a complex and multifaceted concept that has been
debated by philosophers, scientists, and theologians for centuries. At its
core, the question of what it means to be human...

Prefill: 5 tokens in 0.13s (37.3 tok/s)
Generate: 65 tokens in 1.42s (45.7 tok/s)
```

## Performance History

| Version | SmolLM-135M Q8_0 | Improvement |
|---------|------------------|-------------|
| v0.1 (reference interpreter) | 21.6 tok/s | baseline |
| v0.2 (unrolled matmul) | **46.3 tok/s** | **2.1x** |

## Roadmap

| Optimization | Expected Speedup |
|-------------|-----------------|
| ARM NEON SIMD intrinsics | 2-3x |
| Operator fusion | 1.5-2x |
| Quantized inference (skip dequant) | 2-3x |
| Static memory planning | 1.2x |
| **Combined** | **10-30x** |

Target: **200+ tok/s** for SmolLM-135M on Apple Silicon.
