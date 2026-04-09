# Optimizer

The optimizer analyzes and transforms IR computation graphs to improve performance.

## Fusion Detection

The optimizer identifies patterns of operations that can be merged into single fused kernels:

### RMSNorm + MatMul

When an RMSNorm output is consumed by a single MatMul, the normalization can be fused into the matrix multiplication, eliminating the intermediate normalized tensor.

### GateUp + SiLU + Mul

The SiLU-gated FFN pattern (common in Llama, Mistral, Qwen2):
```
gate = MatMul(input, gate_weight)
gate_act = SiLU(gate)
up = MatMul(input, up_weight)
result = Mul(gate_act, up)
```

This can be fused into a single kernel that reads the input once and produces the result directly.

## Dead Code Elimination

The optimizer performs backward liveness analysis from the graph output to identify unreachable nodes. Dead nodes are removed to reduce computation.

## Configuration

Optimization passes can be individually enabled/disabled:

```rust
let config = OptimizeConfig {
    enable_fusion: true,
    enable_dce: true,
};
let optimized = optimize_with_config(&graph, &config);
```

## Current Status

The optimizer currently performs analysis only (identifies fusion opportunities and dead nodes). Graph rewriting with fused operations is planned for future releases.
