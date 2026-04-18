# Attention Got Hard: MMA Flash, FP16 KV Cache, and Dispatch Fusion on Apple Silicon

*How ForgeLLM's v0.7.x series took the long-context prefill wall from 1,000 tok/s to 1,700 tok/s on Llama-3.2-1B, and what we learned along the way.*

---

**TL;DR.** The v0.6.x posts ended with a cliffhanger: MMA-accelerated prefill worked, but long-context prefill plateaued because attention itself wasn't MMA-accelerated. v0.7.0 fixes that with a hardware-MMA flash attention kernel, default-on across three architectures (Llama, Qwen2.5, Phi-3). v0.7.1 halves the KV cache to fp16. v0.7.2 fuses a handful of decode-time dispatches. **+71% prefill at 3K tokens on Llama 1B. +24% decode on SmolLM 135M. +73% prefill at 3K tokens on Phi-3 Mini.** Not every optimization paid off as predicted — the honest post-mortem is below.

## The setup: where v0.6 left us

By the end of v0.6.x, ForgeLLM's generation path was comfortably ahead of MLX and llama.cpp on Apple Silicon across 135M → 3B, and prefill was fast up to ~500 tokens thanks to a tiered `simdgroup_matrix` MMA kernel stack. But prefill throughput **dropped** past 800 tokens:

| Prompt | Llama-3.2-1B v0.6.5 |
|-------:|-------------------:|
| 406 tok | 1,949 tok/s |
| 806 tok | 1,950 tok/s |
| 1,510 tok | 1,531 tok/s |
| 3,006 tok | 995 tok/s |

The culprit was attention. Matmul was running at ~4 TFLOPS on the MMA units; attention was running a scalar-reduce kernel that materialized the full scores row and then softmaxed it. At M=3000 that's O(M²) work at something like 0.3 TFLOPS — and it dominated the wall clock.

## v0.7.0 — MMA flash attention

### The kernel

The design is flash attention with an MMA twist. Each threadgroup handles an 8-token × 1-head block of Q. 128 threads (4 simdgroups) cooperate. K/V stream through threadgroup memory in `K_BLOCK=32`-sized chunks. The online-softmax recurrence runs *per Q row*:

```
m_new  = max(m_old, tile_max)
alpha  = exp(m_old - m_new)
l_new  = alpha * l_old + sum(exp(S - m_new))
O_new  = alpha * O_old + sum(exp(S - m_new) * V)
```

Both the Q·K^T scores and the P·V accumulation use `simdgroup_matrix<half, 8, 8>` + `simdgroup_multiply_accumulate`. Three tricks made this work cleanly:

1. **K is transposed on load.** K is stored as `[K_BLOCK, head_dim]` (natural row-major). We want `K^T` for Q·K^T. `simdgroup_load(K_tile, ptr, stride, origin, transpose=true)` gives us the transpose for free — no separate transpose pass. The Metal MMA units don't care; they just need matching tile shapes.
2. **Output O stays in threadgroup memory across K tiles** (instead of registers). This is unusual — typical flash attention keeps O in registers to avoid TG-memory round trips. But we need to rescale O by `alpha` on every K-tile boundary, which is per-Q-row. Keeping O in TG memory means each thread can update its O slice independently while the scaling factor is broadcast — far simpler than register-level reshuffling.
3. **Dual P storage.** We write softmax probabilities to both an `s_sh` float buffer (for the row-sum reduction that updates `l`) and a `p_sh` half buffer (for the P·V MMA). The precision loss from writing to half matches what the MMA units will consume anyway, and the separate float copy keeps the sum reduction accurate.

Threadgroup memory budget: ~13 KB for head_dim=64, ~24 KB for head_dim=128 — both under Apple's 32 KB/TG limit.

### The payoff

| Prompt | v0.6.5 legacy | **v0.7.0 MMA flash** | Δ |
|-------:|--------------:|---------------------:|--:|
| 406 tok | 1,949 | 2,102 | +8% |
| 806 tok | 1,950 | 2,282 | +17% |
| 1,510 tok | 1,531 | 2,081 | +36% |
| 3,006 tok | 995 | **1,699** | **+71%** |

The gains grow with context length because attention cost is O(M²). Short-prompt numbers are within noise because matmul dominates there (~90% of FLOPs for M < 500 on 1B), and both paths use the same `matmul_q8_mma32*` kernels for Q/K/V/O projections.

### Verification across architectures

Getting the kernel right on a 32-head, head_dim=64 GQA Llama is not the same as getting it right on Qwen2.5 (GQA 7:1 + QKV bias), Phi-3 (MHA, head_dim=96), or anything else with non-standard shapes. The head_dim=96 case is the worst — 96 doesn't tile cleanly into power-of-2 chunks.

Audit result: head_dim=96 works because the Phase 4 P·V loop uses `dims_per_sg = head_dim / 4 = 24` and `tiles_per_sg = dims_per_sg / 8 = 3`. The 4 simdgroups span sg0:0-23, sg1:24-47, sg2:48-71, sg3:72-95 — covers all 96 dims with no remainder. Every simdgroup_load / simdgroup_multiply_accumulate boundary lands on a multiple of 8.

Empirical result on Phi-3 Mini Q8_0:

| Prompt | Legacy | **MMA flash** | Δ |
|-------:|-------:|--------------:|--:|
| 1,201 tok | 465 | **607** | +30% |
| 3,001 tok | 278 | **480** | **+73%** |

Top-5 logits match the legacy path exactly. Output text stays coherent.

### The Phi-3 detour

We lost about a day on a non-bug. Phi-3 GGUFs store attention as a single fused `blk.N.attn_qkv.weight` tensor and MLP as a fused `blk.N.ffn_up.weight` (gate and up concatenated along the output dim). Our weight loader assumed Llama-style split `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj`, so Phi-3 export crashed with `weight not found: model.layers.0.self_attn.q_proj.weight`.

Shipped a fix in v0.6.7: detect the fused layout from GGUF metadata (`{arch}.attention.head_count{,_kv}`, `{arch}.feed_forward_length`) and split the raw bytes on block boundaries post-load. Q8_0 splits work because each section size is a multiple of 32 elements (the Q8_0 block size), so byte-range slicing lands cleanly.

Then we spent an hour convinced Phi-3 Metal was broken after the split. Every test produced `[13][13][13]...` — the newline token repeated forever. Logit dump revealed the actual top-5 were correct (`Paris` won by 4+ on the France test) — the model just fell into greedy-decode repetition after completing the prompt. Not a bug. Rewrote `tail -5` to `tail -30` in the test script and saw the coherent continuation. The lesson: when your CLI uses pure argmax sampling, a 3.8B model will repeat itself on trivial prompts just like a 0.5B model will. That's a confounder, not a regression.

## v0.7.1 — FP16 KV cache

The plan: attention reads the KV cache from HBM every decode step. At 3K-token context, Phi-3 Mini reads ~2.4 GB of K + V per step (32 KV heads × 96 head_dim × 3001 pos × 4 bytes × 32 layers × 2 for K+V). Halving to fp16 should halve that to 1.2 GB. On an M5 Pro with ~400 GB/s memory bandwidth, that's ~3 ms shaved per decode step. Phi-3 at 3K decodes at ~33 ms/token; ~9% improvement predicted.

### What we actually got

| Workload | f32 KV (v0.7.0) | **f16 KV (v0.7.1)** | Δ |
|----------|----------------:|--------------------:|--:|
| Phi-3 Mini decode @ 3001-tok ctx | 29.5 tok/s | 30.8 tok/s | **+4%** |
| Llama-3.2-1B decode @ 2502-tok ctx | 84.1 tok/s | 85.8 tok/s | +2% |

About half the predicted speedup. The error was thinking KV reads dominate. They don't — weights do.

Phi-3 Mini Q8_0 is ~4 GB of weights. Every decode step reads them all (quantized, so ~1 byte/param). Against 4 GB of weight reads, 2.4 GB of KV reads (f32) drops to 1.2 GB (f16) — but 4+2.4 = 6.4 GB total becomes 4+1.2 = 5.2 GB. That's 19% less total bandwidth, which should give ~19% speedup if everything were bandwidth-bound.

We got 4%. Two things eat the gap:

1. **Weight reads are cache-friendly**, KV reads aren't. Weights are read linearly across 32 layers in the same order every token. The L2 cache and streaming prefetcher like that. KV cache is a large random-ish access pattern that mostly misses cache. So the effective bandwidth of a KV read is higher than the bandwidth of a weight read — halving KV saves more time than halving weights would, but only on the KV fraction, which is ~30% of total traffic.

2. **The attention scoring path is compute-bound on small models.** Phi-3 Mini's 3000-token attention is not a pure bandwidth test — there's softmax and row reductions in there. Cutting bandwidth 50% doesn't cut compute at all.

The net of both: the modest +4% is roughly what a careful calculation would predict, if you don't handwave which reads are actually the bottleneck.

### The real win

Memory, not speed. At the same context length, the KV cache now takes half the RAM. On Phi-3 Mini at 4K context:

- Before: 32 layers × 32 KV heads × 96 head_dim × 4096 pos × 4 bytes × 2 (K+V) = **3.0 GB**
- After: same × 2 bytes = **1.5 GB**

Saving 1.5 GB per model is the concurrency story. On an M5 Pro with 24-32 GB unified memory, that's one or two extra models you can keep resident. Or longer contexts. Or more concurrent requests.

### Correctness

f16 KV introduces ULP-level noise in the softmax recurrence. On the Llama 1B "theory of relativity" test, decode is **byte-identical** to f32 KV for 40 tokens. On Phi-3 "Great Wall" test, decode diverges at word 15 ("Mongols" vs "nomadic tribes") — both coherent, both plausible. The noise compounds over long decodes but doesn't tip into incoherence for the models tested.

## v0.7.2 — Fused decode dispatches

### The motivation

By v0.7.1 the decode forward pass had 14 separate GPU dispatches per layer:

```
rms_norm → matmul_qkv → rope_q → rope_k → copy_k → copy_v
         → attention → matmul_o → add_residual → rms_norm
         → matmul_gate_up → silu_mul → matmul_down → add_residual
```

14 × 32 layers = 448 dispatches per decode step on Phi-3. Each dispatch incurs encoder overhead and a memory barrier. Apple's GPU dispatch overhead is typically 5-10 μs per small dispatch. 448 × 7.5 μs ≈ 3.4 ms/step, or ~10% of the 33 ms decode time on Phi-3.

Two fusions were free — the batched prefill path already had `rope_qk_batch` (fused Q and K rotation) and `copy_kv_both_batch` (fused K and V cache writes). Decode just needed to call them with M=1 instead of its own unfused variants.

### The results

| Workload | v0.7.1 | **v0.7.2** | Δ |
|----------|-------:|-----------:|--:|
| SmolLM2-135M, 128-tok decode | 405 | **503** | **+24%** |
| Llama-3.2-1B, short decode | 162 | **173** | +7% |
| Llama-3.2-1B, decode @ 2502-tok ctx | 85.8 | **89.9** | +5% |
| Phi-3 Mini, decode @ 3001-tok ctx | 30.8 | 30.8 | ~0% |

The 24% on SmolLM is the headline. On a 135M model, each dispatch does almost nothing — the kernel launches *are* the work. Cutting 2 of them per layer across 30 layers is proportionally huge.

Phi-3 Mini sees nothing because each of its 32 layers does enough compute per dispatch that the barrier savings are in the noise. That's also why decode speed on Phi-3 is already compute-bound at long context.

### The surprise

I had predicted this would help long-context decode on large models most, because KV cache reads scale with context. The actual data says the opposite: it helps *small models at any context* and has no effect on large models. The fixed per-dispatch overhead (~5-10 μs) is the relevant quantity, not the bandwidth.

This is a useful rule of thumb for Metal: **when a dispatch does less than ~50 μs of real work, the overhead is a significant fraction of its cost**. When it does more than ~500 μs, the overhead is noise. Phi-3's attention dispatch at 3K context takes milliseconds per layer — the overhead is 0.3%.

## Cumulative state (v0.6.5 → v0.7.2)

Four releases, three honest wins, two dead ends.

| Release | Change | Llama-1B 3K prefill | Llama-1B decode | SmolLM decode |
|---------|--------|--------------------:|----------------:|--------------:|
| v0.6.5 | Legacy attention | 995 | — | — |
| v0.7.0 | MMA flash default | 1,699 | — | — |
| v0.7.1 | f16 KV cache | — | 85.8 | — |
| v0.7.2 | Dispatch fusion | — | 89.9 | **503** |

Prefill @ 3K tok: **+71%**. Decode at long context: **+7%**. Decode on small models: **+24%**.

The dead ends:
- FP16 KV cache predicted 9% decode speedup; delivered 4%. Correct direction, wrong magnitude — I was handwaving which reads actually hit cache.
- Phi-3 Metal was "broken for a few hours" because of greedy-decode degeneration being mistaken for a logits bug. Good reminder that 0.5B-style repetition looks the same whether the model has 0.5B or 3.8B params when sampling is argmax.

## What's next

- **Decode-time flash attention.** The current decode attention still materializes the full scores row up to seq_len. Online softmax in registers eliminates that. Helps long-context decode bandwidth but probably not as much as it sounds, per the v0.7.1 analysis — weights, not KV, are the decode bottleneck.
- **RMS-norm + matmul fusion.** Read input from threadgroup memory after normalization. Saves one dispatch per norm. Should benefit all sizes more uniformly than dispatch-level fusion.
- **Gemma / StableLM / Mistral coverage.** These have architectural quirks (Gemma logit softcap, StableLM LayerNorm + parallel residual + partial RoPE, Mistral sliding window) that need real codegen work, not just verification.

## Try it

```bash
cargo install forgellm-cli   # 0.7.2
forge compile --model my-model.gguf --output ./my-model --target metal
forge export-weights --model my-model.gguf --output ./my-model/weights.bin
cp tokenizer.json ./my-model/
cd my-model && cargo build --release
./target/release/my-model weights.bin tokenizer.json "Hello world"
```

MMA flash attention is on by default. `FORGE_MMA_ATTN=0 ./my-model ...` opts out if you want to compare against the legacy scalar attention kernel.

ForgeLLM is MIT-licensed on [GitHub](https://github.com/sauravpanda/forge-llm). The MSL kernels, Rust codegen, and all the regression tests quoted above are in `crates/forgellm-codegen-metal/src/lib.rs`.

---

*Previous post: [How We Beat llama.cpp and MLX](beating-llama-cpp.md) — the v0.5–v0.6 story on generation speed and MMA prefill.*
