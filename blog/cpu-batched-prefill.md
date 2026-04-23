# CPU Caught Up: Batched Prefill Gets 9.5× on Apple Silicon

*How ForgeLLM's v0.8.x series took Llama-3.2-1B Q8_0 CPU prefill from 19 tok/s to 180 tok/s at long contexts, and Q4_0 from 26 to 238 tok/s — without touching the inner sdot kernel.*

---

**TL;DR.** The v0.7.x series put Metal decode ahead of MLX and llama.cpp. CPU prefill was the embarrassing gap: Llama-3.2-1B Q8_0 at 2,500 tokens managed **19 tok/s** in v0.7.13 — and *slowed down* with longer prompts. The v0.8.x series (0.8.0 / 0.8.1 / 0.8.2) restructures the CPU forward pass from per-token to batched: batched matmul amortizes weight bandwidth across all M prompt tokens, Q-tiled flash attention amortizes K/V bandwidth across 16 queries per block, and the Q4_0 path gets the same treatment. **9.5× on Q8_0 at 2,500 tokens. 9.2× on Q4_0 at 1,600 tokens.**  No new inner kernel — just structure.

## The setup: where v0.7.x left us

ForgeLLM's CPU backend has been using the same shape-specialized `matmul_vec_q8_0_KxN` (single-token mat-vec) kernels since v0.3. They were already well-optimized: AArch64 inline asm for SDOT, 4-accumulator ILP for block-scale multiply, `par_chunks_mut(256)` for cross-core parallelism on large matrices. The generation numbers were fine.

But prefill was running at M calls of these per-token kernels in a tight loop:

```rust
for (tok_pos, &token_id) in tokens.iter().enumerate() {
    for layer_idx in 0..NUM_LAYERS {
        rms_norm(...);
        matmul_vec_q8_0_HxN(&mut q, &normed, &lw.q_proj);   // M×L times
        matmul_vec_q8_0_HxN(&mut k, &normed, &lw.k_proj);
        ...
    }
}
```

For Llama-3.2-1B at M=800 tokens:

| Prompt | Prefill tok/s | Direction |
|-------:|--------------:|:----------|
| 100 | ~40 | — |
| 800 | ~30 | **slower with more tokens** |

The "slower with more tokens" is the dead giveaway: Metal prefill gets faster with longer prompts (more batch to amortize fixed cost over). CPU got *slower*. Why? Every matmul in that inner loop re-reads the full weight matrix from DRAM. For Llama-3.2-1B Q8_0 that's ~1.3 GB of weights × M tokens = 1 TB of DRAM bandwidth for a 800-token prefill. Even on M5 Pro's fast unified memory bus, that's bandwidth-bound at tens of tok/s.

## v0.8.0 — Batched matmul

### The restructure

We kept every byte of the inner sdot kernel and rebuilt the outer structure. The generated `forward_prefill` now dispatches to `forward_prefill_batched` when `seq_len ≥ PREFILL_BATCH_THRESHOLD` (=8), which:

1. Heap-allocates `m × dim` batch buffers for each inter-layer tensor (hidden, normed, q/k/v, attn_out, gate/up/down).
2. For each layer, batched QKV / O / gate / up / down matmul runs once.  Per-token rms_norm, RoPE, K/V cache writes, attention, silu_mul, residual_add stay in inner `for r in 0..m` loops.
3. Final norm + lm_head runs on the last token only (matches the existing semantics).

### The kernel

`matmul_mat_q8_0_KxN(output, m, input, weight)` is the new primitive. The key is the loop order:

```rust
(0..n_chunks8).into_par_iter().for_each(|n_block| {
    let j0 = n_block * 8;
    // Load 8 weight rows into L1 — hot for the full m loop.
    let w0 = &weight[(j0+0)*ROW_BYTES..(j0+1)*ROW_BYTES];
    // ... w1..w7 ...
    for r in 0..m {
        let input_q = &input_q8[r*ROW_BYTES_IN..(r+1)*ROW_BYTES_IN];
        let (d0,d1,d2,d3,d4,d5,d6,d7) =
            dot8_q8_0_q8_0(input_q, w0, w1, w2, w3, w4, w5, w6, w7, K);
        // Disjoint output writes per n_block — sound across workers.
        unsafe {
            let p = (out_addr as *mut f32).add(r * N + j0);
            *p.add(0) = d0; /* ... */; *p.add(7) = d7;
        }
    }
});
```

The outer `for n_block` is parallelized over cores.  Within a core, the 8 weight rows (17 KB for K=2048) stay hot in L1 across all M inputs.  Each input row gets streamed through L1 once per n-block.  Bandwidth drops by roughly M× relative to the per-token loop.

Result (Llama-3.2-1B Q8_0, M5 Pro):

| Prompt | v0.7.x | **v0.8.0** | Speedup |
|-------:|-------:|-----------:|--------:|
|   352  |   40.2 |   **129.5** | **3.2×** |
|   902  |   30.9 |    **66.2** | **2.1×** |
|  1603  |   24.3 |    **43.1** | **1.8×** |

The speedup decays with longer prompts because at M=1600 the O(M²) per-token `attention_flash` calls start to dominate — we moved the matmul bottleneck out of the way and found attention behind it.

## v0.8.1 — Batched flash attention

### Same idea, different kernel

`attention_flash_batch` is Q-tiled flash attention.  Per head (parallel via rayon), queries are processed in `Q_TILE=16` chunks.  For each Q tile:

- Stream K/V in `FLASH_ATTN_BLOCK_SIZE=64`-sized blocks.
- **Phase 1**: compute 16 × 64 scores.  K[t] loaded once, dotted with 16 queries.
- **Phase 2**: causal mask (query row `r` sees K positions `0..=base_pos+r`), then per-query online softmax update (running max, rescale, exp).
- **Phase 3**: `P · V` accumulate.  V[t] loaded once, multiplied into all 16 queries' output accumulators.

Per-query state (`m_state`, `l_state`, `acc`) stack-allocates to `16 × HEAD_DIM × f32` — 4 KB for Llama's `head_dim=64`, 16 KB for `head_dim=256`.  No heap allocations inside the kernel.

The key observation: each K/V position is loaded once per `(head, Q_TILE)` pair instead of once per query.  At M=2500, that's ~16× less K/V bandwidth than per-token flash attention.

Result:

| Prompt | v0.7.x | v0.8.0 (matmul) | **v0.8.1 (+attn)** | Total |
|-------:|-------:|----------------:|-------------------:|------:|
|   352  |   40.2 |           129.5 |          **190.6** | **4.7×** |
|   902  |   30.9 |            66.2 |          **234.2** | **7.6×** |
|  1603  |   24.3 |            43.1 |          **207.4** | **8.5×** |
|  2502  |   18.9 |            29.4 |          **180.0** | **9.5×** |

Throughput now *peaks in the middle* (902–1603 tokens) because short prompts still pay the per-token setup cost and very long prompts start to lose to the absolute O(M²) work.

## v0.8.2 — Q4_0 extension

Mechanical.  Q4_0 dot kernels already existed (`dot8_q4_0_q8_0` / `dot4_q4_0_q8_0` — Q4_0 weight × Q8_0-quantized input).  Added `matmul_mat_q4_0_KxN` that parallels `matmul_mat_q8_0_KxN` but with 18-byte weight rows instead of 34.  `forward_prefill_batched` became dtype-aware.  `attention_flash_batch` is reused unchanged (it operates on f32 Q + i8 K/V cache).

Q4_0 actually runs *faster* than Q8_0 at the same prompt length because the per-row weight bytes are smaller (18 vs 34 bytes per 32-element block), so memory bandwidth per forward pass drops proportionally:

| Prompt | Q4_0 per-token | **Q4_0 batched (v0.8.2)** | Speedup |
|-------:|---------------:|--------------------------:|--------:|
|   352  |           43.8 |                 **303.5** | **6.9×** |
|   902  |           33.6 |                 **265.6** | **7.9×** |
|  1603  |           25.7 |                 **237.5** | **9.2×** |

## What we learned

- **The inner kernel was not the bottleneck.**  We spent months polishing sdot ILP, scale-multiply reduction, NEON widening — all table stakes, none of which matter for long-prompt prefill.  The per-token call loop was re-reading the whole model from DRAM M times.  Restructuring gave us a 9.5× win with zero new assembly.
- **Write weight-outer, token-inner.**  It's the same FLOP count either way, but weight-outer keeps the weight rows hot in L1 across M queries.  For sdot-style kernels, 8 weight rows is the natural tile (one `dot8` call).
- **Causal masking during batching is just `valid_end = (q_pos + 1).saturating_sub(block_start).min(block_len)`.**  Don't reach for fancier block scheduling — the per-query valid end inside a block is cheap arithmetic.
- **Output aliasing across rayon workers: use `usize` round-trips.**  Different `n_block`s write disjoint output slices, but raw pointers aren't `Send`.  `let addr = output.as_mut_ptr() as usize;` inside the closure reconstructs as `*mut f32`.  The `Send + Sync` boilerplate on a wrapper struct fights Rust's 2021 disjoint-closure-captures, and debugging why takes longer than it should.

## Where this leaves us

CPU is now in llama.cpp territory for long-context prefill.  Metal is still faster (MPS or MMA flash attention puts us at 1,500–5,000+ tok/s depending on model shape), but CPU is no longer the embarrassing fallback — it's a legitimate deployment target for edge inference where a GPU isn't available.

Next frontiers from here: **batched attention for SWA models** (currently falls back to per-token inside `forward_prefill_batched`), **CPU decode fusion** (decode is inherently per-token, but there's room to fuse rms_norm + matmul across the forward pass), and **Metal + CPU combined scheduling** for the absolute long-context wall.

— the v0.8.2 release,  2026-04-23
