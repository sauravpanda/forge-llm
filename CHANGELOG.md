# Changelog

All notable changes to ForgeLLM are documented here.

## [0.7.4] — 2026-04-18 — Decode Attention V-step Restructure (Metal)

Patch release: the V-weighted-sum loop in `kernel void attention` is rewritten so each simdgroup (32 lanes) cooperatively reduces over `seq_len` for one `d4` output chunk, instead of each thread looping over `seq_len` alone for one `d4`. This fixes a large parallelism underutilization on `head_dim=64` — where only 16 of 256 threads per threadgroup were productive — without changing the Q·K^T or softmax steps.

### Changed

- **V-step outer loop**: `for (uint d4 = tid; d4 < head_dim4; d4 += 256)` → `for (uint d4 = simd_id; d4 < head_dim4; d4 += 8)`. 8 simdgroups handle 8 `d4` values per outer iteration.
- **V-step inner loop**: each lane now handles `s = simd_lane, simd_lane+32, simd_lane+64, ...` across the full `seq_len`, loading one `half4` per `s` per lane.
- **Reduction**: four component-wise `simd_sum` calls reduce the 32 lane partials per `d4`; lane 0 of each simdgroup writes the `float4` output.
- Removed the `s += 4` unroll from the V loop — each lane already strides 32 positions per iter, which gives enough memory-level parallelism.
- Same transformation applied to the scalar fallback for `head_dim % 4 != 0` (unused on current supported models — all use `head_dim ∈ {64, 96, 128}`).

### Performance (Apple M5 Pro, Q8_0, decode tok/s, median of 3)

| Workload | v0.7.3 | **v0.7.4** | Δ |
|----------|-------:|-----------:|--:|
| SmolLM2-135M, ~10-tok decode | 564 | 567 | noise |
| SmolLM2-135M, decode @ ~900-tok ctx | 203 | **296** | **+46%** |
| SmolLM2-135M, decode @ ~2250-tok ctx | 103 | **165** | **+60%** |
| Llama-3.2-1B, ~10-tok decode | 170 | 172 | noise |
| Llama-3.2-1B, decode @ ~2250-tok ctx | 99.6 | **123.9** | **+24%** |

`head_dim=64` (both SmolLM2-135M and Llama-3.2-1B) sees the largest gain because the previous layout kept only 16 / 256 threads productive in the V-step; the new layout keeps all 256 productive.

Profile breakdown for Llama-3.2-1B @ 2502-tok decode, layer time:

| | v0.7.3 | v0.7.4 |
|---|-------:|-------:|
| Layers (GPU) | 8.82 ms | **6.90 ms** |

That -22% on layer time maps cleanly onto the +24% decode throughput after the constant per-token overhead of norm+logits and embedding lookup is folded in.

### Correctness

- Byte-identical output text on SmolLM-135M and Llama-3.2-1B vs v0.7.3 for the "meaning of life" prompt.
- The V-step reduction is a sum over exactly the same terms in a different order; all supported `head_dim` values are multiples of 4 so no scalar-fallback paths fire.
- All 71 Metal codegen tests pass, including the strengthened `decode_attention_uses_half4_vectorized_loads` which now also asserts the `simd_sum` reduction and simdgroup-partitioned `d4` loop.

## [0.7.3] — 2026-04-18 — Vectorized Decode Attention (Metal)

Patch release: the single-token decode attention kernel (`kernel void attention`) now issues `half4` loads against the f16 KV cache, mirroring the prefill path. Both the Q·K^T dot product and the scores·V weighted sum go through vectorized loads, with a scalar fallback for head_dim not divisible by 4 (unused on current supported architectures — all have head_dim ∈ {64, 96, 128}).

### Changed

- **`kernel void attention` Q·K^T** — scalar `half` loads → `float4`/`half4` loads. Each simdgroup lane now does one vector load instead of up to four scalar loads. At head_dim=128 all 32 lanes load in parallel with no inner loop.
- **`kernel void attention` scores·V** — scalar `half` loads → `half4` + `float4` stores. Each thread now accumulates 4 d-dims per iteration. Loop structure over `s` (4-wide unroll) preserved.
- Output writeback via `*(device float4*)` instead of per-element stores.

### Performance (Apple M5 Pro, Q8_0, decode tok/s, median of 3)

| Workload | v0.7.2 | **v0.7.3** | Δ |
|----------|-------:|-----------:|--:|
| SmolLM2-135M, ~10-tok decode | 352 | 356 | noise |
| SmolLM2-135M, decode @ ~900-tok ctx | 174 | **202** | **+16%** |
| SmolLM2-135M, decode @ ~2250-tok ctx | 84 | **99** | **+18%** |
| Llama-3.2-1B, ~10-tok decode | 170 | 172 | noise |
| Llama-3.2-1B, decode @ ~2250-tok ctx | 87 | **96** | **+10%** |

Short-context decode is matmul-bound (projections dominate over attention), so the vectorization is invisible at small KV sizes — expected and correct. Gains scale with context length because decode attention is O(seq_len × head_dim × num_heads), which crosses into the bandwidth-bound regime once KV is large.

### Correctness

- Byte-identical continuation text on SmolLM-135M and Llama-3.2-1B vs v0.7.2 for the "meaning of life" prompt.
- All 71 Metal codegen tests pass, including new regression `decode_attention_uses_half4_vectorized_loads`.

## [0.7.2] — 2026-04-17 — Fused Decode-Path Dispatches (Metal)

Patch release: single-token decode now reuses the existing batched `rope_qk_batch` and `copy_kv_both_batch` helpers (with M=1) instead of issuing four separate dispatches per layer. Two fewer dispatches + barriers per layer per decode step.

### Changed
- **`dispatch_rope_offset` ×2 → `dispatch_rope_qk_batch(M=1)`** in the generated `forward()` and `forward_profile()` paths. Q and K rotate in one kernel launch instead of two.
- **`dispatch_copy_from_offset_f16` ×2 → `dispatch_copy_kv_both_batch(M=1)`** — K and V append to the f16 cache in one dispatch.
- Dropped the now-unused local `k_byte_offset` / `v_byte_offset` (decode path uses float offsets via the batched helper).

### Performance (Apple M5 Pro, Q8_0, decode tok/s)

| Workload | v0.7.1 | **v0.7.2** | Δ |
|----------|-------:|-----------:|--:|
| SmolLM2-135M, 128-tok decode | 405 | **503** | **+24%** |
| Llama-3.2-1B, short decode | 162 | **173** | +7% |
| Llama-3.2-1B, decode @ 2,502-tok ctx | 85.8 | **89.9** | +5% |
| Phi-3 Mini, decode @ 3,001-tok ctx | 30.8 | 30.8 | ~0% |

Small models see the biggest gain because dispatch overhead is a larger fraction of their per-layer work. Phi-3 Mini (32 layers × 32 MHA KV heads) is already dominated by in-kernel compute, so saving 2 dispatches/layer is lost in the noise.

### Correctness

- **Llama-3.2-1B** decode byte-identical to v0.7.1 on the "theory of relativity" test.
- **Qwen2.5-0.5B** coherent on the "history of AI" test.
- **Phi-3 Mini** coherent on the cat story test.

Output identity vs v0.7.1 where the path emitted the same fused kernel numerically (same `rope_qk_batch`, same `copy_kv_both_batch`), just invoked at M=1 instead of M>1.

## [0.7.1] — 2026-04-17 — FP16 KV Cache (Metal)

Patch release: KV cache storage changed from f32 (4 bytes/element) to f16 (2 bytes/element). Halves KV RAM footprint and KV read bandwidth during attention.

### Changed

- **KV cache storage: f32 → f16** across all four Metal attention kernels (`attention`, `attention_batch`, `attention_flash_batch`, `attention_mma_flash_batch`).  K/V are read as `device const half*` and upcast to `float` at the multiply-accumulate.  Batched prefill copy kernels (`copy_kv_batch`, `copy_kv_both_batch`) convert f32→f16 on write.
- **New `copy_f32_to_f16_offset` MSL kernel** + `dispatch_copy_from_offset_f16` helper for the single-token decode path.  The old f32-only `dispatch_copy_from_offset` is no longer used for KV writes.
- **KV buffer allocation halved**: `kv_cache_bytes = effective_seq_len * num_kv_heads * head_dim * 2` (was `* 4`).

### Performance (Apple M5 Pro, Q8_0)

| Workload | v0.7.0 (f32 KV) | v0.7.1 (f16 KV) | Δ |
|----------|----------------:|----------------:|--:|
| Llama-3.2-1B prefill @ 2502 tok | 1,739 | **1,840 tok/s** | +6% |
| Llama-3.2-1B decode @ 2502-tok ctx | 84.1 | **85.8 tok/s** | +2% |
| Phi-3 Mini prefill @ 3001 tok | 476 | **496 tok/s** | +4% |
| Phi-3 Mini decode @ 3001-tok ctx | 29.5 | **30.8 tok/s** | +4% |

Modest speedup at long context — the decode critical path is dominated by weight reads (~1.3 GB for 1B Q8_0 vs ~82 MB f16 KV at 2.5K tokens), so the KV bandwidth savings are ~5-10% of total memory traffic.  The headline win is **~50% KV RAM savings** enabling longer contexts and more concurrent requests.

### Correctness

- Llama-3.2-1B: output **byte-identical** to v0.7.0 on a 40-token decode test ("The theory of relativity was developed by..." → same continuation).
- Phi-3 Mini: outputs diverge at ~15 tokens due to f16 softmax ULP noise ("Mongols" vs "nomadic tribes") — both coherent and semantically correct.

### Migration

No user action needed. Generated Metal binaries automatically use f16 KV after rebuilding:

```bash
cargo install --force forgellm-cli   # 0.7.1
forge compile --model my-model.gguf --output ./my-model --target metal
# weights.bin unchanged; only KV cache storage changed in the binary
```

KV cache is runtime state, not stored on disk — no weights.bin migration needed.

## [0.7.0] — 2026-04-17 — MMA Flash Attention Default-On

Minor release: MMA-accelerated flash attention is now the default Metal attention kernel when `HEAD_DIM ≤ 128` and `num_tokens ≥ 8`. Empirically verified on Llama 3.2, Qwen2.5, and Phi-3 Mini. Set `FORGE_MMA_ATTN=0` as an opt-out escape hatch.

### Changed
- **`dispatch_attention_batch` default flipped** from legacy `attention_batch` to `attention_mma_flash_batch`:
  - Dispatch criterion: `!FORGE_MMA_ATTN=0 && HEAD_DIM <= 128 && num_tokens >= 8`.
  - v0.6.6 required `FORGE_MMA_ATTN=1` to enable.  v0.7.0 inverts the gate.
  - Regression test `attention_mma_flash_batch_kernel_wired` updated to assert the new default and the opt-out conditions.

### Verified (Apple M5 Pro, Q8_0, prefill tok/s)

| Model | Legacy (opt-out) | v0.7.0 default (MMA flash) | Δ |
|-------|-----------------:|--------------------------:|--:|
| Llama-3.2-1B @ 2,501 tok | 1,134 | **1,775** | +56% |
| Llama-3.2-1B @ 3,006 tok | 995 | **1,699** | +71% |
| Qwen2.5-0.5B @ 666 tok | 3,730 | **4,515** | +21% |
| Qwen2.5-0.5B @ 2,501 tok | 2,408 | **3,540** | +47% |
| Phi-3 Mini @ 1,201 tok | 465 | **607** | +30% |
| Phi-3 Mini @ 3,001 tok | 278 | **480** | +73% |

### Correctness

- **Llama 3.2 (GQA 4:1, head_dim=64)**: byte-identical to legacy on SmolLM2-135M @ 203 tok and Llama-3.2-1B @ 406/806/1510/3006 tok (v0.6.6 verification).
- **Qwen2.5 (GQA 7:1 + QKV bias, head_dim=64)**: coherent on both paths; ULP-level softmax divergence ~90 tokens into decode, top-5 token rankings identical at the prefill boundary.
- **Phi-3 Mini (MHA, head_dim=96)**: top-5 logits identical to legacy across all prompts tested; output text coherent and matches the HF reference behavior.

### Documentation correction (supersedes v0.6.7 CHANGELOG)

The v0.6.7 release notes listed Phi-3 Metal AOT as broken ("logits collapse to newline"). That was a misread of greedy-decode degeneration — the forward pass was always correct once the v0.6.7 weight split landed. Phi-3 Metal now tests green end-to-end and is the headline verification target for this release.

### Migration from v0.6.x

No code changes required. Rebuild and redeploy:

```bash
cargo install --force forgellm-cli   # 0.7.0
forge compile --model my-model.gguf --output ./my-model --target metal
# rebuild and run; MMA flash dispatches automatically on prompts ≥ 8 tokens
```

To pin to the v0.6.6 behavior while validating in your environment:

```bash
FORGE_MMA_ATTN=0 ./my-model weights.bin tokenizer.json "..."
```

### Deferred to v0.7.x / v0.8.0

- Gemma (head_dim=256, exercises the legacy fallback path).
- StableLM / Mistral Metal AOT coverage.
- Blog post covering the MMA flash kernel design.

## [0.6.7] — 2026-04-17 — Phi-3 Weight Splitting + Qwen2.5 MMA Flash Verified

Weight-loader support for Phi-3 GGUFs (fused `attn_qkv` / `ffn_up` tensors) so the IR graph, interpreter, and AOT export paths that all expect split Llama-style names can consume them unchanged. Also empirically verified the v0.6.6 MMA flash kernel on Qwen2.5-0.5B.

### Added
- **`split_fused_tensors` / `split_fused_tensors_f32` helpers** in `forgellm-frontend::weight_loader`:
  - Split a fused `model.layers.N.self_attn.qkv_proj.weight` (Q|K|V concatenated along the output dim) into the three HF-named tensors.
  - When `mlp.gate_proj.weight` is absent and `mlp.up_proj.weight` has twice the expected size, split it into `gate_proj` and `up_proj` halves.
  - Q8_0 / Q4_0 splits operate on raw bytes on block boundaries (row counts are always multiples of 32 for current model shapes).
- **Auto-detect hook** in `load_from_file`, `load_from_file_mixed`, and `load_all`: reads GGUF metadata keys (`{arch}.block_count`, `{arch}.attention.head_count{,_kv}`, `{arch}.feed_forward_length`, `{arch}.embedding_length`) and applies the split when a fused tensor is present. No-op for Llama / Qwen2 / Gemma layouts.
- **GGUF→HF name remap** for `attn_qkv.weight` → `self_attn.qkv_proj.weight` (placeholder that the split helper replaces with `q_proj` / `k_proj` / `v_proj`).
- Three new unit tests covering F32 splits, Q8_0 byte-boundary splits, and the no-op path on Llama layouts.

### Fixed
- **Phi-3 interpreter (`forge run`)** now loads coherently. Previously panicked with `weight not found: model.layers.0.self_attn.q_proj.weight` because `blk.N.attn_qkv.weight` fell through the name remap unchanged.

### Verified (Qwen2.5-0.5B, Apple M5 Pro, Q8_0, MMA flash via `FORGE_MMA_ATTN=1`)

| Prompt | Legacy | MMA flash | Δ |
|-------:|-------:|----------:|--:|
| 666 tok | 1,990 | **2,377 tok/s** | +19% |
| 666 tok (long `FOX`) | 3,730 | **4,515 tok/s** | +21% |
| 2,501 tok | 2,408 | **3,540 tok/s** | **+47%** |

Output is coherent on both paths (the `#210` QKV-bias export path works correctly under MMA flash). Not byte-identical to legacy — f16 ULP-level differences in the online-softmax recurrence cause divergence ~90 tokens into decode, as expected.

### Known issues
- **Phi-3 Metal AOT forward pass still produces bad logits** after the weight split — logits collapse to the newline token regardless of prompt. The weight split itself is correct (interpreter output is sane); something else in the Metal codegen path is Phi-3-specific and needs dedicated debugging. Tracked as follow-up work. Phi-3 on the interpreter path works in this release.

### Deferred to v0.7.0
- Phi-3 Metal forward-pass debugging, then empirical MMA flash verification on Phi-3 (head_dim=96).
- Gemma / StableLM coverage.
- Flipping the default from legacy attention → MMA flash.
- README prefill table + blog refresh.

## [0.6.6] — 2026-04-16 — MMA-Accelerated Flash Attention (opt-in)

New Metal attention kernel using hardware `simdgroup_matrix<half, 8, 8>` MMA for both Q·K^T and P·V, with online softmax and causal masking. Gated behind `FORGE_MMA_ATTN=1` for this release while broader model coverage is verified; default dispatch is unchanged.

### Added
- **`attention_mma_flash_batch` MSL kernel** (`feat`, opt-in, closes [#212](https://github.com/sauravpanda/forge-llm/issues/212)):
  - Grid `[ceil(M/8), num_heads, 1]`, 128 threads (4 simdgroups) per TG.
  - Each simdgroup owns one 8×8 S tile and `head_dim/4` columns of O accumulators.
  - Threadgroup memory ~13 KB (`head_dim=64`) / ~24 KB (`head_dim=128`).
  - K loaded with `transpose=true` to get `K^T` view — feeds MMA natively with no pre-transpose pass.
  - Online softmax recurrence runs **per-Q-row** across `K_BLOCK=32` chunks (`m`, `l`, `O` updated per Q row, not globally).
  - Requires `HEAD_DIM ≤ 128` and `num_tokens ≥ 8`; falls back to legacy `attention_batch` otherwise.
- **`FORGE_MMA_ATTN=1` env gate**: set this to route long-prompt attention through the MMA kernel. Unset (default), behavior is identical to v0.6.5.

### Performance (opt-in only)

**Llama-3.2-1B Q8_0 prefill, Apple M5 Pro:**

| Prompt | v0.6.5 | v0.6.6 MMA flash | Δ |
|-------:|-------:|----------------:|--:|
| 406 tok | 1,949 | **2,102 tok/s** | +8% |
| 806 tok | 1,950 | **2,282 tok/s** | +17% |
| 1,510 tok | 1,531 | **2,081 tok/s** | **+36%** |
| 3,006 tok | 995 | **1,699 tok/s** | **+71%** |

Gains grow with prompt length because attention cost is `O(M²)` and the MMA path replaces the scalar per-token reduction in the legacy kernel. Short-prompt numbers are within noise because matmul dominates there and the MMA flash kernel shares the same `matmul_q8_mma32*` path for Q/K/V/O projections.

### Correctness

Output is **byte-identical** vs the legacy `attention_batch` kernel on:

- SmolLM2-135M Q8_0 @ 203-token prompt.
- Llama-3.2-1B Q8_0 @ 406 / 806 / 1,510 / 3,006 token prompts.

### Deferred to v0.7.0

- Verification on Qwen2.5 (QKV-bias path), Phi-3 Mini (`head_dim=96`), and Gemma / StableLM.
- Default-flip from legacy → MMA flash.
- README + blog updates with the new numbers (benchmarks recorded but not yet published in the README).

Regression test: `attention_mma_flash_batch_kernel_wired`.

## [0.6.5] — 2026-04-16 — Docs Polish: Flash Verified, Honest TFLOPS

Docs-only release correcting README claims that were stale after v0.6.4. No code changes to the kernels or dispatch; crates.io package metadata and README are updated.

### Fixed
- **Flash attention status**: v0.6.4's README said the `attention_flash_batch` kernel was "shipped but not yet numerically verified." Post-v0.6.4 verification confirmed the kernel is correct (the earlier garbled output was the `scores[2048]` overflow in the *legacy* kernel, not flash). README now states flash is verified and is ~7–14% slower than legacy until MMA support lands; the legacy kernel remains the dispatch default. See [#212](https://github.com/sauravpanda/forge-llm/issues/212) for MMA-accelerated flash.
- **Inflated TFLOPS claim**: removed "~8.7 TFLOPS sustained on 1B" from the Metal GPU features list. That number originated from the pre-chunking-fix era; the honest v0.6.4 number is ~3.9 TFLOPS at 1,501 tokens (1,580 tok/s × 2·1.24B params).
- **SmolLM2-360M prefill numbers** added to the README prefill table: ~3,900 tok/s at 321 tokens and ~2,350 tok/s at 1,500 tokens.

### Chore
- `cargo fmt --all` applied across `forgellm-cli`, `forgellm-codegen-cpu`, and `forgellm-codegen-metal`. No functional change; `cargo fmt --all -- --check` is now clean.

## [0.6.4] — 2026-04-15 — Long-Prompt Correctness + Honest Numbers

### Fixed
- **Long-prompt truncation in `forward_prefill_batch`** (`fix`, major correctness bug): the batched prefill was silently truncating prompts longer than `MAX_BATCH_SIZE` (512) via `.min(MAX_BATCH_SIZE)` — it processed only the first 512 tokens, filled KV cache for those positions, and left everything past 512 as zero/garbage in the cache. The `forward()` call for the final prompt token then attended over a partially-filled cache, producing subtly-wrong logits without crashing.
  - Real impact: any prompt > 512 tokens produced wrong output. In particular, "needle-in-haystack" queries (fact buried mid-prompt) fail to retrieve. For repetitive prompts the output often looks correct (continues the pattern) because the first 512 tokens are representative.
  - Fix: `forward_prefill_batch` now loops over `tokens.chunks(MAX_BATCH_SIZE)`, running the full attention + matmul pipeline for each chunk and carrying KV-cache state across chunks via `self.pos`.
  - **Previous prefill numbers for prompts > 512 tokens were inflated**: reporting `prompt_len / total_time` where only 512 tokens actually went through the forward pass. Honest numbers on Llama-3.2-1B Q8_0 (M5 Pro): 801-tok ~2,030 tok/s (not 3,390), 1,501-tok ~1,580 tok/s (not 6,320). README table updated. Numbers ≤ 321 tokens were unaffected.
  - Regression test: `forward_prefill_batch_chunks_by_max_batch_size`.
- **`scores[2048]` attention overflow** (`fix`): both the single-token `attention` kernel and the batched `attention_batch` kernel had a hardcoded `threadgroup float scores[2048]` array. For models with `effective_seq_len` up to 4096 (all supported models), a prompt or generation position past 2048 silently overflowed the shared array, corrupting attention output. The scores array is now sized to `ATTN_SCORES_SIZE = min(max_seq_len, 4096)` — templated into the shader source at codegen time.
- **Flash attention kernel temporarily disabled** after the chunking fix exposed a numerical issue (the pre-chunking dispatch never actually exercised flash at long seq because long prompts were being truncated). The kernel source + pipeline are still shipped; the dispatch routes to the (now correctly sized) legacy kernel pending a fix.

## [0.6.3] — 2026-04-15 — Long-Context Attention Fix

### Added
- **`attention_flash_batch` kernel** (`feat`, fixes a long-context correctness bug): a tiled attention kernel with online softmax that streams K/V in `FLASH_K_TILE`-sized chunks, eliminating the `scores[2048]` threadgroup array used by `attention_batch`. The legacy kernel **silently corrupts** attention output when `base_pos + num_tokens > 2048` because the scores array overflows — a real bug for any prompt longer than 2K tokens (the compile-time max_seq_len is 4K for all supported models). Dispatch is tiered: legacy kernel for `max_seq <= 2000` (faster at short seq due to fewer barriers), flash kernel above. Verified coherent output on Llama-3.2-1B at 3,001 prompt tokens (~12K tok/s prefill). Short/medium prefill numbers are roughly unchanged (within thermal variance). Regression test: `attention_flash_batch_kernel_wired`.

## [0.6.2] — 2026-04-15 — Qwen2 AOT Fix

### Fixed
- **Qwen2 QKV bias on AOT backends** (`fix`, closes [#210](https://github.com/sauravpanda/forge-llm/issues/210)): Qwen2 models have bias terms on the Q/K/V projection matrices that Llama does not. Before this release, `cmd_export_weights_impl` never wrote those biases, the Metal codegen had zero handling for them, and the CPU codegen emitted bias fields without populating them — so a Qwen2 AOT binary either failed to compile (CPU) or produced garbled output (Metal).
  - `forgellm-cli` export now writes the Q/K/V bias triplet immediately after the fused Q/K/V weight triplet when `config.qkv_bias` is true.
  - `forgellm-codegen-cpu` `project.rs` weight-loader now reads those biases and populates `LayerWeights`. `forward_prefill` also applies them (previously only `forward` did).
  - `forgellm-codegen-metal` adds a `qkv_bias` buffer to `LayerBuffers`, a new `add_bias_batch` MSL kernel, a `dispatch_add_bias_batch` helper, and calls it after the fused QKV matmul in both `forward` and `forward_prefill_batch`.
  - Verified end-to-end on Qwen2.5-0.5B Q8_0: CPU and Metal AOT both produce coherent output. Llama-3.2-1B Metal prefill is unchanged (6,300 tok/s at 1,501 tokens — no regression).
  - New regression tests: `qwen2_qkv_bias_wired_through_metal_codegen` and `llama_does_not_emit_qkv_bias_machinery` in Metal codegen; the CPU-codegen test for Qwen2 now asserts the bias-add is present in `forward_prefill` as well as `forward`.

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
