# Changelog

All notable changes to ForgeLLM are documented here.

## [0.9.8] — 2026-04-29 — AOT binary CLI parser fix

Surfaced while validating the v0.9.7 fix end-to-end: AOT-compiled binaries
silently appended flag *values* to the prompt.  E.g.
`aot-bin weights.bin tok.json "hi" --temp 0 --max-tokens 30` was processing
the prompt `"hi 0 30"` instead of `"hi"`, because the prompt extractor was
`filter(!starts_with("--"))` over all of `args[3..]` — which retains every
flag value (since values don't start with `--`).  Greedy output then looked
nonsensically "broken" because the model was given a prompt the user never
wrote.

### Fixed

- `crates/forgellm-codegen-cpu/src/project.rs` — both AOT main.rs templates
  (regular and `--embed-weights`) now walk args sequentially with a
  flag-arity table, skipping `(flag, value)` pairs and stand-alone bool
  switches before collecting the prompt.  Verified: `--temp 0
  --repeat-penalty 1.0 --max-tokens 50` no longer leaks `0`, `1.0`, `50`
  into the prompt; AOT-Q8_0 now produces sensible greedy continuations.

### Validation snapshot

Same prompt under interpreter and AOT-Q8_0 (Llama SmolLM2-135M-Q8_0,
greedy, no penalty) agree on the first 4 generated tokens then diverge —
consistent with FP32 numerical noise across the two implementations of the
forward pass (interpreter is token-by-token; AOT uses batched prefill).
Full perplexity-scoring inside the AOT binary is the next step (proposed
v0.9.9) so this can be measured rigorously instead of inferred from
greedy-token agreement.

## [0.9.7] — 2026-04-29 — `f32_to_f16` round-to-nearest-even (fixes Q4_K simulate path)

Fixes the v0.9.6 known-limitation: `--simulate-q4k` no longer produces
catastrophic perplexity.  The bug was a single line in `f32_to_f16` —
`mant >> 13` truncates instead of rounding to nearest even, biasing every
f16 stored scale toward zero.  Per-element error is small (~0.05% relative)
but compounds through stacked matmuls (V·O in attention) into a systematic
~0.03 mean shift on dequant output, which is enough to dominate the signal.

### Result

- `--simulate-q4k qkx2` on Llama-3.2-1B-Q8_0 baseline:
  **ppl 423,338 → 4.57** (~92,000× improvement).  4.57 vs Q8_0 baseline
  4.23 is +8%, the expected Q4_K-without-imatrix gap; the v0.9.5
  `make_qkx2_quants` port is now validated end-to-end.
- `--simulate-q4k naive`: ppl ≈ 4.55 — within statistical noise of qkx2 on
  a 4.9 KB corpus.  Confirms naive (min, max) affine is competitive when
  the underlying f16 scale storage is unbiased; the v0.9.5 quality grind
  was nearly entirely amortized by this f16 bug masking real differences.

### Fixed

- **`crates/forgellm-frontend/src/weight_loader.rs::f32_to_f16`** — now
  does proper round-to-nearest-even with subnormal handling, mantissa
  carry, NaN preservation, and overflow-to-infinity.  Affects every Q4_0
  / Q4_K / Q8_0 quantization scale stored to f16 from the frontend.
- **`crates/forgellm-codegen-cpu/src/emit.rs`** — both copies of the
  emitted `f32_to_f16_bits` helper (Q4_0 and Q8_0 sdot-kernel fallback
  paths) updated to round-to-nearest-even.  Every AOT-compiled binary's
  live activation quantization now uses unbiased scales too.
- **`quantize_f32_to_q4_k`** — cleaned up super-block packing to match
  ggml's reference verbatim: the_min / dmin are stored non-negative
  end-to-end (no sign-juggling), L is re-quantized after 6-bit pack
  using the f16-rounded effective scales.  Both pieces matter; the f16
  fix was the dominant bug, the L refit and clean packing tighten the
  remainder.

### Added

- `quantize_f32_to_q4_k_mean_bias_unbiased` /
  `quantize_f32_to_q4_k_mean_bias_outlier_heavy` unit tests — assert the
  per-element MEAN of `dequant - original` is within 5σ of chance for
  Gaussian and outlier-heavy distributions.  v0.9.6's tests only
  bounded max-err (per-element) and rms (total spread); both passed
  with the f16 truncation bug present.  Mean-bias is what compounds in
  matmuls and is now a regression gate.
- `FORGE_PPL_DUMP_ERRORS=1` env var on `forge bench-perplexity
  --simulate-q4k` — prints per-tensor `mean_x`, `mean_err`, `rms`,
  `max_err`, `max|x|` so future quantizer regressions surface
  immediately.

### Files

- `crates/forgellm-frontend/src/weight_loader.rs` — `f32_to_f16` rewrite,
  Q4_K super-block packing cleanup, two new mean-bias tests.
- `crates/forgellm-codegen-cpu/src/emit.rs` — emitted
  `f32_to_f16_bits` rewrite (×2 sites).
- `crates/forgellm-cli/src/main.rs` — `FORGE_PPL_DUMP_ERRORS` flag,
  removed v0.9.6 catastrophic-perplexity warning text.
- `Cargo.toml` — version 0.9.6 → 0.9.7.

## [0.9.6] — 2026-04-26 — `forge bench-perplexity` infrastructure

Adds an objective quality benchmark for quantization-vs-baseline
comparisons.  Single-prompt eval on a 1B Q4_K_M model is too noisy to
A/B quantizer changes; perplexity-on-corpus is the standard alternative.

### Added

- **`forge bench-perplexity --model X.gguf --tokenizer T.json --corpus
  corpus.txt`** — reads a UTF-8 text corpus, tokenises it, and runs the
  CPU interpreter token-by-token through `chunk_size`-token sliding
  windows.  At each position `i` it computes a numerically-stable
  log-softmax over the logits and accumulates `-log P(tokens[i+1])`.
  Reports total perplexity, bits-per-byte, and tokens/second.
- Flags: `--chunk-size 512` (KV cache resets per chunk),
  `--max-chunks N` for quick A/B runs, `--simulate-q4k {none,naive,qkx2}`
  for round-tripping projection weights through a chosen Q4_K quantizer
  before scoring.
- `pub fn quantize_f32_to_q4_k_naive` — restores the v0.9.4 simple
  `(min, max)`-affine quantizer alongside the v0.9.5 `make_qkx2_quants`
  port, so the two can be compared head-to-head.

### Validated

- Llama-3.2-1B-Instruct-Q8_0 on a 4.9 KB factual-prose corpus:
  **perplexity 4.23, BPB 0.40, 13 tok/s** (interpreter, single thread
  for fairness).  Sane numbers for a small instruct-tuned model on
  coherent prose.
- Same corpus on Llama-3.2-1B-Instruct-Q4_K_M (using the GGUF's
  ggml-stored Q4_K bytes, dequantised by ForgeLLM's reader): **ppl
  4.35** — +2.8% over Q8_0, the expected modest gap.

### Known limitation

- **`--simulate-q4k` produces catastrophic perplexity (~10⁵ vs ~4
  baseline)** for both `naive` and `qkx2` quantizers when applied to
  more than a single tensor.  Per-tensor round-trip max-err looks
  fine in unit tests (~0.03 abs on tensors with 0.1 amplitude); the
  bug is somewhere in the multi-tensor interaction through attention's
  V·O matmul, which amplifies a systematic encoding bias my
  super-block d/dmin packing introduces but ggml's reference quantizer
  apparently doesn't.  Layer-bisection: a single layer's worth of
  attention projections is enough to push ppl to ~50k, so it's a
  per-quantizer-output bias rather than compounding noise across
  layers.  This means **v0.9.5's `make_qkx2_quants` port is still
  unvalidated by perplexity**; the simulate path can't yet do the A/B
  it was designed for.  Tracked as the next session's grind:
  super-block d/dmin search (matching ggml's reference) so the
  per-sub-block (scale, min) computed by `make_qkx2` survives the
  6-bit super-block packing.

### Files

- `crates/forgellm-cli/src/main.rs` — `cmd_bench_perplexity`,
  `log_softmax_at`.
- `crates/forgellm-frontend/src/weight_loader.rs` — restored
  `quantize_f32_to_q4_k_naive` for A/B simulation.

## [0.9.5] — 2026-04-26 — `make_qkx2_quants` port for Q4_K

`quantize_f32_to_q4_k` now uses the same per-sub-block algorithm as
ggml's reference Q4_K quantizer (`make_qkx2_quants`).  This replaces the
naive `(min, max)`-affine sub-block fit shipped in v0.9.2.  Output
quality on Q4_K_M models tightens visibly on prompts that exercise
weight outliers — the 1B-Instruct test model now answers `1+1 = 2`
correctly and stays on-topic for "capital of Japan" instead of
returning empty completions, both of which the naive quantizer
botched on the same weights.

### Algorithm

Per 32-element sub-block:

1. **Initial bracket**: `min = x.min()`, `max = x.max()`, with the ggml
   constraint `if min > 0 { min = 0 }` so the affine encoding's
   unsigned `sub_min` covers the dequant offset cleanly.
2. **Initial L** via `round((x - min) / scale)` clamped to `[0, 15]`.
   Track this as the best-so-far with its squared-error MAD.
3. **Grid search**: for `is = 0..=20`, try `iscale = (rmin + rdelta*is +
   nmax) / (max - min)` (rmin = -1.0, rdelta = 0.1).  This explores
   slightly-shrunken brackets that exclude outliers from forcing the
   bulk into a single L level.
4. For each candidate iscale: re-quantize to `Laux`, then refit
   `(scale, min)` via 2-parameter closed-form least-squares
   (`min = (Σ L² Σ x − Σ L Σ Lx) / det`,
    `scale = (n Σ Lx − Σ L Σ x) / det`,
    with `min` clamped ≤ 0 to keep affine encoding valid).
5. Compute SSE for the candidate; keep only if it beats the current
   best — so the algorithm **never regresses against the naive
   bracket** when no shifted candidate is better.

### Tests

- The simple round-trip tests (smooth ramp, zero block, gaussian-ish
  ±0.1 distribution) all pass with tighter RMS bounds than the
  naive quantizer.
- All 11 Q4_K kernel + quantizer tests still green.

### Performance

Decode / prefill tok/s unchanged from v0.9.4 (Q4_K_M Llama-3.2-1B at
~52 / 56 tok/s).  weights.bin layout / size unchanged — this is a
pure-quality release, encoder-side only.

### Honest caveats

- Single-prompt outputs aren't a reliable quality benchmark on a
  heavily-quantized 1B model — both naive and make_qkx2 produce noisy
  generations in some cases.  The right benchmark is
  perplexity-on-corpus, which is tracked as future work.
- ggml's full pipeline also uses an iterative
  super-block d/dmin search (after make_qkx2_quants per-sub-block);
  this release ports the per-sub-block algorithm only.  The
  super-block d/dmin pick is still the simple
  `max_scale/63` / `max_min/63` from v0.9.2.

## [0.9.4] — 2026-04-25 — Q4_K dot4 ILP: now faster than Q8_0

Q4_K decode is now **faster than the Q8_0 NEON path** at 26% smaller
binary.  The win comes from amortising the Q8_0 input load across four
weight rows: a 32-byte input sub-block is fetched once and dotted against
four Q4_K weight rows in parallel via `dot4_q4_k_q8_0`.  The
`Σ x_q8` reduction (needed for the affine `dmin` correction) is also
shared across the four rows.

### Performance

Llama-3.2-1B-Instruct-Q4_K_M, M5 Pro CPU, prompt = "The capital of France is":

| Variant                  | Prefill tok/s | Decode tok/s | weights.bin |
|--------------------------|--------------:|-------------:|------------:|
| v0.9.1 (Q8_0 requant)    |          47.5 |         47.7 |     2364 MB |
| v0.9.2 (Q4_K scalar)     |          29.7 |         26.6 |     1746 MB |
| v0.9.3 (Q4_K NEON dot1)  |          46.6 |         41.9 |     1746 MB |
| **v0.9.4 (Q4_K dot4)**   |      **56.8** |     **52.3** | **1746 MB** |

Compared to v0.9.3: decode +25%, prefill +22%.  Compared to v0.9.1 Q8_0:
decode +10%, prefill +20%, weights.bin -26%.

### Added

- **`dot4_q4_k_q8_0` (NEON)** — 4-row variant of the Q4_K × Q8_0 dot
  product.  Per Q4_K super-block:
  - Load Q8_0 input for both sub-blocks once (32 bytes × 2 sub-blocks).
  - Compute `Σ x_q8` once per sub-block.
  - For each of 4 weight rows: load 32-byte qs slice, mask/shift to two
    int8x16 nibble vectors, run two `sdot` instructions per sub-block.
  - Apply per-row `(d * sub_scale, dmin * sub_min)` FMAs.

  Inline asm interleaves the four sdots so the issue-width is well
  saturated; in microbench the kernel is 1.5–1.6× faster per output
  element than four separate `dot_q4_k_q8_0` calls.
- **`emit_specialized_q4_k_matmul_functions`** now emits a 4-row cascade
  on aarch64 (`dot4` for groups of 4, `dot1` tail) inside both the
  parallel and serial branches.  Non-aarch64 stays on the scalar path.
- The Q4_K **lm_head dispatch** in both `forward()` and
  `forward_prefill()` uses the same cascade (vocab is large, so dot4 is
  a substantial chunk of the logits projection).
- Codegen test
  `dot4_q4_k_q8_0_matches_four_dot1_calls` — verifies the 4-row kernel
  matches four scalar `dot_q4_k_q8_0` calls within fp rounding (rel err
  < 1e-4) on a synthetic super-block batch.

### Not changed

- f32 → Q4_K quantizer is still simple `(min, max)`-affine — output
  noisier than Q8_0 path on long generations.  Iterative
  `make_qkx2_quants` port is the next quality step.
- `forward_prefill_batched` for Q4_K still pending.  Long-context prefill
  uses the per-token path; for this `m=1` benchmark the cascade above
  *is* the prefill path.

## [0.9.3] — 2026-04-25 — NEON Q4_K kernel: 1.58× decode speedup

The Q4_K decode regression flagged in v0.9.2 is mostly closed.  The
emitted `dot_q4_k_q8_0` now uses inline-asm `sdot` for the int8 × 4-bit
inner reduction and `vpaddlq_s8` + `vaddvq_s16` for the per-sub-block
`Σ x_q8` correction needed by the affine `dmin` term.

### Performance

Llama-3.2-1B-Instruct-Q4_K_M, M5 Pro CPU, prompt = "The capital of France is":

| Variant                  | Decode tok/s | weights.bin |
|--------------------------|-------------:|------------:|
| v0.9.1 (Q8_0 requant)    |         47.7 |      2364 MB |
| v0.9.2 (Q4_K scalar)     |         26.6 |      1746 MB |
| **v0.9.3 (Q4_K + NEON)** |     **41.9** | **1746 MB** |

NEON kernel is **1.58× the scalar Q4_K kernel** and reaches 88% of the Q8_0
NEON sdot path's tok/s — at 26% smaller weights.bin.  Prefill: 46.6 tok/s
on the same prompt (was 29.7 in v0.9.2).

### Added

- **`dot_q4_k_q8_0` (NEON)** — inline `sdot` (`v.4s, v.16b, v.16b`) for the
  per-sub-block dot, plus `vpaddlq_s8 + vaddvq_s16` for the `Σ x_q8`
  reduction.  Two `sdot` invocations per 32-element sub-block (one for
  the 16-byte low half, one for the high half), both halves of one chunk
  (low + high nibbles of the same 32 packed-byte qs slice) processed
  back-to-back.
- Scalar fallback retained as a `cfg(not(target_arch = "aarch64"))`
  branch — the previous reference kernel, still valid for x86 / WASM /
  generic builds.
- **3 NEON correctness tests** in `weight_loader.rs`:
  - `dot_q4_k_q8_0_neon_matches_scalar_uniform` — uniform synthetic
    super-block; NEON ≡ scalar within fp rounding.
  - `dot_q4_k_q8_0_neon_matches_scalar_varied` — varied scales/mins +
    cosine input; rel error < 1e-4.
  - `dot_q4_k_q8_0_neon_multi_superblock` — K=512, two super-blocks
    end-to-end, exercises the per-super-block pointer arithmetic.

### Not changed

- f32 → Q4_K quantizer is still the simple per-sub-block (min, max)
  affine — output noisier than the Q8_0 path.  Iterative quantizer
  (ggml's `make_qkx2_quants`) is the next quality step.
- Q4_K still uses per-token prefill (no `forward_prefill_batched` for
  Q4_K yet) — fine at short contexts, slower than Q4_0/Q8_0 at long
  ones.

## [0.9.2] — 2026-04-25 — Native Q4_K storage + scalar kernel

End-to-end native Q4_K compile path: `Q4_K_M` GGUFs now compile to a binary
where projection weights stay as Q4_K super-blocks (144 bytes per 256
elements) instead of being requantized to Q8_0.  Llama-3.2-1B-Q4_K_M
weights.bin drops **2364 MB → 1746 MB (-26%)** on disk and in memory.

### Added

- **`DType::Q4_K`** in the IR + **`WeightData::Q4_KRaw(Vec<u8>)`** + a
  `get_q4k_raw` accessor.  Q4_K storage is now a first-class dtype.
- **Scalar `dot_q4_k_q8_0` reference kernel** (`emit_q4_k_kernel`) — emits
  `get_scale_min_k4` + the per-super-block dequant-and-dot loop into the
  generated `model.rs`.  Operates on Q4_K weight × Q8_0-quantized input.
- **`emit_specialized_q4_k_matmul_functions`** — shape-specialized
  `matmul_vec_q4_k_KxN` per projection shape, parallelised over output
  rows via rayon when the weight tensor exceeds 1 MB.
- **`load_from_file_mixed_with_target(path, Some(DType::Q4_K))`** — loader
  routing that keeps Q4_K tensors raw and re-quantizes other K-quants
  (Q5_K / Q6_K / Q5_0 / etc.) to Q4_K via `quantize_f32_to_q4_k`.  For
  Q4_K_M files the Q6_K `ffn_down` / `attn_v` per-layer tensors land on
  the Q4_K kernel path alongside the natively-Q4_K projections.
- **`quantize_f32_to_q4_k`** in `forgellm_frontend::weight_loader` — simple
  affine quantizer (per-32-element sub-block (min, max) → 6-bit scale/min;
  super-block d / dmin = absmax/63, signed).  Round-trip max error ≤ 0.05
  on smooth ramps.  Coarser than ggml's iterative `make_qkx2_quants`.
- **`detect_gguf_dtype`** now recognizes a Q4_K-majority projection set
  and returns `DType::Q4_K`; lm_head still reports `DType::Q8_0` for the
  Q6_K output tensor via `detect_gguf_lm_head_dtype` (per-tensor split
  shipped in v0.9.1 carries the load).
- **Codegen dispatch**: `is_q4k` branches added to `forward()` and
  `forward_prefill()` for q/k/v projections, o_proj, gate/up, down, and
  lm_head.  `generate_main` / `generate_main_embedded` compute Q4_K row
  byte sizes (144 bytes per 256 elements) for the binary loader.
- **Unit tests** (`weight_loader`):
  - `dot_q4_k_q8_0_matches_f32_reference_uniform_block` and
    `_varied_block` — verify the kernel matches the f32-dequant reference
    within 1% on synthetic super-blocks.
  - `quantize_f32_to_q4_k_round_trip_smooth` and `_zero_block` — round-trip
    bounds on the new quantizer.

### Validated

- Compile path: `forge compile --target cpu` on
  `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (mixed Q4_K + Q6_K) succeeds and
  emits Q4_K kernels.
- Run path: generated binary loads, runs, and produces coherent output:
  `> Q: What is 2+2? A: 4 B: The answer to ...` (correct first token).
- Binary size: 1746 MB vs 2364 MB on v0.9.1 (-26%).

### Known limitations (tracked for v0.9.3+)

- **Decode tok/s regresses against the Q8_0 path** on AArch64.  v0.9.1
  Q8_0 hits 47.7 tok/s on Llama-3.2-1B-Q4_K_M decode; v0.9.2 Q4_K hits
  ~26.6 tok/s.  The bandwidth saving from reading 144 bytes/superblock
  instead of 272 is real, but the scalar Rust loop is no match for the
  NEON `sdot`-based Q4_0/Q8_0 kernels.  A NEON Q4_K implementation
  (vand/ushr nibble unpack + sdot, per-sub-block scale FMA) is the next
  step.  Set `forge compile` against a Q8_0 GGUF for now if decode
  throughput matters more than binary size.
- **Output quality is slightly noisier** than the Q8_0 path on the same
  GGUF — `quantize_f32_to_q4_k` uses naive per-sub-block (min, max)
  rounding rather than ggml's iterative `make_qkx2_quants`.  Coherent on
  short prompts; long generations hallucinate more.  Replacing the
  quantizer with the iterative variant is a separate follow-up.
- **No batched prefill** for Q4_K yet (`forward_prefill_batched` only
  emitted for Q8_0 / Q4_0).  Q4_K models use the per-token prefill path,
  which is fine at short contexts but loses to the v0.8.x batched
  matmul wins at long ones.

### Not changed

- Q8_0-target compiles (the default and recommended path for production
  decode throughput) are byte-for-byte identical to v0.9.1.
- Q4_0-target compiles unchanged.

## [0.9.1] — 2026-04-24 — Infra: per-tensor lm_head dtype dispatch

Infrastructure release.  No user-visible perf or correctness change on any
currently-supported model; existing Q4_K_M GGUFs compile and run identically
to v0.9.0 (all K-quants → Q8_0 uniform codegen path, ~44–47 tok/s decode on
Llama-3.2-1B Q4_K_M).  The machinery added here is the prerequisite for
native Q4_K / Q6_K decode kernels in a follow-up release.

### Added

- **`ModelConfig::lm_head_dtype: Option<DType>`** — explicit per-tensor storage
  dtype for the output / logits projection, defaulting to `config.dtype` when
  `None`.  Serde-defaulted so existing serialized configs keep parsing.
- **`detect_gguf_lm_head_dtype(gguf, proj_dtype)`** in the CLI — inspects the
  `output.weight` / `lm_head.weight` tensor (if present) and returns a split
  dtype when it differs from projections.  For tied-embedding models (no
  `output.weight` tensor) returns `None`, keeping lm_head in sync with
  projections.

### Changed

- **`emit.rs` codegen** now derives a `lm_dtype = config.lm_head_dtype.unwrap_or(config.dtype)`
  and dispatches the lm_head matmul independently of the projection dispatch.
  All three logits-projection sites (`forward()`, `forward_prefill()`,
  `forward_prefill_batched()`) honour this split.  `lm_head_type` in the
  generated `struct Weights` and `lmh_bytes` in `generate_main` likewise use
  `lm_dtype`, so the weight file layout matches the emitted kernel.
- **`generate()` in `emit.rs`** now emits each quantized-kernel family
  (Q4_0 / Q8_0) whenever *either* projections or lm_head need it.
  `emit_q4_0_sdot_kernel` gained a `skip_shared_helpers` flag so the
  `quantize_to_q8_0_blocks` / `f32_to_f16_bits` helpers are emitted once even
  when both kernel families are in play.

### Explicitly not changed

- **Weight loader** (`load_from_file_mixed`): all K-quants still re-quantize
  to Q8_0.  Q4_K_M files (which commonly mix Q4_K projections with Q6_K
  `ffn_down` / `output.weight`) remain on a single uniform codegen path, so
  kernels never encounter mixed byte layouts within a layer.
- **Weights binary size / decode tok/s**: identical to v0.9.0 for every
  model-dtype combination supported today.

### Rationale

An earlier iteration of this release routed `GGMLType::Q4K → DType::Q4_0`
directly in the loader, expecting ~45% smaller weights and a 1.5× decode
speedup.  That path is **not** safe with current codegen, because Q4_K_M
files mix Q4_K with Q6_K at the tensor granularity — e.g. Llama-3.2-1B
Q4_K_M stores `ffn_down` as Q6_K while `ffn_gate` / `ffn_up` are Q4_K —
and the uniform-dtype per-layer codegen would emit Q4_0 kernels for the
Q6_K bytes (which live as 34-byte Q8_0 blocks after requant), producing
NaNs at the first matmul.  Making that work correctly requires per-tensor
dispatch across *all* projection kernels, which is the v0.9.2+ roadmap.
v0.9.1 ships the lm_head half of that refactor (single-site, well-scoped),
leaving the per-projection-kind dispatch for when the native Q4_K / Q6_K
kernels are ready to slot in.

### Tests

- `ggml_type_conversions` updated: Q4K stays at DType::Q8_0 (matches loader).
- `detect_dtype_k_quants_classified_as_q8_0` renamed + updated.
- All 63 CPU codegen tests pass.  End-to-end Llama-3.2-1B-Q4_K_M compile
  verified: coherent generation at 47.7 tok/s decode (unchanged from v0.9.0).

## [0.9.0] — 2026-04-24 — K-quant compatibility (Q4_K_M and friends)

### Added

- **Q4_K_M / Q5_K_M / Q6_K / Q3_K / Q2_K / Q8_K GGUF models now load and compile**.  Previously the GGUF parser recognized these types but the raw weight loader silently produced bytes the codegen Q4_0 / Q8_0 kernels could not consume — so models like `Llama-3.2-1B-Instruct-Q4_K_M.gguf` would either error out or generate garbage.  The on-disk K-quant superblocks are now dequantized to f32 at load time and re-quantized to Q8_0 (block size 32, 34 bytes/block) before being handed to codegen, so they go through the existing Q8_0 path.

  ```rust
  // crates/forgellm-frontend/src/weight_loader.rs (load_from_file_mixed)
  GGMLType::Q8_0  => WeightData::Q8_0Raw(raw.to_vec()),
  GGMLType::Q4_0  => WeightData::Q4_0Raw(raw.to_vec()),
  GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 => WeightData::F32(...),
  _ => WeightData::Q8_0Raw(quantize_f32_to_q8_0(&dequantize(raw, ...)?)),
  ```

  Verified end-to-end on Llama-3.2-1B-Instruct-Q4_K_M (which mixes Q4_K projection weights with a Q6_K `output.weight`): coherent generation at 44.7 tok/s decode on M5 Pro after `forge compile --target cpu` + native build.

- `quantize_f32_to_q8_0(data: &[f32]) -> Vec<u8>` in `forgellm_frontend::weight_loader` — round-trip-tested f32 → Q8_0 quantizer (max abs error ≈ scale/2, where scale = absmax/127 per 32-element block).

### Changed

- **`GGMLType::to_dtype()`** is now honest about post-load encoding: every quantized type other than native Q4_0 maps to `DType::Q8_0` (because that's what the loader actually produces).  Previously K-quants and their relatives mapped to `DType::Q4_0` / `DType::Q2`, which mismatched what the loader stored — codegen would emit Q4_0 kernels expecting raw Q4_0 bytes and feed them re-quantized data of a different layout.
- **`detect_gguf_dtype`** in the CLI now counts K-quants as Q8_0 for codegen-dispatch purposes (matches the post-load truth above).  Q4_K_M models previously fell through to the `_ => {}` arm and were classified as F16, producing a model that crashed at first matmul.

### Trade-offs

- **Memory cost**: a re-quantized K-quant tensor occupies ~1.9× the bytes of the original (Q4_K is 4.5 bits/wt, Q8_0 is 8.5 bits/wt).  For a 1B-parameter model this is ~1.5 GB extra `weights.bin` size.  Acceptable for the correctness MVP — native Q4_K kernels (preserving on-disk layout for both decode and prefill) are tracked as future work.
- **Perf**: identical to a hand-converted Q8_0 model of the same shape, since the post-load representation is Q8_0.  No accuracy loss vs. Q8_0 (we round-trip through f32 once).

### Tests

- `quantize_f32_to_q8_0_round_trip` and `quantize_f32_to_q8_0_zero_block` in `weight_loader.rs`.
- `ggml_type_conversions` updated to assert K-quants map to `DType::Q8_0`.
- `detect_dtype_treats_k_quants_as_q8_0` (renamed from `detect_dtype_falls_back_to_f16_for_k_quants`).

## [0.8.8] — 2026-04-23 — Docs: v0.8.5 SWA kernel validated on Phi-3

### Changed

- **`attention_sliding_batch` is no longer "unvalidated"**.  v0.8.5 shipped without a live SWA model test because no local Q8_0 SWA model was available at release time.  Validated on **Phi-3.1-mini-4k-instruct Q8_0** (`phi3.attention.sliding_window: 2047`):

  | Prompt | Per-token `attention_sliding` | **`attention_sliding_batch`** | Speedup |
  |-------:|------------------------------:|------------------------------:|--------:|
  |   442  |                          13.7 |                      **96.7** | **7.1×** |
  |  1099  |                           9.6 |                      **78.4** | **8.2×** |
  |  1950  |                           5.8 |                      **67.4** | **11.6×** |

  Both paths produce coherent output; batched path ramps in the same shape as the flash-attention batched path from v0.8.1 (peak in the middle, sustained at long contexts).  `FORGE_BATCHED_PREFILL=0` continues to be the A/B toggle.

### Not changed

- No kernel, codegen, or runtime change.  This release updates the v0.8.5 entry's correctness caveat and nothing else.

## [0.8.7] — 2026-04-23 — UX: `forge bench` clarifies interpreter vs AOT

### Fixed

- **`forge bench` was silently benchmarking the interpreter**, not the AOT-compiled binary.  On Llama-3.2-1B Q8_0 this reported ~10 tok/s prefill and ~14 tok/s generate — numbers that make ForgeLLM look slow, because they're the generic Rust forward pass, not the shape-specialized SIMD kernels that `forge compile` emits.  Users comparing ForgeLLM against llama.cpp or MLX with `forge bench` would see a wildly distorted picture.

  Fixed by adding a loud banner to the output:

  ```
  Mode:   Interpreter (generic Rust path)
  Note:   AOT-compiled binaries are 10–40× faster on this model.
          For AOT numbers: forge compile --target cpu --model <model> --output /tmp/aot
                           cd /tmp/aot && cargo build --release
                           ./target/release/<name> weights.bin tokenizer.json <prompt>
  ```

  Printed before the per-run results so the context is unambiguous.  The actual numbers are unchanged.

### Not changed

- No kernel, codegen, or runtime semantics change.
- Interpreter path itself is unchanged — this is purely a framing/UX fix to keep the default benchmark output from being read as "ForgeLLM is slow".

## [0.8.6] — 2026-04-23 — `FORGE_BATCHED_PREFILL` runtime toggle + blog addendum

Small follow-up to the v0.8.x series.

### Added

- **`FORGE_BATCHED_PREFILL` env var** — set `FORGE_BATCHED_PREFILL=0` before running a generated binary to force the per-token `forward_prefill` path even when the prompt is long enough to normally trigger `forward_prefill_batched`.  Useful for A/B testing the batched path against the baseline, debugging suspected regressions, or measuring the speedup on user workloads without recompiling.  Default (unset or `"1"` / anything else) keeps the batched path active.  Verified: on Llama-3.2-1B Q8_0 at 352 tokens, the toggle swings prefill from **41 tok/s** (per-token) to **324 tok/s** (batched).
- **Blog addendum** in `blog/cpu-batched-prefill.md` covering the v0.8.4 SWA correctness fix, v0.8.5 `attention_sliding_batch`, and this v0.8.6 runtime toggle.

### Not changed

- Default behavior, codegen structure, and kernel semantics are unchanged from v0.8.5.
- No performance or correctness change.

## [0.8.5] — 2026-04-23 — Batched sliding-window attention

Completes the SWA (sliding-window attention) story on the CPU batched prefill path.  v0.8.4 fixed the correctness bug by routing SWA models to a per-token `attention_sliding` fallback; this release gives them the same Q-tiled batched treatment that v0.8.1 gave to flash-attention models.

### Added

- **`attention_sliding_batch`** — Q-tiled batched SWA kernel.  Mirrors `attention_flash_batch` (Q_TILE=16, per-head rayon parallelism, stack-allocated per-query softmax state) but:
  - Applies both causal and sliding-window masks per query (`q_pos` attends to `[max(0, q_pos - window + 1), q_pos]`).
  - Bounds the outer K-block loop to the Q tile's combined valid K range: `[max(0, q_tile_start_pos - window + 1), q_tile_end_pos]`.  Blocks outside that range aren't read at all, so long-context SWA models pay the smaller "window × M" work rather than the full "total_seq × M".
  - Emitted only for Q8_0 / Q4_0 models with `sliding_window_size.is_some()`.
- Codegen test `q8_swa_batched_prefill_uses_sliding_batch` — asserts the new kernel is emitted for Q8_0 Mistral configs with `sliding_window_size = Some(32)`, and that `forward_prefill_batched` dispatches through `attention_sliding_batch(...)` with the correct `window` argument.

### Changed

- **`forward_prefill_batched`** dispatch for SWA models now calls `attention_sliding_batch` once per layer instead of the per-token `attention_sliding` loop added in v0.8.4.  The four-way split in the attention block:
  - `use_flash_attention(config)` → `attention_flash_batch` (v0.8.1).
  - `sliding_window_size.is_some()` → **`attention_sliding_batch`** (this release).
  - Short-context non-SWA → per-token `attention()` fallback.

### Correctness

- Same mask semantics as `attention_sliding`: for any query at `q_pos`, the valid K range is `[max(0, q_pos - window + 1), q_pos]`.  Tile-level block bounds are derived from the earliest `q_pos - window + 1` and the latest `q_pos + 1` across the Q tile.
- 63/63 CPU codegen tests pass.
- Validated in v0.8.8 on Phi-3.1-mini-4k-instruct Q8_0 (sliding_window=2047): 7.1–11.6× speedup over per-token `attention_sliding`, both paths produce coherent output.  See the v0.8.8 entry for numbers.

## [0.8.4] — 2026-04-23 — Fix: SWA correctness in CPU batched prefill

### Fixed

- **Correctness bug** on CPU batched prefill for Q8_0/Q4_0 SWA models (Mistral, Gemma-2 class): the non-flash fallback branch in `forward_prefill_batched` was calling plain `attention()` instead of `attention_sliding()`, so the softmax attended to the full KV cache instead of the configured sliding window.  On a Mistral Q8_0 model with `sliding_window_size = 4096` and a prompt ≥ 8 tokens, the generated output would diverge from the per-token `forward_prefill` path at long contexts.  Introduced in v0.8.0 (batched prefill landing), present in v0.8.1–v0.8.3.
- **Dispatch now splits three ways** inside `forward_prefill_batched`:
  - `use_flash_attention(config)` → `attention_flash_batch` (unchanged, v0.8.1)
  - `config.sliding_window_size.is_some()` → per-token `attention_sliding(pos, WINDOW, ...)` loop
  - Else (short-context non-SWA) → per-token `attention(pos, ...)` loop

### Added

- Codegen regression test `q8_swa_batched_prefill_uses_attention_sliding` — builds a Q8_0 Mistral config with `sliding_window_size = Some(32)`, verifies the generated `forward_prefill_batched` body contains `attention_sliding(` and does *not* use the non-SWA fallback marker.

### Not covered (follow-up)

- The SWA fallback is still per-token (no batched attention for SWA).  A `attention_sliding_batch` kernel would mirror `attention_flash_batch` but skip K/V blocks before `pos − window` — valuable for SWA models at long contexts but deferred until a test model is available locally.

## [0.8.3] — 2026-04-23 — Docs: CPU prefill benchmarks + v0.8.x blog post

No functional or performance changes.  Documentation release covering the v0.8.0 / v0.8.1 / v0.8.2 CPU prefill story.

### Added

- **`blog/cpu-batched-prefill.md`** — walkthrough of the v0.8.x CPU prefill restructure.  Covers the weight-outer / token-inner `matmul_mat_q8_0_KxN` kernel, the Q-tiled flash attention (`attention_flash_batch`), and the Q4_0 extension.  Includes the per-prompt-length tok/s table, the key learnings (inner kernel wasn't the bottleneck; `usize` round-trips for disjoint parallel writes), and what's next.
- **README — "CPU Prefill (v0.8.2)" section** — Apple M5 Pro Llama-3.2-1B Q8_0 and Q4_0 prefill at 352 / 902 / 1603 / 2502 tokens with 4.7–9.5× speedups vs v0.7.x per-token.
- **README header** — "Faster than llama.cpp on Apple Silicon — Metal and CPU" (was Metal-only); added link to the new CPU blog post.
- **`benchmarks/HISTORY.md`** — new top-level "CPU Prefill — Llama-3.2-1B-Instruct" table covering Q8_0 and Q4_0 baselines + v0.8.x batched numbers.

## [0.8.2] — 2026-04-23 — Q4_0 batched prefill

Extends the v0.8.0 / v0.8.1 CPU batched prefill path (batched matmul + Q-tiled flash attention) to Q4_0 models. Previously Q4_0 models fell through to the per-token `forward_prefill` stack path; now they use `forward_prefill_batched` with `matmul_mat_q4_0_KxN` kernels when prompts are ≥ `PREFILL_BATCH_THRESHOLD=8` tokens.

### Performance (Apple M5 Pro, Llama-3.2-1B-Instruct Q4_0)

| Prompt length | Per-token prefill | **v0.8.2 batched** | Speedup |
|---:|---:|---:|---:|
| 352 tokens | 43.8 tok/s | **303.5 tok/s** | **6.9×** |
| 902 tokens | 33.6 tok/s | **265.6 tok/s** | **7.9×** |
| 1603 tokens | 25.7 tok/s | **237.5 tok/s** | **9.2×** |

Q4_0 actually runs *faster* than Q8_0 at the same prompt length because the per-row weight bytes are smaller (18 bytes per 32-element block vs 34), so memory bandwidth per forward pass drops proportionally.

### Added

- **`matmul_mat_q4_0_KxN`** shape-specialized batched Q4_0 matmul kernels, emitted for each unique `(K, N)` pair used by the forward pass. Parallels `matmul_mat_q8_0_KxN`: weight-outer, token-inner loop with the existing `dot8_q4_0_q8_0` / `dot4_q4_0_q8_0` sdot kernels (Q4_0 weight × Q8_0-quantized input). Parallelized over n-blocks via rayon with disjoint output writes.

### Changed

- **`forward_prefill_batched`** is now emitted for both Q8_0 and Q4_0 models (previously Q8_0 only).  Dtype-specific choices — matmul kernel name (`matmul_mat_q8_0_*` vs `matmul_mat_q4_0_*`), lm_head row bytes (34 vs 18), and lm_head dot helpers (`dot8_q8_0_q8_0` vs `dot8_q4_0_q8_0`) — are selected at codegen time based on `config.dtype`.
- **`forward_prefill`** now dispatches to `forward_prefill_batched` when `is_q8 || is_q4 && seq_len >= PREFILL_BATCH_THRESHOLD` (was `is_q8` only).
- **`attention_flash_batch`** (from v0.8.1) is reused unchanged — it operates on f32 Q + i8 K/V cache, so the weight dtype doesn't matter.

### Correctness

- Per-token and batched paths produce byte-identical output on tested Q4_0 prompts (Llama-3.2-1B Q4_0, SmolLM2-135M Q4_0).
- 62/62 CPU codegen tests pass.
- The per-token Q4_0 forward path is unchanged — still available for `seq_len < PREFILL_BATCH_THRESHOLD` and as the fallback for non-AArch64 targets.

## [0.8.1] — 2026-04-23 — CPU batched attention

Follow-on to v0.8.0: adds `attention_flash_batch`, a Q-tiled flash-attention kernel that amortizes K/V scans across Q_TILE=16 query rows per block, parallelized over heads. Wired into `forward_prefill_batched` replacing the per-token `attention_flash` call loop.

### Performance (Apple M5 Pro, Llama-3.2-1B-Instruct Q8_0)

| Prompt | v0.7.x | v0.8.0 matmul | **v0.8.1 + attn** | Total vs v0.7.x |
|---:|---:|---:|---:|---:|
| 352 tokens | 40.2 tok/s | 129.5 | **190.6** | **4.7×** |
| 902 tokens | 30.9 tok/s | 66.2 | **234.2** | **7.6×** |
| 1603 tokens | 24.3 tok/s | 43.1 | **207.4** | **8.5×** |
| 2502 tokens | 18.9 tok/s | 29.4 | **180.0** | **9.5×** |

At longer prompts the prefill is now 180–230 tok/s (vs 19–31 on v0.7.x), which is the range llama.cpp-class runtimes deliver for the same model — matched from pure-Rust Q8_0 with the standard flash softmax recurrence and a GQA-aware K/V layout.  Throughput actually *peaks in the middle* (902–1603) because at short m the per-token setup cost dominates and at very long m the absolute O(M²) work starts to compete with matmul.

### Added

- **`attention_flash_batch`** — Q-tiled batched flash-attention kernel. Algorithm: for each head (parallel via rayon), tile queries into Q_TILE=16 chunks; for each chunk, stream K/V in `FLASH_ATTN_BLOCK_SIZE=64` blocks, compute Q_TILE × block_len scores, apply causal mask per-query, update per-query online softmax state, accumulate P·V. Emitted for Q8_0 models where `use_flash_attention` is true (max_seq_len > 512 and no sliding window).
- **Scope**: K/V each loaded once per (head, Q_TILE), so bandwidth drops by ~Q_TILE (16×) relative to per-token attention. Per-query online softmax state is stack-allocated (16 × HEAD_DIM × f32 ≤ 16 KB) — no heap allocations inside the kernel.

### Changed

- **`forward_prefill_batched`** now calls `attention_flash_batch` once per layer instead of looping `attention_flash` per-token. Per-token RoPE + K/V cache writes still happen first in the existing loop so the full sequence's K/V is available by the time batched attention runs.
- SWA / short-context models (where `use_flash_attention` returns false) keep the per-token `attention` fallback inside `forward_prefill_batched`.

### Correctness

- Per-token and batched paths produce byte-identical output on tested prompts (7–2502 tokens, Llama-3.2-1B Q8_0).
- 62/62 CPU codegen tests pass.

## [0.8.0] — 2026-04-22 — CPU batched prefill (closes #211)

Major CPU backend win: prompt prefill is now batched over all M input tokens with weight-outer, token-inner matmul kernels, amortizing weight-bandwidth across the batch instead of re-reading the full weight matrix per token.  For Q8_0 models at prompts ≥ 8 tokens the generated `forward_prefill` now dispatches to a new batched path.

### Performance (Apple M5 Pro, Llama-3.2-1B-Instruct Q8_0)

| Prompt length | Per-token prefill | **Batched (v0.8.0)** | Speedup |
|---:|---:|---:|---:|
| 352 tokens | 40.2 tok/s | **129.5 tok/s** | **3.2×** |
| 902 tokens | 30.9 tok/s | **66.2 tok/s** | **2.1×** |
| 1603 tokens | 24.3 tok/s | **43.1 tok/s** | **1.8×** |
| 2502 tokens | 18.9 tok/s | **29.4 tok/s** | **1.6×** |

Speedup decays with prompt length as the O(M²) per-token attention path starts to dominate — that's the next optimization target, but the matmul win lifts CPU prefill out of the 20-40 tok/s range that was making CPU deployments impractical for ≥ 500-token prompts.

### Added

- **`matmul_mat_q8_0_KxN`** shape-specialized batched Q8_0 matmul kernels (emitted for each unique `(K, N)` pair used by the forward pass). Weight-outer, token-inner loop keeps each 8-row weight block hot in L1 across all M iterations — the whole point of batched prefill. Uses the existing `dot8_q8_0_q8_0` inline-asm sdot kernel; parallelized over n-blocks via rayon with disjoint `output[r*N + j0..j0+8]` writes.
- **`forward_prefill_batched`** — new public function emitted for Q8_0 models. Heap-allocates `m × dim` batch buffers for each inter-layer tensor (hidden, normed, q/k/v, attn_out, gate/up, ffn_hidden, ffn_out), batches QKV/O/gate/up/down matmul, keeps rms_norm, RoPE, K/V cache writes, attention, silu_mul, and residual_add per-token within the layer loop. Final norm + lm_head run on the last token only (matches existing behavior).
- **`PREFILL_BATCH_THRESHOLD = 8`** const in generated CPU projects — `forward_prefill` dispatches to `forward_prefill_batched` above this threshold, keeps the stack-based per-token path below it (lower fixed cost for short prompts).

### Correctness

- Per-token and batched paths produce byte-identical output across tested prompt lengths (7–2502 tokens on Llama-3.2-1B).
- 62/62 CPU codegen tests pass.
- 53/53 runtime tests pass.

### Not covered

- Q4_0 and f32 models fall through to the existing per-token path — batched `matmul_mat_q4_0` and batched f32 matmul can be added later if needed. Q8_0 covers all GGUF models we currently ship support for, which makes this a practical pragma-default win.
- Attention remains per-token (the `attention_flash` kernel is already tiled with O(FLASH_ATTN_BLOCK_SIZE) memory). At very long prompts (≥ 2K tokens) attention's O(M²) cost starts to dominate the saved matmul bandwidth — batched attention is the next frontier.

## [0.7.13] — 2026-04-22 — MPS attention for MHA models + GQA short-prompt fix

Extends the MPS-materialized attention path from MQA/GQA to MHA (`num_kv_heads == num_attention_heads`) and fixes a latent crash on the GQA path at very short prefill prompts.

### Performance (Apple M5 Pro, Phi-3.1-mini-4k-instruct Q8_0, MHA 32h/32kv)

| Prompt length | MMA flash (prev default) | **MPS attn (v0.7.13)** | Δ |
|---|---:|---:|---:|
| 440 tokens | 1851 tok/s | 1475 tok/s | -20% (MMA still wins at short m) |
| 1097 tokens | 1467 tok/s | 1496 tok/s | ≈ tied |
| 1948 tokens | 1106 tok/s | **1646 tok/s** | **+49%** |
| 3044 tokens | 804 tok/s | **1481 tok/s** | **+84%** |

Crossover on Phi-3 is ~1000 tokens (higher than Llama-3.2-1B's ~500) because the per-kv-head MPS call loop runs 32× on Phi-3 vs 8× on Llama.

### Fixed

- **GQA short-prompt crash**: MPS's small-matrix GEMV kernel overflows its internal `vectorRowPadElements` bit field when K/V row stride is much larger than per-head column count. Fired at m ≤ 2 on Llama-3.2-1B and m ≤ 5 on Phi-3.1-mini. The generated `forward_prefill_batch` now falls back to `dispatch_attention_batch` (MMA flash) below `MPS_ATTN_MIN_M = 8`, which fixes the crash and keeps short-prefill fast.

### Added

- **`MPS_ATTN_MIN_M`** constant in generated Metal projects — codegen emits a runtime `if m >= MPS_ATTN_MIN_M { MPS } else { MMA flash }` branch inside the MPS GQA/MHA prefill block.
- **`FORGE_NO_MPS_ATTN=1`** codegen env var — set before `forge compile` to disable the MPS-materialized attention path while keeping MPS matmul for QKV/O/FFN. Intended for A/B benchmarking MMA-flash vs materialized-MPS attention; not required for production use.

### How

The GQA scaffolding (`extract_q_to_f16_gqa`, `softmax_causal_scale_f16_gqa`, `gqa_out_reshape_f16_to_f32`) already parameterizes on `num_groups = num_heads / num_kv_heads`, so reducing to `num_groups = 1` covers the MHA case without new kernels. The gate change was `num_kv_heads < num_attention_heads` → `num_kv_heads <= num_attention_heads`.

The pre-attention QKV prep (f16→f32 convert, optional bias, RoPE, copy_kv) is now emitted in its own encoder and shared between the MPS-attn path and the MMA-flash fallback, so the runtime branch only switches the attention op itself.

### Correctness

- Llama-3.2-1B GQA: short prompts (m = 1..7) no longer crash — previously `vectorRowPadElements will overflow its fc bit allocation`.
- Phi-3.1-mini MHA: MPS and MMA flash variants produce byte-identical output on "The Eiffel Tower is located in" → "the city of Cairo, Egypt" (Phi-3 hallucination, but the same hallucination).
- Gemma-1.1-2B MQA path unchanged (different code path, not touched).
- All 71 Metal codegen tests pass.

## [0.7.12] — 2026-04-22 — CI cleanup for Rust 1.95

Mechanical fixes for new clippy lints introduced in Rust 1.94/1.95 that broke the `-D warnings` CI job (and by extension the PyPI wheel publish workflow). No functional or performance changes.

### Fixed

- `manual_is_multiple_of` → `.is_multiple_of()` (three sites in Metal codegen gates for MPS GQA/MQA).
- `literal_empty_format` → inline variables in `writeln!` format strings (Metal MPS buffer sizing).
- `manual_range_contains` → `RangeInclusive::contains` (Metal codegen test).
- `needless_range_loop` → iterator-based loops (weight_loader K-quant round-trip tests).
- `useless_vec` → stack array (emit.rs quantization round-trip test).
- Bench `ModelConfig` literals missing `hidden_activation` field (benches/codegen.rs).
- `manual_checked_ops` → `.checked_div(...).unwrap_or(...)` (safetensors loader head-count inference).

### CI

- `cargo clippy --workspace --exclude forgellm-python -- -D warnings` passes on Rust 1.95.
- `cargo fmt --all -- --check` passes (one drift fix in Metal codegen carried over from v0.7.11).

## [0.7.11] — 2026-04-21 — MPS Attention for GQA Models

Extends the MPS-materialized attention path from MQA only (v0.7.9) to any Q8_0 model with hidden ≥ 2048 and an integer-divisible `num_heads / num_kv_heads` ratio — in particular Llama-3.2-1B (32 heads / 8 KV heads, num_groups = 4).

### Performance (Apple M5 Pro, Llama-3.2-1B-Instruct Q8_0)

| Prompt length | MPS matmul only (pre-v0.7.11) | **v0.7.11 (MPS attn)** | Δ |
|---|---:|---:|---:|
| 1002 tokens | 3060 tok/s | **6934 tok/s** | **+127%** |
| 2602 tokens | 3490 tok/s | **6778 tok/s** | **+94%** |

### Added

- **`extract_q_to_f16_gqa`** kernel — writes Q in kv-head-major layout `[num_kv_heads, M*num_groups, head_dim]` so each kv-head's Q lives in a contiguous buffer slice.
- **`softmax_causal_scale_f16_gqa`** kernel — row-major softmax for the GQA layout; derives the per-row token via `tok = (row / num_groups) % M` for causal masking.
- **`gqa_out_reshape_f16_to_f32`** kernel — unshuffles the P@V output from kv-head-major back into `[M, num_heads, head_dim]` f32.
- **`mps::matrix_with_offset`** objc helper — wraps `-[MPSMatrix initWithBuffer:offset:descriptor:]`.
- **`mps_matmul_f16_offsets`** rust helper — per-kv-head MPS matmul with explicit offsets into the Q / K / V / scores buffers.

### How

Per attention block: one loop over `num_kv_heads` issuing a MPS Q·K^T matmul (alpha = 1/√head_dim) into the attn_scores buffer, one GQA softmax dispatch, one loop issuing MPS P·V matmuls into mps_out, and a final f16→f32 reshape.  Each MPS matmul for GQA is smaller (M·num_groups × head_dim × seq_len) than the MQA single call, but there are `num_kv_heads` of them — aggregate compute matches.

Why offsets instead of batched MPS? The K/V caches have `num_kv_heads` heads interleaved within each row (`rowBytes = num_kv_heads * head_dim * 2`).  Trying to use MPS `matrixBytes` striding on overlapping slices of the same rows produced wrong numerics; MPS's batch model assumes non-overlapping matrix regions.

Gate: `is_q8 && hidden_size >= 2048 && num_kv_heads > 1 && num_kv_heads < num_heads && num_heads % num_kv_heads == 0`.  MQA (`num_kv_heads == 1`) keeps the existing path; MHA (`num_kv_heads == num_heads`) continues on flash attention.

### Correctness

- Gemma-1.1-2B MQA unchanged (5195 tok/s @ 2601, byte-identical "meaning of life" output to v0.7.10).
- Llama-3.2-1B GQA: "The capital of France is Paris. The Eiffel Tower is located in..." matches the flash-attention baseline.
- 71/71 Metal codegen tests pass.

## [0.7.10] — 2026-04-21 — f16-In/Out GeLU·Mul Fusion

Eliminates two convert kernels per layer around the FFN activation on the MPS prefill path.  The convert-kernel no-op ablation measured ~6% headroom; fusing the post-gate_up `convert_f16→f32` and pre-down `convert_f32→f16` into a single f16-in/f16-out activation kernel captures it.

### Performance (Apple M5 Pro, Gemma-1.1-2B Q8_0)

| Prompt length | v0.7.9 | **v0.7.10** | Δ | vs llama.cpp |
|---|---:|---:|---:|---:|
| 1001 tokens | 4116 tok/s | **4776 tok/s** | +16% | **+40%** |
| 2601 tokens | 4541 tok/s | **5225 tok/s** | +15% | **+54%** |

Cumulative v0.7.6 → v0.7.10: **5.1× @ 2601 tokens, 4.2× @ 1001 tokens.**

### Added

- **`silu_mul_fused_batch_f16io` / `gelu_mul_fused_batch_f16io`** — reads the gate_up matmul output directly as f16 from `mps_out_f16` and writes the FFN-hidden directly as f16 into `mps_in_f16` for the next MPS matmul.  All arithmetic in f32 internally; only the load/store are f16.
- **`dispatch_silu_mul_fused_batch_f16io` helper** — wired into the MPS prefill layer body, replacing the three-step (convert f16→f32, activation, convert f32→f16) sequence.

### Bandwidth saved per layer @ 2601 tokens

- gate_up post-convert: 164 MB read + 327 MB write skipped
- down pre-convert: 82 MB read + 41 MB write skipped
- Total: 614 MB per layer × 18 layers = 11 GB of device-memory traffic removed per prefill.

### Correctness

- Byte-identical output on Gemma-1.1-2B "meaning of life" prompt vs v0.7.9.
- 71/71 Metal codegen tests pass.

## [0.7.9] — 2026-04-21 — MPS-Materialized Attention (Faster Than llama.cpp)

For MQA (num_kv_heads = 1) models on the MPS prefill path, attention is now materialized as two MPS matmuls plus a dedicated causal-masked softmax kernel — replacing the streaming flash-attention kernel for the batched prefill.

### Performance (Apple M5 Pro, Gemma-1.1-2B Q8_0)

| Prompt length | v0.7.6 | v0.7.8 (MPS matmul) | **v0.7.9 (MPS attn)** | Cumulative Δ | vs llama.cpp |
|---|---:|---:|---:|---:|---:|
| 1001 tokens | 1147 tok/s | 3088 tok/s | **4116 tok/s** | **3.6×** | **~117%** |
| 2601 tokens | 1031 tok/s | 2136 tok/s | **4541 tok/s** | **4.4×** | **~134%** |

ForgeLLM Metal prefill now beats llama.cpp on Gemma-1.1-2B at both tested prompt lengths on the same hardware.

### How

- **Q extraction** (`extract_q_to_f16_mqa`): one kernel copies the Q slice out of the fused `[M, qkv_stride]` f32 buffer into a contiguous `[M*num_heads, head_dim]` f16 buffer so MPS can treat all Q heads as a single matrix.
- **Q @ K^T via MPS** with `alpha = 1/sqrt(head_dim)`: K cache (MQA: `[seq_len, head_dim]` f16) is used directly with no copy.  Output written into `attn_scores_f16` (stride = `MAX_SEQ_LEN`).
- **Causal softmax** (`softmax_causal_scale_f16`): one simdgroup per (token, head) row; three-pass row-max / row-sum / normalize using `simd_max` / `simd_sum`.  Mask is applied inline.
- **P @ V via MPS**: scores f16 times the V cache f16 directly into `mps_out_f16`.
- **Post-convert** to f32 for the downstream O projection.

Gated on `is_q8 && hidden_size >= 2048 && num_kv_heads == 1` (GQA and MHA models keep the existing flash-MMA kernel).

### Correctness

- Byte-identical output on Gemma-1.1-2B "meaning of life" prompt vs v0.7.8.
- 71/71 Metal codegen tests pass.

### Memory

+32 MB `attn_scores_f16` scratch (`MAX_BATCH × NUM_HEADS × MAX_SEQ_LEN × 2`) plus +2 MB `q_f16`.  Both allocated from private Metal storage once at load.

## [0.7.8] — 2026-04-20 — MPSMatrixMultiplication Prefill Path

Major prefill performance jump for Gemma-class models (hidden ≥ 2048) via Apple's Metal Performance Shaders.  Non-qualifying models are unaffected.

### Performance (Apple M5 Pro, Gemma-1.1-2B Q8_0)

| Prompt length | v0.7.7 | **v0.7.8** | Δ vs v0.7.7 | Δ vs v0.7.6 |
|---|---:|---:|---:|---:|
| 1001 tokens | 1304 tok/s | **3088 tok/s** | **+137%** | +169% |
| 2601 tokens | 1155 tok/s | **2136 tok/s** | **+85%** | +107% |

That's 87% of llama.cpp's Gemma-1-2B throughput at 1001 tokens, 63% at 2601 tokens on the same hardware (up from 35–40% at v0.7.7).

### How

- **Pre-dequantization at load**: `dequant_q8_to_f16` runs once per prefill weight and writes a contiguous `[rows, cols]` row-major f16 buffer alongside the existing Q8_0 one.  Adds ~2.6 GB of additional weight memory for Gemma-1-2B.
- **MPSMatrixMultiplication dispatch**: all four prefill matmuls (QKV, O, gate_up, down) run through `MPSMatrixMultiplication` f16×f16→f16.  Called via raw `objc` — the `metal` crate does not expose MPS matrix bindings.
- **f32↔f16 convert kernels** (`convert_f32_to_f16`, `convert_f16_to_f32`) wrap every MPS call: residual stream stays f32, but MPS inputs/outputs are f16.  The compute encoder is split into 4 sub-encoders per layer (one per matmul phase) because MPS creates its own internal encoder.
- **Gated on `is_q8 && hidden >= 2048`** — smaller Q8 models (SmolLM-135M at hidden=576) keep the existing `hh4_wide` path unchanged; the conversion-kernel overhead dominates at small hidden.

### Correctness

- Byte-identical output on Gemma-1.1-2B "meaning of life" prompt vs v0.7.7: *"a profound question that has captivated philosophers and theologians for centuries. While there is no definitive answer, exploring this question can lead to profound insights into the nature of existence."*
- 71/71 Metal codegen tests pass.

### Why this works where hh4 tuning didn't

Prior v0.7.x matmul work extracted the local maximum of our 8-accumulator/12 KB-TG-memory algorithm family at ~17% of f16 peak.  The remaining 3× gap was structural — MPS uses a fundamentally different tile/scheduling strategy.  This release doesn't tune our kernel; it delegates to a kernel that already runs at ~54% of peak.

## [0.7.7] — 2026-04-20 — Gemma-1 Prefill MMA Throughput

Minor release focused on Apple Silicon prefill throughput for Gemma-class models (hidden ≥ 2048). No API changes; non-Gemma models are unaffected (dispatch conditions `cols ≥ 2048 && num_tokens ≥ 256 && cols % 64 == 0`).

### Performance (Apple M5 Pro, Gemma-1.1-2B Q8_0)

| Prompt length | v0.7.6 | **v0.7.7** | Δ |
|---|---:|---:|--:|
| 1001 tokens | 1147 tok/s | **1304 tok/s** | **+14%** |
| 2601 tokens | 1031 tok/s | **1155 tok/s** | **+12%** |

### Added

- **`matmul_q8_mma32_hh4`** — K=64 tile variant of the all-FP16 MMA matmul (two Q8_0 blocks per threadgroup barrier). One `device const half*` scale read per row (vs eight). `float4` token loads for the inline f32→f16 conversion. 32-lane parallel C-tile store (vs lane-0 serial).
- **`matmul_q8_mma32_hh4_wide`** — 32 tok × 64 row threadgroup tile with 4 simdgroups laid 1×4 along the row dim; each SG produces a 32×16 sub-tile with 8 accumulators, doubling MMAs per barrier. Dispatched when `rows % 64 == 0` (all four Gemma-1-2B matmul shapes qualify).
- **Aliased threadgroup memory in `hh4_wide`** — the 2048-half token tile and the 2048-half C-store scratch are never simultaneously live; the trailing outer-loop barrier orders the reuse. Drops the kernel's TG-mem footprint from 16 KB to 12 KB and improves concurrent-TG occupancy.
- **`attention_mma_flash_batch` extended to `head_dim=256`** — `K_BLOCK=32` with `v_sh` dropped keeps total TG memory under 32 KB (V is read directly from the cache in Phase 4). Closes head_dim=256 to MMA flash attention for Gemma-1.

### Correctness

- Byte-identical output vs v0.7.6 on Gemma-1.1-2B across sampled prompts.
- 71/71 Metal codegen tests pass.

### Honest framing

v0.7.7 extracts the local maximum of the hh4 algorithm at 12 KB TG memory. Further explorations this cycle (K=128 outer tile, 8-SG layout, double-buffered ping-pong tiles, attention K_BLOCK=16 + `v_sh`, hardcoded-constant shape specialization) each regressed or broke even. The remaining ~2.5× gap to llama.cpp on Gemma prefill is split roughly matmul 61% / attention 29% / other 10% (measured via kernel-ablation binaries); closing it further likely requires either a fundamentally different matmul algorithm (e.g. pre-dequantization to f16 + MPSMatrixMultiplication-class kernels) or structural attention work (phase merging / SG-local reductions).

## [0.7.6] — 2026-04-18 — Gemma-1 AOT Metal Support

Minor release: Gemma-1-2B now compiles and runs on Metal with `forge compile --target metal`. Output is byte-identical to the v0.7.5 interpreter at 32 tokens for the "meaning of life" prompt, at 12× the throughput (89.8 tok/s vs 7.3 tok/s on M5 Pro, Q8_0).

### Fixed

- **`matmul_q8` / `matmul_q8_batch` vec_tile overflow on `cols > 8192`**: both kernels cached the input vector in a `threadgroup float vec_tile[VEC_TILE_SIZE]` array capped at 8192 floats (32 KB — the device TG memory limit on Apple Silicon). For architectures with `intermediate_size > 8192` (Gemma-2B uses 16384), the down-projection matmul's `for (uint i = tid; i < cols; i += 256) vec_tile[i] = input[i];` loop silently wrote past the end of the array. The dispatch helpers now route large-col calls (`cols > 8192`) through `matmul_q8_gemm_batch`, which reads inputs directly from device memory without caching. Single-token decode path (`dispatch_matmul_q8`) dispatches gemm with `M = 1`; batched prefill path (`dispatch_matmul_q8_batch`) extends its existing gemm-path condition to `num_tokens >= TOKENS_PER_TG_Q8 || cols > 8192`.

### Added

- **Metal GELU kernels** — `gelu_mul_fused` and `gelu_mul_fused_batch` use tanh-approximate GELU (matches HF's `gelu_pytorch_tanh`, Gemma-1's default). Correctness verified against a CPU reference with max abs error `1.19e-7` over `g ∈ [-3, 3]`.
- **Codegen conditional activation pipeline** — when `config.hidden_activation == GeluApprox`, the `silu_mul_fused*` struct fields are populated with the GELU kernel compiled variants. The dispatch helper signatures and FFN forward code are unchanged (the kernel behind the pipeline just differs).
- **`scale_buffer` kernel** + `dispatch_scale_buffer` helper — Gemma-1 multiplies input embeddings by `sqrt(hidden_size)` right after lookup. Single-token `forward()` / `forward_profile()` paths apply this scaling via a tight Rust loop in the embed-lookup block (unified memory makes this free). Batched prefill applies it via the new `scale_buffer` compute kernel after `dispatch_copy_embedding_batch`. Both paths are gated on `HiddenActivation::GeluApprox` so non-Gemma models are unaffected.

### Performance (Apple M5 Pro, Q8_0)

| Workload | Interpreter | **Metal** | Δ |
|----------|------------:|----------:|--:|
| Gemma-1.1-2B, short decode | 7.3 tok/s | **89.8 tok/s** | **12.3×** |

### Correctness

- **Byte-identical output** vs v0.7.5 interpreter at 32 tokens on Gemma-1.1-2B ("meaning of life" prompt): *"a profound question that has captivated philosophers and theologians for centuries. While there is no definitive answer, exploring this question can lead to profound insights into the nature of existence"*.
- All 71 Metal codegen tests pass. Non-Gemma models (SmolLM-135M, Llama-3.2-1B) unaffected — the vec_tile routing only triggers on `cols > 8192`, which none of them hit.

### Debug path

The root-cause bug took most of the session to isolate. The GELU kernel tested correct in isolation (max error `1.19e-7`); the embedding scale was applied in both forward paths. The final bisection: bypassing `forward_prefill_batch` for the whole prompt produced coherent Gemma output; sweeping the prefill-batch cutoff `M` showed the bug appeared at `M ≥ 4` — matching the `num_tokens >= TOKENS_PER_TG_Q8 (= 4)` dispatch threshold, which kicks in `matmul_q8_gemm_batch`. But disabling that path still produced garbage at every `M` because the fallback `matmul_q8_batch` also overflows `vec_tile` on Gemma's down-proj. The fix above routes `cols > 8192` through gemm at every `M`.

## [0.7.5] — 2026-04-18 — Gemma-1 Interpreter Support

Minor release: `forge run` (the interpreter) now produces coherent output for Gemma-1 GGUFs. The previous "✅ Verified" claim in the README was false — the interpreter was silently running the Llama forward pass on Gemma weights, which is the wrong activation function (SiLU vs GeLU-approx) and misses Gemma's post-lookup embedding scale.

### Changed

- **`ModelConfig`** gains a `hidden_activation: HiddenActivation` field (`SiLU` | `GeluApprox`), `#[serde(default)]` so existing configs deserialize unchanged.
- **CLI GGUF + HFConfig parsers** set `hidden_activation = GeluApprox` when `architecture == Gemma`; everything else stays `SiLU`.
- **Interpreter FFN** dispatches on `config.hidden_activation` for the gate activation (`silu_mul` → `gelu_mul` when Gemma).
- **Interpreter embedding lookup** multiplies the hidden state by `sqrt(hidden_size)` when the model is Gemma, matching HF's reference forward pass. This happens after the lookup (not via weight rewrite) so tied-embedding layouts — where `lm_head` shares the `embed_tokens` buffer — keep the logit projection unscaled.
- **Prompt encoding** uses `Tokenizer::encode_with_special` instead of `encode` so the tokenizer's post_processor can add the BOS / template tokens that Gemma requires for coherent generation. Verified to not regress SmolLM.

### Not handled by this release

- Gemma's RMS-norm `+1` offset. Standard `llama.cpp`-converted Gemma GGUFs already bake `w' = w + 1` into stored norm weights (see `convert_hf_to_gguf.py`), so no runtime adjustment is needed for GGUF-origin weights. A public `apply_gemma_weight_tweaks` helper is exposed for future Safetensors-origin support, where the tweak would be needed.
- Gemma-2 and Gemma-3 (softcap, dual norms per sublayer, sliding window alternation). Left intentionally unsupported — the `gemma2` GGUF arch string is recognized at the enum level but no forward-pass logic handles the additional ops.
- AOT-Metal support for Gemma-1 (planned for a follow-up release).

### Verified

- `bartowski/gemma-1.1-2b-it-Q8_0.gguf` produces coherent text via `forge run` on the "meaning of life" prompt: *"a profound question that has captivated philosophers and theologians for centuries. While there is no definitive answer, exploring this question can lead to profound insights into the nature of existence"*.
- `HuggingFaceTB/SmolLM2-135M-Instruct-Q8_0.gguf` (Llama arch) still produces the same coherent text as v0.7.4.

### Correctness

- All 259 workspace tests pass (including 71 Metal, 112 frontend, 62 CPU codegen, 53 runtime).

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
