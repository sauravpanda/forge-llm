# Forge LLM - Development Guidelines

## Project Overview

Forge is a Rust-native ahead-of-time (AOT) ML compiler for small LLMs (1-7B parameters). It compiles model weights + architecture into optimized, self-contained binaries with no runtime interpreter.

## Repository Structure

Rust workspace with crates under `crates/`:
- `forge-frontend` — Model parsing (GGUF, SafeTensors) and IR construction
- `forge-optimizer` — Graph optimizations (fusion, layout, quantization, memory planning)
- `forge-codegen-cpu` — CPU code generation (x86 AVX2/512, ARM NEON, Apple AMX)
- `forge-codegen-wasm` — WASM + WebGPU code generation
- `forge-codegen-gpu` — GPU code generation via wgpu/WGSL
- `forge-runtime` — Minimal runtime (KV cache, sampling, tokenizer, API server)
- `forge-cli` — CLI (`forge compile`, `forge run`, `forge bench`, `forge serve`)

## Build & Test

```bash
cargo build                    # Build all crates
cargo test                     # Run all tests
cargo test -p forge-frontend   # Test a specific crate
cargo clippy                   # Lint
cargo fmt --check              # Format check
```

## Development Rules

### Public Repository — Security First
- **Never commit secrets, API keys, tokens, or credentials**
- **Never include personal names or identifying info in code, comments, or commit messages**
- Review all code for OWASP top 10 before committing
- No hardcoded paths or machine-specific configuration
- Use environment variables or config files for any runtime configuration

### Code Style
- Follow standard Rust conventions (`rustfmt`, `clippy`)
- Use `thiserror` for library errors, `anyhow` for binary/CLI errors
- Prefer `#[inline]` only where benchmarks justify it
- Write doc comments for all public APIs
- Keep `unsafe` blocks minimal and always document the safety invariant

### Architecture Principles
- Each crate should have a clear, single responsibility
- The IR is the central abstraction — all frontends produce it, all backends consume it
- Optimization passes are independent and composable
- Backend code generation emits Rust source (compiled by cargo) or shader source (WGSL/PTX)
- Zero allocations during inference is the target — static memory planning

### Testing Strategy
- Unit tests in each crate for core logic
- Integration tests that compile small models end-to-end
- Benchmark tests comparing against reference implementations
- Test quantization round-trip accuracy

### Commit Messages
- Use conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `perf:`
- Keep subject line under 72 characters
- No personal names or identifying information in commits
