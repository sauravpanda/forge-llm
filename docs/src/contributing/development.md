# Development Guide

## Building

```bash
git clone https://github.com/sauravpanda/forge-llm.git
cd forge-llm
cargo build
cargo test
```

## Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p forgellm-frontend

# With output
cargo test --workspace -- --nocapture
```

## Code Quality

```bash
# Lint
cargo clippy --workspace --all-targets -- -D warnings

# Format
cargo fmt --all -- --check
```

CI runs all four checks (build, test, clippy, fmt) on every PR.

## Project Structure

```
forge-llm/
├── crates/
│   ├── forgellm-frontend/     # Parsing, IR, graph building, weights
│   ├── forgellm-optimizer/    # Fusion detection, DCE
│   ├── forgellm-codegen-cpu/  # CPU Rust source generation
│   ├── forgellm-codegen-wasm/ # WASM generation (planned)
│   ├── forgellm-codegen-gpu/  # GPU generation (planned)
│   ├── forgellm-runtime/      # Interpreter, KV cache, sampling, tokenizer
│   └── forgellm-cli/          # CLI tool
├── docs/                      # This documentation (mdBook)
├── .github/workflows/         # CI configuration
└── Cargo.toml                 # Workspace definition
```

## Commit Conventions

- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code restructuring
- `test:` — test additions
- `docs:` — documentation
- `perf:` — performance improvement
- `chore:` — maintenance

## Adding a New Model Architecture

1. Add architecture variant to `Architecture` enum in `ir.rs`
2. Add detection logic in `config.rs` (`detect_architecture`)
3. Add graph builder in `graph_builder.rs` (or extend `build_llama_graph` if structurally similar)
4. Add GGUF name mappings in `weight_loader.rs` if needed
5. Add tests
