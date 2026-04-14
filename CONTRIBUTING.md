# Contributing to ForgeLLM

## Quick Start

```bash
git clone https://github.com/sauravpanda/forge-llm.git
cd forge-llm
cargo build
cargo test --workspace --exclude forgellm-python
```

## Development Setup

- Rust stable (1.75+)
- macOS with Apple Silicon (for Metal codegen testing)
- C compiler (for tokenizers crate — `xcode-select --install` on macOS)
- Optional: Python 3.11+ (for Python bindings)

## Running Tests

```bash
cargo test --workspace --exclude forgellm-python  # All tests (~286)
cargo test -p forgellm-codegen-metal              # Metal codegen only
cargo test -p forgellm-codegen-cpu                # CPU codegen only
cargo test -p forgellm-frontend                   # Frontend/parser only
cargo test -p forgellm-optimizer                  # Optimizer passes only
cargo test -p forgellm-runtime                    # Runtime/interpreter only
```

## Code Quality

```bash
cargo clippy --workspace --exclude forgellm-python -- -D warnings
cargo fmt --all -- --check
```

CI runs build, test, clippy, and fmt checks on every PR.

## Making a PR

1. Fork and create a branch from `main`
2. Write code + tests
3. Run `cargo test --workspace --exclude forgellm-python` and `cargo clippy`
4. Submit PR with a conventional commit title (`feat:`, `fix:`, `perf:`, `docs:`, `test:`)
5. Keep the PR focused — one feature or fix per PR

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full compiler pipeline, or browse the [mdbook docs](https://sauravpanda.github.io/forge-llm/) for detailed per-crate documentation.

## Commit Style

We use conventional commits:

- `feat:` — new feature
- `fix:` — bug fix
- `perf:` — performance improvement
- `docs:` — documentation
- `test:` — test addition
- `refactor:` — code restructuring
- `chore:` — maintenance

Keep the subject line under 72 characters.

## Adding a New Model Architecture

1. Add architecture variant to `Architecture` enum in `crates/forgellm-frontend/src/ir.rs`
2. Add detection logic in `crates/forgellm-frontend/src/config.rs` (`detect_architecture`)
3. Add or extend graph builder in `crates/forgellm-frontend/src/graph_builder.rs`
4. Add GGUF name mappings in `crates/forgellm-frontend/src/weight_loader.rs` if needed
5. Add tests

## Adding a New Codegen Backend

1. Create a new crate `crates/forgellm-codegen-<target>/`
2. Accept `&Graph` as input (the IR is the central contract)
3. Emit a complete Cargo project (or shader source) to an output directory
4. Wire it into the CLI's `compile` subcommand in `crates/forgellm-cli/src/main.rs`
5. Add tests that verify the generated code compiles and produces correct output

## Security

This is a public repository. Before submitting a PR:

- Never commit secrets, API keys, tokens, or credentials
- No hardcoded paths or machine-specific configuration
- Review for OWASP top 10 concerns
