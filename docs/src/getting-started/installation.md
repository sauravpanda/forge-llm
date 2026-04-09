# Installation

## From Source

ForgeLLM requires Rust 1.75+ (for `div_ceil` stabilization).

```bash
git clone https://github.com/sauravpanda/forge-llm.git
cd forge-llm
cargo build --release
```

The binary will be at `target/release/forge`.

## From crates.io

```bash
cargo install forgellm-cli
```

## Dependencies

ForgeLLM has minimal dependencies:

- **Rust toolchain** (stable, 1.75+)
- **C compiler** (for the `tokenizers` crate's oniguruma regex engine)
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Ubuntu: `apt install build-essential`
  - Arch: `pacman -S base-devel`

No Python, no CUDA toolkit, no large frameworks.
