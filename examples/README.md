# ForgeLLM Examples

Runnable examples covering the main ways to use ForgeLLM.

| # | Example | What it does | Best for |
|---|---------|--------------|----------|
| 01 | `01-compile-cpu.sh` | AOT compile to native CPU binary | Edge devices, CI/CD |
| 02 | `02-compile-metal.sh` | AOT compile to Metal GPU binary (Apple Silicon) | Max throughput on Mac |
| 03 | `03-run-interpreter.sh` | Direct interpreter, no compile | Quick testing |
| 04 | `04-api-server.sh` | OpenAI-compatible HTTP server | Drop-in llama.cpp replacement |
| 05 | `05-python-inference.py` | Python bindings with generate/stream/chat | ML pipelines |

## Prerequisites

- Rust 1.75+ (`rustup install stable`)
- ForgeLLM built: `cargo build --release` from the repo root
- `forge` on PATH or reference `./target/release/forge` directly
- Optional: Python 3.11+ for example 05
- Optional: macOS with Apple Silicon for example 02

## Running

All shell examples auto-download SmolLM2-135M Q8_0 (~140 MB) on first run to `~/.cache/forgellm-examples/`. Subsequent runs are instant.

```bash
# From repo root
./examples/01-compile-cpu.sh
./examples/02-compile-metal.sh         # macOS only
./examples/03-run-interpreter.sh
./examples/04-api-server.sh

# Python (needs forgellm installed: cd crates/forgellm-python && maturin develop)
python3 examples/05-python-inference.py
```

## Environment variables

Shell scripts respect:
- `PROMPT` — override the prompt (default: "The meaning of life is")
- `MAX_TOKENS` — override max tokens (default: 50)
- `PORT` — API server port (default: 8080)
- `TEMPERATURE` — sampling temperature (default: 0.7)

## Choosing an approach

- **Need fastest inference on Apple Silicon?** Use `02-compile-metal.sh`
- **Deploying to Linux/embedded?** Use `01-compile-cpu.sh`
- **Quick prototype/debug?** Use `03-run-interpreter.sh`
- **Serving multiple requests?** Use `04-api-server.sh`
- **Integrating into Python ML code?** Use `05-python-inference.py`
- **Embedding in a Rust app?** Depend on `forgellm-frontend` + `forgellm-runtime` directly — see the `forge run` command implementation in `crates/forgellm-cli/src/main.rs` for a complete reference

## Troubleshooting

- **"Model not found"**: Check `~/.cache/forgellm-examples/` — delete and re-run to force redownload
- **"forge: command not found"**: Run `cargo build --release` from the repo root, then either add `target/release` to PATH or use the absolute path
- **Metal build fails**: Ensure you're on macOS with Apple Silicon and Xcode command line tools installed
- **Python import fails**: Build and install with `cd crates/forgellm-python && maturin develop` (requires Python 3.11+)
