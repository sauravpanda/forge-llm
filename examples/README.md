# ForgeLLM Examples

Minimal, self-contained examples demonstrating common ForgeLLM use cases.
Each example is independent — pick whichever matches your workflow.

All examples use **SmolLM2-135M-Instruct** (Q8_0, ~135 MB) as a small, fast
model for quick iteration. You can swap in any supported GGUF model.

## Prerequisites

- Rust toolchain (stable): <https://rustup.rs>
- `forge` CLI built and on `$PATH`:
  ```bash
  cargo install --path crates/forgellm-cli
  # or during development:
  cargo build --release -p forgellm-cli
  export PATH="$PWD/target/release:$PATH"
  ```
- `curl` for downloading models
- ~200 MB of disk space under `~/.cache/forgellm-examples`

Shell-script examples cache the model and tokenizer under
`~/.cache/forgellm-examples/` so re-running is instant.

## Index

| # | File | What it shows | When to use |
|---|------|---------------|-------------|
| 1 | [`01-compile-cpu.sh`](01-compile-cpu.sh) | AOT-compile a GGUF model to a self-contained CPU binary. | You want a native binary with zero runtime interpreter — maximum performance on x86/ARM CPUs. |
| 2 | [`02-compile-metal.sh`](02-compile-metal.sh) | AOT-compile a GGUF model to a Metal (Apple GPU) binary. | You are on Apple Silicon and want GPU-accelerated inference. |
| 3 | [`03-run-interpreter.sh`](03-run-interpreter.sh) | Run a GGUF model directly via the built-in interpreter (no compile step). | You want the fastest path from "have a GGUF file" to "generating text" — no cargo build. Best for experimenting. |
| 4 | [`04-api-server.sh`](04-api-server.sh) | Start an OpenAI-compatible HTTP API server. | You want to serve a model over HTTP for existing OpenAI SDK clients. |
| 5 | [`05-python-inference.py`](05-python-inference.py) | Use the `forgellm` Python bindings for generate / stream / chat. | You are writing Python code and want to call ForgeLLM as a library. |
| 6 | [`06-library-usage/`](06-library-usage/) | Standalone Cargo project embedding `forgellm-runtime` as a library. | You are writing a Rust app and want programmatic inference without the CLI. |
| 7 | [`07-custom-sampling.rs`](07-custom-sampling.rs) | Implement a custom sampler (Mirostat-style temperature scheduler) against the runtime API. | You want to experiment with novel sampling strategies. |

## Running an example

```bash
# From the repo root
chmod +x examples/*.sh
./examples/01-compile-cpu.sh
```

## Choosing an approach

```
Need fastest raw inference?          ->  01 (CPU compile) or 02 (Metal compile)
Just want to try a model quickly?    ->  03 (interpreter)
Serving to external clients?         ->  04 (OpenAI API server)
Calling from Python?                 ->  05 (Python bindings)
Calling from Rust?                   ->  06 (library usage)
Researching sampling algorithms?     ->  07 (custom sampling)
```

## Expected output

All examples generate text continuing the prompt `"The meaning of life is"`
(or a similar short prompt) and print it to stdout along with timing stats
(tokens/second, prefill time, etc.).

## Troubleshooting

- **`forge: command not found`** — make sure `cargo build --release -p forgellm-cli`
  has completed and `target/release` is on your `$PATH`.
- **`tokenizer.json not found`** — examples download it automatically; make sure
  your network can reach `huggingface.co`.
- **Metal example fails on Linux/Windows** — `02-compile-metal.sh` only works on
  macOS with Apple Silicon. Use `01-compile-cpu.sh` elsewhere.
- **Python example `ImportError`** — build the Python bindings first:
  `cd crates/forgellm-python && pip install maturin && maturin develop --release`.
