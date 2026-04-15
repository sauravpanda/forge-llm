"""
05-python-inference.py — Minimal example using the forgellm Python bindings.

Demonstrates:
  1. Loading a GGUF model + tokenizer into a `forgellm.Model`.
  2. One-shot generation with `model.generate(...)`.
  3. Token-by-token streaming with `model.stream(...)`.
  4. Chat-templated generation with `model.chat(...)`.

Prerequisites
-------------
Build and install the forgellm Python bindings once from the repo root:

    cd crates/forgellm-python
    pip install maturin
    maturin develop --release

The model and tokenizer are downloaded on first run into
``~/.cache/forgellm-examples`` (same cache used by the shell examples).

Run
---
    python examples/05-python-inference.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "forgellm-examples"
GGUF_PATH = MODEL_DIR / "SmolLM2-135M-Instruct-Q8_0.gguf"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

GGUF_URL = (
    "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/"
    "SmolLM2-135M-Instruct-Q8_0.gguf"
)
TOKENIZER_URL = (
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/"
    "tokenizer.json"
)


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> Downloading {dest.name}...", flush=True)
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    try:
        import forgellm
    except ImportError:
        print(
            "error: the 'forgellm' Python package is not installed.\n"
            "       Build it from the repo root with:\n"
            "           cd crates/forgellm-python && maturin develop --release",
            file=sys.stderr,
        )
        return 1

    _download_if_missing(GGUF_URL, GGUF_PATH)
    _download_if_missing(TOKENIZER_URL, TOKENIZER_PATH)

    print(f"forgellm version: {forgellm.__version__}")
    print(f"Loading model from {GGUF_PATH}...")

    model = forgellm.Model(
        str(GGUF_PATH),
        str(TOKENIZER_PATH),
    )
    print(repr(model))
    print()

    # --- 1. one-shot generate -------------------------------------------------
    print("=== generate ===")
    prompt = "The meaning of life is"
    text = model.generate(prompt, max_tokens=50, temperature=0.7)
    print(f"{prompt}{text}")
    print()

    # --- 2. streaming ---------------------------------------------------------
    print("=== stream ===")
    print("Hello world", end="", flush=True)
    for token in model.stream("Hello world", max_tokens=20, temperature=0.7):
        print(token, end="", flush=True)
    print("\n")

    # --- 3. chat --------------------------------------------------------------
    print("=== chat ===")
    response = model.chat(
        [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
        max_tokens=50,
        temperature=0.7,
    )
    print(f"assistant: {response}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
