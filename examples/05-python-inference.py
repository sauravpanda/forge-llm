"""Minimal example using the forgellm Python bindings.

Install first:
    cd crates/forgellm-python
    maturin develop --release   # or: maturin build --release

Requires Python 3.11+ with dev headers.
"""
import os
import sys
from pathlib import Path

try:
    import forgellm
except ImportError:
    print("Error: forgellm not installed. Run:")
    print("  cd crates/forgellm-python && maturin develop")
    sys.exit(1)


def main():
    cache = Path.home() / ".cache" / "forgellm-examples"
    cache.mkdir(parents=True, exist_ok=True)

    gguf_path = cache / "SmolLM2-135M-Instruct-Q8_0.gguf"
    tok_path = cache / "smollm2-tokenizer.json"

    # Auto-download if missing
    if not gguf_path.exists():
        print("Downloading SmolLM2-135M Q8_0...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf",
            gguf_path,
        )

    if not tok_path.exists():
        print("Downloading tokenizer...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json",
            tok_path,
        )

    # Load model
    print(f"Loading {gguf_path.name}...")
    model = forgellm.Model(str(gguf_path), str(tok_path))

    # Generate (full response at once)
    print("\n=== Generate ===")
    text = model.generate("The meaning of life is", max_tokens=50)
    print(text)

    # Stream (token by token)
    print("\n=== Stream ===")
    for token in model.stream("Hello world, my name is", max_tokens=30):
        print(token, end="", flush=True)
    print()

    # Chat (with role-based messages)
    print("\n=== Chat ===")
    response = model.chat(
        [
            {"role": "user", "content": "What is 2+2? Answer in one sentence."},
        ],
        max_tokens=50,
    )
    print(response)


if __name__ == "__main__":
    main()
