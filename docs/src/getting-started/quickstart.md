# Quick Start

## 1. Download a Model

ForgeLLM works with GGUF model files. SmolLM2-135M is a great starting point:

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Download Q8_0 quantized SmolLM2-135M
python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('bartowski/SmolLM2-135M-Instruct-GGUF', 'SmolLM2-135M-Instruct-Q8_0.gguf'))
"

# Download the tokenizer
python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('HuggingFaceTB/SmolLM2-135M-Instruct', 'tokenizer.json'))
"
```

## 2. Inspect the Model

```bash
forge info path/to/SmolLM2-135M-Instruct-Q8_0.gguf
```

Output:
```
Architecture: Llama
Hidden size:  576
Intermediate: 1536
Layers:       30
Attn heads:   9
KV heads:     3
Head dim:     64
Vocab size:   49152
Max seq len:  8192
```

## 3. Run Inference

```bash
forge run \
  --model path/to/SmolLM2-135M-Instruct-Q8_0.gguf \
  --tokenizer path/to/tokenizer.json \
  --prompt "Explain quantum computing in simple terms"
```

### Sampling Options

```bash
# Greedy (default, deterministic)
forge run --model model.gguf --tokenizer tokenizer.json --prompt "Hello"

# Creative (temperature + top-p)
forge run --model model.gguf --tokenizer tokenizer.json \
  --prompt "Write a poem about" \
  --temperature 0.8 --top-p 0.9

# Top-k sampling
forge run --model model.gguf --tokenizer tokenizer.json \
  --prompt "Once upon a time" \
  --temperature 0.7 --top-k 40
```

## 4. Compile to Rust Source (Experimental)

Generate standalone Rust code from a model:

```bash
forge compile \
  --model path/to/SmolLM2-135M-Instruct-Q8_0.gguf \
  --target cpu \
  --output smollm_model.rs
```

This generates a `.rs` file with all transformer kernels baked in, specialized to the exact model dimensions.
