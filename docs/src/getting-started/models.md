# Supported Models

ForgeLLM supports Llama-family transformer architectures. Any model in GGUF format with one of the supported architectures should work.

## Tested Models

| Model | Params | Architecture | Quantization | Status |
|-------|--------|-------------|--------------|--------|
| SmolLM2-135M-Instruct | 135M | Llama | Q8_0 | Verified |
| SmolLM2-360M-Instruct | 360M | Llama | Q8_0 | Expected |
| SmolLM2-1.7B-Instruct | 1.7B | Llama | Q8_0, Q4_0 | Expected |
| TinyLlama-1.1B | 1.1B | Llama | Q8_0, Q4_0 | Expected |
| Llama-3.2-1B-Instruct | 1B | Llama | Q8_0, Q4_0 | Expected |
| Llama-3.2-3B-Instruct | 3B | Llama | Q8_0, Q4_0 | Expected |
| Qwen2.5-0.5B-Instruct | 0.5B | Qwen2 | Q8_0 | Expected |
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen2 | Q8_0, Q4_0 | Expected |
| Mistral-7B-Instruct | 7B | Mistral | Q4_0 | Expected |

## Supported Quantization Formats

| Format | Description | Size (1B model) |
|--------|-------------|-----------------|
| F32 | Full precision | ~4 GB |
| F16 | Half precision | ~2 GB |
| BF16 | Brain floating point | ~2 GB |
| Q8_0 | 8-bit quantized | ~1 GB |
| Q4_0 | 4-bit quantized | ~0.5 GB |
| Q4_1 | 4-bit with min | ~0.55 GB |

## Adding New Architectures

ForgeLLM uses a modular architecture detection system. Adding a new Llama-variant model typically requires:

1. Adding architecture detection in `forgellm-frontend/src/config.rs`
2. If the model uses non-standard layer names, adding GGUF name mappings in `weight_loader.rs`

Models with different attention mechanisms (e.g., sliding window, MQA with different grouping) may require additional interpreter changes.
