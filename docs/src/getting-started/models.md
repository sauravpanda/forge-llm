# Supported Models

ForgeLLM supports Llama-family transformer architectures. Any model in GGUF format with one of the supported architectures should work.

## Tested Models

| Model | Params | Architecture | Quantization | Status |
|-------|--------|-------------|--------------|--------|
| SmolLM2-135M-Instruct | 135M | Llama | Q8_0 | **Verified** (46.3 tok/s) |
| SmolLM2-360M-Instruct | 360M | Llama | Q8_0 | **Verified** (17.5 tok/s) |
| SmolLM2-1.7B-Instruct | 1.7B | Llama | Q8_0, Q4_0 | Expected |
| TinyLlama-1.1B | 1.1B | Llama | Q8_0, Q4_0 | Expected |
| Llama-3.2-1B-Instruct | 1B | Llama | Q8_0, Q4_0 | Expected |
| Llama-3.2-3B-Instruct | 3B | Llama | Q8_0, Q4_0 | Expected |
| Qwen2.5-0.5B-Instruct | 0.5B | Qwen2 | Q8_0 | **Verified** (12.0 tok/s) |
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen2 | Q8_0, Q4_0 | Expected |
| Mistral-7B-Instruct | 7B | Mistral | Q4_0 | Expected |

## Supported Quantization Formats

| Format | Description | Size (1B model) |
|--------|-------------|-----------------|
| F32 | Full precision | ~4 GB |
| F16 | Half precision | ~2 GB |
| BF16 | Brain floating point | ~2 GB |
| Q8_0 | 8-bit quantized | ~1 GB |
| Q8_K | 8-bit K-quant | ~1 GB |
| Q6_K | 6-bit K-quant | ~0.82 GB |
| Q5_K | 5-bit K-quant | ~0.69 GB |
| Q4_0 | 4-bit quantized | ~0.5 GB |
| Q4_1 | 4-bit with min | ~0.55 GB |
| Q4_K | 4-bit K-quant | ~0.56 GB |
| Q3_K | 3-bit K-quant | ~0.43 GB |
| Q2_K | 2-bit K-quant | ~0.33 GB |

## Adding New Architectures

ForgeLLM uses a modular architecture detection system. Adding a new Llama-variant model typically requires:

1. Adding architecture detection in `forgellm-frontend/src/config.rs`
2. If the model uses non-standard layer names, adding GGUF name mappings in `weight_loader.rs`

Models with different attention mechanisms (e.g., sliding window, MQA with different grouping) may require additional interpreter changes.
