# Frontends

ForgeLLM supports two model file formats: GGUF and SafeTensors.

## GGUF

GGUF (GGML Universal File) is the standard format for quantized LLM weights, used by llama.cpp and most quantized model distributions.

### Parsing

The GGUF parser reads:
- **Header**: magic number, version (v2/v3), tensor count, metadata count
- **Metadata**: key-value pairs (architecture, hyperparameters, tokenizer data)
- **Tensor descriptors**: name, shape, quantization type, data offset

### Weight Loading

Tensor data is loaded on-demand and dequantized to f32:

| GGML Type | Block Size | Dequantization |
|-----------|-----------|----------------|
| F32 | 1 | Direct copy |
| F16 | 1 | IEEE 754 half→float conversion |
| BF16 | 1 | Upper 16 bits of f32 |
| Q8_0 | 32 | f16 scale × int8 values |
| Q4_0 | 32 | f16 scale × 4-bit signed values |
| Q4_1 | 32 | f16 scale × 4-bit unsigned + f16 min |

### Name Remapping

GGUF uses different tensor names than HuggingFace. The weight loader automatically remaps:

| GGUF | HuggingFace |
|------|-------------|
| `token_embd.weight` | `model.embed_tokens.weight` |
| `blk.N.attn_q.weight` | `model.layers.N.self_attn.q_proj.weight` |
| `blk.N.ffn_gate.weight` | `model.layers.N.mlp.gate_proj.weight` |
| `output_norm.weight` | `model.norm.weight` |
| `output.weight` | `lm_head.weight` |

## SafeTensors

SafeTensors is a simple format used by HuggingFace for non-quantized models.

### Parsing

The parser reads the JSON header (tensor metadata) without loading tensor data. Combined with a `config.json` file, it provides the same information as a GGUF file.
