//! 06-library-usage — Embed `forgellm-runtime` as a library in a Rust app.
//!
//! This standalone Cargo project shows the minimal code path to:
//!   1. Parse a GGUF model and build a `ModelConfig`.
//!   2. Construct the computation graph via `forgellm-frontend`.
//!   3. Load quantized weights with `weight_loader`.
//!   4. Run the interpreter manually with a `KVCache` + sampler.
//!
//! It is the Rust analogue of `examples/05-python-inference.py`. Unlike the
//! AOT-compiled binary produced by `forge compile`, this path walks the IR
//! graph at runtime and is the simplest way to embed inference in an
//! existing Rust application without spawning a subprocess.
//!
//! Run it with:
//!
//! ```bash
//! cd examples/06-library-usage
//! cargo run --release -- \
//!     ~/.cache/forgellm-examples/SmolLM2-135M-Instruct-Q8_0.gguf \
//!     ~/.cache/forgellm-examples/tokenizer.json \
//!     "The meaning of life is"
//! ```

use std::env;
use std::io::{self, Cursor, Write};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use forgellm_frontend::gguf;
use forgellm_frontend::graph_builder;
use forgellm_frontend::ir::{Architecture, DType, ModelConfig};
use forgellm_frontend::weight_loader;
use forgellm_runtime::interpreter;
use forgellm_runtime::kv_cache::KVCache;
use forgellm_runtime::sampling::{self, SamplingConfig};
use forgellm_runtime::tokenizer::Tokenizer;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <model.gguf> <tokenizer.json> <prompt> [max_tokens]",
            args.first().map(String::as_str).unwrap_or("example")
        );
        std::process::exit(2);
    }
    let model_path = &args[1];
    let tokenizer_path = &args[2];
    let prompt = &args[3];
    let max_tokens: usize = args
        .get(4)
        .map(|s| s.parse())
        .transpose()
        .context("max_tokens must be a positive integer")?
        .unwrap_or(50);

    // ---- 1. Model config ----
    println!("Loading config from {model_path}...");
    let config = load_config_from_gguf(model_path)?;
    println!(
        "Model: {:?} | {} layers | hidden={} | vocab={}",
        config.architecture, config.num_layers, config.hidden_size, config.vocab_size
    );

    // ---- 2. Build computation graph ----
    let graph = graph_builder::build_graph(&config)
        .map_err(|e| anyhow::anyhow!("failed to build graph: {e}"))?;

    // ---- 3. Load weights (mmap + dequantize as needed) ----
    println!("Loading weights...");
    let load_start = Instant::now();
    let (_gguf_file, weights) = weight_loader::load_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("failed to load weights: {e}"))?;
    println!(
        "  {} tensors ({:.1} MB) in {:.2}s",
        weights.len(),
        weights.memory_bytes() as f64 / 1e6,
        load_start.elapsed().as_secs_f64()
    );

    // ---- 4. Tokenizer ----
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    // ---- 5. KV cache + sampler ----
    let mut cache = KVCache::with_capacity(
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        config.max_seq_len,
    );
    let sampling_config = SamplingConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.95,
        repetition_penalty: 1.1,
    };

    // ---- 6. Prefill ----
    let prompt_tokens = tokenizer
        .encode(prompt)
        .map_err(|e| anyhow::anyhow!("failed to encode prompt: {e}"))?;
    if prompt_tokens.len() >= config.max_seq_len {
        bail!(
            "prompt ({} tokens) exceeds max context length ({})",
            prompt_tokens.len(),
            config.max_seq_len
        );
    }

    print!("{prompt}");
    io::stdout().flush().ok();

    let mut next_token = 0u32;
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        let logits = interpreter::forward(tok, pos, &graph, &weights, &mut cache);
        cache.advance();
        next_token = sampling::sample(&logits, &sampling_config, pos as u64);
    }

    // ---- 7. Decode loop ----
    let eos = tokenizer.eos_token_id();
    let effective_max = max_tokens.min(config.max_seq_len - prompt_tokens.len());
    let gen_start = Instant::now();
    let mut generated = 0usize;

    for i in 0..effective_max {
        if let Ok(text) = tokenizer.decode_one(next_token) {
            print!("{text}");
            io::stdout().flush().ok();
        }
        if eos.is_some_and(|e| next_token == e) {
            break;
        }

        let pos = prompt_tokens.len() + i;
        let logits = interpreter::forward(next_token, pos, &graph, &weights, &mut cache);
        cache.advance();
        next_token = sampling::sample(&logits, &sampling_config, (pos + 1) as u64);
        generated += 1;
    }
    println!();

    let elapsed = gen_start.elapsed().as_secs_f64();
    eprintln!(
        "\n--- {generated} tokens in {elapsed:.2}s ({:.1} tok/s) ---",
        generated as f64 / elapsed.max(1e-9),
    );

    Ok(())
}

/// Extract a `ModelConfig` from GGUF metadata.
///
/// Mirrors the logic used internally by the `forge` CLI and Python bindings,
/// kept here so this example is fully self-contained and shows the frontend API.
fn load_config_from_gguf(path: &str) -> Result<ModelConfig> {
    let data = std::fs::read(path).with_context(|| format!("failed to read {path}"))?;
    let g = gguf::parse(Cursor::new(&data)).context("failed to parse GGUF file")?;

    let arch_name = g.architecture().unwrap_or("unknown").to_string();
    let p = &arch_name;

    let hidden_size = g
        .get_u32(&format!("{p}.embedding_length"))
        .map(|v| v as usize)
        .unwrap_or(0);
    let num_layers = g
        .get_u32(&format!("{p}.block_count"))
        .map(|v| v as usize)
        .unwrap_or(0);
    let num_heads = g
        .get_u32(&format!("{p}.attention.head_count"))
        .map(|v| v as usize)
        .unwrap_or(0);
    let num_kv_heads = g
        .get_u32(&format!("{p}.attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(num_heads);

    if hidden_size == 0 || num_layers == 0 || num_heads == 0 {
        bail!("could not extract model dimensions from GGUF metadata");
    }

    let head_dim = hidden_size / num_heads;
    let intermediate_size = g
        .get_u32(&format!("{p}.feed_forward_length"))
        .map(|v| v as usize)
        .unwrap_or(hidden_size * 4);
    let vocab_size = g
        .get_array_len("tokenizer.ggml.tokens")
        .or_else(|| g.get_u64(&format!("{p}.vocab_size")).map(|v| v as usize))
        .unwrap_or(32000);
    let max_seq_len = g
        .get_u64(&format!("{p}.context_length"))
        .map(|v| v as usize)
        .unwrap_or(2048);
    let rms_norm_eps = g
        .get_f32(&format!("{p}.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);
    let rope_theta = g.get_f32(&format!("{p}.rope.freq_base")).unwrap_or(10000.0);

    let architecture = match arch_name.as_str() {
        "llama" => Architecture::Llama,
        "qwen2" => Architecture::Qwen2,
        "mistral" => Architecture::Mistral,
        "phi3" | "phi" => Architecture::Phi3,
        "gemma" | "gemma2" => Architecture::Gemma,
        other => bail!("unsupported GGUF architecture '{other}'"),
    };

    let sliding_window_size = g
        .get_u32(&format!("{p}.attention.sliding_window"))
        .map(|v| v as usize);
    let qkv_bias = matches!(architecture, Architecture::Qwen2);

    Ok(ModelConfig {
        architecture,
        hidden_size,
        intermediate_size,
        num_layers,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        vocab_size,
        max_seq_len,
        rms_norm_eps,
        rope_theta,
        dtype: DType::F16,
        sliding_window_size,
        qkv_bias,
    })
}
