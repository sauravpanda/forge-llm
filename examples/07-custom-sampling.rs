//! 07-custom-sampling.rs — Implement a custom sampler against the runtime API.
//!
//! Demonstrates how to bypass `forgellm_runtime::sampling::sample` and drive
//! the logits yourself. Here we implement a simple adaptive-temperature
//! sampler inspired by Mirostat: we track the recent entropy of sampled
//! tokens and adjust the temperature to keep it close to a target value.
//!
//! The goal of the example is not to ship a production sampler — it is to
//! show exactly which types and functions you touch to plug a novel
//! sampling strategy into the ForgeLLM runtime.
//!
//! This file is built as a binary of the neighbouring `06-library-usage`
//! project so that both share a single `Cargo.toml`. Run it with:
//!
//! ```bash
//! cd examples/06-library-usage
//! cargo run --release --bin custom-sampling -- \
//!     ~/.cache/forgellm-examples/SmolLM2-135M-Instruct-Q8_0.gguf \
//!     ~/.cache/forgellm-examples/tokenizer.json \
//!     "The meaning of life is"
//! ```

use std::env;
use std::io::{self, Cursor, Write};

use anyhow::{bail, Context, Result};
use forgellm_frontend::gguf;
use forgellm_frontend::graph_builder;
use forgellm_frontend::ir::{Architecture, DType, ModelConfig};
use forgellm_frontend::weight_loader;
use forgellm_runtime::interpreter;
use forgellm_runtime::kv_cache::KVCache;
use forgellm_runtime::tokenizer::Tokenizer;

/// Adaptive-temperature sampler that tracks the entropy of recent picks
/// and adjusts temperature to steer toward a target entropy.
struct AdaptiveSampler {
    /// Target entropy (nats). Higher = more creative, lower = more focused.
    target_entropy: f32,
    /// Current temperature; clamped to `[min_temp, max_temp]`.
    temperature: f32,
    min_temp: f32,
    max_temp: f32,
    /// Learning rate for temperature updates.
    lr: f32,
    /// RNG seed (monotonically incremented per step).
    seed: u64,
}

impl AdaptiveSampler {
    fn new(target_entropy: f32) -> Self {
        Self {
            target_entropy,
            temperature: 1.0,
            min_temp: 0.1,
            max_temp: 2.0,
            lr: 0.1,
            seed: 0xC0FFEE,
        }
    }

    /// Sample a token from raw logits using the current temperature,
    /// then update the temperature based on the observed entropy.
    fn sample(&mut self, logits: &[f32]) -> u32 {
        // 1. Temperature-scaled softmax.
        let inv_t = 1.0 / self.temperature.max(1e-4);
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits
            .iter()
            .map(|&l| ((l - max_logit) * inv_t).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // 2. Measure entropy in nats: H = -sum(p * ln p).
        let entropy: f32 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        // 3. Sample one token from the (p_i) distribution.
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = ((self.seed >> 33) as f32) / ((1u64 << 31) as f32);
        let mut cumulative = 0.0f32;
        let mut picked = (probs.len() - 1) as u32;
        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                picked = i as u32;
                break;
            }
        }

        // 4. Online update: if entropy is too high, cool down (temperature↓);
        //    if too low, heat up (temperature↑).
        let error = entropy - self.target_entropy;
        self.temperature = (self.temperature - self.lr * error).clamp(self.min_temp, self.max_temp);

        picked
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <model.gguf> <tokenizer.json> <prompt> [max_tokens] [target_entropy]",
            args.first()
                .map(String::as_str)
                .unwrap_or("custom-sampling")
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
        .unwrap_or(60);
    let target_entropy: f32 = args
        .get(5)
        .map(|s| s.parse())
        .transpose()
        .context("target_entropy must be a float (nats)")?
        .unwrap_or(3.0);

    let config = load_config_from_gguf(model_path)?;
    let graph = graph_builder::build_graph(&config)
        .map_err(|e| anyhow::anyhow!("build_graph failed: {e}"))?;
    let (_gguf_file, weights) = weight_loader::load_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("load_from_file failed: {e}"))?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

    let mut cache = KVCache::with_capacity(
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        config.max_seq_len,
    );

    let mut sampler = AdaptiveSampler::new(target_entropy);

    let prompt_tokens = tokenizer
        .encode(prompt)
        .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?;
    if prompt_tokens.len() + max_tokens >= config.max_seq_len {
        bail!(
            "prompt + max_tokens ({} + {}) would exceed max context length {}",
            prompt_tokens.len(),
            max_tokens,
            config.max_seq_len
        );
    }

    print!("{prompt}");
    io::stdout().flush().ok();

    let mut next_token = 0u32;
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        let logits = interpreter::forward(tok, pos, &graph, &weights, &mut cache);
        cache.advance();
        next_token = sampler.sample(&logits);
    }

    let eos = tokenizer.eos_token_id();
    for i in 0..max_tokens {
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
        next_token = sampler.sample(&logits);
    }
    println!();
    eprintln!(
        "\n--- final temperature: {:.3} (target entropy {:.2} nats) ---",
        sampler.temperature, sampler.target_entropy
    );

    Ok(())
}

/// Extract a `ModelConfig` from GGUF metadata. See `06-library-usage/src/main.rs`
/// for a line-by-line walkthrough of this helper.
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
