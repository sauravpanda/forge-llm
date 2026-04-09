use std::fs;
use std::io::{self, Cursor, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use forgellm_frontend::{config::HFConfig, gguf, graph_builder, ir::ModelConfig, weight_loader};
use forgellm_runtime::{interpreter, kv_cache::KVCache, sampling, tokenizer::Tokenizer};

#[derive(Parser)]
#[command(name = "forge")]
#[command(about = "Compile your LLMs, don't interpret them.")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a model into optimized Rust source
    Compile {
        /// Path to GGUF model file or directory with config.json
        #[arg(long)]
        model: String,

        /// Target backend: cpu, wasm, gpu
        #[arg(long, default_value = "cpu")]
        target: String,

        /// Output path for the generated source
        #[arg(long)]
        output: String,
    },

    /// Inspect a model file and show its architecture info
    Info {
        /// Path to GGUF model file or directory with config.json
        model: String,
    },

    /// Run inference on a GGUF model
    Run {
        /// Path to GGUF model file
        #[arg(long)]
        model: String,

        /// Path to tokenizer.json
        #[arg(long)]
        tokenizer: String,

        /// Input prompt
        #[arg(long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "128")]
        max_tokens: usize,

        /// Temperature (0 = greedy)
        #[arg(long, default_value = "0.0")]
        temperature: f32,

        /// Top-k sampling (0 = disabled)
        #[arg(long, default_value = "0")]
        top_k: usize,

        /// Top-p nucleus sampling (1.0 = disabled)
        #[arg(long, default_value = "1.0")]
        top_p: f32,
    },

    /// Benchmark a compiled model (not yet implemented)
    Bench {
        /// Path to compiled model
        model: String,

        /// Number of iterations
        #[arg(long, default_value = "100")]
        iterations: usize,
    },

    /// Start an OpenAI-compatible API server (not yet implemented)
    Serve {
        /// Path to compiled model
        model: String,

        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            model,
            target,
            output,
        } => cmd_compile(&model, &target, &output)?,

        Commands::Info { model } => cmd_info(&model)?,

        Commands::Run {
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
        } => cmd_run(
            &model,
            &tokenizer,
            &prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
        )?,

        Commands::Bench {
            model, iterations, ..
        } => {
            println!("forge bench is not yet implemented (model={model}, iterations={iterations})");
        }

        Commands::Serve { model, port } => {
            println!("forge serve is not yet implemented (model={model}, port={port})");
        }
    }

    Ok(())
}

/// Load a ModelConfig from a GGUF file or HF config.json directory.
fn load_model_config(model_path: &str) -> Result<ModelConfig> {
    let path = Path::new(model_path);

    if path.extension().is_some_and(|ext| ext == "gguf") {
        // GGUF file
        let data = fs::read(path).with_context(|| format!("failed to read {model_path}"))?;
        let gguf_file =
            gguf::parse(Cursor::new(&data)).with_context(|| "failed to parse GGUF file")?;

        let arch_name = gguf_file.architecture().unwrap_or("unknown").to_string();

        // Extract config from GGUF metadata
        let prefix = &arch_name;
        let hidden_size = gguf_file
            .get_u32(&format!("{prefix}.embedding_length"))
            .map(|v| v as usize)
            .unwrap_or(0);
        let num_layers = gguf_file
            .get_u32(&format!("{prefix}.block_count"))
            .map(|v| v as usize)
            .unwrap_or(0);
        let num_heads = gguf_file
            .get_u32(&format!("{prefix}.attention.head_count"))
            .map(|v| v as usize)
            .unwrap_or(0);
        let num_kv_heads = gguf_file
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .map(|v| v as usize)
            .unwrap_or(num_heads);

        if hidden_size == 0 || num_layers == 0 || num_heads == 0 {
            bail!("could not extract model config from GGUF metadata");
        }

        let head_dim = hidden_size / num_heads;
        let intermediate_size = gguf_file
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 4);
        let vocab_size = gguf_file
            .get_array_len("tokenizer.ggml.tokens")
            .or_else(|| {
                gguf_file
                    .get_u64(&format!("{prefix}.vocab_size"))
                    .map(|v| v as usize)
            })
            .unwrap_or(32000);
        let max_seq_len = gguf_file
            .get_u64(&format!("{prefix}.context_length"))
            .map(|v| v as usize)
            .unwrap_or(2048);
        let rms_norm_eps = gguf_file
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let rope_theta = gguf_file
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10000.0);

        let architecture = match arch_name.as_str() {
            "llama" => forgellm_frontend::ir::Architecture::Llama,
            "qwen2" => forgellm_frontend::ir::Architecture::Qwen2,
            "mistral" => forgellm_frontend::ir::Architecture::Mistral,
            "phi3" | "phi" => forgellm_frontend::ir::Architecture::Phi3,
            "gemma" | "gemma2" => forgellm_frontend::ir::Architecture::Gemma,
            _ => bail!("unsupported GGUF architecture: {arch_name}"),
        };

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
            dtype: forgellm_frontend::ir::DType::F16,
        })
    } else if path.is_dir() {
        // HuggingFace directory with config.json
        let config_path = path.join("config.json");
        let json =
            fs::read(&config_path).with_context(|| format!("failed to read {config_path:?}"))?;
        let hf_config =
            HFConfig::from_json(&json).with_context(|| "failed to parse config.json")?;
        hf_config
            .to_model_config()
            .ok_or_else(|| anyhow::anyhow!("could not build model config from config.json"))
    } else if path.file_name().is_some_and(|f| f == "config.json") {
        // Direct config.json path
        let json = fs::read(path).with_context(|| format!("failed to read {model_path}"))?;
        let hf_config =
            HFConfig::from_json(&json).with_context(|| "failed to parse config.json")?;
        hf_config
            .to_model_config()
            .ok_or_else(|| anyhow::anyhow!("could not build model config from config.json"))
    } else {
        bail!("unrecognized model path: expected .gguf file or directory with config.json");
    }
}

fn cmd_compile(model_path: &str, target: &str, output_path: &str) -> Result<()> {
    println!("Loading model config from {model_path}...");
    let config = load_model_config(model_path)?;

    println!(
        "Model: {} | {} layers | hidden={} | heads={} | kv_heads={} | vocab={}",
        config.architecture,
        config.num_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.vocab_size,
    );

    println!("Building computation graph...");
    let graph =
        graph_builder::build_graph(&config).with_context(|| "failed to build computation graph")?;
    println!(
        "Graph: {} nodes, {} weights",
        graph.len(),
        graph.weights.len()
    );

    println!("Optimizing...");
    let optimized = forgellm_optimizer::optimize(&graph);

    println!("Generating {target} code...");
    let code = match target {
        "cpu" => forgellm_codegen_cpu::generate(&optimized)
            .map_err(|e| anyhow::anyhow!("CPU codegen failed: {e}"))?,
        _ => bail!("unsupported target: {target} (supported: cpu)"),
    };

    fs::write(output_path, &code)
        .with_context(|| format!("failed to write output to {output_path}"))?;

    println!("Wrote {} bytes to {output_path}", code.len());
    println!("Done.");

    Ok(())
}

fn cmd_info(model_path: &str) -> Result<()> {
    let config = load_model_config(model_path)?;

    println!("Architecture: {}", config.architecture);
    println!("Hidden size:  {}", config.hidden_size);
    println!("Intermediate: {}", config.intermediate_size);
    println!("Layers:       {}", config.num_layers);
    println!("Attn heads:   {}", config.num_attention_heads);
    println!("KV heads:     {}", config.num_kv_heads);
    println!("Head dim:     {}", config.head_dim);
    println!("Vocab size:   {}", config.vocab_size);
    println!("Max seq len:  {}", config.max_seq_len);
    println!("RMS norm eps: {:e}", config.rms_norm_eps);
    println!("RoPE theta:   {:e}", config.rope_theta);
    println!("Dtype:        {}", config.dtype);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_run(
    model_path: &str,
    tokenizer_path: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<()> {
    // Load tokenizer
    eprintln!("Loading tokenizer from {tokenizer_path}...");
    let tokenizer =
        Tokenizer::from_file(tokenizer_path).with_context(|| "failed to load tokenizer")?;
    eprintln!("Vocab size: {}", tokenizer.vocab_size());

    // Load model config from GGUF
    eprintln!("Loading model from {model_path}...");
    let config = load_model_config(model_path)?;
    eprintln!(
        "Model: {} | {} layers | hidden={} | vocab={}",
        config.architecture, config.num_layers, config.hidden_size, config.vocab_size,
    );

    // Build computation graph
    let graph =
        graph_builder::build_graph(&config).with_context(|| "failed to build computation graph")?;

    // Load weights
    eprintln!("Loading weights...");
    let start = Instant::now();
    let data = fs::read(model_path).with_context(|| format!("failed to read {model_path}"))?;
    let gguf_file = gguf::parse(Cursor::new(&data)).with_context(|| "failed to parse GGUF file")?;
    let weights = weight_loader::load_all(&mut Cursor::new(&data), &gguf_file)
        .with_context(|| "failed to load weights")?;
    eprintln!(
        "Loaded {} tensors ({:.1} MB) in {:.1}s",
        weights.len(),
        weights.memory_bytes() as f64 / 1e6,
        start.elapsed().as_secs_f64(),
    );

    // Initialize KV cache
    let mut cache = KVCache::with_capacity(
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        config.max_seq_len,
    );

    // Sampling config
    let sampling_config = if temperature == 0.0 {
        sampling::SamplingConfig::greedy()
    } else {
        sampling::SamplingConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.0,
        }
    };

    // Encode prompt
    let prompt_tokens = tokenizer
        .encode(prompt)
        .with_context(|| "failed to encode prompt")?;
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    // Process prompt tokens (prefill)
    eprintln!("Generating...\n");
    print!("{prompt}");
    io::stdout().flush()?;

    let prefill_start = Instant::now();
    let mut next_token = 0u32;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let logits = interpreter::forward(token, pos, &graph, &weights, &mut cache);
        cache.advance();
        next_token = sampling::sample(&logits, &sampling_config, pos as u64);
    }
    let prefill_time = prefill_start.elapsed();

    // Generate tokens
    let mut generated_tokens = vec![next_token];
    let eos_id = tokenizer.eos_token_id();
    let gen_start = Instant::now();

    for i in 0..max_tokens {
        // Decode and print the token
        if let Ok(text) = tokenizer.decode_one(next_token) {
            print!("{text}");
            io::stdout().flush()?;
        }

        // Check for EOS
        if eos_id.is_some_and(|eos| next_token == eos) {
            break;
        }

        // Forward pass
        let pos = prompt_tokens.len() + i;
        let logits = interpreter::forward(next_token, pos, &graph, &weights, &mut cache);
        cache.advance();

        // Sample next token
        next_token = sampling::sample(&logits, &sampling_config, (pos + 1) as u64);
        generated_tokens.push(next_token);
    }

    let gen_time = gen_start.elapsed();
    println!();

    // Stats
    let total_gen = generated_tokens.len();
    let tok_per_sec = if gen_time.as_secs_f64() > 0.0 {
        total_gen as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    eprintln!("\n--- Stats ---");
    eprintln!(
        "Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        prompt_tokens.len(),
        prefill_time.as_secs_f64(),
        prompt_tokens.len() as f64 / prefill_time.as_secs_f64(),
    );
    eprintln!(
        "Generate: {total_gen} tokens in {:.2}s ({tok_per_sec:.1} tok/s)",
        gen_time.as_secs_f64(),
    );

    Ok(())
}
