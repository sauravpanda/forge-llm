use std::fs;
use std::io::{self, BufRead, BufReader, Cursor, Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use forgellm_frontend::{
    config::HFConfig, gguf, graph_builder, hub, ir::ModelConfig, weight_loader,
};
use forgellm_runtime::{
    chat::ChatTemplate, interpreter, kv_cache::KVCache, sampling, tokenizer::Tokenizer,
};

#[derive(Parser)]
#[command(name = "forge")]
#[command(about = "ForgeLLM — Compile your LLMs, don't interpret them.")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(long_version = concat!(
    env!("CARGO_PKG_VERSION"),
    "\n",
    "Architectures: Llama, Qwen2, Mistral, Phi3, Gemma, StableLM\n",
    "Quantizations: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q2_K-Q8_K\n",
    "Homepage:      https://forgellm.dev\n",
    "Repository:    https://github.com/sauravpanda/forge-llm",
))]
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

    /// List GGUF models in a directory
    Models {
        /// Directory to search (default: current dir, or FORGE_MODEL_DIR)
        #[arg(default_value = ".")]
        dir: String,
    },

    /// Run inference on a GGUF model
    Run {
        /// Path to GGUF model file or HuggingFace model ID
        #[arg(long)]
        model: String,

        /// Path to tokenizer.json (auto-detected if omitted)
        #[arg(long)]
        tokenizer: Option<String>,

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

        /// Repetition penalty (1.0 = disabled, >1.0 = penalize repeats)
        #[arg(long, default_value = "1.1")]
        repeat_penalty: f32,

        /// System prompt (enables chat template formatting)
        #[arg(long)]
        system: Option<String>,

        /// Use chat template formatting
        #[arg(long)]
        chat: bool,
    },

    /// Benchmark model inference performance
    Bench {
        /// Path to GGUF model file
        #[arg(long)]
        model: String,

        /// Path to tokenizer.json (auto-detected if omitted)
        #[arg(long)]
        tokenizer: Option<String>,

        /// Number of tokens to generate per run
        #[arg(long, default_value = "128")]
        num_tokens: usize,

        /// Number of benchmark runs
        #[arg(long, default_value = "3")]
        runs: usize,

        /// Prompt for benchmarking
        #[arg(long, default_value = "The quick brown fox jumps over the lazy dog")]
        prompt: String,
    },

    /// Interactive chat mode
    Chat {
        /// Path to GGUF model file
        #[arg(long)]
        model: String,

        /// Path to tokenizer.json (auto-detected if omitted)
        #[arg(long)]
        tokenizer: Option<String>,

        /// System prompt
        #[arg(long, default_value = "You are a helpful assistant.")]
        system: String,

        /// Maximum tokens per response
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Temperature (0 = greedy)
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },

    /// Start an OpenAI-compatible API server
    Serve {
        /// Path to GGUF model file
        #[arg(long)]
        model: String,

        /// Path to tokenizer.json (auto-detected if omitted)
        #[arg(long)]
        tokenizer: Option<String>,

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

        Commands::Models { dir } => cmd_models(&dir)?,

        Commands::Info { model } => cmd_info(&model)?,

        Commands::Run {
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            system,
            chat,
        } => {
            // Apply chat template if --chat or --system is provided
            let effective_prompt = if chat || system.is_some() {
                let config = load_model_config(&model)?;
                let template = forgellm_runtime::chat::ChatTemplate::from_architecture(
                    &config.architecture.to_string(),
                );
                if let Some(ref sys) = system {
                    template.format_with_system(sys, &prompt)
                } else {
                    template.format_prompt(&prompt)
                }
            } else {
                prompt.clone()
            };
            let tok = resolve_tokenizer(&tokenizer, &model)?;
            cmd_run(
                &model,
                &tok,
                &effective_prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                repeat_penalty,
            )?
        }

        Commands::Bench {
            model,
            tokenizer,
            num_tokens,
            runs,
            prompt,
        } => {
            let tok = resolve_tokenizer(&tokenizer, &model)?;
            cmd_bench(&model, &tok, &prompt, num_tokens, runs)?
        }

        Commands::Chat {
            model,
            tokenizer,
            system,
            max_tokens,
            temperature,
        } => {
            let tok = resolve_tokenizer(&tokenizer, &model)?;
            cmd_chat(&model, &tok, &system, max_tokens, temperature)?
        }

        Commands::Serve {
            model,
            tokenizer,
            port,
        } => {
            let tok = resolve_tokenizer(&tokenizer, &model)?;
            cmd_serve(&model, &tok, port)?
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

/// Resolve model and tokenizer paths — handles HF model IDs.
/// If model is a HF ID (org/model), downloads GGUF + tokenizer.
/// Returns (model_path, tokenizer_path).
fn resolve_paths(model: &str, tokenizer: &str) -> Result<(String, String)> {
    if hub::is_hf_model_id(model) {
        eprintln!("Detected HuggingFace model ID: {model}");
        let (gguf_path, tok_path) =
            hub::resolve_model(model).with_context(|| "failed to resolve HF model")?;
        eprintln!("Model: {}", gguf_path.display());
        eprintln!("Tokenizer: {}", tok_path.display());
        Ok((
            gguf_path.to_string_lossy().into(),
            tok_path.to_string_lossy().into(),
        ))
    } else {
        Ok((model.to_string(), tokenizer.to_string()))
    }
}

/// Find tokenizer.json near a model file.
/// Searches: same directory, parent directory, sibling directories.
fn find_tokenizer(model_path: &str) -> Option<String> {
    let model = Path::new(model_path);

    // Same directory as model
    if let Some(dir) = model.parent() {
        let candidate = dir.join("tokenizer.json");
        if candidate.exists() {
            return Some(candidate.to_string_lossy().into());
        }
        // Parent directory
        if let Some(parent) = dir.parent() {
            let candidate = parent.join("tokenizer.json");
            if candidate.exists() {
                return Some(candidate.to_string_lossy().into());
            }
        }
    }

    None
}

/// Resolve tokenizer path — use provided path, or auto-find.
fn resolve_tokenizer(tokenizer: &Option<String>, model_path: &str) -> Result<String> {
    if let Some(tok) = tokenizer {
        return Ok(tok.clone());
    }

    // Try to auto-find
    if let Some(found) = find_tokenizer(model_path) {
        eprintln!("Auto-detected tokenizer: {found}");
        return Ok(found);
    }

    bail!(
        "no tokenizer found. Provide --tokenizer path/to/tokenizer.json\n\
         Tip: download with: python3 -c \"from huggingface_hub import hf_hub_download; \
         print(hf_hub_download('MODEL_ID', 'tokenizer.json'))\""
    )
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
    model_path_raw: &str,
    tokenizer_path_raw: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
) -> Result<()> {
    let (model_path, tokenizer_path) = resolve_paths(model_path_raw, tokenizer_path_raw)?;
    let model_path = model_path.as_str();
    let tokenizer_path = tokenizer_path.as_str();

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

    // Load weights (mmap)
    eprintln!("Loading weights...");
    let start = Instant::now();
    let (_gguf_file, weights) =
        weight_loader::load_from_file(model_path).with_context(|| "failed to load weights")?;
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
        sampling::SamplingConfig {
            repetition_penalty: repeat_penalty,
            ..sampling::SamplingConfig::greedy()
        }
    } else {
        sampling::SamplingConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: repeat_penalty,
        }
    };

    // Encode prompt
    let prompt_tokens = tokenizer
        .encode(prompt)
        .with_context(|| "failed to encode prompt")?;
    let total_budget = config.max_seq_len.min(prompt_tokens.len() + max_tokens);
    eprintln!(
        "Prompt: {} tokens | Budget: {}/{} context",
        prompt_tokens.len(),
        total_budget,
        config.max_seq_len,
    );
    if prompt_tokens.len() >= config.max_seq_len {
        bail!(
            "prompt ({} tokens) exceeds max context length ({})",
            prompt_tokens.len(),
            config.max_seq_len,
        );
    }
    let effective_max_tokens = max_tokens.min(config.max_seq_len - prompt_tokens.len());

    // Process prompt tokens (prefill)
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
    eprintln!(
        "Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        prompt_tokens.len(),
        prefill_time.as_secs_f64(),
        prompt_tokens.len() as f64 / prefill_time.as_secs_f64(),
    );

    // Generate tokens
    let mut generated_tokens: Vec<u32> = prompt_tokens.clone();
    generated_tokens.push(next_token);
    let eos_id = tokenizer.eos_token_id();
    let gen_start = Instant::now();

    for i in 0..effective_max_tokens {
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
        let mut logits = interpreter::forward(next_token, pos, &graph, &weights, &mut cache);
        cache.advance();

        // Apply repetition penalty
        if repeat_penalty != 1.0 {
            sampling::apply_repetition_penalty(&mut logits, &generated_tokens, repeat_penalty);
        }

        // Sample next token
        next_token = sampling::sample(&logits, &sampling_config, (pos + 1) as u64);
        generated_tokens.push(next_token);
    }

    let gen_time = gen_start.elapsed();
    println!();

    // Stats
    let gen_count = generated_tokens.len() - prompt_tokens.len();
    let tok_per_sec = if gen_time.as_secs_f64() > 0.0 {
        gen_count as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    eprintln!("\n--- Stats ---");
    eprintln!(
        "Generate: {gen_count} tokens in {:.2}s ({tok_per_sec:.1} tok/s)",
        gen_time.as_secs_f64(),
    );
    eprintln!(
        "Context: {}/{} tokens used",
        prompt_tokens.len() + gen_count,
        config.max_seq_len,
    );

    Ok(())
}

fn cmd_serve(model_path: &str, tokenizer_path: &str, port: u16) -> Result<()> {
    // Load model and tokenizer
    eprintln!("Loading tokenizer from {tokenizer_path}...");
    let tokenizer =
        Tokenizer::from_file(tokenizer_path).with_context(|| "failed to load tokenizer")?;

    eprintln!("Loading model from {model_path}...");
    let config = load_model_config(model_path)?;
    let graph =
        graph_builder::build_graph(&config).with_context(|| "failed to build computation graph")?;

    let (_gguf_file, weights) =
        weight_loader::load_from_file(model_path).with_context(|| "failed to load weights")?;

    eprintln!(
        "Model loaded: {} | {} layers | {:.0} MB",
        config.architecture,
        config.num_layers,
        weights.memory_bytes() as f64 / 1e6,
    );

    // Start HTTP server
    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).with_context(|| format!("failed to bind {addr}"))?;
    eprintln!("Serving on http://localhost:{port}");
    eprintln!("Endpoints:");
    eprintln!("  POST /v1/completions");
    eprintln!("  POST /v1/chat/completions");
    eprintln!("  GET  /v1/models");
    eprintln!("  GET  /health");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Connection error: {e}");
                continue;
            }
        };
        let mut reader = BufReader::new(&stream);

        // Read request line
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() {
            continue;
        }
        let parts: Vec<&str> = request_line.trim().split(' ').collect();
        if parts.len() < 2 {
            continue;
        }
        let method = parts[0];
        let path = parts[1];

        // Read headers
        let mut content_length: usize = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            if line.trim().is_empty() {
                break;
            }
            if let Some(val) = line.strip_prefix("Content-Length: ") {
                content_length = val.trim().parse().unwrap_or(0);
            }
            if let Some(val) = line.strip_prefix("content-length: ") {
                content_length = val.trim().parse().unwrap_or(0);
            }
        }

        // Read body
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body)?;
        }

        // Route
        let (status, response_body) = match (method, path) {
            ("OPTIONS", _) => {
                // CORS preflight
                let cors = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nContent-Length: 0\r\n\r\n";
                stream.write_all(cors.as_bytes())?;
                continue;
            }

            ("GET", "/health") => {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                (
                    "200 OK",
                    format!(
                        r#"{{"status":"ok","model":"{}","timestamp":{ts}}}"#,
                        config.architecture
                    ),
                )
            }

            ("GET", "/v1/models") => {
                let model_name = config.architecture.to_string();
                (
                    "200 OK",
                    format!(
                        r#"{{"object":"list","data":[{{"id":"{model_name}","object":"model","owned_by":"forgellm","layers":{},"hidden_size":{}}}]}}"#,
                        config.num_layers, config.hidden_size,
                    ),
                )
            }

            ("POST", "/v1/completions") => {
                // Check for streaming
                let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
                    .ok()
                    .and_then(|v| v["stream"].as_bool())
                    .unwrap_or(false);

                if is_stream {
                    // Stream response directly
                    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
                    stream.write_all(header.as_bytes())?;
                    if let Err(e) = handle_completion_stream(
                        &body,
                        &config,
                        &graph,
                        &weights,
                        &tokenizer,
                        &mut stream,
                    ) {
                        let err_data = format!("data: {{\"error\":\"{e}\"}}\n\n");
                        let _ = stream.write_all(err_data.as_bytes());
                    }
                    let _ = stream.write_all(b"data: [DONE]\n\n");
                    continue;
                }

                match handle_completion(&body, &config, &graph, &weights, &tokenizer) {
                    Ok(resp) => ("200 OK", resp),
                    Err(e) => (
                        "400 Bad Request",
                        format!(r#"{{"error":{{"message":"{}"}}}}"#, e),
                    ),
                }
            }

            ("POST", "/v1/chat/completions") => {
                match handle_chat_completion(&body, &config, &graph, &weights, &tokenizer) {
                    Ok(resp) => ("200 OK", resp),
                    Err(e) => (
                        "400 Bad Request",
                        format!(r#"{{"error":{{"message":"{}"}}}}"#, e),
                    ),
                }
            }

            _ => (
                "404 Not Found",
                r#"{"error":{"message":"not found"}}"#.to_string(),
            ),
        };

        // Write response
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{response_body}",
            response_body.len()
        );
        stream.write_all(response.as_bytes())?;
    }

    Ok(())
}

fn handle_completion(
    body: &[u8],
    config: &ModelConfig,
    graph: &forgellm_frontend::ir::Graph,
    weights: &weight_loader::ModelWeights,
    tokenizer: &Tokenizer,
) -> Result<String> {
    let req: serde_json::Value = serde_json::from_slice(body)?;

    let prompt = req["prompt"].as_str().unwrap_or("Hello");
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(64) as usize;
    let temperature = req["temperature"].as_f64().unwrap_or(0.0) as f32;

    let sampling_config = if temperature == 0.0 {
        sampling::SamplingConfig::greedy()
    } else {
        sampling::SamplingConfig {
            temperature,
            top_k: req["top_k"].as_u64().unwrap_or(0) as usize,
            top_p: req["top_p"].as_f64().unwrap_or(1.0) as f32,
            repetition_penalty: 1.0,
        }
    };

    let prompt_tokens = tokenizer.encode(prompt)?;
    let mut cache = KVCache::with_capacity(
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        config
            .max_seq_len
            .min(prompt_tokens.len() + max_tokens + 16),
    );

    // Prefill
    let mut next_token = 0u32;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let logits = interpreter::forward(token, pos, graph, weights, &mut cache);
        cache.advance();
        next_token = sampling::sample(&logits, &sampling_config, pos as u64);
    }

    // Generate
    let mut generated_text = String::new();
    let eos_id = tokenizer.eos_token_id();
    let mut generated_count = 0;

    for i in 0..max_tokens {
        if let Ok(text) = tokenizer.decode_one(next_token) {
            generated_text.push_str(&text);
        }
        generated_count += 1;

        if eos_id.is_some_and(|eos| next_token == eos) {
            break;
        }

        let pos = prompt_tokens.len() + i;
        let logits = interpreter::forward(next_token, pos, graph, weights, &mut cache);
        cache.advance();
        next_token = sampling::sample(&logits, &sampling_config, (pos + 1) as u64);
    }

    // Build OpenAI-compatible response
    let response = serde_json::json!({
        "id": "cmpl-forge",
        "object": "text_completion",
        "model": config.architecture.to_string(),
        "choices": [{
            "text": generated_text,
            "index": 0,
            "finish_reason": if eos_id.is_some_and(|eos| next_token == eos) { "stop" } else { "length" },
        }],
        "usage": {
            "prompt_tokens": prompt_tokens.len(),
            "completion_tokens": generated_count,
            "total_tokens": prompt_tokens.len() + generated_count,
        }
    });

    Ok(response.to_string())
}

fn cmd_bench(
    model_path: &str,
    tokenizer_path: &str,
    prompt: &str,
    num_tokens: usize,
    runs: usize,
) -> Result<()> {
    eprintln!("Loading tokenizer...");
    let tokenizer =
        Tokenizer::from_file(tokenizer_path).with_context(|| "failed to load tokenizer")?;

    eprintln!("Loading model...");
    let config = load_model_config(model_path)?;
    let graph = graph_builder::build_graph(&config)?;

    let data = fs::read(model_path)?;
    let gguf_file = gguf::parse(Cursor::new(&data))?;
    let weights = weight_loader::load_all(&mut Cursor::new(&data), &gguf_file)?;

    let prompt_tokens = tokenizer.encode(prompt)?;

    println!(
        "Benchmark: {} | {} layers | hidden={}",
        config.architecture, config.num_layers, config.hidden_size
    );
    println!(
        "Prompt: {} tokens | Generate: {} tokens | Runs: {}",
        prompt_tokens.len(),
        num_tokens,
        runs
    );
    println!();

    let sampling_config = sampling::SamplingConfig::greedy();
    let mut prefill_times = Vec::with_capacity(runs);
    let mut gen_times = Vec::with_capacity(runs);
    let mut gen_counts = Vec::with_capacity(runs);

    for run in 0..runs {
        let mut cache = KVCache::with_capacity(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            config
                .max_seq_len
                .min(prompt_tokens.len() + num_tokens + 16),
        );

        // Prefill
        let prefill_start = Instant::now();
        let mut next_token = 0u32;
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let logits = interpreter::forward(token, pos, &graph, &weights, &mut cache);
            cache.advance();
            next_token = sampling::sample(&logits, &sampling_config, pos as u64);
        }
        let prefill_time = prefill_start.elapsed();

        // Generate
        let gen_start = Instant::now();
        let eos_id = tokenizer.eos_token_id();
        let mut gen_count = 0;
        for i in 0..num_tokens {
            if eos_id.is_some_and(|eos| next_token == eos) {
                break;
            }
            let pos = prompt_tokens.len() + i;
            let logits = interpreter::forward(next_token, pos, &graph, &weights, &mut cache);
            cache.advance();
            next_token = sampling::sample(&logits, &sampling_config, (pos + 1) as u64);
            gen_count += 1;
        }
        let gen_time = gen_start.elapsed();

        let prefill_tps = prompt_tokens.len() as f64 / prefill_time.as_secs_f64();
        let gen_tps = gen_count as f64 / gen_time.as_secs_f64();

        println!(
            "Run {}: prefill {:.1} tok/s ({:.2}s) | generate {:.1} tok/s ({:.2}s, {} tokens)",
            run + 1,
            prefill_tps,
            prefill_time.as_secs_f64(),
            gen_tps,
            gen_time.as_secs_f64(),
            gen_count,
        );

        prefill_times.push(prefill_time.as_secs_f64());
        gen_times.push(gen_time.as_secs_f64());
        gen_counts.push(gen_count);
    }

    println!();
    println!("--- Summary ---");

    let avg_prefill_tps: f64 = prefill_times
        .iter()
        .map(|t| prompt_tokens.len() as f64 / t)
        .sum::<f64>()
        / runs as f64;
    let avg_gen_tps: f64 = gen_times
        .iter()
        .zip(gen_counts.iter())
        .map(|(t, c)| *c as f64 / t)
        .sum::<f64>()
        / runs as f64;

    println!("Avg prefill: {avg_prefill_tps:.1} tok/s");
    println!("Avg generate: {avg_gen_tps:.1} tok/s");
    println!(
        "Memory: {:.1} MB (weights)",
        weights.memory_bytes() as f64 / 1e6
    );

    Ok(())
}

fn handle_chat_completion(
    body: &[u8],
    config: &ModelConfig,
    graph: &forgellm_frontend::ir::Graph,
    weights: &weight_loader::ModelWeights,
    tokenizer: &Tokenizer,
) -> Result<String> {
    let req: serde_json::Value = serde_json::from_slice(body)?;

    // Parse messages
    let messages = req["messages"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing 'messages' array"))?;

    let chat_messages: Vec<forgellm_runtime::chat::ChatMessage> = messages
        .iter()
        .map(|m| forgellm_runtime::chat::ChatMessage {
            role: m["role"].as_str().unwrap_or("user").to_string(),
            content: m["content"].as_str().unwrap_or("").to_string(),
        })
        .collect();

    // Format with appropriate template
    let template = ChatTemplate::from_architecture(&config.architecture.to_string());
    let prompt = template.format(&chat_messages);

    let max_tokens = req["max_tokens"].as_u64().unwrap_or(128) as usize;
    let temperature = req["temperature"].as_f64().unwrap_or(0.0) as f32;

    let sampling_config = if temperature == 0.0 {
        sampling::SamplingConfig::greedy()
    } else {
        sampling::SamplingConfig {
            temperature,
            top_k: req["top_k"].as_u64().unwrap_or(0) as usize,
            top_p: req["top_p"].as_f64().unwrap_or(1.0) as f32,
            repetition_penalty: 1.1,
        }
    };

    let prompt_tokens = tokenizer.encode(&prompt)?;
    let mut cache = KVCache::with_capacity(
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        config
            .max_seq_len
            .min(prompt_tokens.len() + max_tokens + 16),
    );

    // Prefill
    let mut next_token = 0u32;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let logits = interpreter::forward(token, pos, graph, weights, &mut cache);
        cache.advance();
        next_token = sampling::sample(&logits, &sampling_config, pos as u64);
    }

    // Generate
    let mut generated_text = String::new();
    let eos_id = tokenizer.eos_token_id();
    let mut generated_count = 0;
    let mut all_tokens: Vec<u32> = prompt_tokens.clone();

    for i in 0..max_tokens {
        if let Ok(text) = tokenizer.decode_one(next_token) {
            generated_text.push_str(&text);
        }
        generated_count += 1;
        all_tokens.push(next_token);

        if eos_id.is_some_and(|eos| next_token == eos) {
            break;
        }

        let pos = prompt_tokens.len() + i;
        let mut logits = interpreter::forward(next_token, pos, graph, weights, &mut cache);
        cache.advance();
        sampling::apply_repetition_penalty(&mut logits, &all_tokens, 1.1);
        next_token = sampling::sample(&logits, &sampling_config, (pos + 1) as u64);
    }

    let response = serde_json::json!({
        "id": "chatcmpl-forge",
        "object": "chat.completion",
        "model": config.architecture.to_string(),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text,
            },
            "finish_reason": if eos_id.is_some_and(|eos| next_token == eos) { "stop" } else { "length" },
        }],
        "usage": {
            "prompt_tokens": prompt_tokens.len(),
            "completion_tokens": generated_count,
            "total_tokens": prompt_tokens.len() + generated_count,
        }
    });

    Ok(response.to_string())
}

fn handle_completion_stream(
    body: &[u8],
    config: &ModelConfig,
    graph: &forgellm_frontend::ir::Graph,
    weights: &weight_loader::ModelWeights,
    tokenizer: &Tokenizer,
    stream: &mut impl Write,
) -> Result<()> {
    let req: serde_json::Value = serde_json::from_slice(body)?;
    let prompt = req["prompt"].as_str().unwrap_or("Hello");
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(64) as usize;
    let temperature = req["temperature"].as_f64().unwrap_or(0.0) as f32;

    let sampling_config = if temperature == 0.0 {
        sampling::SamplingConfig::greedy()
    } else {
        sampling::SamplingConfig {
            temperature,
            top_k: req["top_k"].as_u64().unwrap_or(0) as usize,
            top_p: req["top_p"].as_f64().unwrap_or(1.0) as f32,
            repetition_penalty: 1.1,
        }
    };

    let prompt_tokens = tokenizer.encode(prompt)?;
    let mut cache = KVCache::with_capacity(
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
        config
            .max_seq_len
            .min(prompt_tokens.len() + max_tokens + 16),
    );

    let mut next_token = 0u32;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let logits = interpreter::forward(token, pos, graph, weights, &mut cache);
        cache.advance();
        next_token = sampling::sample(&logits, &sampling_config, pos as u64);
    }

    let eos_id = tokenizer.eos_token_id();
    let mut all_tokens: Vec<u32> = prompt_tokens;

    for i in 0..max_tokens {
        let text = tokenizer.decode_one(next_token).unwrap_or_default();
        all_tokens.push(next_token);

        let chunk = serde_json::json!({
            "id": "cmpl-forge",
            "object": "text_completion",
            "choices": [{"text": text, "index": 0, "finish_reason": serde_json::Value::Null}]
        });
        write!(stream, "data: {}\n\n", chunk)?;
        stream.flush()?;

        if eos_id.is_some_and(|eos| next_token == eos) {
            break;
        }

        let pos = all_tokens.len() - 1;
        let mut logits = interpreter::forward(next_token, pos, graph, weights, &mut cache);
        cache.advance();
        sampling::apply_repetition_penalty(&mut logits, &all_tokens, 1.1);
        next_token = sampling::sample(&logits, &sampling_config, (i + 1) as u64);
    }

    Ok(())
}

fn cmd_chat(
    model_path: &str,
    tokenizer_path: &str,
    system_prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    use forgellm_runtime::chat::{ChatMessage, ChatTemplate};

    eprintln!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    eprintln!("Loading model...");
    let config = load_model_config(model_path)?;
    let graph = graph_builder::build_graph(&config)?;
    eprintln!("Loading weights...");
    let (_gguf, weights) = weight_loader::load_from_file(model_path)?;
    eprintln!(
        "Ready! {} | {} layers | {:.0} MB\n",
        config.architecture,
        config.num_layers,
        weights.memory_bytes() as f64 / 1e6
    );

    let template = ChatTemplate::from_architecture(&config.architecture.to_string());
    let sampling_config = if temperature == 0.0 {
        sampling::SamplingConfig {
            repetition_penalty: 1.1,
            ..sampling::SamplingConfig::greedy()
        }
    } else {
        sampling::SamplingConfig {
            temperature,
            top_k: 0,
            top_p: 0.9,
            repetition_penalty: 1.1,
        }
    };

    let mut history: Vec<ChatMessage> = vec![ChatMessage::system(system_prompt)];

    loop {
        eprint!("You: ");
        io::stderr().flush()?;
        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input)? == 0 {
            break;
        }
        let user_input = user_input.trim();
        if user_input.is_empty() {
            continue;
        }
        if user_input == "/quit" || user_input == "/exit" {
            break;
        }
        if user_input == "/clear" {
            history.truncate(1);
            eprintln!("History cleared.\n");
            continue;
        }
        if user_input == "/help" {
            eprintln!("Commands: /clear /quit /exit /help\n");
            continue;
        }

        history.push(ChatMessage::user(user_input));
        let prompt = template.format(&history);
        let prompt_tokens = tokenizer.encode(&prompt)?;
        let prompt_len = prompt_tokens.len();

        let mut cache = KVCache::with_capacity(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            config.max_seq_len.min(prompt_len + max_tokens + 16),
        );

        // Prefill
        let prefill_start = Instant::now();
        let mut next_token = 0u32;
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let logits = interpreter::forward(token, pos, &graph, &weights, &mut cache);
            cache.advance();
            next_token = sampling::sample(&logits, &sampling_config, pos as u64);
        }
        let _prefill_time = prefill_start.elapsed();

        // Generate
        eprint!("\nAssistant: ");
        let gen_start = Instant::now();
        let eos_id = tokenizer.eos_token_id();
        let mut response_text = String::new();
        let mut all_tokens: Vec<u32> = prompt_tokens;
        let mut gen_count = 0usize;

        for i in 0..max_tokens {
            let text = tokenizer.decode_one(next_token).unwrap_or_default();
            eprint!("{text}");
            io::stderr().flush()?;
            response_text.push_str(&text);
            all_tokens.push(next_token);
            gen_count += 1;
            if eos_id.is_some_and(|eos| next_token == eos) {
                break;
            }
            let pos = all_tokens.len() - 1;
            let mut logits = interpreter::forward(next_token, pos, &graph, &weights, &mut cache);
            cache.advance();
            sampling::apply_repetition_penalty(&mut logits, &all_tokens, 1.1);
            next_token = sampling::sample(&logits, &sampling_config, (i + 1) as u64);
        }
        let gen_time = gen_start.elapsed();

        let gen_tps = if gen_time.as_secs_f64() > 0.0 {
            gen_count as f64 / gen_time.as_secs_f64()
        } else {
            0.0
        };
        eprintln!(
            "\n[{gen_count} tokens, {gen_tps:.0} tok/s | context: {}/{} tokens]\n",
            prompt_len + gen_count,
            config.max_seq_len,
        );
        history.push(ChatMessage::assistant(response_text.trim()));
    }

    Ok(())
}

fn cmd_models(dir: &str) -> Result<()> {
    let search_dir = if dir == "." {
        // Check FORGE_MODEL_DIR env var
        std::env::var("FORGE_MODEL_DIR").unwrap_or_else(|_| ".".to_string())
    } else {
        dir.to_string()
    };

    let path = Path::new(&search_dir);
    if !path.is_dir() {
        bail!("{search_dir} is not a directory");
    }

    println!("Searching for GGUF models in {search_dir}...\n");

    let mut found = 0;
    for entry in std::fs::read_dir(path)?.flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|ext| ext == "gguf") {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            let size_mb = p.metadata().map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);

            // Try to get model info
            match load_model_config(&p.to_string_lossy()) {
                Ok(config) => {
                    println!(
                        "  {} ({:.0} MB) — {} | {} layers | hidden={}",
                        name, size_mb, config.architecture, config.num_layers, config.hidden_size,
                    );
                }
                Err(_) => {
                    println!("  {} ({:.0} MB)", name, size_mb);
                }
            }
            found += 1;
        }
    }

    if found == 0 {
        println!("No GGUF models found.");
        println!("Tip: set FORGE_MODEL_DIR to your model directory");
    } else {
        println!("\n{found} model(s) found.");
    }

    Ok(())
}
