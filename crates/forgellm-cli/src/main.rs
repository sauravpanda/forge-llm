use clap::{Parser, Subcommand};

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
    /// Compile a model into an optimized binary
    Compile {
        /// Model path or HuggingFace model ID
        #[arg(long)]
        model: String,

        /// Target backend: cpu, wasm, gpu
        #[arg(long, default_value = "cpu")]
        target: String,

        /// Quantization format: q4, q8, fp8, f16, f32
        #[arg(long)]
        quantize: Option<String>,

        /// Output path for the compiled artifact
        #[arg(long)]
        output: String,
    },

    /// Run a compiled model
    Run {
        /// Path to compiled model
        model: String,

        /// Input prompt
        #[arg(long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,
    },

    /// Benchmark a compiled model
    Bench {
        /// Path to compiled model
        model: String,

        /// Number of iterations
        #[arg(long, default_value = "100")]
        iterations: usize,
    },

    /// Start an OpenAI-compatible API server
    Serve {
        /// Path to compiled model
        model: String,

        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            model,
            target,
            quantize,
            output,
        } => {
            tracing::info!(
                model = %model,
                target = %target,
                quantize = ?quantize,
                output = %output,
                "compilation not yet implemented"
            );
            println!("forge compile is not yet implemented — see PR 5+");
        }
        Commands::Run {
            model,
            prompt,
            max_tokens,
        } => {
            tracing::info!(
                model = %model,
                prompt = %prompt,
                max_tokens = %max_tokens,
                "run not yet implemented"
            );
            println!("forge run is not yet implemented — see PR 7+");
        }
        Commands::Bench {
            model, iterations, ..
        } => {
            tracing::info!(
                model = %model,
                iterations = %iterations,
                "bench not yet implemented"
            );
            println!("forge bench is not yet implemented — see PR 10");
        }
        Commands::Serve { model, port } => {
            tracing::info!(
                model = %model,
                port = %port,
                "serve not yet implemented"
            );
            println!("forge serve is not yet implemented — see PR 10");
        }
    }

    Ok(())
}
