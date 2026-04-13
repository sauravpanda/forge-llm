//! Python bindings for ForgeLLM via PyO3.
//!
//! Exposes two top-level APIs:
//!
//! ```python
//! import forgellm
//!
//! # AOT compilation: model + weights → standalone Rust project
//! forgellm.compile('smollm2.gguf', output_dir='./compiled', target='cpu')
//!
//! # Interpreter inference: load once, call generate() repeatedly
//! model = forgellm.Model('smollm2.gguf')
//! print(model.generate('The meaning of life is', max_tokens=64))
//! ```

use std::path::Path;

use anyhow::{bail, Context};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use forgellm_frontend::{gguf, graph_builder, ir::*, weight_loader};
use forgellm_runtime::{interpreter, kv_cache::KVCache, sampling, tokenizer::Tokenizer};

// ─── Error conversion ────────────────────────────────────────────────────────

fn to_py(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

// ─── GGUF config extraction ───────────────────────────────────────────────────

/// Extract a `ModelConfig` from a GGUF file by reading its metadata section.
///
/// This mirrors the logic in `forgellm-cli`'s `load_model_config` so the
/// Python bindings do not depend on the CLI crate.
fn config_from_gguf(path: &str) -> anyhow::Result<ModelConfig> {
    let data = std::fs::read(path).with_context(|| format!("failed to read '{path}'"))?;
    let g = gguf::parse(std::io::Cursor::new(&data))
        .with_context(|| format!("failed to parse GGUF file '{path}'"))?;

    let arch_name = g.architecture().unwrap_or("unknown").to_string();
    let p = &arch_name; // metadata prefix

    let hidden = g
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

    if hidden == 0 || num_layers == 0 || num_heads == 0 {
        bail!("could not extract model dimensions from GGUF metadata in '{path}'");
    }

    let head_dim = hidden / num_heads;
    let intermediate = g
        .get_u32(&format!("{p}.feed_forward_length"))
        .map(|v| v as usize)
        .unwrap_or(hidden * 4);
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
        hidden_size: hidden,
        intermediate_size: intermediate,
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

// ─── Tokenizer auto-detection ─────────────────────────────────────────────────

/// Search for `tokenizer.json` near the model file.
///
/// Tries the model's directory, then its parent, following the same convention
/// as the `forge run` CLI command.
fn find_tokenizer(model_path: &str) -> Option<String> {
    let model = Path::new(model_path);
    if let Some(dir) = model.parent() {
        let candidate = dir.join("tokenizer.json");
        if candidate.exists() {
            return Some(candidate.to_string_lossy().into_owned());
        }
        if let Some(parent) = dir.parent() {
            let candidate = parent.join("tokenizer.json");
            if candidate.exists() {
                return Some(candidate.to_string_lossy().into_owned());
            }
        }
    }
    None
}

// ─── compile() ───────────────────────────────────────────────────────────────

/// Compile a GGUF model into a standalone Rust project.
///
/// Runs the full AOT pipeline (parse → IR → optimize → codegen) and writes
/// a self-contained Rust project to *output_dir*.  The resulting project can
/// be built with ``cargo build --release`` to produce a zero-dependency binary.
///
/// Parameters
/// ----------
/// model_path : str
///     Path to a GGUF model file.
/// output_dir : str
///     Directory where the generated Rust project will be written.
///     Created if it does not exist.
/// target : str, optional
///     Backend target. Currently only ``"cpu"`` is supported (default).
///
/// Raises
/// ------
/// RuntimeError
///     If the model cannot be loaded, the architecture is unsupported, or
///     code generation fails.
///
/// Examples
/// --------
/// >>> import forgellm
/// >>> forgellm.compile('smollm2-135m.gguf', output_dir='./compiled')
#[pyfunction]
#[pyo3(signature = (model_path, output_dir, target = "cpu"))]
fn compile(model_path: &str, output_dir: &str, target: &str) -> PyResult<()> {
    if target != "cpu" {
        return Err(to_py(anyhow::anyhow!(
            "unsupported target '{target}'; only 'cpu' is supported"
        )));
    }

    let config = config_from_gguf(model_path).map_err(to_py)?;

    let graph = graph_builder::build_graph(&config).map_err(|e| to_py(anyhow::anyhow!("{e}")))?;

    let optimized = forgellm_optimizer::optimize(&graph);

    let out = Path::new(output_dir);
    std::fs::create_dir_all(out)
        .map_err(|e| to_py(anyhow::anyhow!("failed to create '{output_dir}': {e}")))?;

    let model_name = out
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());

    forgellm_codegen_cpu::generate_project(&optimized, out, &model_name, false)
        .map_err(|e| to_py(anyhow::anyhow!("{e}")))?;

    Ok(())
}

// ─── Model class ─────────────────────────────────────────────────────────────

/// Interpreter-backed LLM loaded from a GGUF file.
///
/// Loads the model weights into memory and provides a ``generate()`` method
/// for text generation using the ForgeLLM interpreter (not an AOT binary).
/// The interpreter is convenient for prototyping and correctness testing;
/// for production throughput, compile with :func:`forgellm.compile` instead.
///
/// Parameters
/// ----------
/// model_path : str
///     Path to a GGUF model file.
/// tokenizer_path : str or None, optional
///     Path to ``tokenizer.json``.  If *None*, the file is auto-detected in
///     the same directory as the model (or its parent).
///
/// Attributes
/// ----------
/// architecture : str
///     Architecture name (e.g. ``"Llama"``, ``"Mistral"``).
/// num_layers : int
///     Number of transformer layers.
/// hidden_size : int
///     Hidden state dimensionality.
/// vocab_size : int
///     Vocabulary size.
///
/// Examples
/// --------
/// >>> import forgellm
/// >>> model = forgellm.Model('smollm2-135m.gguf')
/// >>> output = model.generate('The meaning of life is', max_tokens=64)
/// >>> print(output)
#[pyclass]
pub struct Model {
    graph: forgellm_frontend::ir::Graph,
    weights: forgellm_frontend::weight_loader::ModelWeights,
    tokenizer: Tokenizer,
    config: ModelConfig,
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path = None))]
    fn new(model_path: &str, tokenizer_path: Option<&str>) -> PyResult<Self> {
        // Load model configuration from GGUF metadata
        let config = config_from_gguf(model_path).map_err(to_py)?;

        // Build computation graph from config
        let graph =
            graph_builder::build_graph(&config).map_err(|e| to_py(anyhow::anyhow!("{e}")))?;

        // Load and dequantize weights
        let (_gguf_file, weights) = weight_loader::load_from_file(model_path).map_err(|e| {
            to_py(anyhow::anyhow!(
                "failed to load weights from '{model_path}': {e}"
            ))
        })?;

        // Resolve tokenizer path
        let tok_path = if let Some(p) = tokenizer_path {
            p.to_string()
        } else {
            find_tokenizer(model_path).ok_or_else(|| {
                to_py(anyhow::anyhow!(
                    "tokenizer.json not found near '{}'; \
                     pass tokenizer_path= explicitly or place tokenizer.json \
                     in the same directory as the model",
                    model_path
                ))
            })?
        };

        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| {
            to_py(anyhow::anyhow!(
                "failed to load tokenizer from '{}': {e}",
                tok_path
            ))
        })?;

        Ok(Self {
            graph,
            weights,
            tokenizer,
            config,
        })
    }

    /// Generate text continuation for *prompt*.
    ///
    /// Parameters
    /// ----------
    /// prompt : str
    ///     Input text.
    /// max_tokens : int, optional
    ///     Maximum number of new tokens to generate (default: ``64``).
    ///     The actual count may be less if an EOS token is produced.
    /// temperature : float, optional
    ///     Sampling temperature (default: ``0.7``).  Pass ``0.0`` for
    ///     deterministic greedy decoding.
    ///
    /// Returns
    /// -------
    /// str
    ///     Generated text (the prompt is not included in the output).
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the prompt exceeds the model's context length, or if encoding
    ///     or decoding fails.
    #[pyo3(signature = (prompt, max_tokens = 64, temperature = 0.7))]
    fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32) -> PyResult<String> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| to_py(anyhow::anyhow!("failed to encode prompt: {e}")))?;

        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        let max_ctx = self.config.max_seq_len;
        if prompt_tokens.len() >= max_ctx {
            return Err(to_py(anyhow::anyhow!(
                "prompt is {} tokens but model context is {}; shorten the prompt",
                prompt_tokens.len(),
                max_ctx,
            )));
        }

        let sampling_config = if temperature == 0.0 {
            sampling::SamplingConfig::greedy()
        } else {
            sampling::SamplingConfig {
                temperature,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
            }
        };

        // Fresh KV cache for this generation (stateless)
        let mut cache = KVCache::with_capacity(
            self.config.num_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
            max_ctx,
        );

        // Prefill: process the prompt tokens
        let mut next_token = 0u32;
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            let logits =
                interpreter::forward(token_id, pos, &self.graph, &self.weights, &mut cache);
            cache.advance();
            next_token = sampling::sample(&logits, &sampling_config, pos as u64);
        }

        // Generate up to max_tokens new tokens
        let budget = max_tokens.min(max_ctx - prompt_tokens.len());
        let stop_ids = self.tokenizer.stop_token_ids();
        let mut generated: Vec<u32> = Vec::with_capacity(budget);

        for i in 0..budget {
            if stop_ids.contains(&next_token) {
                break;
            }
            generated.push(next_token);

            let pos = prompt_tokens.len() + i;
            let logits =
                interpreter::forward(next_token, pos, &self.graph, &self.weights, &mut cache);
            cache.advance();
            next_token = sampling::sample(&logits, &sampling_config, (pos + 1) as u64);
        }

        self.tokenizer
            .decode(&generated)
            .map_err(|e| to_py(anyhow::anyhow!("failed to decode output: {e}")))
    }

    /// Architecture name (e.g. ``"Llama"``, ``"Mistral"``).
    #[getter]
    fn architecture(&self) -> String {
        self.config.architecture.to_string()
    }

    /// Number of transformer layers.
    #[getter]
    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Hidden state dimensionality.
    #[getter]
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn __repr__(&self) -> String {
        format!(
            "forgellm.Model(architecture='{}', layers={}, hidden={}, vocab={})",
            self.config.architecture,
            self.config.num_layers,
            self.config.hidden_size,
            self.config.vocab_size,
        )
    }
}

// ─── Module registration ──────────────────────────────────────────────────────

/// ForgeLLM Python bindings.
///
/// Compile and run small LLMs entirely from Python.
#[pymodule]
fn forgellm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_class::<Model>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
