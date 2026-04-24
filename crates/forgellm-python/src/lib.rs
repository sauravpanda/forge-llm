//! Python bindings for ForgeLLM via PyO3.
//!
//! Exposes two top-level APIs:
//!
//! ```python
//! import forgellm
//!
//! # AOT compilation: model + weights -> standalone Rust project
//! forgellm.compile('smollm2.gguf', output_dir='./compiled', target='cpu')
//! forgellm.compile('smollm2.gguf', output_dir='./compiled', target='metal')
//!
//! # Interpreter inference: load once, call generate() repeatedly
//! model = forgellm.Model('smollm2.gguf', device='cpu')
//! print(model.generate('The meaning of life is', max_tokens=64))
//!
//! # Streaming
//! for token in model.stream('Hello world', max_tokens=100):
//!     print(token, end='', flush=True)
//!
//! # Chat API
//! response = model.chat([
//!     {"role": "user", "content": "What is 2+2?"}
//! ], max_tokens=100)
//! ```

use std::path::Path;

use anyhow::{bail, Context};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use forgellm_frontend::{gguf, graph_builder, ir::*, weight_loader};
use forgellm_runtime::{
    chat::{ChatMessage, ChatTemplate},
    interpreter,
    kv_cache::KVCache,
    sampling,
    tokenizer::Tokenizer,
};

// --- Error conversion --------------------------------------------------------

fn to_py(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

// --- GGUF config extraction --------------------------------------------------

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
        hidden_activation: HiddenActivation::SiLU,
    })
}

// --- Tokenizer auto-detection ------------------------------------------------

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

// --- compile() ---------------------------------------------------------------

/// Compile a GGUF model into a standalone Rust project.
///
/// Runs the full AOT pipeline (parse -> IR -> optimize -> codegen) and writes
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
///     Backend target: ``"cpu"`` (default) or ``"metal"`` (Apple Silicon GPU).
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
/// >>> forgellm.compile('smollm2-135m.gguf', output_dir='./compiled', target='metal')
#[pyfunction]
#[pyo3(signature = (model_path, output_dir, target = "cpu"))]
fn compile(model_path: &str, output_dir: &str, target: &str) -> PyResult<()> {
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

    match target {
        "cpu" => {
            forgellm_codegen_cpu::generate_project(&optimized, out, &model_name, false)
                .map_err(|e| to_py(anyhow::anyhow!("{e}")))?;
        }
        "metal" => {
            forgellm_codegen_metal::generate_metal_project(&optimized, out, &model_name)
                .map_err(|e| to_py(anyhow::anyhow!("{e}")))?;
        }
        _ => {
            return Err(to_py(anyhow::anyhow!(
                "unsupported target '{target}'; supported targets: 'cpu', 'metal'"
            )));
        }
    }

    Ok(())
}

// --- TokenIterator (for streaming) -------------------------------------------

/// Iterator that yields generated tokens one at a time.
///
/// Created by :meth:`Model.stream`.  Each call to ``__next__`` runs one step
/// of the autoregressive loop and returns the decoded token string.
#[pyclass]
struct TokenIterator {
    graph: std::sync::Arc<Graph>,
    weights: std::sync::Arc<weight_loader::ModelWeights>,
    tokenizer: std::sync::Arc<Tokenizer>,
    cache: KVCache,
    next_token: u32,
    pos: usize,
    remaining: usize,
    stop_ids: Vec<u32>,
    sampling_config: sampling::SamplingConfig,
    done: bool,
}

#[pymethods]
impl TokenIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<String>> {
        if self.done || self.remaining == 0 {
            return Ok(None);
        }

        if self.stop_ids.contains(&self.next_token) {
            self.done = true;
            return Ok(None);
        }

        let token_text = self
            .tokenizer
            .decode(&[self.next_token])
            .map_err(|e| to_py(anyhow::anyhow!("failed to decode token: {e}")))?;

        let logits = interpreter::forward(
            self.next_token,
            self.pos,
            &self.graph,
            &self.weights,
            &mut self.cache,
        );
        self.cache.advance();
        self.next_token = sampling::sample(&logits, &self.sampling_config, (self.pos + 1) as u64);
        self.pos += 1;
        self.remaining -= 1;

        Ok(Some(token_text))
    }
}

// --- Model class -------------------------------------------------------------

/// Interpreter-backed LLM loaded from a GGUF file.
///
/// Loads the model weights into memory and provides ``generate()``,
/// ``stream()``, and ``chat()`` methods for text generation using the
/// ForgeLLM interpreter (not an AOT binary).  The interpreter is convenient
/// for prototyping and correctness testing; for production throughput,
/// compile with :func:`forgellm.compile` instead.
///
/// Parameters
/// ----------
/// model_path : str
///     Path to a GGUF model file.
/// tokenizer_path : str or None, optional
///     Path to ``tokenizer.json``.  If *None*, the file is auto-detected in
///     the same directory as the model (or its parent).
/// device : str, optional
///     Device for inference: ``"cpu"`` (default) or ``"metal"``.
///     The ``"metal"`` device is currently accepted for forward-compatibility
///     but inference runs on the CPU interpreter.  Use :func:`forgellm.compile`
///     with ``target="metal"`` for true GPU inference.
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
/// device : str
///     Device being used for inference.
///
/// Examples
/// --------
/// >>> import forgellm
/// >>> model = forgellm.Model('smollm2-135m.gguf')
/// >>> output = model.generate('The meaning of life is', max_tokens=64)
/// >>> print(output)
#[pyclass]
pub struct Model {
    graph: std::sync::Arc<Graph>,
    weights: std::sync::Arc<weight_loader::ModelWeights>,
    tokenizer: std::sync::Arc<Tokenizer>,
    config: ModelConfig,
    device: String,
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path = None, device = "cpu"))]
    fn new(model_path: &str, tokenizer_path: Option<&str>, device: &str) -> PyResult<Self> {
        // Validate device
        match device {
            "cpu" | "metal" => {}
            other => {
                return Err(to_py(anyhow::anyhow!(
                    "unsupported device '{other}'; supported devices: 'cpu', 'metal'"
                )));
            }
        }

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
            graph: std::sync::Arc::new(graph),
            weights: std::sync::Arc::new(weights),
            tokenizer: std::sync::Arc::new(tokenizer),
            config,
            device: device.to_string(),
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

    /// Stream generated tokens one at a time.
    ///
    /// Returns a Python iterator that yields one decoded token string per
    /// iteration.  This is useful for displaying output progressively.
    ///
    /// Parameters
    /// ----------
    /// prompt : str
    ///     Input text.
    /// max_tokens : int, optional
    ///     Maximum number of new tokens to generate (default: ``64``).
    /// temperature : float, optional
    ///     Sampling temperature (default: ``0.7``).  Pass ``0.0`` for
    ///     deterministic greedy decoding.
    ///
    /// Returns
    /// -------
    /// TokenIterator
    ///     An iterator yielding ``str`` tokens.
    ///
    /// Examples
    /// --------
    /// >>> for token in model.stream('Hello world', max_tokens=100):
    /// ...     print(token, end='', flush=True)
    #[pyo3(signature = (prompt, max_tokens = 64, temperature = 0.7))]
    fn stream(&self, prompt: &str, max_tokens: usize, temperature: f32) -> PyResult<TokenIterator> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| to_py(anyhow::anyhow!("failed to encode prompt: {e}")))?;

        if prompt_tokens.is_empty() {
            return Ok(TokenIterator {
                graph: self.graph.clone(),
                weights: self.weights.clone(),
                tokenizer: self.tokenizer.clone(),
                cache: KVCache::with_capacity(
                    self.config.num_layers,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    self.config.max_seq_len,
                ),
                next_token: 0,
                pos: 0,
                remaining: 0,
                stop_ids: vec![],
                sampling_config: sampling::SamplingConfig::greedy(),
                done: true,
            });
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

        let budget = max_tokens.min(max_ctx - prompt_tokens.len());
        let stop_ids = self.tokenizer.stop_token_ids();

        Ok(TokenIterator {
            graph: self.graph.clone(),
            weights: self.weights.clone(),
            tokenizer: self.tokenizer.clone(),
            cache,
            next_token,
            pos: prompt_tokens.len(),
            remaining: budget,
            stop_ids,
            sampling_config,
            done: false,
        })
    }

    /// Generate a response for a chat conversation.
    ///
    /// Applies the model's chat template to format the messages into a prompt,
    /// then generates a response.  Supports multi-turn conversations.
    ///
    /// Parameters
    /// ----------
    /// messages : list of dict
    ///     Conversation messages.  Each dict must have ``"role"`` (``"system"``,
    ///     ``"user"``, or ``"assistant"``) and ``"content"`` keys.
    /// max_tokens : int, optional
    ///     Maximum number of new tokens to generate (default: ``64``).
    /// temperature : float, optional
    ///     Sampling temperature (default: ``0.7``).
    ///
    /// Returns
    /// -------
    /// str
    ///     The assistant's response text.
    ///
    /// Examples
    /// --------
    /// >>> response = model.chat([
    /// ...     {"role": "user", "content": "What is 2+2?"}
    /// ... ], max_tokens=100)
    /// >>> print(response)
    #[pyo3(signature = (messages, max_tokens = 64, temperature = 0.7))]
    fn chat(
        &self,
        messages: Vec<Bound<'_, PyDict>>,
        max_tokens: usize,
        temperature: f32,
    ) -> PyResult<String> {
        let chat_messages: Vec<ChatMessage> = messages
            .iter()
            .map(|msg| {
                let role: String = msg
                    .get_item("role")
                    .map_err(|e| to_py(anyhow::anyhow!("missing 'role' key: {e}")))?
                    .ok_or_else(|| to_py(anyhow::anyhow!("missing 'role' key in message")))?
                    .extract()
                    .map_err(|e| to_py(anyhow::anyhow!("'role' must be a string: {e}")))?;
                let content: String = msg
                    .get_item("content")
                    .map_err(|e| to_py(anyhow::anyhow!("missing 'content' key: {e}")))?
                    .ok_or_else(|| to_py(anyhow::anyhow!("missing 'content' key in message")))?
                    .extract()
                    .map_err(|e| to_py(anyhow::anyhow!("'content' must be a string: {e}")))?;
                Ok(ChatMessage { role, content })
            })
            .collect::<PyResult<Vec<_>>>()?;

        let arch_name = self.config.architecture.to_string().to_lowercase();
        let template = ChatTemplate::from_architecture(&arch_name);
        let prompt = template.format(&chat_messages);

        self.generate(&prompt, max_tokens, temperature)
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

    /// Device used for inference.
    #[getter]
    fn device(&self) -> &str {
        &self.device
    }

    fn __repr__(&self) -> String {
        format!(
            "forgellm.Model(architecture='{}', layers={}, hidden={}, vocab={}, device='{}')",
            self.config.architecture,
            self.config.num_layers,
            self.config.hidden_size,
            self.config.vocab_size,
            self.device,
        )
    }
}

// --- Module registration -----------------------------------------------------

/// ForgeLLM Python bindings.
///
/// Compile and run small LLMs entirely from Python.
#[pymodule]
fn forgellm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_class::<Model>()?;
    m.add_class::<TokenIterator>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
