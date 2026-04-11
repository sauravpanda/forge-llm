//! End-to-end correctness test: compile a tiny deterministic model,
//! build the generated binary, run it, check token output.
//!
//! The slow `e2e_tiny_model_produces_deterministic_output` test is marked
//! `#[ignore]` so it only runs when explicitly requested (e.g. in nightly CI
//! or via `cargo test -- --ignored`).
//!
//! The fast `generated_source_is_valid_rust` test runs on every `cargo test`
//! and validates that the emitted code parses as syntactically correct Rust.

use forgellm_codegen_cpu::generate;
use forgellm_frontend::{
    graph_builder,
    ir::{Architecture, DType, ModelConfig},
};
use std::path::Path;

// ─── Shared config ────────────────────────────────────────────────────────────

/// A tiny, fully deterministic model config used by all tests in this file.
///
/// Uses F32 dtype so weight serialisation is straightforward (raw `f32` bytes).
/// `num_kv_heads` equals `num_attention_heads` so the weight layout is uniform.
fn tiny_deterministic_config() -> ModelConfig {
    ModelConfig {
        architecture: Architecture::Llama,
        hidden_size: 64,
        intermediate_size: 128,
        num_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 4,
        head_dim: 16,
        vocab_size: 256,
        max_seq_len: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        dtype: DType::F32,
        sliding_window_size: None,
        qkv_bias: false,
    }
}

// ─── Fast test (runs every `cargo test`) ─────────────────────────────────────

/// Verify that the source code emitted for the tiny model parses as valid Rust.
///
/// This does **not** invoke `cargo build` — it only uses `syn` to parse the
/// generated string.  If `emit.rs` produces invalid Rust the test fails
/// immediately and cheaply.
#[test]
fn generated_source_is_valid_rust() {
    let config = tiny_deterministic_config();
    let graph = graph_builder::build_graph(&config).unwrap();
    let code = generate(&graph).unwrap();
    syn::parse_file(&code).expect("generated code should be valid Rust syntax");
}

// ─── Slow E2E test (ignored by default) ──────────────────────────────────────

/// Write a `weights.bin` for the tiny F32 model in the exact layout that the
/// generated `main.rs` expects:
///
/// ```text
/// embed_tokens   [vocab * hidden]  f32 LE
/// for each layer:
///   attn_norm    [hidden]          f32 LE
///   q_proj       [qk_size*hidden]  f32 LE   (qk_size = num_heads * head_dim)
///   k_proj       [kv_size*hidden]  f32 LE   (kv_size = num_kv_heads * head_dim)
///   v_proj       [kv_size*hidden]  f32 LE
///   o_proj       [hidden*qk_size]  f32 LE
///   ffn_norm     [hidden]          f32 LE
///   gate_proj    [inter*hidden]    f32 LE
///   up_proj      [inter*hidden]    f32 LE
///   down_proj    [hidden*inter]    f32 LE
/// final_norm     [hidden]          f32 LE
/// lm_head        [vocab*hidden]    f32 LE
/// ```
fn write_weights_bin(config: &ModelConfig, path: &Path) {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;
    let qk_size = config.num_attention_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;

    // Helper: write `n` f32 values, cycling through [0.01, 0.02, ..., 0.10].
    // Small, bounded values keep activations non-NaN while remaining
    // deterministic across platforms.
    let make_tensor =
        |n: usize| -> Vec<f32> { (0..n).map(|i| ((i % 10) as f32 + 1.0) * 0.01).collect() };

    let mut buf: Vec<u8> = Vec::new();

    let write_tensor = |buf: &mut Vec<u8>, n: usize| {
        for v in make_tensor(n) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    };

    // embed_tokens
    write_tensor(&mut buf, vocab * hidden);

    // per-layer weights
    for _ in 0..config.num_layers {
        write_tensor(&mut buf, hidden); // attn_norm
        write_tensor(&mut buf, qk_size * hidden); // q_proj
        write_tensor(&mut buf, kv_size * hidden); // k_proj
        write_tensor(&mut buf, kv_size * hidden); // v_proj
        write_tensor(&mut buf, hidden * qk_size); // o_proj
        write_tensor(&mut buf, hidden); // ffn_norm
        write_tensor(&mut buf, inter * hidden); // gate_proj
        write_tensor(&mut buf, inter * hidden); // up_proj
        write_tensor(&mut buf, hidden * inter); // down_proj
    }

    // final_norm
    write_tensor(&mut buf, hidden);
    // lm_head
    write_tensor(&mut buf, vocab * hidden);

    std::fs::write(path, &buf).expect("failed to write weights.bin");
}

/// Write a minimal `tokenizer.json` using the `WordLevel` model so that every
/// single-character input token round-trips correctly.
///
/// Vocabulary: single ASCII characters 0x00–0xFF mapped to token IDs 0–255.
/// An `<unk>` token is included as token ID 256 (required by the WordLevel
/// model even though we never generate it in tests).
fn write_tokenizer_json(path: &Path) {
    // Build vocab: single-byte strings → token ID
    let mut vocab_entries = String::new();
    for i in 0u32..=255 {
        let ch = char::from_u32(i).unwrap_or('\u{FFFD}');
        // Escape the character for JSON
        let escaped = match ch {
            '"' => r#"\""#.to_string(),
            '\\' => r"\\".to_string(),
            '\n' => r"\n".to_string(),
            '\r' => r"\r".to_string(),
            '\t' => r"\t".to_string(),
            c if (c as u32) < 0x20 => format!(r"\u{:04X}", c as u32),
            c => c.to_string(),
        };
        if !vocab_entries.is_empty() {
            vocab_entries.push_str(", ");
        }
        vocab_entries.push_str(&format!(r#""{escaped}": {i}"#));
    }
    // Add the mandatory <unk> token
    vocab_entries.push_str(r#", "<unk>": 256"#);

    let json = format!(
        r#"{{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {{
    "type": "WordLevel",
    "vocab": {{{vocab_entries}}},
    "unk_token": "<unk>"
  }}
}}"#
    );

    std::fs::write(path, json).expect("failed to write tokenizer.json");
}

/// Build and run the generated project, returning stdout as a `String`.
///
/// Returns `None` if building fails (the test then fails with a clear message
/// rather than panicking in a confusing location).
#[cfg(test)]
fn build_and_run(
    project_dir: &Path,
    weights_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    max_tokens: usize,
) -> Option<std::process::Output> {
    use std::process::Command;

    let build_status = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(project_dir)
        .status()
        .expect("failed to spawn cargo build");

    if !build_status.success() {
        eprintln!("cargo build failed with status: {build_status}");
        return None;
    }

    // Locate the compiled binary.
    // On Unix the binary is at `target/release/<name>`, on Windows it has `.exe`.
    let bin_name = {
        let cargo_toml =
            std::fs::read_to_string(project_dir.join("Cargo.toml")).expect("read Cargo.toml");
        // Extract `name = "..."` from the [package] section.
        cargo_toml
            .lines()
            .find(|l| l.trim_start().starts_with("name"))
            .and_then(|l| l.split('"').nth(1))
            .unwrap_or("e2e-test")
            .to_string()
    };

    let binary_path = project_dir.join("target").join("release").join(&bin_name);
    #[cfg(windows)]
    let binary_path = binary_path.with_extension("exe");

    let output = Command::new(&binary_path)
        .args([
            weights_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            prompt,
            "--max-tokens",
            &max_tokens.to_string(),
            "--temp",
            "0",
            "--top-k",
            "1",
            "--seed",
            "42",
            "--quiet",
        ])
        .output()
        .expect("failed to spawn inference binary");

    Some(output)
}

/// Full end-to-end correctness test:
/// 1. Generate a Cargo project from the tiny model.
/// 2. Write deterministic weights and a minimal tokenizer.
/// 3. Build the project with `cargo build --release`.
/// 4. Run the binary twice with the same prompt and `--temp 0`.
/// 5. Assert the output is non-empty, exits successfully, and is identical
///    on both runs (determinism).
///
/// Marked `#[ignore]` because it invokes `cargo build`, which takes ~30 s.
/// Run explicitly with:
/// ```
/// cargo test --test e2e_correctness -- --ignored --nocapture
/// ```
#[test]
#[ignore = "slow: compiles and runs a full cargo project (~30 s)"]
fn e2e_tiny_model_produces_deterministic_output() {
    use forgellm_codegen_cpu::generate_project;
    use tempfile::tempdir;

    let config = tiny_deterministic_config();
    let graph = graph_builder::build_graph(&config).unwrap();

    let dir = tempdir().expect("failed to create tempdir");

    // 1. Generate the project source
    generate_project(&graph, dir.path(), "e2e-test", false).expect("generate_project failed");

    // 2. Write weights and tokenizer
    let weights_path = dir.path().join("weights.bin");
    let tokenizer_path = dir.path().join("tokenizer.json");
    write_weights_bin(&config, &weights_path);
    write_tokenizer_json(&tokenizer_path);

    // 3. Build + run (first run)
    let output1 = build_and_run(dir.path(), &weights_path, &tokenizer_path, "hello", 5)
        .expect("first build+run failed");

    assert!(
        output1.status.success(),
        "binary exited with non-zero status on first run: {}\nstderr: {}",
        output1.status,
        String::from_utf8_lossy(&output1.stderr)
    );

    let stdout1 = String::from_utf8_lossy(&output1.stdout).into_owned();
    assert!(
        !stdout1.is_empty(),
        "binary produced no stdout on first run"
    );

    // 4. Run a second time to verify determinism
    let output2 = {
        use std::process::Command;
        let bin_name = "e2e-test";
        let binary_path = dir.path().join("target").join("release").join(bin_name);
        #[cfg(windows)]
        let binary_path = binary_path.with_extension("exe");

        Command::new(&binary_path)
            .args([
                weights_path.to_str().unwrap(),
                tokenizer_path.to_str().unwrap(),
                "hello",
                "--max-tokens",
                "5",
                "--temp",
                "0",
                "--top-k",
                "1",
                "--seed",
                "42",
                "--quiet",
            ])
            .output()
            .expect("failed to spawn binary for second run")
    };

    assert!(
        output2.status.success(),
        "binary exited with non-zero status on second run"
    );

    let stdout2 = String::from_utf8_lossy(&output2.stdout).into_owned();

    // 5. Determinism check
    assert_eq!(
        stdout1, stdout2,
        "output differs between runs — forward pass is not deterministic"
    );

    // Sanity: stderr should not contain "NaN" or "Inf"
    let stderr = String::from_utf8_lossy(&output1.stderr).into_owned();
    assert!(
        !stderr.contains("NaN") && !stderr.contains("Inf"),
        "NaN/Inf detected in stderr:\n{stderr}"
    );

    eprintln!(
        "E2E output (first 200 chars): {}",
        &stdout1[..stdout1.len().min(200)]
    );
}
