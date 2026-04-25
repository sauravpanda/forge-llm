//! Micro-benchmarks for ForgeLLM codegen performance.
//!
//! Measures IR graph building and Rust source emission speed.
//! Used for perf regression detection in CI.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use forgellm_codegen_cpu::{generate, generate_project};
use forgellm_frontend::{
    graph_builder,
    ir::{Architecture, DType, HiddenActivation, ModelConfig},
};
use std::hint::black_box;

fn tiny_config() -> ModelConfig {
    ModelConfig {
        architecture: Architecture::Llama,
        hidden_size: 64,
        intermediate_size: 128,
        num_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 4,
        head_dim: 16,
        vocab_size: 512,
        max_seq_len: 512,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        dtype: DType::F32,
        lm_head_dtype: None,
        sliding_window_size: None,
        qkv_bias: false,
        hidden_activation: HiddenActivation::SiLU,
    }
}

fn small_config() -> ModelConfig {
    ModelConfig {
        architecture: Architecture::Llama,
        hidden_size: 576,
        intermediate_size: 1536,
        num_layers: 30,
        num_attention_heads: 9,
        num_kv_heads: 3,
        head_dim: 64,
        vocab_size: 49152,
        max_seq_len: 4096,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        dtype: DType::F32,
        lm_head_dtype: None,
        sliding_window_size: None,
        qkv_bias: false,
        hidden_activation: HiddenActivation::SiLU,
    }
}

fn bench_graph_build(c: &mut Criterion) {
    let mut g = c.benchmark_group("graph_build");
    g.bench_function("tiny", |b| {
        b.iter(|| black_box(graph_builder::build_graph(black_box(&tiny_config()))))
    });
    g.bench_function("smollm2_135m", |b| {
        b.iter(|| black_box(graph_builder::build_graph(black_box(&small_config()))))
    });
    g.finish();
}

fn bench_emit(c: &mut Criterion) {
    let mut g = c.benchmark_group("emit");
    let tiny_graph = graph_builder::build_graph(&tiny_config()).unwrap();
    g.bench_function("tiny", |b| {
        b.iter(|| black_box(generate(black_box(&tiny_graph)).unwrap()))
    });
    let small_graph = graph_builder::build_graph(&small_config()).unwrap();
    g.bench_function("smollm2_135m", |b| {
        b.iter(|| black_box(generate(black_box(&small_graph)).unwrap()))
    });
    g.finish();
}

fn bench_generate_project(c: &mut Criterion) {
    let mut g = c.benchmark_group("generate_project");
    let tiny_graph = graph_builder::build_graph(&tiny_config()).unwrap();
    g.bench_function("tiny_f32", |b| {
        b.iter(|| {
            let dir = tempfile::tempdir().unwrap();
            generate_project(black_box(&tiny_graph), dir.path(), "bench-model", false).unwrap();
            black_box(dir)
        })
    });
    g.finish();
}

fn bench_emit_sizes(c: &mut Criterion) {
    let mut g = c.benchmark_group("emit_output_size");
    for (name, config) in [("tiny", tiny_config()), ("smollm2_135m", small_config())] {
        g.bench_with_input(BenchmarkId::new("bytes", name), &config, |b, cfg| {
            let graph = graph_builder::build_graph(cfg).unwrap();
            b.iter(|| black_box(generate(black_box(&graph)).unwrap().len()))
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_graph_build,
    bench_emit,
    bench_generate_project,
    bench_emit_sizes,
);
criterion_main!(benches);
