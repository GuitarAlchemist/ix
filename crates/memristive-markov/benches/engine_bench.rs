use criterion::{black_box, criterion_group, criterion_main, Criterion};
use memristive_markov::serde_state::EngineConfig;
use memristive_markov::MemristiveEngine;
use rand::SeedableRng;

fn bench_observe(c: &mut Criterion) {
    let mut engine = MemristiveEngine::new(EngineConfig::default());
    let mut state = 0usize;
    c.bench_function("observe_1000_states", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                engine.observe(black_box(state % 64));
                state += 1;
            }
        })
    });
}

fn bench_predict(c: &mut Criterion) {
    let mut engine = MemristiveEngine::new(EngineConfig {
        min_observations: 1,
        ..EngineConfig::default()
    });
    for i in 0..1000 {
        engine.observe(i % 32);
    }
    c.bench_function("predict", |b| {
        b.iter(|| {
            let _ = black_box(engine.predict());
        })
    });
}

fn bench_sample(c: &mut Criterion) {
    let mut engine = MemristiveEngine::new(EngineConfig {
        min_observations: 1,
        ..EngineConfig::default()
    });
    for i in 0..1000 {
        engine.observe(i % 32);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    c.bench_function("sample", |b| {
        b.iter(|| {
            let _ = black_box(engine.sample(&mut rng));
        })
    });
}

fn bench_state_roundtrip(c: &mut Criterion) {
    let mut engine = MemristiveEngine::new(EngineConfig::default());
    for i in 0..500 {
        engine.observe(i % 16);
    }
    let json = engine.export_state();
    c.bench_function("state_roundtrip", |b| {
        b.iter(|| {
            let _ = black_box(MemristiveEngine::from_state(&json));
        })
    });
}

criterion_group!(
    benches,
    bench_observe,
    bench_predict,
    bench_sample,
    bench_state_roundtrip
);
criterion_main!(benches);
