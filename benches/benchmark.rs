// This file contains benchmarks for the purpose of guarding against
// performance regressions. To run them, use `cargo bench`.

use rustic::trace::*;

use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Performance regression tests");
    group.sample_size(10);
    group.bench_function("Startup time (GPU)", |b| { // 3.021s
        b.iter(|| trace_gpu("scenes/BreakTime.glb", None, setup_trace(1280, 720, 0)))
    });
    group.bench_function("Startup time (CPU)", |b| { // 2.855s
        b.iter(|| trace_cpu("scenes/BreakTime.glb", None, setup_trace(1280, 720, 0)))
    });
    group.bench_function("160 samples (GPU)", |b| { // 2.408s
        b.iter(|| trace_gpu("scenes/DarkCornell.glb", None, setup_trace(1280, 720, 160)))
    });
    group.bench_function("32 samples (CPU)", |b| { // 12.891s
        b.iter(|| trace_cpu("scenes/DarkCornell.glb", None, setup_trace(1280, 720, 32)))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
