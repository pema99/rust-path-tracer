// This file contains benchmarks for the purpose of guarding against
// performance regressions. To run them, use the following command:
//
//     cargo test -- --nocapture --test-threads=1

#[cfg(test)]
mod benchmark {
    use crate::trace::*;
    use std::{
        io::{Read, Write},
        sync::{atomic::Ordering, Arc},
    };

    fn time<F, T>(f: F) -> (T, f64)
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let res = f();
        let end = std::time::Instant::now();

        let runtime_nanos = (end - start).as_nanos();
        let runtime_secs = runtime_nanos as f64 / 1_000_000_000.0;
        (res, runtime_secs)
    }

    fn setup_trace(width: u32, height: u32, samples: u32) -> Arc<TracingState> {
        let state = Arc::new(TracingState::new(width, height));
        state.running.store(true, Ordering::Relaxed);
        {
            let state = state.clone();
            std::thread::spawn(move || {
                while state.samples.load(Ordering::Relaxed) < samples {
                    std::thread::yield_now();
                }
                state.running.store(false, Ordering::Relaxed);
            });
        }
        state
    }

    fn benchmark<I, R, T>(tolerated_increase_percent: f64, num_runs: u32, init: I, run: R)
    where
        I: Fn() -> T,
        R: Fn(T),
    {
        // Run benchmark
        let mut sum_elapsed = 0.0;
        for _ in 0..num_runs {
            let data = init();
            let (_, elapsed) = time(|| run(data));
            sum_elapsed += elapsed;
        }
        let elapsed = sum_elapsed / num_runs as f64;
        println!("Average runtime: {}s", elapsed);

        // Get test file path
        let current_thread = std::thread::current();
        let raw_name = current_thread.name().unwrap();
        let test_name = raw_name.split("::").last().unwrap();
        let test_file_path_raw = format!(".benchmarks/{}", test_name);
        let test_file_path = std::path::Path::new(&test_file_path_raw);

        // Check if it already exists
        if !test_file_path.exists() {
            // Write data to new file
            std::fs::create_dir_all(".benchmarks").unwrap();
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(test_file_path)
                .unwrap();
            file.write(format!("{}\n", elapsed).as_bytes()).unwrap();
        } else {
            // Read data from file
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .open(test_file_path)
                .unwrap();
            let mut buf = String::new();
            file.read_to_string(&mut buf).unwrap();
            let expected = buf.trim().parse::<f64>().unwrap();

            let percent_change = (elapsed - expected) / expected * 100.0;
            assert!(
                percent_change < tolerated_increase_percent,
                "Benchmark failed: {}% change in performance.",
                percent_change
            );
        }
    }

    #[test]
    fn trace_160_gpu() {
        benchmark(
            10.0,
            10,
            || setup_trace(1280, 720, 160),
            |tracing_state| trace_gpu("scenes/DarkCornell.glb", None, tracing_state),
        );
    }

    #[test]
    fn trace_32_cpu() {
        benchmark(
            10.0,
            10,
            || setup_trace(1280, 720, 32),
            |tracing_state| trace_cpu("scenes/DarkCornell.glb", None, tracing_state),
        );
    }

    #[test]
    fn startup_time_gpu() {
        benchmark(
            10.0,
            10,
            || setup_trace(1280, 720, 0),
            |tracing_state| trace_cpu("scenes/BreakTime.glb", None, tracing_state),
        );
    }

    #[test]
    fn startup_time_cpu() {
        benchmark(
            10.0,
            10,
            || setup_trace(1280, 720, 0),
            |tracing_state| trace_cpu("scenes/BreakTime.glb", None, tracing_state),
        );
    }
}
