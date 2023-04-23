# rust-gpu-compute-example
Minimal example of using rust-gpu and wgpu to dispatch compute shaders written in rust.

Adapted from rust-gpu example source.

To run, simply `cargo run`. A build script will compile all kernel crates in the `kernels` directory. The function `execute_kernel` in `src/main.rs` shows how to run such a compiled kernel.
