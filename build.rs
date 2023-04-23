use spirv_builder::SpirvBuilder;

fn main() {
    for kernel in std::fs::read_dir("kernels").expect("Error finding kernels folder") {
        let path = kernel.expect("Invalid path in kernels folder").path();
        SpirvBuilder::new(path, "spirv-unknown-vulkan1.1")
            .extra_arg("--no-spirt")
            .build()
            .expect("Kernel failed to compile");
    }
}
