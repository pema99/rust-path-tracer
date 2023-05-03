use spirv_builder::SpirvBuilder;

#[cfg(feature = "oidn")]
use std::{
    env,
    path::{Path, PathBuf},
};

#[cfg(feature = "oidn")]
fn get_output_path() -> PathBuf {
    let manifest_dir_string = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_type = env::var("PROFILE").unwrap();
    let path = Path::new(&manifest_dir_string)
        .join("target")
        .join(build_type);
    return PathBuf::from(path);
}

fn main() {
    SpirvBuilder::new("kernels", "spirv-unknown-vulkan1.1")
        .extra_arg("--no-spirt")
        .build()
        .expect("Kernel failed to compile");

    #[cfg(feature = "oidn")]
    {
        let oidn_dir = std::env::var("OIDN_DIR").expect("OIDN_DIR environment variable not set. Please set this to the OIDN install directory root.");
        let oidn_path = Path::new(&oidn_dir).join("bin");
        for entry in std::fs::read_dir(oidn_path).expect("Error finding OIDN binaries") {
            let path = entry.expect("Invalid path in OIDN binaries folder").path();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            let mut output_path = get_output_path();
            output_path.push(file_name);
            std::fs::copy(path, output_path).expect("Failed to copy OIDN binary");
        }
    }
}
