use spirv_builder::SpirvBuilder;

use std::{
    env,
    path::{Path, PathBuf},
};

fn get_output_path() -> PathBuf {
    let manifest_dir_string = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_type = env::var("PROFILE").unwrap();
    let path = Path::new(&manifest_dir_string)
        .join("target")
        .join(build_type);
    return PathBuf::from(path);
}

fn copy_assimp_shared_library() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir: &Path = out_dir.parent().unwrap().parent().unwrap();
    for entry in std::fs::read_dir(build_dir).unwrap().flatten() {
        if !entry.file_name().to_string_lossy().starts_with("russimp-sys-") {
            continue;
        }
        let Ok(bins) = std::fs::read_dir(entry.path().join("out").join("dylib").join("bin")) else {
            continue;
        };
        for bin in bins.flatten() {
            let path = bin.path();
            let mut output_path = get_output_path();
            output_path.push(path.file_name().unwrap());
            if std::fs::copy(&path, &output_path).is_err() && !output_path.exists() {
                panic!("Failed to copy {}", path.display());
            }
        }
    }
}

fn main() {
    let mut builder = SpirvBuilder::new("kernels", "spirv-unknown-vulkan1.1");
    builder.build_script.defaults = true;
    builder.build_script.env_shader_spv_path = Some(true);
    builder.build().expect("Kernel failed to compile");

    copy_assimp_shared_library();

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
