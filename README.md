# Yet another path tracer

GPU accelerated using compute shaders, but everything is written in Rust - both the GPU and CPU code. Uses `rust-gpu` to transpile Rust code to SPIR-V, and then uses `wgpu` to execute that SPIR-V. Pass feature flag `oidn` to enable denoising. Requires OpenImageDenoise to be installed.

![](image.png)
