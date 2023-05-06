# Yet another path tracer

GPU accelerated using compute shaders, but everything is written in Rust - both the GPU and CPU code. Uses `rust-gpu` to transpile Rust code to SPIR-V, and then uses `wgpu` to execute that SPIR-V. Pass feature flag `oidn` to enable denoising. Requires OpenImageDenoise to be installed.

![](image.png)
![image](https://user-images.githubusercontent.com/11212115/236580283-10b90b04-48fd-4863-95df-ca5f27afff26.png)
![image](https://user-images.githubusercontent.com/11212115/236580256-e1bda1b2-37fb-461d-919d-3a3c037eb955.png)
