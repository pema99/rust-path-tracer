# Yet another path tracer

Yet another GPU accelerated toy path tracer, but everything is written in Rust (:rocket:) - both the GPU and CPU code. Uses [rust-gpu](https://github.com/EmbarkStudios/rust-gpu) to transpile Rust (:rocket:) code to SPIR-V, and then uses [wgpu](https://github.com/gfx-rs/wgpu) to execute that SPIR-V. 

# Features
- Simple GPU accelerated path tracing.
- Supports PBR materials with roughness/metallic workflow. These can be set on a per-mesh basis.
- Supports texture mapping. Can load albedo, normal, roughness and metallic maps from scene file.
- Ray intersections are made fast using a [BVH](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) built in a binned manner using the [surface area heuristic](https://en.wikipedia.org/wiki/Bounding_interval_hierarchy#Construction).
- Convergence rate is improved by the use of a [low-discrepancy sequence](http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/) in place of uniform random sampling.
- Basic [next event estimation](https://www.youtube.com/watch?v=FU1dbi827LY) (direct light sampling).
- Uses [assimp](https://github.com/assimp/assimp) for scene loading, so can load many scene and model file formats, such as glTF, FBX, obj, etc.
- Uses a nice procedural atmospheric skybox (thanks @nyrox). Alternatively, can load HDR images to use as the skybox.
- Cross platform. Tested on Windows 10 and Arch Linux.
- All the GPU code can be run on the CPU via a dropdown in the UI. Mostly useful for debugging.

# How to build and run
```sh
# builds without denoising support
cargo run
```

The path tracer optionally supports denoising via OpenImageDenoise, via feature flag `oidn`. To use this feature, first [install OpenImageDenoise 1.4.3](https://github.com/OpenImageDenoise/oidn/releases/tag/v1.4.3) and ensure that the `OIDN_DIR` environment variable points to your install location.

```sh
# with denoising (requires OIDN to be installed and available on PATH)
export OIDN_DIR <path_to_oidn_install>
cargo run -F oidn
```

Once built and launched, to start rendering, simply drag any compatible scene file onto the window, or use the file picker. Holding right click and using WASD will let you move the camera.

I've only tested using Vulkan. If `wgpu` for whatever reason defaults to a different backend on your system, you can fix this by setting the `WGPU_BACKEND` environment variable to `"vulkan"`.

GPU kernel code is in `kernels/`, code shared between GPU and CPU is in `shared_structs/`, pure CPU code is in `src/`.

# Pretty pictures
![image](https://github.com/pema99/rust-path-tracer/assets/11212115/4f6e0936-77b7-40bf-917c-0424b37b8c74)
![image](https://user-images.githubusercontent.com/11212115/236666588-51cb006b-a1c6-4688-b49a-9dfc906cfa6c.png)
![image](https://github.com/pema99/rust-path-tracer/assets/11212115/7ba5c1bc-5d85-4e8e-b155-bd2233bad7e2)
![image](https://github.com/pema99/rust-path-tracer/assets/11212115/ba63c245-60a2-4eb5-990c-9d2a2e4d5449)
![image](https://user-images.githubusercontent.com/11212115/236580283-10b90b04-48fd-4863-95df-ca5f27afff26.png)
![image](https://user-images.githubusercontent.com/11212115/236580256-e1bda1b2-37fb-461d-919d-3a3c037eb955.png)
