#![cfg_attr(target_arch = "spirv", no_std)]

use bsdf::BSDF;
use glam::*;
use intersection::BVHReference;
use shared_structs::{TracingConfig, BVHNode};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::{glam, spirv};

mod bsdf;
mod rng;
mod util;
mod intersection;
mod vec;
mod skybox;

#[spirv(compute(threads(8, 8, 1)))]
pub fn main_material(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &TracingConfig,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rng: &mut [UVec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] vertex_buffer: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] index_buffer: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] normal_buffer: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] nodes_buffer: &[BVHNode],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] indirect_indices_buffer: &[u32],
) {
    let index = (id.y * config.width + id.x) as usize;

    // Handle non-divisible workgroup sizes.
    if id.x > config.width || id.y > config.height {
        return;
    }

    let mut rng_state = rng::RngState::new(&mut rng[index]);

    // Get anti-aliased pixel coordinates.
    let suv = id.xy().as_vec2() + rng_state.gen_float_pair();
    let mut uv = Vec2::new(
        suv.x as f32 / config.width as f32,
        1.0 - suv.y as f32 / config.height as f32,
    ) * 2.0
        - 1.0;
    uv.y *= config.height as f32 / config.width as f32;

    // Setup camera.
    let mut ray_origin = Vec3::new(0.0, 1.0, -5.0);
    let mut ray_direction = Vec3::new(uv.x, uv.y, 1.0).normalize();

    let bvh = BVHReference {
        nodes: nodes_buffer,
        indirect_indices: indirect_indices_buffer,
    };

    let mut throughput = Vec3::ONE;
    for _ in 0..4 {
        let trace_result = bvh.intersect_front_to_back(vertex_buffer, index_buffer, ray_origin, ray_direction);
        //let trace_result = trace_slow_as_shit(vertex_buffer, index_buffer, ray_origin, ray_direction);
        let hit = ray_origin + ray_direction * trace_result.t;

        if !trace_result.hit {
            // skybox
            output[index] += (throughput * skybox::scatter(ray_origin, ray_direction)).extend(1.0);
            return;
        } else {
            let norm_a = normal_buffer[trace_result.triangle.x as usize].xyz();
            let norm_b = normal_buffer[trace_result.triangle.y as usize].xyz();
            let norm_c = normal_buffer[trace_result.triangle.z as usize].xyz();
            let norm = (norm_a + norm_b + norm_c) / 3.0; // flat shading

            let albedo = norm * 0.5 + 0.5;
            let bsdf = bsdf::PBR {
                albedo: albedo,
                roughness: 0.5,
                metallic: 0.3,
            };
            let bsdf_sample = bsdf.sample(-ray_direction, norm, &mut rng_state);
            throughput *= bsdf_sample.spectrum / bsdf_sample.pdf;

            ray_direction = bsdf_sample.sampled_direction;
            ray_origin = hit + norm * 0.01;
        }
    }
}
