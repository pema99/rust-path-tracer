#![cfg_attr(target_arch = "spirv", no_std)]

use bsdf::BSDF;
use glam::*;
use intersection::BVHReference;
use shared_structs::{TracingConfig, BVHNode, MaterialData, PerVertexData};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::{glam, spirv};

mod bsdf;
mod rng;
mod util;
mod intersection;
mod vec;
mod skybox;

// TODO: Use flat buffers instead of textures, so we can use the same code for CPU and GPU

#[spirv(compute(threads(8, 8, 1)))]
pub fn main_material(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &TracingConfig,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rng: &mut [UVec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] per_vertex_buffer: &[PerVertexData],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] index_buffer: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] nodes_buffer: &[BVHNode],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] indirect_indices_buffer: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] material_data_buffer: &[MaterialData],
    #[spirv(descriptor_set = 0, binding = 8)] sampler: &spirv_std::Sampler,
    #[spirv(descriptor_set = 0, binding = 9)] albedo_atlas: &spirv_std::Image!(2D, type=f32, sampled),
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
    let mut ray_origin = config.cam_position.xyz();
    let mut ray_direction = Vec3::new(uv.x, uv.y, 1.0).normalize();
    let euler_mat = Mat3::from_rotation_y(config.cam_rotation.y) * Mat3::from_rotation_x(config.cam_rotation.x);
    ray_direction = euler_mat * ray_direction;

    let bvh = BVHReference {
        nodes: nodes_buffer,
        indirect_indices: indirect_indices_buffer,
    };

    let mut throughput = Vec3::ONE;
    for _ in 0..4 {
        let trace_result = bvh.intersect_front_to_back(per_vertex_buffer, index_buffer, ray_origin, ray_direction);
        //let trace_result = trace_slow_as_shit(vertex_buffer, index_buffer, ray_origin, ray_direction);
        let hit = ray_origin + ray_direction * trace_result.t;

        if !trace_result.hit {
            // skybox
            output[index] += (throughput * skybox::scatter(ray_origin, ray_direction)).extend(1.0);
            return;
        } else {
            let norm_a = per_vertex_buffer[trace_result.triangle.x as usize].normal.xyz();
            let norm_b = per_vertex_buffer[trace_result.triangle.y as usize].normal.xyz();
            let norm_c = per_vertex_buffer[trace_result.triangle.z as usize].normal.xyz();
            let vert_a = per_vertex_buffer[trace_result.triangle.x as usize].vertex.xyz();
            let vert_b = per_vertex_buffer[trace_result.triangle.y as usize].vertex.xyz();
            let vert_c = per_vertex_buffer[trace_result.triangle.z as usize].vertex.xyz();
            let bary = util::barycentric(hit, vert_a, vert_b, vert_c);
            let norm = bary.x * norm_a + bary.y * norm_b + bary.z * norm_c;

            let material_index = trace_result.triangle.w;
            let material = material_data_buffer[material_index as usize];
            let albedo = if material.has_albedo_texture() {
                let uv_a = per_vertex_buffer[trace_result.triangle.x as usize].uv0;
                let uv_b = per_vertex_buffer[trace_result.triangle.y as usize].uv0;
                let uv_c = per_vertex_buffer[trace_result.triangle.z as usize].uv0;
                let uv = bary.x * uv_a + bary.y * uv_b + bary.z * uv_c;
                let uv = Vec2::new(uv.x, uv.y);
                let scaled_uv = material.albedo.xy() + uv * material.albedo.zw();
                let albedo = albedo_atlas.sample_by_lod(*sampler, scaled_uv, 0.0);
                albedo.xyz()
            } else {
                material.albedo.xyz()
            };

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
