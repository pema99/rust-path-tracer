#![cfg_attr(target_arch = "spirv", no_std)]

use bsdf::BSDF;
use glam::*;
use intersection::BVHReference;
use shared_structs::{TracingConfig, BVHNode, MaterialData, PerVertexData, LightPickEntry, NextEventEstimation};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::{glam, spirv, Sampler, Image};

mod bsdf;
mod rng;
mod util;
mod intersection;
mod vec;
mod skybox;
mod light_pick;

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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] material_data_buffer: &[MaterialData],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] light_pick_buffer: &[LightPickEntry],
    #[spirv(descriptor_set = 0, binding = 8)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 9)] atlas: &Image!(2D, type=f32, sampled),
) {
    let index = (id.y * config.width + id.x) as usize;

    // Handle non-divisible workgroup sizes.
    if id.x > config.width || id.y > config.height {
        return;
    }

    let nee_mode = NextEventEstimation::from_u32(config.nee);
    let nee = nee_mode.uses_nee();
    let mut rng_state = rng::RngState::new(rng[index]);

    // Get anti-aliased pixel coordinates.
    let suv = id.xy().as_vec2() + rng_state.gen_r2();
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
    };

    let mut last_sampled_lobe = bsdf::LobeType::SpecularReflection;
    let mut throughput = Vec3::ONE;
    for bounce in 0..config.max_bounces {
        let trace_result = bvh.intersect_front_to_back(per_vertex_buffer, index_buffer, ray_origin, ray_direction);
        //let trace_result = trace_slow_as_shit(vertex_buffer, index_buffer, ray_origin, ray_direction);
        let hit = ray_origin + ray_direction * trace_result.t;

        if !trace_result.hit {
            // skybox
            output[index] += (throughput * skybox::scatter(ray_origin, ray_direction)).extend(1.0);
            break;
        } else {
            // Get material
            let material_index = trace_result.triangle.w;
            let material = material_data_buffer[material_index as usize];

            // Add emission
            if material.emissive.xyz() != Vec3::ZERO {
                // We want to add emissive light if any of these are true:
                // - We are not doing NEE.
                // - This is the first bounce (so light sources don't look black).
                // - This is a non-diffuse bounce (so we don't double count emissive light).
                // AND we aren't hitting a backface (to match direct light sampling behavior).
                if !trace_result.backface && (!nee || bounce == 0 || last_sampled_lobe != bsdf::LobeType::DiffuseReflection) {
                    output[index] += util::mask_nan(throughput * material.emissive.xyz()).extend(1.0);
                }
                // Emissives don't bounce light
                break;
            }

            // Interpolate vertex data
            let vert_a = per_vertex_buffer[trace_result.triangle.x as usize].vertex.xyz();
            let vert_b = per_vertex_buffer[trace_result.triangle.y as usize].vertex.xyz();
            let vert_c = per_vertex_buffer[trace_result.triangle.z as usize].vertex.xyz();
            let norm_a = per_vertex_buffer[trace_result.triangle.x as usize].normal.xyz();
            let norm_b = per_vertex_buffer[trace_result.triangle.y as usize].normal.xyz();
            let norm_c = per_vertex_buffer[trace_result.triangle.z as usize].normal.xyz();
            let uv_a = per_vertex_buffer[trace_result.triangle.x as usize].uv0;
            let uv_b = per_vertex_buffer[trace_result.triangle.y as usize].uv0;
            let uv_c = per_vertex_buffer[trace_result.triangle.z as usize].uv0;
            let bary = util::barycentric(hit, vert_a, vert_b, vert_c);
            let mut normal = bary.x * norm_a + bary.y * norm_b + bary.z * norm_c;
            let mut uv = bary.x * uv_a + bary.y * uv_b + bary.z * uv_c;
            if uv.clamp(Vec2::ZERO, Vec2::ONE) != uv {
                uv = uv.fract(); // wrap UVs
            }

            // Apply normal map
            if material.has_normal_texture() {
                let scaled_uv = material.normals.xy() + uv * material.normals.zw();
                let normal_map = atlas.sample_by_lod(*sampler, scaled_uv, 0.0) * 2.0 - 1.0;
                let tangent_a = per_vertex_buffer[trace_result.triangle.x as usize].tangent.xyz();
                let tangent_b = per_vertex_buffer[trace_result.triangle.y as usize].tangent.xyz();
                let tangent_c = per_vertex_buffer[trace_result.triangle.z as usize].tangent.xyz();
                let tangent = bary.x * tangent_a + bary.y * tangent_b + bary.z * tangent_c;
                let tbn = Mat3::from_cols(tangent, tangent.cross(normal), normal);
                normal = (tbn * normal_map.xyz()).normalize();
            }
            
            // Sample BSDF
            let bsdf = bsdf::get_pbr_bsdf(&material, uv, atlas, sampler);
            let bsdf_sample = bsdf.sample(-ray_direction, normal, &mut rng_state);
            last_sampled_lobe = bsdf_sample.sampled_lobe;

            // Add direct lighting
            if nee && bsdf_sample.sampled_lobe == bsdf::LobeType::DiffuseReflection {
                let direct_lighting = light_pick::sample_direct_lighting(
                    nee_mode,
                    index_buffer,
                    per_vertex_buffer,
                    material_data_buffer,
                    light_pick_buffer,
                    &bvh,
                    &bsdf,
                    hit,
                    normal,
                    &bsdf_sample,
                    ray_direction,
                    &mut rng_state
                );
                output[index] += (throughput * direct_lighting).extend(1.0);
            }

            // Attenuate by BSDF
            throughput *= bsdf_sample.spectrum / bsdf_sample.pdf;

            // Update ray
            ray_direction = bsdf_sample.sampled_direction;
            ray_origin = hit + ray_direction * util::EPS;

            // Russian roulette
            if bounce > config.min_bounces {
                let prob = throughput.max_element();
                if rng_state.gen_r1() > prob {
                    break;
                }
                throughput *= 1.0 / prob;
            }
        }
    }

    rng[index] = rng_state.next_state();
}
