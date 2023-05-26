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
    #[spirv(local_invocation_id)] lid: UVec3,
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
    #[spirv(workgroup)] scratch: &mut [u32; 8 * 8],
) {
    let index = (id.y * config.width + id.x) as usize;

    // Handle non-divisible workgroup sizes.
    if id.x > config.width || id.y > config.height {
        return;
    }

    let mut sobel = [0.0; 9];
    for y in 0..3 {
        for x in 0..3 {
            let sobel_id = id.xy().as_ivec2() + IVec2::new(x - 1, y - 1);
            let sobel_index = (sobel_id.y * config.width as i32 + sobel_id.x) as usize;
            let color = output[sobel_index].xyz() / (rng[sobel_index].x+1) as f32;
            let luminance = color.dot(Vec3::new(0.2126, 0.7152, 0.0722));
            sobel[(y * 3 + x) as usize] = luminance;
        }
    }
    let gx = (sobel[0] - sobel[2] + 2.0 * sobel[3] - 2.0 * sobel[5] + sobel[6] - sobel[8]).abs();
    let gy = (sobel[0] + 2.0 * sobel[1] + sobel[2] - sobel[6] - 2.0 * sobel[7] - sobel[8]).abs();
    let threshold = 0.1; 

    const RATE_1X1: u32 = 0b000;
    const RATE_1X2: u32 = 0b001;
    const RATE_2X1: u32 = 0b100;
    const RATE_2X2: u32 = 0b101;
    let mut my_rate = RATE_2X2;
    if gx > threshold && gy > threshold {
        my_rate = RATE_1X1;
    } else if gx > threshold {
        my_rate = RATE_2X1;
    } else if gy > threshold {
        my_rate = RATE_1X2;
    }
    scratch[(lid.y * 8 + lid.x) as usize] = my_rate;
    unsafe { spirv_std::arch::workgroup_memory_barrier_with_group_sync(); }
    if lid.x == 0 && lid.y == 0 {
        let mut rate = my_rate;
        for i in 0..64 {
            let other_rate = scratch[i];
            rate = if other_rate < rate { other_rate } else { rate };
        }
        scratch[0] = rate;
    }
    unsafe { spirv_std::arch::workgroup_memory_barrier_with_group_sync(); }
    let warp_rate = scratch[0];
    output[index].w = warp_rate as f32;

    let mut rng_state = rng::RngState::new(rng[index]);
    if (warp_rate & RATE_2X1 == 1) && (id.x & 1 == 1) {
        output[index] = output[index + 1];
        rng[index] = rng_state.next_state();
        return;
    } else if (warp_rate & RATE_1X2 == 1) && (id.y & 1 == 1) {
        output[index] = output[index + config.width as usize];
        rng[index] = rng_state.next_state();
        return;
    }

    let nee_mode = NextEventEstimation::from_u32(config.nee);
    let nee = nee_mode.uses_nee();
    
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
    
    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;
    let mut last_bsdf_sample = bsdf::BSDFSample::default();
    let mut last_light_sample = light_pick::DirectLightSample::default(); 
    
    for bounce in 0..config.max_bounces {
        let trace_result = bvh.intersect_front_to_back(per_vertex_buffer, index_buffer, ray_origin, ray_direction);
        let hit = ray_origin + ray_direction * trace_result.t;

        if !trace_result.hit {
            // skybox
            radiance += throughput * skybox::scatter(ray_origin, ray_direction);
            break;
        } else {
            // Get material
            let material_index = trace_result.triangle.w;
            let material = material_data_buffer[material_index as usize];

            // Add emission
            if material.emissive.xyz() != Vec3::ZERO {
                // Emissive triangles are single-sided
                if trace_result.backface {
                    break; // Break since emissives don't bounce light
                }

                // We want to add emissive contribution if:
                // - We are not doing NEE at all.
                // - This is the first bounce (so light sources don't look black).
                // - This is a non-diffuse bounce (so we don't double count emissive light).
                // AND we aren't hitting a backface (to match direct light sampling behavior).
                if !nee || bounce == 0 || last_bsdf_sample.sampled_lobe != bsdf::LobeType::DiffuseReflection {
                    radiance += util::mask_nan(throughput * material.emissive.xyz());
                    break;
                }

                // If we have hit a light source, and we are using NEE with MIS, we use last bounces data
                // to add the BSDF contribution, weighted by MIS.
                if nee_mode.uses_mis() && last_bsdf_sample.sampled_lobe == bsdf::LobeType::DiffuseReflection {
                    let direct_contribution = light_pick::calculate_bsdf_mis_contribution(&trace_result, &last_bsdf_sample, &last_light_sample);
                    radiance += util::mask_nan(direct_contribution);
                    break;
                }
            }

            // Interpolate vertex data
            let vertex_data_a = per_vertex_buffer[trace_result.triangle.x as usize];
            let vertex_data_b = per_vertex_buffer[trace_result.triangle.y as usize];
            let vertex_data_c = per_vertex_buffer[trace_result.triangle.z as usize];
            let vert_a = vertex_data_a.vertex.xyz();
            let vert_b = vertex_data_b.vertex.xyz();
            let vert_c = vertex_data_c.vertex.xyz();
            let norm_a = vertex_data_a.normal.xyz();
            let norm_b = vertex_data_b.normal.xyz();
            let norm_c = vertex_data_c.normal.xyz();
            let uv_a = vertex_data_a.uv0;
            let uv_b = vertex_data_b.uv0;
            let uv_c = vertex_data_c.uv0;
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
                let tangent_a = vertex_data_a.tangent.xyz();
                let tangent_b = vertex_data_b.tangent.xyz();
                let tangent_c = vertex_data_c.tangent.xyz();
                let tangent = bary.x * tangent_a + bary.y * tangent_b + bary.z * tangent_c;
                let tbn = Mat3::from_cols(tangent, tangent.cross(normal), normal);
                normal = (tbn * normal_map.xyz()).normalize();
            }
            
            // Sample BSDF
            let bsdf = bsdf::get_pbr_bsdf(&material, uv, atlas, sampler);
            let bsdf_sample = bsdf.sample(-ray_direction, normal, &mut rng_state);
            last_bsdf_sample = bsdf_sample;

            // Sample lights directly
            if nee && bsdf_sample.sampled_lobe == bsdf::LobeType::DiffuseReflection {
                last_light_sample = light_pick::sample_direct_lighting(
                    nee_mode,
                    index_buffer,
                    per_vertex_buffer,
                    material_data_buffer,
                    light_pick_buffer,
                    &bvh,
                    throughput,
                    &bsdf,
                    hit,
                    normal,
                    ray_direction,
                    &mut rng_state
                );
                radiance += util::mask_nan(last_light_sample.direct_light_contribution);
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
    output[index] += radiance.extend(0.0);
}
