#![cfg_attr(target_arch = "spirv", no_std)]

use bsdf::BSDF;
use glam::*;
use intersection::BVHReference;
use shared_structs::{TracingConfig, BVHNode, MaterialData, PerVertexData, LightPickEntry, NextEventEstimation, Ray};
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


#[spirv(compute(threads(8, 8, 1)))]
pub fn main_raygen(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &TracingConfig,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rng: &mut [UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] ray_buffer: &mut [Ray],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] throughput_buffer: &mut [Vec4],
) {
    let index = (id.y * config.width + id.x) as usize;

    // Handle non-divisible workgroup sizes.
    if id.x > config.width || id.y > config.height {
        return;
    }

    // Get RNG, advance for new sample.
    let mut rng_state = rng::RngState::new(rng[index]);
    rng_state.advance_sample();

    // Get anti-aliased pixel coordinates.
    let suv = id.xy().as_vec2() + rng_state.gen_r2();
    let mut uv = Vec2::new(
        suv.x as f32 / config.width as f32,
        1.0 - suv.y as f32 / config.height as f32,
    ) * 2.0
        - 1.0;
    uv.y *= config.height as f32 / config.width as f32;

    // Setup camera.
    let ray_origin = config.cam_position.xyz();
    let mut ray_direction = Vec3::new(uv.x, uv.y, 1.0).normalize();
    let euler_mat = Mat3::from_rotation_y(config.cam_rotation.y) * Mat3::from_rotation_x(config.cam_rotation.x);
    ray_direction = euler_mat * ray_direction;

    // Write ray to buffer.
    ray_buffer[index] = Ray::new(ray_origin, ray_direction);

    // Initialize throughput buffer.
    throughput_buffer[index] = Vec4::ONE;

    // Write next rng state back to buffer.
    rng[index] = rng_state.current_state();
}

fn oct_wrap(v: Vec2) -> Vec2 {
    ( 1.0 - v.yx().abs() ) * Vec2::new(if v.x >= 0.0 { 1.0 } else { -1.0 }, if v.y >= 0.0 { 1.0 } else { -1.0 })
}
 
fn encode(mut n: Vec3) -> Vec2 {
    n /= n.x.abs() + n.y.abs() + n.z.abs();
    let nnxy = if n.z >= 0.0 { n.xy() } else { oct_wrap(n.xy()) };
    n.x = nnxy.x;
    n.y = nnxy.y;
    n.x = n.x * 0.5 + 0.5;
    n.y = n.y * 0.5 + 0.5;
    n.xy()
}
 
fn decode(mut f: Vec2) -> Vec3 {
    f = f * 2.0 - 1.0;
    let mut n = Vec3::new( f.x, f.y, 1.0 - f.x.abs() - f.y.abs());
    let t = -n.z.clamp(0.0, 1.0);
    n.x += if n.x >= 0.0 { -t } else { t };
    n.y += if n.y >= 0.0 { -t } else { t };
    return n.normalize();
}

#[spirv(compute(threads(8, 8, 1)))]
pub fn main_intersect(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &TracingConfig,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rng: &mut [UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] ray_buffer: &mut [Ray],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] output: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] throughput_buffer: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] per_vertex_buffer: &[PerVertexData],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] index_buffer: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] nodes_buffer: &[BVHNode],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] material_data_buffer: &[MaterialData],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] light_pick_buffer: &[LightPickEntry],
    #[spirv(descriptor_set = 0, binding = 10)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 11)] atlas: &Image!(2D, type=f32, sampled),
    #[spirv(workgroup)] workgroup: &mut [u32; 32 * 32],
) {
    let id = lid + wid * UVec3::new(8, 8, 1);
    
    // Handle non-divisible workgroup sizes.
    if id.x > config.width || id.y > config.height {
        return;
    }

    //////
    let index = (id.y * config.width + id.x) as usize;

    let bin_size = 32;
    let bin_count = bin_size * bin_size;

    let ray = ray_buffer[index];
    let oct_encoded = encode(ray.direction());
    let bin_coord = (bin_size as f32 * oct_encoded).as_uvec2();
    let bin_index = bin_coord.y * bin_size + bin_coord.x;

    let mut ray_bin_index = 0;
    if bin_index < bin_count {
        unsafe {
            ray_bin_index = spirv_std::arch::atomic_i_add::<u32, 2, 0x400>(&mut workgroup[bin_index as usize], 1);
        }
    }

    output[index] += if workgroup[bin_index as usize] != 0 { Vec4::ONE } else { Vec4::ZERO };
    if ray.depth() > 0 { 
        return;
    }





    // If ray is dead, we are done
    let mut ray = ray_buffer[index];
    if !ray.alive() {
        return;
    }

    let nee_mode = NextEventEstimation::from_u32(config.nee);
    let nee = nee_mode.uses_nee();
    let mut rng_state = rng::RngState::new(rng[index]);

    // Setup trace state
    let bvh = BVHReference {
        nodes: nodes_buffer,
    };
    let mut throughput = throughput_buffer[index].xyz();
    let mut radiance = Vec3::ZERO;
    let mut last_bsdf_sample = bsdf::BSDFSample::default();
    let mut last_light_sample = light_pick::DirectLightSample::default(); 

    // Trace ray
    let trace_result = bvh.intersect_front_to_back(per_vertex_buffer, index_buffer, ray.origin(), ray.direction());
    let hit = ray.origin() + ray.direction() * trace_result.t;

    let bounce = 0;
    'trace: {
        if !trace_result.hit {
            // skybox
            radiance += throughput * skybox::scatter(ray.origin(), ray.direction());
            ray.kill();
            break 'trace;
        } else {
            // Get material
            let material_index = trace_result.triangle.w;
            let material = material_data_buffer[material_index as usize];

            // Add emission
            if material.emissive.xyz() != Vec3::ZERO {
                // Emissive triangles are single-sided
                if trace_result.backface {
                    ray.kill(); // Kill since emissives don't bounce light
                    break 'trace;
                }
                
                // We want to add emissive contribution if:
                // - We are not doing NEE at all.
                // - This is the first bounce (so light sources don't look black).
                // - This is a non-diffuse bounce (so we don't double count emissive light).
                // AND we aren't hitting a backface (to match direct light sampling behavior).
                if !nee || bounce == 0 || last_bsdf_sample.sampled_lobe != bsdf::LobeType::DiffuseReflection {
                    radiance += util::mask_nan(throughput * material.emissive.xyz());
                    ray.kill();
                    break 'trace;
                }

                // If we have hit a light source, and we are using NEE with MIS, we use last bounces data
                // to add the BSDF contribution, weighted by MIS.
                if nee_mode.uses_mis() && last_bsdf_sample.sampled_lobe == bsdf::LobeType::DiffuseReflection {
                    let direct_contribution = light_pick::calculate_bsdf_mis_contribution(&trace_result, &last_bsdf_sample, &last_light_sample);
                    radiance += util::mask_nan(direct_contribution);
                    ray.kill();
                    break 'trace;
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
            let bsdf_sample = bsdf.sample(-ray.direction(), normal, &mut rng_state);
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
                    ray.direction(),
                    &mut rng_state
                );
                radiance += util::mask_nan(last_light_sample.direct_light_contribution);
            }

            // Attenuate by BSDF
            throughput *= bsdf_sample.spectrum / bsdf_sample.pdf;

            // Update ray
            ray.set_direction(bsdf_sample.sampled_direction);
            ray.set_origin(hit + ray.direction() * util::EPS);
            ray.bounce();

            // Russian roulette
            if bounce > config.min_bounces {
                let prob = throughput.max_element();
                if rng_state.gen_r1() > prob {
                    break 'trace;
                }
                throughput *= 1.0 / prob;
            }
        }
    }

    ray_buffer[index] = ray;
    throughput_buffer[index] = throughput.extend(1.0);
    output[index] += radiance.extend(1.0);
    rng[index] = rng_state.current_state();
}


// TODO: Use flat buffers instead of textures, so we can use the same code for CPU and GPU

#[spirv(compute(threads(8, 8, 1)))]
pub fn main_material(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &TracingConfig,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rng: &mut [UVec4],
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

    let mut throughput = Vec3::ONE;
    let mut radiance = Vec3::ZERO;
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

    output[index] += radiance.extend(1.0);
    rng_state.advance_sample();
    rng[index] = rng_state.current_state();
}
