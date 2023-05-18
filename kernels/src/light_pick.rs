use shared_structs::{LightPickEntry, PerVertexData, MaterialData};
use spirv_std::glam::{Vec3, UVec4, Vec4Swizzles};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use crate::{rng::RngState, util, bsdf::{self, BSDF, BSDFSample}, intersection::BVHReference};

pub fn pick_light(table: &[LightPickEntry], rng_state: &mut RngState) -> (u32, f32, f32) {
    let rng = rng_state.gen_r2();
    let entry = table[(rng.x * table.len() as f32) as usize];
    if rng.y < entry.ratio {
        (entry.triangle_index_a, entry.triangle_area_a, entry.triangle_pick_pdf_a)
    } else {
        (entry.triangle_index_b, entry.triangle_area_b, entry.triangle_pick_pdf_b)
    }
}

// https://www.cs.princeton.edu/~funk/tog02.pdf equation 1
pub fn pick_triangle_point(a: Vec3, b: Vec3, c: Vec3, rng_state: &mut RngState) -> Vec3 {
    let rng = rng_state.gen_r2();
    let r1_sqrt = rng.x.sqrt();
    (1.0 - r1_sqrt) * a + (r1_sqrt * (1.0 - rng.y)) * b + (r1_sqrt * rng.y) * c
}

// PDF of picking a point on a light source w.r.t area
// - light_area is the area of the light source
// - light_distance is the distance from the chosen point to the point being shaded
// - light_normal is the normal of the light source at the chosen point
// - light_direction is the direction from the light source to the point being shaded
pub fn calculate_light_pdf(light_area: f32, light_distance: f32, light_normal: Vec3, light_direction: Vec3) -> f32 {
    let cos_theta = light_normal.dot(-light_direction);
    if cos_theta <= 0.0 {
        return 0.0;
    }
    light_distance.powi(2) / (light_area * cos_theta)
}

pub fn sample_direct_lighting(
    index_buffer: &[UVec4],
    per_vertex_buffer: &[PerVertexData],
    material_data_buffer: &[MaterialData],
    light_pick_buffer: &[LightPickEntry],
    bvh: &BVHReference,
    surface_bsdf: &impl BSDF,
    surface_point: Vec3,
    surface_normal: Vec3,
    surface_sample: &BSDFSample,
    ray_direction: Vec3,
    rng_state: &mut RngState
) -> Vec3 {
    // If the first entry is a sentinel, there are no lights
    let mut direct = Vec3::ZERO;
    if light_pick_buffer[0].is_sentinel() {
        return direct;
    }

    // Pick a light, get its surface properties
    let (light_index, light_area, light_pick_pdf) = pick_light(&light_pick_buffer, rng_state);
    let light_triangle = index_buffer[light_index as usize];
    let light_vert_a = per_vertex_buffer[light_triangle.x as usize].vertex.xyz();
    let light_vert_b = per_vertex_buffer[light_triangle.y as usize].vertex.xyz();
    let light_vert_c = per_vertex_buffer[light_triangle.z as usize].vertex.xyz();
    let light_norm_a = per_vertex_buffer[light_triangle.x as usize].normal.xyz();
    let light_norm_b = per_vertex_buffer[light_triangle.y as usize].normal.xyz();
    let light_norm_c = per_vertex_buffer[light_triangle.z as usize].normal.xyz();
    let light_normal = (light_norm_a + light_norm_b + light_norm_c) / 3.0; // lights can use flat shading, no need to pay for interpolation
    let light_material = material_data_buffer[light_triangle.w as usize];
    let light_emission = light_material.emissive.xyz();

    // Pick a point on the light
    let light_point = pick_triangle_point(light_vert_a, light_vert_b, light_vert_c, rng_state);
    let light_direction_unorm = light_point - surface_point;
    let light_distance = light_direction_unorm.length();
    let light_direction = light_direction_unorm / light_distance;

    // Sample the light directly using MIS
    let light_trace = bvh.intersect_front_to_back(
        per_vertex_buffer,
        index_buffer,
        surface_point + light_direction * util::EPS,
        light_direction
    );
    if light_trace.hit && light_trace.triangle_index == light_index {
        // Calculate light pdf for this sample
        let light_pdf = calculate_light_pdf(light_area, light_distance, light_normal, light_direction);
        if light_pdf > 0.0 {
            // Calculate BSDF attenuation for this sample
            let bsdf_attenuation = surface_bsdf.evaluate(-ray_direction, surface_normal, light_direction, bsdf::LobeType::DiffuseReflection);
            // Calculate BSDF pdf for this sample
            let bsdf_pdf = surface_bsdf.pdf(-ray_direction, surface_normal, light_direction, bsdf::LobeType::DiffuseReflection);
            if bsdf_pdf > 0.0 {
                // MIS - add the weighted sample
                let weight = util::power_heuristic(light_pdf, bsdf_pdf);
                direct += bsdf_attenuation * light_emission * weight / light_pdf;
            }
        }
    }
    
    // TODO: Don't do this trace twice! We will do it next iteration of the tracing loop
    // Sample the BSDF using MIS
    let bsdf_trace = bvh.intersect_front_to_back(
        per_vertex_buffer,
        index_buffer,
        surface_point + surface_sample.sampled_direction * util::EPS,
        surface_sample.sampled_direction
    );
    if bsdf_trace.hit && bsdf_trace.triangle_index == light_index {
        // Calculate the light pdf for this sample
        let light_pdf = calculate_light_pdf(light_area, bsdf_trace.t, light_normal, surface_sample.sampled_direction);
        if light_pdf > 0.0 {
            // MIS - add the weighted sample
            let weight = util::power_heuristic(surface_sample.pdf, light_pdf);
            direct += surface_sample.spectrum * light_emission * weight / surface_sample.pdf;
        }
    }

    // Add the direct contribution, weighted by the probability of picking this light
    direct / light_pick_pdf
}