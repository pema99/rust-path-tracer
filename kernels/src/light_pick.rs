use shared_structs::{LightPickEntry, PerVertexData, MaterialData};
use spirv_std::glam::{Vec3, UVec4, Vec4Swizzles};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use crate::{rng::RngState, util, bsdf::{self, BSDF}, intersection::BVHReference};

pub fn pick_light(table: &[LightPickEntry], rng_state: &mut RngState) -> (u32, f32) {
    let rng = rng_state.gen_r2();
    let entry = table[(rng.x * table.len() as f32) as usize];
    if rng.y < entry.ratio {
        (entry.triangle_index_a, entry.triangle_area_a)
    } else {
        (entry.triangle_index_b, entry.triangle_area_b)
    }
}

// https://www.cs.princeton.edu/~funk/tog02.pdf equation 1
pub fn pick_triangle_point(a: Vec3, b: Vec3, c: Vec3, rng_state: &mut RngState) -> Vec3 {
    let rng = rng_state.gen_r2();
    let r1_sqrt = rng.x.sqrt();
    (1.0 - r1_sqrt) * a + (r1_sqrt * (1.0 - rng.y)) * b + (r1_sqrt * rng.y) * c
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
    ray_direction: Vec3,
    rng_state: &mut RngState
) -> Vec3 {
    let (light_index, light_area) = pick_light(&light_pick_buffer, rng_state);
    let light_triangle = index_buffer[light_index as usize];
    let light_vert_a = per_vertex_buffer[light_triangle.x as usize].vertex.xyz();
    let light_vert_b = per_vertex_buffer[light_triangle.y as usize].vertex.xyz();
    let light_vert_c = per_vertex_buffer[light_triangle.z as usize].vertex.xyz();
    let light_norm_a = per_vertex_buffer[light_triangle.x as usize].normal.xyz();
    let light_norm_b = per_vertex_buffer[light_triangle.y as usize].normal.xyz();
    let light_norm_c = per_vertex_buffer[light_triangle.z as usize].normal.xyz();
    let light_material = material_data_buffer[light_triangle.w as usize];
    let light_emission = light_material.emissive.xyz();

    let light_pick_pdf = 0.5; // TODO: Hardcoded 2 equal sized lights
    //let light_direction = 
    //bsdf.sample(view_direction, normal, rng)
    let light_point = pick_triangle_point(light_vert_a, light_vert_b, light_vert_c, rng_state);
    let light_direction_unorm = light_point - surface_point;
    let light_distance = light_direction_unorm.length();
    let light_direction = light_direction_unorm / light_distance;

    let light_bary = util::barycentric(light_point, light_vert_a, light_vert_b, light_vert_c);
    let light_normal = light_bary.x * light_norm_a + light_bary.y * light_norm_b + light_bary.z * light_norm_c;

    let light_trace = bvh.intersect_front_to_back(
        per_vertex_buffer,
        index_buffer,
        surface_point + light_direction * util::EPS,
        light_direction);
    if light_trace.hit && light_trace.triangle_index == light_index {
        let bsdf_attenuation = surface_bsdf.evaluate(-ray_direction, surface_normal, light_direction, bsdf::LobeType::DiffuseReflection);
        let light_pdf = light_distance.powi(2) / (light_area * light_normal.dot(-light_direction));
        if light_pdf > 0.0 && bsdf_attenuation != Vec3::ZERO {
            return bsdf_attenuation * light_emission / (light_pdf * light_pick_pdf)
        }
    }
    Vec3::ZERO
}