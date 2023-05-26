use shared_structs::{LightPickEntry, PerVertexData, MaterialData, NextEventEstimation};
use spirv_std::glam::{Vec3, UVec4, Vec4Swizzles};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use crate::{rng::RngState, util, bsdf::{self, BSDF}, intersection::{BVHReference, self}};

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
    /* This warrants some explanation for my future dumb self:
    (In case anyone but me reads this, I use "mathover" VSCode extension to render the LaTeX inline)
    When we estimate the rendering equation by monte carlo integration, we typically integrate over the solid angle domain,
    but with direct light sampling, we use the (surface) area domain of the light sources, and need to convert between the 2.
    
    An integral for direct lighting can be written as so:
    # Math: \int_{\Omega_d} f_r(x, w_i, w_o) L_i(x, w_i) w_i \cdot w_n dw
    Where \Omega_d crucially denotes only the part of the hemisphere - the part which the visible lights project onto.
    This is similar to the rendering equation, except we know that the incoming ray is coming from a light source.

    We can instead integrate over area domain like this:
    # Math: \int_\triangle f_r(x, w_i, w_o) L_i(x, w_i) w_i \cdot w_n \frac{-w_i \cdot n}{r^2} dA
    Where r is the distance to the light source, n is the surface normal of the light source, and \triangle denotes the
    domain of light source surface area. dA is thus projected differential area, centered around the point on the light source which we hit.
    This uses the fact that:
    # Math: dA = r^2 \frac{1}{-w \cdot n} dw = \frac{r^2}{-w \cdot n} dw
    Which can be rewritten to:
    # Math: dw = \frac{-w \cdot n}{r^2} dA
    The intuition behind this is fairly straight forward. As we move the light source further from the shading point,
    by a distance of r, the projected area grows by a factor of r^2. This is the inverse square law, and the reason for the r^2 term.
    As we tilt the light source away from the shading point, the projected area also grows. The factor with which it grows is determined
    by the cosine of the angle between the surface normal on the light source and the negated incoming light direction. As the angle grows,
    the cosine shrinks. Since the area should grow and not shrink when this happens, we have the \frac{1}{-w_i \cdot n} term. Since both 
    vectors are normalized, that dot product is exactly the cosine of the angle. See https://arxiv.org/abs/1205.4447 for more details.

    We can write out an estimator for our integral over area domain, assuming uniform sampling over the surface of the light source:
    # Math: \frac{1}{n}\sum^n \frac{f_r(x, w_i, w_o) L_i(x, w_i) w_i \cdot w_n \frac{-w_i \cdot n}{r^2}}{\frac{1}{|\triangle|}}
    Where |\triangle| is the surface area of the light source.
    We can further simplify to:
    # Math: \frac{1}{n}\sum^n f_r(x, w_i, w_o) L_i(x, w_i) w_i \cdot w_n \frac{(-w_i \cdot n) |\triangle|}{r^2}
    Note now that this last part:
    # Math: \frac{(-w_i \cdot n) |\triangle|}{r^2}
    Is precisely the reciprocal of the formula you see written out in code below. The reason I use the reciprocal
    is because I am dividing rather than multiplying this weighting term. It's an implementation detail.

    Note what this tell us about direct light sampling - to evaluate direct lighting, for a given bounce, we choose a direction towards
    a light source, and multiply the incoming radiance (light source emission) by the BRDF, by the regular cosine term (often folded into the BRDF),
    and by the weighting factor described just above - there are 2 cosine at terms at play!

    This explanation doesn't include visibility checks or weight for multiple different sources, though, so let me briefly describe.
    When we don't pass a visibility check (ie. the chosen light point is occluded), we simply don't add the contribution, since the
    probability of hitting that point is 0. When we have multiple light sources, we simply pick one at random and divide the contribution
    by the probability of picking the given light source. This is just splitting the estimator into multiple addends. */
    let cos_theta = light_normal.dot(-light_direction);
    if cos_theta <= 0.0 {
        return 0.0;
    }
    light_distance.powi(2) / (light_area * cos_theta)
}

pub fn get_weight(nee_mode: NextEventEstimation, p1: f32, p2: f32) -> f32 {
    match nee_mode {
        NextEventEstimation::None => 1.0,
        NextEventEstimation::MultipleImportanceSampling => util::power_heuristic(p1, p2),
        NextEventEstimation::DirectLightSampling => 1.0,
    }
}

#[derive(Default, Copy, Clone)]
pub struct DirectLightSample {
    pub light_area: f32,
    pub light_normal: Vec3,
    pub light_pick_pdf: f32,
    pub light_emission: Vec3,
    pub light_triangle_index: u32,
    pub throughput: Vec3,
    pub direct_light_contribution: Vec3,
}

pub fn sample_direct_lighting(
    nee_mode: NextEventEstimation,
    index_buffer: &[UVec4],
    per_vertex_buffer: &[PerVertexData],
    material_data_buffer: &[MaterialData],
    light_pick_buffer: &[LightPickEntry],
    bvh: &BVHReference,
    throughput: Vec3,
    surface_bsdf: &impl BSDF,
    surface_point: Vec3,
    surface_normal: Vec3,
    ray_direction: Vec3,
    rng_state: &mut RngState,
) -> DirectLightSample {
    // If the first entry is a sentinel, there are no lights
    let mut info = DirectLightSample::default();
    if light_pick_buffer[0].is_sentinel() {
        return info;
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
    let mut direct = Vec3::ZERO;
    let light_trace = bvh.intersect_any(
        per_vertex_buffer,
        index_buffer,
        surface_point + light_direction * util::EPS,
        light_direction,
        light_distance - util::EPS * 2.0,
    );
    if !light_trace.hit {
        // Calculate light pdf for this sample
        let light_pdf = calculate_light_pdf(light_area, light_distance, light_normal, light_direction);
        if light_pdf > 0.0 {
            // Calculate BSDF attenuation for this sample
            let bsdf_attenuation = surface_bsdf.evaluate(-ray_direction, surface_normal, light_direction, bsdf::LobeType::DiffuseReflection);
            // Calculate BSDF pdf for this sample
            let bsdf_pdf = surface_bsdf.pdf(-ray_direction, surface_normal, light_direction, bsdf::LobeType::DiffuseReflection);
            if bsdf_pdf > 0.0 {
                // MIS - add the weighted sample
                let weight = get_weight(nee_mode, light_pdf, bsdf_pdf);
                direct = (bsdf_attenuation * light_emission * weight / light_pdf) / light_pick_pdf;
            }
        }
    }

    // Write out data for the next bounce to use
    info.light_area = light_area;
    info.light_normal = light_normal;
    info.light_pick_pdf = light_pick_pdf;
    info.light_emission = light_emission;
    info.light_triangle_index = light_index;
    info.throughput = throughput;
    info.direct_light_contribution = throughput * direct;
    info
}

// If this is being called, the assumption is that:
// - We are using NEE with MIS
// - We have hit a light source
// - That last bounce was diffuse, so we did direct light sampling
pub fn calculate_bsdf_mis_contribution(
    trace_result: &intersection::TraceResult,
    last_bsdf_sample: &bsdf::BSDFSample,
    last_light_sample: &DirectLightSample
) -> Vec3 {
    // If we haven't hit the same light as we sampled directly, no contribution
    if trace_result.triangle_index != last_light_sample.light_triangle_index {
        return Vec3::ZERO;
    }

    // Calculate the light pdf for this sample
    let light_pdf = calculate_light_pdf(last_light_sample.light_area, trace_result.t, last_light_sample.light_normal, last_bsdf_sample.sampled_direction);
    if light_pdf > 0.0 {
        // MIS - add the weighted sample
        let weight = get_weight(NextEventEstimation::MultipleImportanceSampling, last_bsdf_sample.pdf, light_pdf);
        let direct = (last_bsdf_sample.spectrum * last_light_sample.light_emission * weight / last_bsdf_sample.pdf) / last_light_sample.light_pick_pdf;
        last_light_sample.throughput * direct
    } else {
        Vec3::ZERO
    }
}