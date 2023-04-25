#![cfg_attr(target_arch = "spirv", no_std)]

use bsdf::BSDF;
use glam::*;
use shared_structs::TracingConfig;
#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::{glam, spirv};

mod bsdf;
mod rng;
mod util;

fn map(p: Vec3) -> f32 {
    // metaball
    let mut d = (p - vec3(-0.4, -0.2, 0.5)).length() - 0.5;
    d = d.min((p - vec3(-0.7, -0.4, 0.3)).length() - 0.4);

    // ball
    d = d.min((p - Vec3::new(0.5, -0.2, -0.3)).length() - 0.5);

    // walls
    d = d.min((p.y + 1.0).abs());
    d = d.min((p.y - 1.0).abs());
    d = d.min((p.x + 1.5).abs());
    d = d.min((p.x - 1.5).abs());
    d = d.min((p.z + 3.5).abs());
    d = d.min((p.z - 2.0).abs());

    d
}

fn normal(p: Vec3) -> Vec3 {
    let h = Vec2::new(0.0, 0.0001);
    Vec3::new(
        map(p + h.yxx()) - map(p - h.yxx()),
        map(p + h.xyx()) - map(p - h.xyx()),
        map(p + h.xxy()) - map(p - h.xxy()),
    )
    .normalize()
}

fn march(ro: Vec3, rd: Vec3) -> f32 {
    let mut t = 0.0;
    for _ in 0..128 {
        let d = map(ro + rd * t);
        if d < 0.001 {
            return t;
        }
        t += d;
    }
    -1.0
}

// Adapted from raytri.c
fn muller_trumbore(ro: Vec3, rd: Vec3, a: Vec3, b: Vec3, c: Vec3, out_t: &mut f32) -> bool
{
    *out_t = 0.0;

    let edge1 = b - a;
    let edge2 = c - a;
 
    // begin calculating determinant - also used to calculate U parameter
    let pv = rd.cross(edge2);
 
    // if determinant is near zero, ray lies in plane of triangle
    let det = edge1.dot(pv);
 
    if det.abs() < 1e-6 {
        return false;
    }
 
    let inv_det = 1.0 / det;
 
    // calculate distance from vert0 to ray origin
    let tv = ro - a;
 
    // calculate U parameter and test bounds
    let u = tv.dot(pv) * inv_det;
    if u < 0.0 || u > 1.0 {
        return false;
    }
 
    // prepare to test V parameter
    let qv = tv.cross(edge1);
 
    // calculate V parameter and test bounds
    let v = rd.dot(qv) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
 
    let t = edge2.dot(qv) * inv_det;
    if t < 0.0 {
        return false;
    }
    *out_t = t;
 
    return true;
}

const MAX_DIST: f32 = 1000000.0;

struct TraceResult {
    t: f32,
    triangle: UVec4,
    hit: bool,
}

impl Default for TraceResult {
    fn default() -> Self {
        Self {
            t: 1000000.0,
            triangle: UVec4::splat(0),
            hit: false,
        }
    }
}

fn trace_slow_as_shit(
    vertex_buffer: &[Vec4],
    index_buffer: &[UVec4],
    ro: Vec3,
    rd: Vec3
) -> TraceResult {
    let mut result = TraceResult::default();
    for i in 0..index_buffer.len() {
        let triangle = index_buffer[i];
        let a = vertex_buffer[triangle.x as usize].xyz();
        let b = vertex_buffer[triangle.y as usize].xyz();
        let c = vertex_buffer[triangle.z as usize].xyz();

        let mut t = 0.0;
        if muller_trumbore(ro, rd, a, b, c, &mut t) && t > 0.001 && t < result.t {
            result.t = result.t.min(t);
            result.triangle = triangle;
            result.hit = true;
        }
    }
    result
}

#[spirv(compute(threads(8, 8, 1)))]
pub fn main_material(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &TracingConfig,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rng: &mut [UVec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] vertex_buffer: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] index_buffer: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] normal_buffer: &[Vec4],
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

    let mut throughput = Vec3::ONE;
    for _ in 0..4 {
        let trace_result = trace_slow_as_shit(vertex_buffer, index_buffer, ray_origin, ray_direction);
        let hit = ray_origin + ray_direction * trace_result.t;
        //let dist = march(ray_origin, ray_direction);
        //let hit = ray_origin + ray_direction * dist;


        if !trace_result.hit {
            // skybox
            output[index] += (throughput * Vec3::ONE).extend(1.0);
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
