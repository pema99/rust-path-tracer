#![cfg_attr(target_arch = "spirv", no_std)]

use bsdf::BSDF;
use shared_structs::TracingConfig;
#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::{glam, spirv};
use glam::*;
use util::gen_random;

mod util;
mod bsdf;

fn map(p: Vec3) -> f32 {
    // metaball
    let mut d = (p - vec3(-0.4, -0.2, 0.5)).length() - 0.5;
    d = d.min((p - vec3(-0.7, -0.4, 0.3)).length() - 0.4);

    // ball
    d = d.min((p - Vec3::new(0.5, -0.2, -0.3)).length() - 0.5);

    // walls
    d = d.min((p.y+1.0).abs());
    d = d.min((p.y-1.0).abs());
    d = d.min((p.x+1.5).abs());
    d = d.min((p.x-1.5).abs());
    d = d.min((p.z+3.5).abs());
    d = d.min((p.z-2.0).abs());

    d
}

fn normal(p: Vec3) -> Vec3 {
    let h = Vec2::new(0.0, 0.0001);
    Vec3::new(
        map(p + h.yxx()) - map(p - h.yxx()),
        map(p + h.xyx()) - map(p - h.xyx()),
        map(p + h.xxy()) - map(p - h.xxy()),
    ).normalize()
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

#[spirv(compute(threads(8, 8, 1)))]
pub fn main_material(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &TracingConfig,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rng: &mut [UVec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [Vec4],
) {
    let index = (id.y * config.width + id.x) as usize;
    
    // Handle non-divisible workgroup sizes.
    if id.x > config.width || id.y > config.height {
        return;
    }

    // Get anti-aliased pixel coordinates.
    let suv = id.xy().as_vec2() + gen_random(&mut rng[index]);
    let mut uv = Vec2::new(suv.x as f32 / config.width as f32, 1.0 - suv.y as f32 / config.height as f32) * 2.0 - 1.0;
    uv.y *= config.height as f32 / config.width as f32;

    // Setup camera.
    let mut ray_origin = Vec3::new(0.0, 0.0, -2.0);
    let mut ray_direction = Vec3::new(uv.x, uv.y, 1.0).normalize();

    let mut throughput = Vec3::ONE;
    for _ in 0..4 {
        let dist = march(ray_origin, ray_direction);
        let hit = ray_origin + ray_direction * dist;

        // ceiling emissive
        if hit.y > 0.95 {
            // white light
            output[index] += (throughput * Vec3::ONE).extend(1.0);
            return;
        } else {
            let rng_sample = gen_random(&mut rng[index]);

            let norm = normal(hit);
            let bsdf = bsdf::Lambertian { albedo: norm * 0.5 + 0.5 };
            let bsdf_sample = bsdf.sample(ray_direction, norm, rng_sample);
            throughput *= bsdf_sample.spectrum / bsdf_sample.pdf;
            
            ray_direction = bsdf_sample.sampled_direction;
            ray_origin = hit + norm * 0.01;
        }
    }
}