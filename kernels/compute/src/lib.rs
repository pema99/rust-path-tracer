#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::{glam, spirv};
use spirv_std::num_traits::Float;
use glam::*;

#[spirv(compute(threads(64, 1, 1)))]
pub fn main_raygen(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] ray_origins: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] ray_dirs: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] throughput: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rng: &[UVec2],
) {
    let index = (id.y * 1280 + id.x) as usize;
    let r1 = rng[index].x as f32 / u32::MAX as f32;
    let r2 = rng[index].y as f32 / u32::MAX as f32;
    
    let suv = id.xy().as_vec2() + Vec2::new(r1, r2);
    let mut uv = Vec2::new(suv.x as f32 / 1280.0, 1.0 - suv.y as f32 / 720.0) * 2.0 - 1.0;
    uv.y *= 720.0 / 1280.0;

    let ro = Vec3::new(0.0, 0.0, -2.0);
    let rd = Vec3::new(uv.x, uv.y, 1.0).normalize();

    ray_origins[index] = ro.extend(0.0);
    ray_dirs[index] = rd.extend(0.0);
    throughput[index] = Vec4::ONE;
}

fn map(p: Vec3) -> f32 {
    // metaball
    let mut d = (p + vec3(0.4, 0.0, -0.5)).length() - 0.5;
    d = d.min((p + vec3(0.7, -0.2, -0.3)).length() - 0.4);

    // ball
    d = d.min((p + Vec3::new(-0.5, 0.0, 0.3)).length() - 0.5);

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

#[spirv(compute(threads(64, 1, 1)))]
pub fn main_raytrace(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] ray_origins: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] ray_dirs: &mut [Vec4],
) {
    let index = (id.y * 1280 + id.x) as usize;
    if ray_origins[index].w < 0.0 {
        return;
    }

    let dist = march(ray_origins[index].truncate(), ray_dirs[index].truncate());

    ray_origins[index].w = dist;
}

fn uniform_sample_sphere(r1: f32, r2: f32) -> Vec3 {
    let cos_phi = 2.0 * r1 - 1.0;
    let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
    let theta = 2.0 * core::f32::consts::PI * r2;
    Vec3::new(sin_phi * theta.cos(), cos_phi, sin_phi * theta.cos())
}

fn uniform_sample_hemisphere(r1: f32, r2: f32) -> Vec3 {
    let sin_theta = (1.0 - r1 * r1).sqrt();
    let phi = 2.0 * core::f32::consts::PI * r2;
    let x = sin_theta * phi.cos();
    let z = sin_theta * phi.sin();
    Vec3::new(x, r1, z)
}

fn cosine_sample_hemisphere(r1: f32, r2: f32) -> Vec3 {
    let theta = r1.sqrt().acos();
    let phi = 2.0 * core::f32::consts::PI * r2;
    Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin())
}

fn create_cartesian(up: Vec3) -> (Vec3, Vec3, Vec3) {
    let arbitrary = Vec3::new(0.1, 0.5, 0.9);
    let temp_vec = up.cross(arbitrary).normalize();
    let right = temp_vec.cross(up).normalize();
    let forward = up.cross(right).normalize();
    (up, right, forward)
}

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u32 + 2891336453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277803737u32;
    (word >> 22u32) ^ word
}

#[spirv(compute(threads(64, 1, 1)))]
pub fn main_material(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] ray_origins: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] ray_dirs: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] throughput: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rng: &mut [UVec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] output: &mut [Vec4],
) {
    let index = (id.y * 1280 + id.x) as usize;
    if ray_origins[index].w < 0.0 {
        return;
    }

    let ray_origin = ray_origins[index].xyz();
    let ray_dir = ray_dirs[index].xyz();
    let dist = ray_origins[index].w;
    let hit = ray_origin + ray_dir * dist;

    // ceiling emissive
    if hit.y > 0.95 {
        // white light
        output[index] += throughput[index] * Vec4::ONE;
        ray_origins[index].w = -1.0;
    } else {
        let r1 = rng[index].x as f32 / u32::MAX as f32;
        let r2 = rng[index].y as f32 / u32::MAX as f32;

        let norm = normal(hit);
        let (up, nt, nb) = create_cartesian(norm);
        let sample = uniform_sample_hemisphere(r1, r2);
        let sample_dir = Vec3::new(
            sample.x * nb.x + sample.y * up.x + sample.z * nt.x,
            sample.x * nb.y + sample.y * up.y + sample.z * nt.y,
            sample.x * nb.z + sample.y * up.z + sample.z * nt.z).normalize();

        // albedo = normal
        throughput[index] = throughput[index] * (norm * 0.5 + 0.5).extend(1.0);
        
        ray_dirs[index] = sample_dir.extend(0.0);
        ray_origins[index] = (hit + norm * 0.01).extend(0.0);
    }

    rng[index] = UVec2::new(pcg_hash(rng[index].x), pcg_hash(rng[index].y));
}