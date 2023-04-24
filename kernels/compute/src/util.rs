#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::glam::{Vec3, UVec2, Vec2};

pub fn uniform_sample_sphere(r1: f32, r2: f32) -> Vec3 {
    let cos_phi = 2.0 * r1 - 1.0;
    let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
    let theta = 2.0 * core::f32::consts::PI * r2;
    Vec3::new(sin_phi * theta.cos(), cos_phi, sin_phi * theta.cos())
}

pub fn uniform_sample_hemisphere(r1: f32, r2: f32) -> Vec3 {
    let sin_theta = (1.0 - r1 * r1).sqrt();
    let phi = 2.0 * core::f32::consts::PI * r2;
    let x = sin_theta * phi.cos();
    let z = sin_theta * phi.sin();
    Vec3::new(x, r1, z)
}

pub fn cosine_sample_hemisphere(r1: f32, r2: f32) -> Vec3 {
    let theta = r1.sqrt().acos();
    let phi = 2.0 * core::f32::consts::PI * r2;
    Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin())
}

pub fn create_cartesian(up: Vec3) -> (Vec3, Vec3, Vec3) {
    let arbitrary = Vec3::new(0.1, 0.5, 0.9);
    let temp_vec = up.cross(arbitrary).normalize();
    let right = temp_vec.cross(up).normalize();
    let forward = up.cross(right).normalize();
    (up, right, forward)
}

pub fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u32 + 2891336453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277803737u32;
    (word >> 22u32) ^ word
}

type RngState = UVec2;
pub fn gen_random(rng: &mut RngState) -> Vec2 {
    rng.x = pcg_hash(rng.x);
    rng.y = pcg_hash(rng.y);
    Vec2::new(rng.x as f32 / u32::MAX as f32, rng.y as f32 / u32::MAX as f32)
}