#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::glam::{Vec3, Vec2};

use crate::util;

type Spectrum = Vec3;

#[derive(Copy, Clone)]
pub enum LobeType {
    Diffuse,
    Specular,
}

#[derive(Copy, Clone)]
pub struct BSDFSample {
    pub pdf: f32,
    pub sampled_lobe: LobeType,
    pub spectrum: Spectrum,
    pub sampled_direction: Vec3,
}

pub trait BSDF {
    fn evaluate(&self, view_direction: Vec3, normal: Vec3, sample_direction: Vec3, lobe_type: LobeType) -> Spectrum;
    fn sample(&self, view_direction: Vec3, normal: Vec3, rng: Vec2) -> BSDFSample;
    fn pdf(&self, view_direction: Vec3, normal: Vec3, sample_direction: Vec3, lobe_type: LobeType) -> f32;
}

pub struct Lambertian {
    pub albedo: Spectrum,
}

impl BSDF for Lambertian {
    fn evaluate(&self, view_direction: Vec3, normal: Vec3, sample_direction: Vec3, lobe_type: LobeType) -> Spectrum {
        let cos_theta = normal.dot(sample_direction).max(0.0);
        self.albedo / core::f32::consts::PI * cos_theta
    }

    fn sample(&self, view_direction: Vec3, normal: Vec3, rng: Vec2) -> BSDFSample {
        let (up, nt, nb) = util::create_cartesian(normal);
        let sample = util::cosine_sample_hemisphere(rng.x, rng.y);
        let sampled_direction = Vec3::new(
            sample.x * nb.x + sample.y * up.x + sample.z * nt.x,
            sample.x * nb.y + sample.y * up.y + sample.z * nt.y,
            sample.x * nb.z + sample.y * up.z + sample.z * nt.z).normalize();

        let sampled_lobe = LobeType::Diffuse;
        let pdf = self.pdf(view_direction, normal, sampled_direction, sampled_lobe);
        let spectrum = self.evaluate(view_direction, normal, sampled_direction, sampled_lobe);
        BSDFSample {
            pdf,
            sampled_lobe,
            spectrum,
            sampled_direction,
        }
    }

    fn pdf(&self, view_direction: Vec3, normal: Vec3, sample_direction: Vec3, lobe_type: LobeType) -> f32 {
        let cos_theta = normal.dot(sample_direction).max(0.0);
        cos_theta / core::f32::consts::PI
    }
}