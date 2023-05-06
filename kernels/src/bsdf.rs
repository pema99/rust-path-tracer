use shared_structs::MaterialData;
use spirv_std::{glam::{Vec3, Vec2, Vec4Swizzles}, Sampler, Image};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use crate::{rng, util};

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
    fn evaluate(
        &self,
        view_direction: Vec3,
        normal: Vec3,
        sample_direction: Vec3,
        lobe_type: LobeType,
    ) -> Spectrum;
    fn sample(&self, view_direction: Vec3, normal: Vec3, rng: &mut rng::RngState) -> BSDFSample;
    fn pdf(
        &self,
        view_direction: Vec3,
        normal: Vec3,
        sample_direction: Vec3,
        lobe_type: LobeType,
    ) -> f32;
}

pub struct Lambertian {
    pub albedo: Spectrum,
}

impl Lambertian {
    fn pdf_fast(&self, cos_theta: f32) -> f32 {
        cos_theta / core::f32::consts::PI
    }

    fn evaluate_fast(&self, cos_theta: f32) -> Spectrum {
        self.albedo / core::f32::consts::PI * cos_theta
    }
}

impl BSDF for Lambertian {
    fn evaluate(
        &self,
        _view_direction: Vec3,
        normal: Vec3,
        sample_direction: Vec3,
        _lobe_type: LobeType,
    ) -> Spectrum {
        let cos_theta = normal.dot(sample_direction).max(0.0);
        self.evaluate_fast(cos_theta)
    }

    fn sample(&self, _view_direction: Vec3, normal: Vec3, rng: &mut rng::RngState) -> BSDFSample {
        let (up, nt, nb) = util::create_cartesian(normal);
        let rng_sample = rng.gen_r3();
        let sample = util::cosine_sample_hemisphere(rng_sample.x, rng_sample.y);
        let sampled_direction = Vec3::new(
            sample.x * nb.x + sample.y * up.x + sample.z * nt.x,
            sample.x * nb.y + sample.y * up.y + sample.z * nt.y,
            sample.x * nb.z + sample.y * up.z + sample.z * nt.z,
        )
        .normalize();

        let sampled_lobe = LobeType::Diffuse;
        let cos_theta = normal.dot(sampled_direction).max(0.0);
        let pdf = self.pdf_fast(cos_theta);
        let spectrum = self.evaluate_fast(cos_theta);
        BSDFSample {
            pdf,
            sampled_lobe,
            spectrum,
            sampled_direction,
        }
    }

    fn pdf(
        &self,
        _view_direction: Vec3,
        normal: Vec3,
        sample_direction: Vec3,
        _lobe_type: LobeType,
    ) -> f32 {
        let cos_theta = normal.dot(sample_direction).max(0.0);
        self.pdf_fast(cos_theta)
    }
}

pub struct PBR {
    pub albedo: Spectrum,
    pub roughness: f32,
    pub metallic: f32,
}

impl PBR {
    fn evaluate_diffuse_fast(
        &self,
        cos_theta: f32,
        diffuse_specular_ratio: f32,
        ks: Vec3,
    ) -> Spectrum {
        let kd = (Vec3::splat(1.0) - ks) * (1.0 - self.metallic);
        let diffuse = kd * self.albedo / core::f32::consts::PI;
        diffuse * cos_theta / (1.0 - diffuse_specular_ratio)
    }

    fn evaluate_specular_fast(
        &self,
        view_direction: Vec3,
        normal: Vec3,
        sample_direction: Vec3,
        cos_theta: f32,
        d_term: f32,
        diffuse_specular_ratio: f32,
        ks: Vec3,
    ) -> Spectrum {
        let roughness = self.roughness.max(util::EPS);
        let g_term = util::geometry_smith(normal, view_direction, sample_direction, roughness);
        let specular_numerator = d_term * g_term * ks;
        let specular_denominator = 4.0 * normal.dot(view_direction).max(0.0) * cos_theta;
        let specular = specular_numerator / specular_denominator.max(util::EPS);
        specular * cos_theta / diffuse_specular_ratio
    }

    fn pdf_diffuse_fast(&self, cos_theta: f32) -> f32 {
        cos_theta / core::f32::consts::PI
    }

    fn pdf_specular_fast(
        &self,
        view_direction: Vec3,
        normal: Vec3,
        halfway: Vec3,
        d_term: f32,
    ) -> f32 {
        (d_term * normal.dot(halfway)) / (4.0 * view_direction.dot(halfway))
    }
}

impl BSDF for PBR {
    fn evaluate(
        &self,
        view_direction: Vec3,
        normal: Vec3,
        sample_direction: Vec3,
        lobe_type: LobeType,
    ) -> Spectrum {
        let diffuse_specular_ratio = 0.5 + 0.5 * self.metallic; // wtf is this

        let cos_theta = normal.dot(sample_direction).max(0.0);
        let halfway = (view_direction + sample_direction).normalize();

        let f0 = Vec3::splat(0.04).lerp(self.albedo, self.metallic);
        let ks = util::fresnel_schlick(halfway.dot(view_direction).max(0.0), f0);

        match lobe_type {
            LobeType::Diffuse => self.evaluate_diffuse_fast(cos_theta, diffuse_specular_ratio, ks),
            LobeType::Specular => {
                let roughness = self.roughness.max(util::EPS);
                let d_term = util::ggx_distribution(normal, halfway, roughness);
                self.evaluate_specular_fast(
                    view_direction,
                    normal,
                    sample_direction,
                    cos_theta,
                    d_term,
                    diffuse_specular_ratio,
                    ks,
                )
            }
        }
    }

    fn sample(&self, view_direction: Vec3, normal: Vec3, rng: &mut rng::RngState) -> BSDFSample {
        let rng_sample = rng.gen_r3();
        let diffuse_specular_ratio = 0.5 + 0.5 * self.metallic; // wtf is this

        let roughness = self.roughness.max(util::EPS);

        let (sampled_direction, sampled_lobe) = if rng_sample.z > diffuse_specular_ratio {
            let (up, nt, nb) = util::create_cartesian(normal);
            let sample = util::cosine_sample_hemisphere(rng_sample.x, rng_sample.y);
            let sampled_direction = Vec3::new(
                sample.x * nb.x + sample.y * up.x + sample.z * nt.x,
                sample.x * nb.y + sample.y * up.y + sample.z * nt.y,
                sample.x * nb.z + sample.y * up.z + sample.z * nt.z,
            )
            .normalize();
            (sampled_direction, LobeType::Diffuse)
        } else {
            let reflection_direction = util::reflect(-view_direction, normal);
            let sampled_direction = util::sample_ggx(
                rng_sample.x,
                rng_sample.y,
                reflection_direction,
                roughness,
            );
            (sampled_direction, LobeType::Specular)
        };

        let cos_theta = normal.dot(sampled_direction).max(0.0);
        let halfway = (view_direction + sampled_direction).normalize();

        let f0 = Vec3::splat(0.04).lerp(self.albedo, self.metallic);
        let ks = util::fresnel_schlick(halfway.dot(view_direction).max(0.0), f0);

        let (sampled_direction, sampled_lobe, pdf, spectrum) = match sampled_lobe {
            LobeType::Diffuse => {
                let pdf = self.pdf_diffuse_fast(cos_theta);
                let spectrum = self.evaluate_diffuse_fast(cos_theta, diffuse_specular_ratio, ks);
                (sampled_direction, LobeType::Diffuse, pdf, spectrum)
            }
            LobeType::Specular => {
                let d_term = util::ggx_distribution(normal, halfway, roughness);
                let pdf = self.pdf_specular_fast(view_direction, normal, halfway, d_term);
                let spectrum = self.evaluate_specular_fast(
                    view_direction,
                    normal,
                    sampled_direction,
                    cos_theta,
                    d_term,
                    diffuse_specular_ratio,
                    ks,
                );
                (sampled_direction, LobeType::Specular, pdf, spectrum)
            }
        };

        BSDFSample {
            pdf,
            sampled_lobe,
            spectrum,
            sampled_direction,
        }
    }

    fn pdf(
        &self,
        view_direction: Vec3,
        normal: Vec3,
        sample_direction: Vec3,
        lobe_type: LobeType,
    ) -> f32 {
        match lobe_type {
            LobeType::Diffuse => {
                let cos_theta = normal.dot(sample_direction).max(0.0);
                self.pdf_diffuse_fast(cos_theta)
            }
            LobeType::Specular => {
                let halfway = (view_direction + sample_direction).normalize();
                let roughness = self.roughness.max(util::EPS);
                let d_term = util::ggx_distribution(normal, halfway, roughness);
                self.pdf_specular_fast(view_direction, normal, halfway, d_term)
            }
        }
    }
}

pub fn get_pbr_bsdf(material: &MaterialData, uv: Vec2, atlas: &Image!(2D, type=f32, sampled), sampler: &Sampler) -> PBR {
    let albedo = if material.has_albedo_texture() {
        let scaled_uv = material.albedo.xy() + uv * material.albedo.zw();
        let albedo = atlas.sample_by_lod(*sampler, scaled_uv, 0.0);
        albedo.xyz()
    } else {
        material.albedo.xyz()
    };
    let roughness = if material.has_roughness_texture() {
        let scaled_uv = material.roughness.xy() + uv * material.roughness.zw();
        let roughness = atlas.sample_by_lod(*sampler, scaled_uv, 0.0);
        roughness.x
    } else {
        material.roughness.x
    };
    let metallic = if material.has_metallic_texture() {
        let scaled_uv = material.metallic.xy() + uv * material.metallic.zw();
        let metallic = atlas.sample_by_lod(*sampler, scaled_uv, 0.0);
        metallic.x
    } else {
        material.metallic.x
    };

    PBR {
        albedo,
        roughness,
        metallic,
    }
}