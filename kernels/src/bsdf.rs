use shared_structs::{MaterialData, TracingConfig};
use spirv_std::{glam::{Vec3, Vec2, Vec4Swizzles}};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use crate::{rng, util::{self}};
use shared_structs::{Image, Sampler};

type Spectrum = Vec3;

#[derive(Default, Copy, Clone, PartialEq)]
#[repr(u32)]
pub enum LobeType {
    #[default] DiffuseReflection,
    SpecularReflection,
    #[allow(dead_code)] DiffuseTransmission,
    SpecularTransmission,
}

#[derive(Default, Copy, Clone)]
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

        let sampled_lobe = LobeType::DiffuseReflection;
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

pub struct Glass {
    pub albedo: Spectrum,
    pub ior: f32,
    pub roughness: f32,
}

impl BSDF for Glass {
    fn evaluate(
        &self,
        _view_direction: Vec3,
        _normal: Vec3,
        _sample_direction: Vec3,
        lobe_type: LobeType,
    ) -> Spectrum {
        if lobe_type == LobeType::SpecularReflection {
            Vec3::ONE // This is 1 because glass is fully non-metallic
        } else {
            self.albedo
        }
    }

    fn sample(&self, view_direction: Vec3, normal: Vec3, rng: &mut rng::RngState) -> BSDFSample {
        let rng_sample = rng.gen_r3();

        let inside = normal.dot(view_direction) < 0.0;
        let normal = if inside { -normal } else { normal };
        let in_ior = if inside { self.ior } else { 1.0 };
        let out_ior = if inside { 1.0 } else { self.ior }; 

        let microsurface_normal = util::sample_ggx_microsurface_normal(rng_sample.x, rng_sample.y, normal, self.roughness);
        let fresnel = util::fresnel_schlick_scalar(in_ior, out_ior, microsurface_normal.dot(view_direction).max(0.0));
        if rng_sample.z <= fresnel {
            // Reflection
            let sampled_direction = (2.0 * view_direction.dot(microsurface_normal).abs() * microsurface_normal - view_direction).normalize();
            let pdf = 1.0;
            let sampled_lobe = LobeType::SpecularReflection;
            let spectrum = Vec3::ONE;
            BSDFSample {
                pdf,
                sampled_lobe,
                spectrum,
                sampled_direction,
            }
        } else {
            // Refraction
            let eta = in_ior / out_ior;
            let c = view_direction.dot(microsurface_normal);
            let sampled_direction = ((eta * c - (view_direction.dot(normal)).signum() * (1.0 + eta * (c * c - 1.0)).max(0.0).sqrt()) * microsurface_normal - eta * view_direction).normalize();
            let pdf = 1.0;
            let sampled_lobe = LobeType::SpecularTransmission;
            let spectrum = self.albedo;
            BSDFSample {
                pdf,
                sampled_lobe,
                spectrum,
                sampled_direction,
            }
        }
    }

    fn pdf(
        &self,
        _view_direction: Vec3,
        _normal: Vec3,
        _sample_direction: Vec3,
        _lobe_type: LobeType,
    ) -> f32 {
        1.0 // Delta distribution
    }
}

// Assume IOR of 1.5 for dielectrics, which works well for most.
const DIELECTRIC_IOR: f32 = 1.5;

// Fresnel at normal incidence for dielectrics, with air as the other medium.
const DIELECTRIC_F0_SQRT: f32 = (DIELECTRIC_IOR - 1.0) / (DIELECTRIC_IOR + 1.0);
const DIELECTRIC_F0: f32 = DIELECTRIC_F0_SQRT * DIELECTRIC_F0_SQRT;

pub struct PBR {
    pub albedo: Spectrum,
    pub roughness: f32,
    pub metallic: f32,
    pub specular_weight_clamp: Vec2,
}

impl PBR {
    fn evaluate_diffuse_fast(
        &self,
        cos_theta: f32,
        specular_weight: f32,
        ks: Vec3,
    ) -> Spectrum {
        let kd = (Vec3::splat(1.0) - ks) * (1.0 - self.metallic);
        let diffuse = kd * self.albedo / core::f32::consts::PI;
        diffuse * cos_theta / (1.0 - specular_weight)
    }

    fn evaluate_specular_fast(
        &self,
        view_direction: Vec3,
        normal: Vec3,
        sample_direction: Vec3,
        cos_theta: f32,
        d_term: f32,
        specular_weight: f32,
        ks: Vec3,
    ) -> Spectrum {
        let g_term = util::geometry_smith_schlick_ggx(normal, view_direction, sample_direction, self.roughness);
        let specular_numerator = d_term * g_term * ks;
        let specular_denominator = 4.0 * normal.dot(view_direction).max(0.0) * cos_theta;
        let specular = specular_numerator / specular_denominator.max(util::EPS);
        specular * cos_theta / specular_weight
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
        let approx_fresnel = util::fresnel_schlick_scalar(1.0, DIELECTRIC_IOR, normal.dot(view_direction).max(0.0));
        let mut specular_weight = util::lerp(approx_fresnel, 1.0, self.metallic);
        if specular_weight != 0.0 && specular_weight != 1.0 {
            specular_weight = specular_weight.clamp(self.specular_weight_clamp.x, self.specular_weight_clamp.y);
        }

        let cos_theta = normal.dot(sample_direction).max(0.0);
        let halfway = (view_direction + sample_direction).normalize();

        let f0 = Vec3::splat(DIELECTRIC_F0).lerp(self.albedo, self.metallic);
        let ks = util::fresnel_schlick(halfway.dot(view_direction).max(0.0), f0);

        if lobe_type == LobeType::DiffuseReflection {
            self.evaluate_diffuse_fast(cos_theta, specular_weight, ks)
        } else {
            let d_term = util::ggx_distribution(normal, halfway, self.roughness);
            self.evaluate_specular_fast(
                view_direction,
                normal,
                sample_direction,
                cos_theta,
                d_term,
                specular_weight,
                ks,
            )
        }
    }

    fn sample(&self, view_direction: Vec3, normal: Vec3, rng: &mut rng::RngState) -> BSDFSample {
        let rng_sample = rng.gen_r3();

        let approx_fresnel = util::fresnel_schlick_scalar(1.0, DIELECTRIC_IOR, normal.dot(view_direction).max(0.0));
        let mut specular_weight = util::lerp(approx_fresnel, 1.0, self.metallic);
        // Clamp specular weight to prevent firelies. See Jakub Boksansky and Adam Marrs in RT gems 2 chapter 14.
        if specular_weight != 0.0 && specular_weight != 1.0 {
            specular_weight = specular_weight.clamp(self.specular_weight_clamp.x, self.specular_weight_clamp.y);
        }

        let (sampled_direction, sampled_lobe) = if rng_sample.z >= specular_weight {
            let (up, nt, nb) = util::create_cartesian(normal);
            let sample = util::cosine_sample_hemisphere(rng_sample.x, rng_sample.y);
            let sampled_direction = Vec3::new(
                sample.x * nb.x + sample.y * up.x + sample.z * nt.x,
                sample.x * nb.y + sample.y * up.y + sample.z * nt.y,
                sample.x * nb.z + sample.y * up.z + sample.z * nt.z,
            )
            .normalize();
            (sampled_direction, LobeType::DiffuseReflection)
        } else {
            let reflection_direction = util::reflect(-view_direction, normal);
            let sampled_direction = util::sample_ggx(
                rng_sample.x,
                rng_sample.y,
                reflection_direction,
                self.roughness,
            );
            (sampled_direction, LobeType::SpecularReflection)
        };

        let cos_theta = normal.dot(sampled_direction).max(util::EPS);
        let halfway = (view_direction + sampled_direction).normalize();

        let f0 = Vec3::splat(DIELECTRIC_F0).lerp(self.albedo, self.metallic);
        let ks = util::fresnel_schlick(halfway.dot(view_direction).max(0.0), f0);

        let (sampled_direction, sampled_lobe, pdf, spectrum) = if sampled_lobe == LobeType::DiffuseReflection {
            let pdf = self.pdf_diffuse_fast(cos_theta);
            let spectrum = self.evaluate_diffuse_fast(cos_theta, specular_weight, ks);
            (sampled_direction, LobeType::DiffuseReflection, pdf, spectrum)
        } else {
            let d_term = util::ggx_distribution(normal, halfway, self.roughness);
            let pdf = self.pdf_specular_fast(view_direction, normal, halfway, d_term);
            let spectrum = self.evaluate_specular_fast(
                view_direction,
                normal,
                sampled_direction,
                cos_theta,
                d_term,
                specular_weight,
                ks,
            );
            (sampled_direction, LobeType::SpecularReflection, pdf, spectrum)
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
        if lobe_type == LobeType::DiffuseReflection {
            let cos_theta = normal.dot(sample_direction).max(0.0);
            self.pdf_diffuse_fast(cos_theta)
        } else {
            let halfway = (view_direction + sample_direction).normalize();
            let d_term = util::ggx_distribution(normal, halfway, self.roughness);
            self.pdf_specular_fast(view_direction, normal, halfway, d_term)
        }
    }
}

pub fn get_pbr_bsdf(config: &TracingConfig, material: &MaterialData, uv: Vec2, atlas: &Image!(2D, type=f32, sampled), sampler: &Sampler) -> PBR {
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

    // Clamp values to avoid NaNs :P
    let roughness = roughness.max(util::EPS);
    let metallic = metallic.min(1.0 - util::EPS);

    PBR {
        albedo,
        roughness,
        metallic,
        specular_weight_clamp: config.specular_weight_clamp,
    }
}