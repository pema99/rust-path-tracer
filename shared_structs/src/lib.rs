#![no_std]

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4, Vec4Swizzles, Vec2};

mod image_polyfill;
pub use image_polyfill::polyfill::{Image, Sampler};
#[cfg(not(target_arch = "spirv"))]
pub use image_polyfill::polyfill::CpuImage;


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TracingConfig {
    pub cam_position: Vec4,
    pub cam_rotation: Vec4,
    pub width: u32,
    pub height: u32,
    pub min_bounces: u32,
    pub max_bounces: u32,
    pub sun_direction: Vec4,
    pub nee: u32,
    pub has_skybox: u32,
    pub specular_weight_clamp: Vec2,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self { 
            cam_position: Vec4::new(0.0, 1.0, -5.0, 0.0),
            cam_rotation: Vec4::ZERO,
            width: 1280,
            height: 720,
            min_bounces: 3,
            max_bounces: 4,
            sun_direction: Vec3::new(0.5, 1.3, 1.0).normalize().extend(15.0),
            nee: 0,
            has_skybox: 0,
            specular_weight_clamp: Vec2::new(0.1, 0.9),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct MaterialData { // each Vec4 is either a color or an atlas location
    pub emissive: Vec4,
    pub albedo: Vec4,
    pub roughness: Vec4,
    pub metallic: Vec4,
    pub normals: Vec4,
    has_albedo_texture: u32,
    has_metallic_texture: u32,
    has_roughness_texture: u32,
    has_normal_texture: u32,
}

impl MaterialData {
    pub fn has_albedo_texture(&self) -> bool {
        self.has_albedo_texture != 0
    }

    pub fn set_has_albedo_texture(&mut self, has_albedo_texture: bool) {
        self.has_albedo_texture = if has_albedo_texture { 1 } else { 0 };
    }

    pub fn has_metallic_texture(&self) -> bool {
        self.has_metallic_texture != 0
    }

    pub fn set_has_metallic_texture(&mut self, has_metallic_texture: bool) {
        self.has_metallic_texture = if has_metallic_texture { 1 } else { 0 };
    }

    pub fn has_roughness_texture(&self) -> bool {
        self.has_roughness_texture != 0
    }

    pub fn set_has_roughness_texture(&mut self, has_roughness_texture: bool) {
        self.has_roughness_texture = if has_roughness_texture { 1 } else { 0 };
    }

    pub fn has_normal_texture(&self) -> bool {
        self.has_normal_texture != 0
    }

    pub fn set_has_normal_texture(&mut self, has_normal_texture: bool) {
        self.has_normal_texture = if has_normal_texture { 1 } else { 0 };
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct PerVertexData {
    pub vertex: Vec4,
    pub normal: Vec4,
    pub tangent: Vec4,
    pub uv0: Vec2,
    pub uv1: Vec2,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct LightPickEntry {
    pub triangle_index_a: u32,
    pub triangle_area_a: f32,
    pub triangle_pick_pdf_a: f32,
    pub triangle_index_b: u32,
    pub triangle_area_b: f32,
    pub triangle_pick_pdf_b: f32,
    pub ratio: f32,
}

// wgpu doesn't allow 0-sized buffers, so we use negative ratios to indicate sentinel values
impl LightPickEntry {
    pub fn is_sentinel(&self) -> bool {
        self.ratio < 0.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BVHNode {
    aabb_min: Vec4, // w = triangle count
    aabb_max: Vec4, // w = left_node if triangle_count is 0, first_triangle_index if triangle_count is 1
}

impl Default for BVHNode {
    fn default() -> Self {
        Self {
            aabb_min: Vec4::new(f32::INFINITY, f32::INFINITY, f32::INFINITY, 0.0),
            aabb_max: Vec4::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0),
        }
    }
}

impl BVHNode {
    // Immutable access
    pub fn triangle_count(&self) -> u32 {
        unsafe { core::mem::transmute(self.aabb_min.w) }
    }

    pub fn left_node_index(&self) -> u32 {
        unsafe { core::mem::transmute(self.aabb_max.w) }
    }

    pub fn right_node_index(&self) -> u32 {
        self.left_node_index() + 1
    }

    pub fn first_triangle_index(&self) -> u32 {
        unsafe { core::mem::transmute(self.aabb_max.w) }
    }

    pub fn aabb_min(&self) -> Vec3 {
        self.aabb_min.xyz()
    }

    pub fn aabb_max(&self) -> Vec3 {
        self.aabb_max.xyz()
    }

    pub fn is_leaf(&self) -> bool {
        self.triangle_count() > 0
    }

    // Mutable access
    pub fn set_triangle_count(&mut self, triangle_count: u32) {
        self.aabb_min.w = unsafe { core::mem::transmute(triangle_count) };
    }

    pub fn set_left_node_index(&mut self, left_node_index: u32) {
        self.aabb_max.w = unsafe { core::mem::transmute(left_node_index) };
    }

    pub fn set_first_triangle_index(&mut self, first_triangle_index: u32) {
        self.aabb_max.w = unsafe { core::mem::transmute(first_triangle_index) };
    }

    pub fn set_aabb_min(&mut self, aabb_min: &Vec3) {
        self.aabb_min.x = aabb_min.x;
        self.aabb_min.y = aabb_min.y;
        self.aabb_min.z = aabb_min.z;
    }

    pub fn set_aabb_max(&mut self, aabb_max: &Vec3) {
        self.aabb_max.x = aabb_max.x;
        self.aabb_max.y = aabb_max.y;
        self.aabb_max.z = aabb_max.z;
    }
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum NextEventEstimation {
    None,
    MultipleImportanceSampling,
    DirectLightSampling,
}

impl core::fmt::Debug for NextEventEstimation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NextEventEstimation::None => write!(f, "None"),
            NextEventEstimation::MultipleImportanceSampling => write!(f, "MIS"),
            NextEventEstimation::DirectLightSampling => write!(f, "Direct only"),
        }
    }
}

impl NextEventEstimation {
    pub fn to_u32(self) -> u32 {
        match self {
            NextEventEstimation::None => 0,
            NextEventEstimation::MultipleImportanceSampling => 1,
            NextEventEstimation::DirectLightSampling => 2,
        }
    }

    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => NextEventEstimation::None,
            1 => NextEventEstimation::MultipleImportanceSampling,
            2 => NextEventEstimation::DirectLightSampling,
            _ => NextEventEstimation::None,
        }
    }

    pub fn uses_mis(&self) -> bool {
        self == &NextEventEstimation::MultipleImportanceSampling
    }

    pub fn uses_nee(&self) -> bool {
        self != &NextEventEstimation::None
    }
}