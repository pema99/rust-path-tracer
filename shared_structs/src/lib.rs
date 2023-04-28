#![no_std]

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4, Vec4Swizzles};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct TracingConfig {
    pub width: u32,
    pub height: u32,
    pub max_bounces: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct BVHNode {
    aabb_min: Vec4, // w = triangle count
    aabb_max: Vec4, // w = left_node if triangle_count is 0, first_triangle_index if triangle_count is 1
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

pub struct BVHReference<'a> {
    pub nodes: &'a [BVHNode],
    pub indirect_indices: &'a [u32],
}