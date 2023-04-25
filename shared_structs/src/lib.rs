#![no_std]

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct TracingConfig {
    pub width: u32,
    pub height: u32,
    pub max_bounces: u32,
}
