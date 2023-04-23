#![no_std]

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TracingConfig {
    pub width: u32,
    pub height: u32
}