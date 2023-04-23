#![no_std]

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MyVec
{
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32
}