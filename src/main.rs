#![feature(int_roundings)]

pub mod window;
pub mod trace;
pub mod bvh;
pub mod atlas;
pub mod asset;

fn main() {
    window::open_window();
}