#![feature(int_roundings)]

pub mod window;
pub mod trace;
pub mod bvh;
pub mod atlas;
pub mod asset;

use window::open_window;

fn main() {
    open_window();
}
